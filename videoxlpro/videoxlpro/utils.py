import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
import cv2
import requests

from videoxlpro.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
except ImportError:
    print("Please install pyav to use video processing functions.")

def process_video_with_pyav(video_file,scale,data_args):
    #print(video_file)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_file}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = total_frame_num / fps if fps > 0 else 0

    video_fps = data_args.video_fps

    avg_fps = round(fps / video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0 and len(frame_idx) > data_args.frames_upbound:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()

    # dict.fromkeys de-dups while preserving order: one frame per wanted index, which is
    # what the old membership test gave us (a repeated index could only ever match once).
    frame_idx = list(dict.fromkeys(frame_idx))

    # 读取视频帧
    video_frames = []
    timestamps = []
    if frame_idx:
        # Two ways to reach the wanted frames, and neither wins in both sampling regimes:
        #   seek       -- jump to each index; costs a keyframe jump + decode-forward per
        #                 frame, so it is flat in the video's length but scales with how
        #                 MANY frames we want.
        #   sequential -- walk the video once, grab() past unwanted frames (decode only,
        #                 no color-convert) and retrieve() only the ones we keep. Scales
        #                 with the video's length, not the frame count.
        # Seeking wins only when the wanted frames are further apart than a GOP, otherwise
        # the seeks land in the same GOPs we would have walked through anyway and repeat
        # that work. Measured on a 15.6k-frame h264 clip (seek vs sequential): 32 frames
        # 0.66s vs 2.65s, 128 frames 1.95s vs 1.83s, 522 frames 7.85s vs 2.30s -- so the
        # crossover sits near a stride of ~150. 200 keeps us on the safe side of it, and
        # both paths return bit-identical frames either way.
        stride = total_frame_num / len(frame_idx)
        if stride >= 200:
            for index in frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转换为 RGB
                timestamps.append(round(index / fps, 1))
        else:
            wanted = set(frame_idx)
            for index in range(frame_idx[-1] + 1):   # stop at the last frame we want
                if not cap.grab():
                    break
                if index in wanted:
                    ret, frame = cap.retrieve()
                    if not ret:
                        break
                    video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转换为 RGB
                    timestamps.append(round(index / fps, 1))
    cap.release()

    # 将帧堆叠为 numpy 数组
    video = np.stack(video_frames)
    #print(scale,'*',timestamps)
    if scale!=1:
        video_duration_seconds=video_duration_seconds*scale
        timestamps = (np.array(timestamps) * scale).tolist()
    #print(timestamps)
    #print(video_duration_seconds,timestamps)
    return video, video_duration_seconds,timestamps
# def process_video_with_pyav(video_file,scale,data_args):
#     container = av.open(video_file)
#     # !!! This is the only difference. Using auto threading
#     container.streams.video[0].thread_type = "AUTO"

#     video_frames = []
#     for packet in container.demux():
#         if packet.stream.type == 'video':
#             for frame in packet.decode():
#                 video_frames.append(frame)
    
#     if not video_frames:
#         print(f"Error: 无法从视频文件 {video_file} 中获取帧")
#         return None, None, None
    
#     total_frame_num = len(video_frames)
#     video_time = video_frames[-1].time  # 视频总时长(秒)
#     fps = total_frame_num / video_time if video_time > 0 else 0
    
#     video_fps = data_args.video_fps
#     avg_fps = round(fps / video_fps)
#     frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

#     if data_args.frames_upbound > 0 and len(frame_idx) > data_args.frames_upbound:
#         uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
#         frame_idx = uniform_sampled_frames.tolist()

#     # 获取选中的帧和时间戳
#     frames = []
#     timestamps = []
#     for idx in frame_idx:
#         frame = video_frames[idx]
#         frames.append(frame.to_ndarray(format="rgb24"))
#         timestamps.append(round(frame.time, 1))  # 使用帧的实际时间戳
    
#     video_duration_seconds = video_time
#     video = np.stack(frames)
#     #print(scale,timestamps)
#     if scale!=1:
#         video_duration_seconds=video_duration_seconds*scale
#         timestamps = (np.array(timestamps) * scale).tolist()
#     #print(timestamps)
#     return video, video_duration_seconds, timestamps

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
