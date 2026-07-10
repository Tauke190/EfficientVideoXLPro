"""Periodic MLVU evaluation of training checkpoints via the lmms-eval harness.

Registered from train.py when --mlvu_eval_on_save is set; the knobs it reads
(mlvu_eval_*) are declared on that module's TrainingArguments.
"""

import json
import os
import pathlib

import torch
import transformers

from videoxlpro.utils import rank0_print


class MLVUCheckpointEvalCallback(transformers.TrainerCallback):
    """Run the real lmms-eval MLVU harness against each saved checkpoint, in a subprocess.

    Why a subprocess rather than evaluating the live model: lmms-eval's VideoXLPro
    wrapper loads via load_image_processor(), which calls tokenizer.add_tokens() and
    casts the vision tower to fp16 -- either would corrupt a bf16 run in progress --
    and generating under Zero-3 would gather sharded weights inside the live engine
    under inference_mode, poisoning the params for the next backward. A separate
    process reuses the exact eval path the published numbers came from and cannot
    touch training state. The cost is one checkpoint write per eval.

    Ordering is safe: Trainer fires on_save AFTER save_model() and after
    _rotate_checkpoints(), so the directory is complete and the newest checkpoint is
    never the one rotation deletes. With save_total_limit=N an eval on checkpoint k
    stays on disk until save k+N.

    Rank discipline is the sharp edge. on_save runs on EVERY rank. Only rank 0 spawns,
    but every rank must sit in the barriers below -- otherwise the others return, walk
    into the next Zero-3 all-gather, and NCCL aborts the whole job once the process
    group's ddp_timeout (default 1800s) expires. An MLVU pass takes longer than that,
    so --ddp_timeout must comfortably exceed the eval wall clock.

    empty_cache() is not optional. The training procs keep their allocator high-water
    mark reserved from the driver while idle here, so the child OOMs on a GPU that is
    in fact mostly free. Both ranks must release before the spawn.
    """

    # torchrun exports these into our environment. lmms_eval reads RANK/WORLD_SIZE
    # straight off os.environ (evaluator.py) and accelerate re-derives its own, so a
    # child that inherits them mis-shards its work silently instead of crashing.
    _LAUNCHER_VARS = frozenset({
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
        "MASTER_ADDR", "MASTER_PORT", "OMP_NUM_THREADS",
    })

    def __init__(self, trainer, baseline_path=None):
        self._trainer = trainer
        self._cpu_pg = None
        # What --mlvu_eval_at_start scores: the weights training is about to start from.
        # Faithful even though the eval subprocess re-inits apt_patch_attn randomly --
        # apt_zero_conv is zero-init in weight AND bias, so it gates patch_attn's output
        # to exactly 0 at step 0 (llava_arch.get_apt_zero_conv). The step-0 number is
        # therefore "APT token merging, no learned compensation", which is the baseline
        # the training curve has to beat.
        self._baseline_path = baseline_path

    def _barrier(self):
        """Block every rank until rank 0's eval subprocess finishes, without using the GPU.

        Deliberately NOT the default NCCL group: an NCCL barrier launches a device
        kernel that spin-waits until all ranks arrive, so the idle ranks would burn SMs
        on the very GPUs the eval subprocess needs, for its whole runtime. A gloo group
        waits on the CPU instead. Created lazily on first use -- new_group() is itself a
        collective, and every rank reaches this method.

        The timeout must be passed: new_group() ignores the default group's timeout and
        falls back to torch's 30-minute default, which is shorter than an MLVU pass.
        """
        dist = torch.distributed
        if not (dist.is_available() and dist.is_initialized()):
            return
        if self._cpu_pg is None:
            from datetime import timedelta
            self._cpu_pg = dist.new_group(
                backend="gloo", timeout=timedelta(seconds=self._trainer.args.ddp_timeout)
            )
        dist.barrier(group=self._cpu_pg)

    @staticmethod
    def _free_port():
        import socket
        with socket.socket() as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @staticmethod
    def _repo_root():
        return pathlib.Path(__file__).resolve().parents[3]

    @classmethod
    def _eval_cwd(cls):
        # The cwd the working eval script (lmms-eval/scripts/eval_videoxl_pro.slurm) uses.
        return str(cls._repo_root() / "lmms-eval")

    def _child_env(self):
        env = {
            k: v for k, v in os.environ.items()
            if k not in self._LAUNCHER_VARS
            and not k.startswith("TORCHELASTIC_")
            and not k.startswith("ACCELERATE_")
        }
        # lmms_eval would otherwise attach to this run's wandb process and interleave
        # its own step counter with the trainer's.
        env["WANDB_MODE"] = "disabled"

        # The checkpoint's auto_map routes model loading through the CACHED HUB module
        # (transformers_modules/MINT-SJTU/Video-XL-Pro-3B/<hash>/modeling_*.py), which
        # imports the local package -- but which spelling it uses differs per machine,
        # because that cache is edited in place: `videoxlpro.model...` on one box,
        # `videoxlpro.videoxlpro.model...` on another. Putting the repo ROOT on the path
        # makes `videoxlpro` a namespace package spanning both, so either spelling
        # resolves. This is exactly what eval_videoxl_pro.slurm already exports, and cwd
        # alone does not supply it.
        root = str(self._repo_root())
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{root}{os.pathsep}{prev}" if prev else root
        return env

    def _model_args(self, pretrained):
        cfg = self._trainer.model.config
        args = [
            f"pretrained={pretrained}",
            f"max_frames_num={self._trainer.args.mlvu_eval_frames}",
            f"attn_implementation={self._trainer.args.attn_implementation}",
        ]
        # lmms_eval splits --model_args on commas, so threshold lists travel colon-separated.
        thresholds = ":".join(str(t) for t in getattr(cfg, "apt_thresholds", []) or [])
        if getattr(cfg, "use_apt", False):
            args += ["use_apt=True", f"apt_threshold={thresholds}", f"apt_num_scales={cfg.apt_num_scales}"]
        elif getattr(cfg, "use_apt_temporal", False):
            args += ["use_apt_temporal=True", f"apt_threshold={thresholds}",
                     f"apt_num_scales={cfg.apt_num_scales}", f"rlt_threshold={cfg.rlt_threshold}"]
        elif getattr(cfg, "use_rlt", False):
            args += ["use_rlt=True", f"rlt_threshold={cfg.rlt_threshold}"]
        return ",".join(args)

    def _build_cmd(self, pretrained, out_dir, args):
        import sys
        nproc = args.mlvu_eval_num_processes or max(1, args.world_size)
        cmd = [
            args.mlvu_eval_python or sys.executable,
            "-m", "accelerate.commands.launch",
            "--num_processes", str(nproc),
            "--main_process_port", str(self._free_port()),
            "-m", "lmms_eval",
            "--model", "videoxlpro",
            "--tasks", args.mlvu_eval_task,
            "--model_args", self._model_args(pretrained),
            "--batch_size", "1",
            "--log_samples",
            "--output_path", out_dir,
            "--verbosity", "WARNING",
        ]
        if args.mlvu_eval_limit > 0:
            cmd += ["--limit", str(args.mlvu_eval_limit)]
        return cmd

    def _parse(self, out_dir, task):
        """Aggregate score plus per-category accuracy recovered from the samples file.

        The harness only ever returns the macro-average; the per-category split lives
        in the jsonl. Worth recovering: APT merges visual tokens, so needleQA and count
        are where it would regress first, and a rising average can hide them falling.
        """
        out = pathlib.Path(out_dir)
        results = sorted(out.glob("**/*_results.json"))
        if not results:
            raise FileNotFoundError(f"no *_results.json under {out_dir}")
        with open(results[-1]) as f:
            blob = json.load(f)
        metrics = {"mlvu/macro_avg": blob["results"][task]["mlvu_percetion_score,none"]}

        samples = sorted(out.glob(f"**/*_samples_{task}.jsonl"))
        if samples:
            tally = {}
            with open(samples[-1]) as f:
                for line in f:
                    d = json.loads(line).get("mlvu_percetion_score")
                    if not isinstance(d, dict):
                        continue
                    hit, seen = tally.get(d["task_type"], (0, 0))
                    tally[d["task_type"]] = (hit + int(d["pred_answer"] == d["answer"]), seen + 1)
            for cat, (hit, seen) in sorted(tally.items()):
                metrics[f"mlvu/{cat}"] = 100.0 * hit / seen
            metrics["mlvu/n_scored"] = sum(seen for _, seen in tally.values())
            metrics["mlvu/n_categories"] = len(tally)
        return metrics

    @staticmethod
    def _write_child_logs(out_dir, proc):
        for name, text in (("eval_stdout.log", proc.stdout), ("eval_stderr.log", proc.stderr)):
            try:
                with open(os.path.join(out_dir, name), "w") as f:
                    f.write(text or "")
            except OSError:
                pass

    @staticmethod
    def _failure_report(step, exc, cmd, proc, out_dir):
        """Everything needed to diagnose the eval without re-running training."""
        import shlex
        lines = [f"[MLVU] step={step} eval FAILED, training continues: {exc}"]
        if cmd:
            lines.append("  rerun: " + " ".join(shlex.quote(c) for c in cmd))
        lines.append(f"  logs:  {os.path.join(out_dir, 'eval_stderr.log')}")
        stderr = getattr(proc, "stderr", None) or getattr(exc, "stderr", None)
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        if stderr:
            tail = stderr.strip().splitlines()[-30:]
            lines.append("  child stderr (last 30 lines):")
            lines += ["    " + ln for ln in tail]
        return "\n".join(lines)

    def _run_eval(self, step, pretrained, args):
        """Score `pretrained` (a checkpoint dir, or a hub id for the step-0 baseline).

        Every rank must enter this: only rank 0 spawns, but the barriers below are
        collectives. See the class docstring on why skipping them aborts the job.
        """
        import subprocess

        out_dir = os.path.join(args.output_dir, "mlvu_eval", f"step-{step}")

        # Hand the reserved-but-unused blocks back to the driver, then make sure BOTH
        # ranks have done so before anyone spawns a child onto these GPUs.
        torch.cuda.empty_cache()
        self._barrier()

        if args.local_rank in (-1, 0):
            rank0_print(f"[MLVU] step={step} evaluating {pretrained}")
            # Everything rank 0 does here is inside the try: an exception escaping
            # would strand the other ranks in the barrier below until ddp_timeout.
            cmd, proc = None, None
            try:
                os.makedirs(out_dir, exist_ok=True)
                cmd = self._build_cmd(pretrained, out_dir, args)
                proc = subprocess.run(
                    cmd,
                    env=self._child_env(),
                    cwd=self._eval_cwd(),
                    timeout=args.mlvu_eval_timeout,
                    capture_output=True,
                    text=True,
                )
                # lmms-eval catches evaluation errors, prints the traceback, and STILL
                # exits 0 (__main__.cli_evaluate, unless --verbosity=DEBUG), so a clean
                # return code proves nothing. Keep the child's output either way -- the
                # traceback is the only record of why an eval produced no results.
                self._write_child_logs(out_dir, proc)
                if proc.returncode != 0:
                    raise RuntimeError(f"lmms-eval exited {proc.returncode}")
                metrics = self._parse(out_dir, args.mlvu_eval_task)
            except Exception as e:
                # A broken eval must never take the training run down with it.
                rank0_print(self._failure_report(step, e, cmd, proc, out_dir))
                metrics = None

            if metrics:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(metrics, step=step)
                except Exception:
                    pass
                summary = " ".join(f"{k.split('/')[-1]}={v:.1f}" for k, v in sorted(metrics.items()))
                rank0_print(f"[step={step}] MLVU {summary}")

        # Rank 0 was gone for the whole eval; the others rejoin the training loop here.
        self._barrier()

    def on_train_begin(self, args, state, control, **kwargs):
        """Baseline before a single optimizer step, so the curve has a zero point.

        Cheapest eval of the run to run here: the model is sharded and the optimizer
        exists, but no activations have been allocated yet, so GPU headroom is at its
        maximum. Costs one MLVU pass of wall clock before training starts.
        """
        if not args.mlvu_eval_at_start:
            return
        if not self._baseline_path:
            rank0_print("[MLVU] --mlvu_eval_at_start set but no baseline path was passed; skipping.")
            return
        self._run_eval(state.global_step, self._baseline_path, args)

    def on_save(self, args, state, control, **kwargs):
        if not args.mlvu_eval_on_save:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self._run_eval(state.global_step, ckpt_dir, args)
