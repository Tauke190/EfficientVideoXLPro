from videoxlpro.train.train import train
import os
import wandb
wandb.init(mode="disabled")

wandb.login(key='2b6c46299a43da99a3708813d1ffcd9a469cd192')
# 设置超时时间为60秒
os.environ["WANDB__SERVICE_WAIT"] = "600"
if __name__ == "__main__":
    train()
