from videoxlpro.train.train import train
import os
os.environ["WANDB__SERVICE_WAIT"] = "600"
if __name__ == "__main__":
    train()
