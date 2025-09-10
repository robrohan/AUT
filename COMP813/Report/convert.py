from cifar10craw.cifar10c import CIFAR10C
from datasets import dataset
import os

os.makedirs("./temp", exist_ok=True)
builder = CIFAR10C()
builder.download_and_prepare()
dataset = builder.as_dataset(split="test")
dataset.save_to_disk("./cifar10-c-parquet")
