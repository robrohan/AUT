"""
Simple training loop; Boilerplate that could apply to any arbitrary neural
network, so nothing in this file really has anything to do with GPT
specifically.
"""

import time
import math
from typing import Tuple
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from gpt_mini.utils import CfgNode as CN


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, val_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def perplexity(self, avg_loss) -> float:
        return math.exp(avg_loss)

    def calculate_accuracy(self, logits, y):
        # Generate predictions by taking the argmax of the logits
        predictions = torch.argmax(logits, dim=-1)
        # Compare predictions to labels and calculate accuracy
        correct_predictions = (predictions == y).sum().item()
        total_predictions = y.size(0)
        accuracy = correct_predictions / total_predictions
        return accuracy

    def validate(self, val_loader) -> Tuple[float, float, float]:
        model = self.model
        model.eval()  # Set the model to evaluation mode

        val_losses = []
        val_accuracies = []
        num_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = [t.to(self.device) for t in batch]
                x, y = batch
                logits, loss = model(x, y)
                val_losses.append(loss.item())
                num_tokens += x.size(0)

                # Calculate accuracy for the batch
                batch_accuracy = self.calculate_accuracy(logits, y)
                val_accuracies.append(batch_accuracy)

        # Calculate the average validation loss, perplexity, and accuracy
        avg_val_loss = sum(val_losses) / num_tokens
        val_pp = self.perplexity(avg_val_loss)
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)

        model.train()  # Set the model back to training mode
        return avg_val_loss, val_pp, avg_val_accuracy

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=self.config.max_sample_size
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        # keep track of the number of iterations between validations
        validation_interval = config.validation_interval

        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # Calculate perplexity for this batch
            avg_loss = self.loss.item() / x.size(0)  # Average loss per token
            batch_pp = self.perplexity(avg_loss)
            self.batch_pp = batch_pp
            # Calculate accuracy for the batch
            batch_accuracy = self.calculate_accuracy(logits, y)
            self.batch_accuracy = batch_accuracy

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            # Model validation
            if self.iter_num % validation_interval == 0:
                val_loss, val_pp, val_acc = self.validate(val_loader)
                self.val_loss = val_loss
                self.val_pp = val_pp
                self.val_acc = val_acc

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
