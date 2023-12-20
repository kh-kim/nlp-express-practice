import os
from tqdm import tqdm

import wandb

import torch


class Trainer:

    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config

    def _train(self, train_loader):
        # Set model to train mode
        self.model.train()

        epoch_loss = 0

        for batch in tqdm(train_loader):
            # Initialize gradient
            self.optimizer.zero_grad()

            loss = self.model.compute_loss(batch, self.criterion)
            loss.backward() # Compute gradient

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.optimizer.step() # Update parameters

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)
    
    def _validate(self, valid_loader):
        # Set model to evaluation mode
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader):
                loss = self.model.compute_loss(batch, self.criterion)

                epoch_loss += loss.item()

        return epoch_loss / len(valid_loader)

    def train(self, train_loader, valid_loader, model_name):
        best_valid_loss = float("inf")

        for epoch_idx in range(self.config.n_epochs):
            train_loss = self._train(train_loader)
            valid_loss = self._validate(valid_loader)

            print(f"Epoch {epoch_idx + 1} | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}")

            wandb.log({
                "train/loss": train_loss,
                "eval/loss": valid_loss,
            })

            # Save best model, if best_valid_loss is updated
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({
                    "model": self.model.state_dict(),
                    "config": self.config,
                    "label2idx": train_loader.dataset.label2idx,
                    "idx2label": train_loader.dataset.idx2label,
                }, os.path.join(self.config.output_dir, f"{model_name}.pt"))

        # Load best model
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.output_dir, f"{model_name}.pt"))["model"]
        )

        return self.model

    def test(self, test_loader):
        self.model.eval()

        epoch_loss = 0

        total_correct_cnt = 0
        total_sample_cnt = 0

        with torch.no_grad():
            for batch in tqdm(test_loader):
                loss = self.model.compute_loss(batch, self.criterion)

                epoch_loss += loss.item()

                log_prob = self.model(
                    batch["input_ids"].to(
                        next(self.model.parameters()).device
                    )
                )
                # |log_prob| = (batch_size, output_dim)

                y_hat = torch.argmax(log_prob, -1)
                # |y_hat| = (batch_size, )

                correct_cnt = torch.sum(batch["labels"].to(y_hat.device) == y_hat)
                sample_cnt = len(batch["labels"])

                total_correct_cnt += correct_cnt.item()
                total_sample_cnt += sample_cnt

        print(f"Test Loss: {epoch_loss / len(test_loader):.3f}")
        print(f"Test Accuracy: {total_correct_cnt / total_sample_cnt * 100:.2f}%")
        print(f"Correct / Total: {total_correct_cnt} / {total_sample_cnt}")

        wandb.log({
            "test/loss": epoch_loss / len(test_loader),
            "test/accuracy": total_correct_cnt / total_sample_cnt * 100,
        })
