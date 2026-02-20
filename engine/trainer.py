import torch
import torch.optim as optim
import gc
from torch import nn
from torch.utils.data import DataLoader


class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.logs = []

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        torch.cuda.empty_cache()
        gc.collect()
        return total_loss / len(loader)

    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                total_loss += self.criterion(output, y).item()
        return total_loss / len(loader)

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float):
        metrics = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "memory_allocated": torch.cuda.memory_allocated()
        }
        self.logs.append(metrics)
        print(f"Epoch {epoch} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")