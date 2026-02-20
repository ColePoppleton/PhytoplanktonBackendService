import torch
import torch.optim as optim
import gc
from torch import nn
from torch.utils.data import DataLoader
from engine.logger import TrainingLogger
from pathlib import Path


class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0
        epoch_bytes = 0
        for x, y in loader:
            batch_bytes = x.element_size() * x.nelement() + y.element_size() * y.nelement()
            epoch_bytes += batch_bytes

            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        torch.cuda.empty_cache()
        gc.collect()
        return total_loss / len(loader), epoch_bytes

    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                total_loss += self.criterion(output, y).item()
        return total_loss / len(loader)

    async def run_full_training(self, loaders, epochs=50, save_path=None, status_callback=None):
        logger = TrainingLogger()
        for epoch in range(1, epochs + 1):
            t_loss, processed_bytes = self.train_epoch(loaders["train"])
            v_loss = self.validate(loaders["val"])

            logger.log_step(
                epoch,
                "training",
                {"train_mse": t_loss, "val_mse": v_loss},
                batch_bytes=processed_bytes
            )

            print(f"Epoch {epoch} | Train Loss: {t_loss:.6f} | Val Loss: {v_loss:.6f}")

            if status_callback:
                await status_callback(epoch, epochs)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)

        return str(logger.metrics_file)