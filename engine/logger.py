import json, torch, time, psutil, os
from pathlib import Path


class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_path = Path(log_dir)
        self.log_path.mkdir(exist_ok=True)
        self.metrics_file = self.log_path / f"metrics_{time.strftime('%Y%m%d-%H%M%S')}.json"
        self.history = []
        self.total_bytes_processed = 0

    def log_step(self, epoch: int, stage: str, metrics: dict, batch_bytes: int = 0):
        self.total_bytes_processed += batch_bytes

        gpu_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        ram_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        total_gb = self.total_bytes_processed / (1024 ** 3)

        entry = {
            "epoch": epoch,
            "stage": stage,
            "timestamp": time.time(),
            "memory_gpu_mb": round(gpu_mb, 2),
            "memory_ram_mb": round(ram_mb, 2),
            "data_processed_gb": round(total_gb, 4),
            **metrics
        }
        self.history.append(entry)
        with open(self.metrics_file, "w") as f:
            json.dump(self.history, f, indent=4)