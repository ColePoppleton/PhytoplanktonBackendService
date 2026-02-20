import json
import torch
import time
from pathlib import Path

class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_path = Path(log_dir)
        self.log_path.mkdir(exist_ok=True)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.metrics_file = self.log_path / f"metrics_{self.run_id}.json"
        self.history = []

    def log_step(self, epoch: int, stage: str, metrics: dict):
        entry = {
            "epoch": epoch,
            "stage": stage,
            "timestamp": time.time(),
            "gpu_mem_alloc": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "gpu_mem_res": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            **metrics
        }
        self.history.append(entry)
        self._save()

    def _save(self):
        with open(self.metrics_file, "w") as f:
            json.dump(self.history, f, indent=4)