import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')


def plot_training_results(json_path: str):
    df = pd.read_json(json_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.lineplot(data=df, x='epoch', y='train_mse', label='Train', ax=ax1)
    sns.lineplot(data=df, x='epoch', y='val_mse', label='Val', ax=ax1, linestyle='--')
    ax1.set_title('Convergence (MSE)')

    mem_col = 'memory_gpu_mb' if df['memory_gpu_mb'].max() > 0 else 'memory_ram_mb'
    label = "GPU Memory (MB)" if mem_col == 'memory_gpu_mb' else "System RAM (MB)"

    sns.lineplot(data=df, x='epoch', y=mem_col, color='green', ax=ax2)
    ax2.set_title(f'Computational Efficiency ({label})')
    ax2.set_ylabel('MB')

    plt.tight_layout()
    plt.savefig(Path(json_path).stem + "_plot.png", dpi=300)


if __name__ == "__main__":
    import glob

    log_files = glob.glob('logs/*.json')
    if log_files:
        latest = max(log_files, key=lambda x: Path(x).stat().st_mtime)
        plot_training_results(latest)