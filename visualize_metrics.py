import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)


def plot_training_results(csv_path: str):
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.lineplot(data=df, x='epoch', y='train_mse', label='Training Loss', ax=ax1, linewidth=2)
    sns.lineplot(data=df, x='epoch', y='val_mse', label='Validation Loss', ax=ax1, linewidth=2, linestyle='--')
    ax1.set_title('Model Convergence (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    sns.lineplot(data=df, x='epoch', y='gpu_mem_mb', color='green', ax=ax2, linewidth=2)
    ax2.set_title('Computational Efficiency')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('GPU Memory Usage (MB)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_filename = Path(csv_path).stem + "_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    log_file = "logs/SwinPredictor_metrics_1740071112.csv"
    if Path(log_file).exists():
        plot_training_results(log_file)
    else:
        print("Log file not found. Please ensure training has completed.")