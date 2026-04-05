import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. Load the data
# If you have the data in a file, use: df = pd.read_csv('your_stats_file.csv')
# Using the raw data provided in your prompt
# (Replace the string below with your actual file path if needed)
df = pd.read_csv('outputs/bdd100k_UKAN_ddp/log.csv') 

# 2. Set up the plotting area (1 row, 2 columns)
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Training and Validation Loss Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=300)

# --- Graph 2: Val Dice and Val IoU ---
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['val_dice'], label='Val Dice', color='#2ca02c', linewidth=2)
plt.plot(df['epoch'], df['val_iou'], label='Val IoU', color='#d62728', linestyle='--', linewidth=2)
plt.title('Validation Accuracy Metrics (Dice & IoU)', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Metric Score (0-1)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('metrics_plot.png', dpi=300)
