import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_equity_curve(results_dir, checkpoint_name):
    equity_curve = np.load(os.path.join(results_dir, "equity_curve.npy"))
    # Normalize to starting balance of 10,000 for cleaner visualization if needed, 
    # but the data likely already starts at 10,000.
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label=f"{checkpoint_name} Equity", color='#00ff88', linewidth=2)
    plt.title(f"NAS100 Equity Curve - {checkpoint_name}", fontsize=14, pad=20)
    plt.xlabel("Steps (15m intervals)", fontsize=12)
    plt.ylabel("Balance ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # Modern aesthetics
    plt.gca().set_facecolor('#1a1a1a')
    plt.gcf().set_facecolor('#1a1a1a')
    plt.gca().tick_params(colors='white')
    plt.gca().xaxis.label.set_color('white')
    plt.gca().yaxis.label.set_color('white')
    plt.gca().title.set_color('white')
    for spine in plt.gca().spines.values():
        spine.set_color('#444444')
    
    save_path = f"/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/equity_plot_{checkpoint_name.replace(',', '').replace(' ', '_')}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")
    return save_path

if __name__ == "__main__":
    checkpoints = [
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_141800", "17M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_161219", "35M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_203734", "60M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_203555", "65M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_203235", "75M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_203047", "80M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_202905", "93M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_203905", "40M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_161428", "43M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_141525", "22M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_142549", "24M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_160147", "40.5M (0% Conf)"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_160404", "30M Checkpoint"),
        ("/Users/gervaciusjr/Desktop/AI Trading Bot/NAS100/results/20251223_155921", "40.5M Checkpoint")
    ]
    
    for folder, name in checkpoints:
        if os.path.exists(folder):
            plot_equity_curve(folder, name)
        else:
            print(f"Folder not found: {folder}")
