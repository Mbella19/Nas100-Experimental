
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
plt.style.use('dark_background')
sns.set_context("poster")  # Larger fonts for readability

def create_evolution_dashboard():
    # 1. Data Definition
    # ------------------
    data = [
        {'Steps': 200_000, 'Return': 6.00, 'DD': 10.97, 'Trades': 6614, 'Phase': 'Junior'},
        {'Steps': 300_000, 'Return': 19.59, 'DD': 6.96, 'Trades': 2895, 'Phase': 'Junior'},
        {'Steps': 500_000, 'Return': 1.88, 'DD': 8.51, 'Trades': 746, 'Phase': 'Junior'},
        {'Steps': 1_000_000, 'Return': 1.17, 'DD': 3.63, 'Trades': 265, 'Phase': 'Mid'},
        {'Steps': 1_400_000, 'Return': 3.84, 'DD': 2.52, 'Trades': 229, 'Phase': 'Mid'},
        {'Steps': 1_500_000, 'Return': 9.27, 'DD': 1.84, 'Trades': 196, 'Phase': 'Golden'},
        {'Steps': 1_600_000, 'Return': 3.25, 'DD': 1.25, 'Trades': 134, 'Phase': 'Mid'},
        {'Steps': 1_700_000, 'Return': 2.32, 'DD': 2.26, 'Trades': 211, 'Phase': 'Mid'},
        {'Steps': 2_000_000, 'Return': 4.77, 'DD': 2.07, 'Trades': 144, 'Phase': 'Mid'},
        {'Steps': 2_500_000, 'Return': 9.25, 'DD': 2.89, 'Trades': 331, 'Phase': 'Mid'},
        {'Steps': 3_000_000, 'Return': -0.18, 'DD': 2.26, 'Trades': 174, 'Phase': 'Senior'},
        {'Steps': 3_500_000, 'Return': -0.13, 'DD': 3.25, 'Trades': 181, 'Phase': 'Senior'},
        {'Steps': 4_000_000, 'Return': -6.26, 'DD': 6.52, 'Trades': 182, 'Phase': 'Senior'},
        {'Steps': 4_400_000, 'Return': -5.24, 'DD': 7.33, 'Trades': 201, 'Phase': 'Senior'},
        {'Steps': 5_000_000, 'Return': 1.31, 'DD': 3.38, 'Trades': 204, 'Phase': 'Senior'},
        {'Steps': 6_000_000, 'Return': 1.31, 'DD': 4.15, 'Trades': 256, 'Phase': 'Senior'},
        {'Steps': 6_500_000, 'Return': 0.78, 'DD': 2.90, 'Trades': 142, 'Phase': 'Senior'},
        {'Steps': 7_000_000, 'Return': -2.80, 'DD': 3.72, 'Trades': 207, 'Phase': 'Senior'},
        {'Steps': 7_400_000, 'Return': 3.54, 'DD': 1.21, 'Trades': 196, 'Phase': 'Senior'},
    ]
    df = pd.DataFrame(data)
    
    stress_test_data = [
        {'Scenario': 'Baseline\n(Conf=95%)', 'Return': 9.27, 'DD': 1.84, 'Color': '#00ff00'},
        {'Scenario': 'No Filter\n(Conf=0%)', 'Return': 9.27, 'DD': 1.84, 'Color': '#00cc00'},
        {'Scenario': 'High Risk\n(7x Lev)', 'Return': 64.87, 'DD': 9.28, 'Color': '#ff3333'},
    ]
    df_stress = pd.DataFrame(stress_test_data)

    # 2. Setup Figure
    # ---------------
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

    # 3. Plot 1: Return Evolution (Top Left)
    # --------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    
    # Gradient line
    colors = ['red' if x < 0 else '#00ff00' for x in df['Return']]
    ax1.plot(df['Steps'], df['Return'], color='white', alpha=0.3, linewidth=2, zorder=1)
    scatter = ax1.scatter(df['Steps'], df['Return'], c=colors, s=150, zorder=2, edgecolors='white', linewidth=1.5)
    
    # Highlight 1.5M
    row_1_5m = df[df['Steps'] == 1_500_000].iloc[0]
    ax1.scatter([1_500_000], [row_1_5m['Return']], color='#00ff00', s=500, marker='*', zorder=3, label='The Champion (1.5M)')
    
    # Highlight 300k
    row_300k = df[df['Steps'] == 300_000].iloc[0]
    ax1.scatter([300_000], [row_300k['Return']], color='#ffaa00', s=300, marker='o', zorder=3, label='The Gambler (300k)')
    
    # Annotate points
    for idx, row in df.iterrows():
        if row['Steps'] in [300_000, 1_500_000, 7_400_000]:
            label = f"{row['Return']:+.2f}%"
            ax1.annotate(label, (row['Steps'], row['Return']), xytext=(0, 15), textcoords='offset points', ha='center', fontweight='bold', color='white')

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Evolution of Intelligence: Net Return by Checkpoint', fontsize=24, pad=20, color='#00ff00')
    ax1.set_ylabel('Net Return (%)', fontsize=18)
    ax1.set_xlabel('Training Steps', fontsize=18)
    ax1.grid(True, alpha=0.1)
    ax1.legend(loc='upper right', frameon=True, facecolor='black', framealpha=0.8)

    # Format X axis
    ax1.ticklabel_format(style='plain', axis='x')
    def human_format(num, pos):
        if num >= 1_000_000: return f'{num/1_000_000:.1f}M'
        return f'{num/1_000:.0f}k'
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(human_format))


    # 4. Plot 2: Max Drawdown (Bottom Left)
    # -------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Bar chart for DD
    bars = ax2.bar(range(len(df)), df['DD'], color='#ff3333', alpha=0.7)
    
    # Highlight 1.5M DD
    idx_1_5m = df.index[df['Steps'] == 1_500_000][0]
    bars[idx_1_5m].set_color('#00ff00')
    bars[idx_1_5m].set_alpha(1.0)
    
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([human_format(x,0) for x in df['Steps']], rotation=45, ha='right')
    ax2.set_title('Risk Profile: Max Drawdown (%)', fontsize=20, pad=15)
    ax2.set_ylabel('Max Drawdown (%)', fontsize=16)
    ax2.grid(True, axis='y', alpha=0.1)
    
    # Add value labels for key bars
    for i, bar in enumerate(bars):
        if df.iloc[i]['Steps'] in [300_000, 1_500_000, 7_400_000]:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', color='white', fontweight='bold')


    # 5. Plot 3: 1.5M Stress Test Results (Bottom Right)
    # --------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    
    x = np.arange(len(df_stress))
    width = 0.35
    
    rects1 = ax3.bar(x - width/2, df_stress['Return'], width, label='Return', color=df_stress['Color'])
    rects2 = ax3.bar(x + width/2, df_stress['DD'], width, label='Max DD', color='gray', alpha=0.5)
    
    ax3.set_title('Checkpoint 1.5M: Stress & Scalability Tests', fontsize=20, pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_stress['Scenario'], fontsize=14)
    ax3.set_ylabel('Percentage (%)', fontsize=16)
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.1)
    
    # Label bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3) if height >=0 else (0, -15),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # 6. Final Layout
    # ---------------
    plt.suptitle("NAS100 Hybrid Agent: Evaluation & Selection Report", fontsize=32, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.01, "Generated by Antigravity | Zero-Cost Out-of-Sample Testing", ha="center", fontsize=12, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = Path('results/evaluation_dashboard.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='black')
    print(f"Dashboard saved to {output_path.absolute()}")

if __name__ == "__main__":
    create_evolution_dashboard()
