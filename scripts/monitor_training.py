#!/usr/bin/env python3
"""
PPO Training Stability Monitor

Plots key metrics to monitor training health:
- KL Divergence: Should stay under 0.02 (PPO target). Spikes indicate unstable updates.
- Entropy: Should gradually decrease. Too fast = premature convergence.
- Value Loss: Should decrease over time.
- Explained Variance: Should approach 1.0.

Usage:
    python scripts/monitor_training.py [log_dir]
    
If no log_dir provided, uses the most recent one in models/agent/ppo_logs/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def find_latest_log_dir():
    """Find the most recent PPO log directory."""
    log_base = Path("models/agent/ppo_logs")
    if not log_base.exists():
        raise FileNotFoundError("No ppo_logs directory found")
    
    dirs = sorted([d for d in log_base.iterdir() if d.is_dir()], reverse=True)
    if not dirs:
        raise FileNotFoundError("No log directories found")
    
    return dirs[0]

def plot_training_stability(log_dir: Path):
    """Plot training stability metrics."""
    progress_file = log_dir / "progress.csv"
    
    if not progress_file.exists():
        raise FileNotFoundError(f"progress.csv not found in {log_dir}")
    
    df = pd.read_csv(progress_file)
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PPO Training Stability Monitor\n{log_dir.name}', fontsize=14, fontweight='bold')
    
    # Get timesteps for x-axis
    if 'time/total_timesteps' in df.columns:
        timesteps = df['time/total_timesteps'] / 1e6  # In millions
        xlabel = 'Timesteps (M)'
    else:
        timesteps = df.index
        xlabel = 'Update'
    
    # 1. KL Divergence (Most Important!)
    ax1 = axes[0, 0]
    if 'train/approx_kl' in df.columns:
        kl = df['train/approx_kl'].dropna()
        ts = timesteps[:len(kl)]
        ax1.plot(ts, kl, 'b-', alpha=0.7, linewidth=0.5)
        ax1.axhline(y=0.02, color='r', linestyle='--', label='Target KL (0.02)')
        ax1.axhline(y=0.05, color='orange', linestyle='--', label='Warning (0.05)')
        
        # Rolling mean
        if len(kl) > 100:
            kl_smooth = kl.rolling(100).mean()
            ax1.plot(ts, kl_smooth, 'b-', linewidth=2, label='Rolling Mean (100)')
        
        ax1.set_title('KL Divergence (Stability Indicator)', fontweight='bold')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Approx KL')
        ax1.legend()
        ax1.set_ylim(0, min(0.1, kl.max() * 1.1))
        
        # Add status indicator
        recent_kl = kl.tail(100).mean() if len(kl) > 100 else kl.mean()
        status = "✅ STABLE" if recent_kl < 0.02 else ("⚠️ WARNING" if recent_kl < 0.05 else "🚨 UNSTABLE")
        ax1.text(0.98, 0.95, f'{status}\nRecent: {recent_kl:.4f}', 
                 transform=ax1.transAxes, ha='right', va='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, 'No KL data available', ha='center', va='center')
    
    # 2. Entropy (Exploration vs Exploitation)
    ax2 = axes[0, 1]
    if 'train/entropy_loss' in df.columns:
        entropy = -df['train/entropy_loss'].dropna()  # Negate because it's stored as negative
        ts = timesteps[:len(entropy)]
        ax2.plot(ts, entropy, 'g-', alpha=0.7, linewidth=0.5)
        
        if len(entropy) > 100:
            entropy_smooth = entropy.rolling(100).mean()
            ax2.plot(ts, entropy_smooth, 'g-', linewidth=2, label='Rolling Mean (100)')
        
        ax2.set_title('Policy Entropy (Exploration)', fontweight='bold')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Entropy')
        ax2.legend()
        
        # Add trend indicator
        if len(entropy) > 100:
            start_ent = entropy.head(100).mean()
            end_ent = entropy.tail(100).mean()
            change = (end_ent - start_ent) / start_ent * 100
            trend = f"↓ {abs(change):.1f}%" if change < 0 else f"↑ {change:.1f}%"
            ax2.text(0.98, 0.95, f'Trend: {trend}\nCurrent: {end_ent:.3f}', 
                     transform=ax2.transAxes, ha='right', va='top',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No entropy data available', ha='center', va='center')
    
    # 3. Value Loss
    ax3 = axes[1, 0]
    if 'train/value_loss' in df.columns:
        vloss = df['train/value_loss'].dropna()
        ts = timesteps[:len(vloss)]
        ax3.plot(ts, vloss, 'purple', alpha=0.7, linewidth=0.5)
        
        if len(vloss) > 100:
            vloss_smooth = vloss.rolling(100).mean()
            ax3.plot(ts, vloss_smooth, 'purple', linewidth=2, label='Rolling Mean (100)')
        
        ax3.set_title('Value Loss', fontweight='bold')
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel('Value Loss')
        ax3.set_yscale('log')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No value loss data available', ha='center', va='center')
    
    # 4. Episode Reward
    ax4 = axes[1, 1]
    if 'rollout/ep_rew_mean' in df.columns:
        rewards = df['rollout/ep_rew_mean'].dropna()
        ts = timesteps[:len(rewards)]
        ax4.plot(ts, rewards, 'orange', alpha=0.7, linewidth=0.5)
        
        if len(rewards) > 100:
            rewards_smooth = rewards.rolling(100).mean()
            ax4.plot(ts, rewards_smooth, 'orange', linewidth=2, label='Rolling Mean (100)')
        
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_title('Episode Reward', fontweight='bold')
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel('Mean Episode Reward')
        ax4.legend()
        
        # Add trend indicator
        if len(rewards) > 100:
            start_rew = rewards.head(100).mean()
            end_rew = rewards.tail(100).mean()
            ax4.text(0.98, 0.95, f'Start: {start_rew:.2f}\nCurrent: {end_rew:.2f}', 
                     transform=ax4.transAxes, ha='right', va='top',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No reward data available', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save figure
    output_path = log_dir / "training_stability.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING STABILITY SUMMARY")
    print("="*60)
    
    if 'train/approx_kl' in df.columns:
        kl = df['train/approx_kl'].dropna()
        print(f"\n📊 KL Divergence:")
        print(f"   Mean: {kl.mean():.4f}")
        print(f"   Max:  {kl.max():.4f}")
        print(f"   Spikes >0.05: {(kl > 0.05).sum()} / {len(kl)}")
        
        if kl.max() > 0.05:
            print("   ⚠️  HIGH KL SPIKES detected! Consider:")
            print("      - Reducing learning rate")
            print("      - Reducing reward_scaling")
            print("      - Increasing batch size")
    
    if 'train/entropy_loss' in df.columns:
        entropy = -df['train/entropy_loss'].dropna()
        print(f"\n📊 Entropy:")
        print(f"   Start: {entropy.head(100).mean():.3f}")
        print(f"   End:   {entropy.tail(100).mean():.3f}")
    
    if 'rollout/ep_rew_mean' in df.columns:
        rewards = df['rollout/ep_rew_mean'].dropna()
        print(f"\n📊 Episode Reward:")
        print(f"   Start: {rewards.head(100).mean():.2f}")
        print(f"   End:   {rewards.tail(100).mean():.2f}")
        print(f"   Max:   {rewards.max():.2f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_dir = Path(sys.argv[1])
    else:
        log_dir = find_latest_log_dir()
    
    print(f"📂 Analyzing: {log_dir}")
    plot_training_stability(log_dir)
