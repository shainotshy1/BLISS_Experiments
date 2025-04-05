import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import floor, ceil

def display_eval(df, target_protein = None):
    if target_protein is not None:
        df = df[df['protein_name'] == target_protein + '.pdb']
    rewards_eval = df['rewards_eval'].values
    scRMSD_eval = df['gen_true_bb_rmsd'].values
    loglikelihoods = df['loglikelihood'].values

    pred_ddg_med = np.median(rewards_eval)
    pos_ddg_prop = np.mean(rewards_eval > 0)
    scRMSD_med = np.median(scRMSD_eval)
    low_rmsd_prop = np.mean(scRMSD_eval < 2)
    success_rate = np.mean((rewards_eval > 0) & (scRMSD_eval < 2))
    log_likelihood_med = np.median(loglikelihoods)

    print(f"Pred-ddG (median)↑: {pred_ddg_med:.3f}")
    print(f"%(ddG > 0) (%)↑: {pos_ddg_prop * 100:.1f}")
    print(f"scRMSD (median)↓: {scRMSD_med:.3f}")
    print(f"%(scRMSD < 2)(%)↑: {low_rmsd_prop * 100:.1f}")
    print(f"Success Rate (%)↑: {success_rate * 100:.1f}")
    print(f"Log Likelihood (median): {log_likelihood_med:.3f}")

    stats = {
        "pred_ddg_med" : pred_ddg_med,
        "pos_ddg_prop" : pos_ddg_prop,
        "scRMSD_med" : scRMSD_med,
        "low_rmsd_prop" : low_rmsd_prop,
        "success_rate" : success_rate,
        "ll_med" : log_likelihood_med
    }
    return stats

def plot_line_prot_comp(key, groups, group_labels, item_labels, markers=None, title='', x_label='', y_label='', h_line_threshold=None, h_line_label=''):
    if markers == None:
        markers = ['o' for _ in range(len(groups))]
    for group, label, marker in zip(groups, group_labels, markers):
        values = [t[key] for t in group]
        plt.style.use('default')
        plt.plot(item_labels, values, marker=marker, label=label)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    if h_line_threshold is not None:
        plt.axhline(y=0, color='#E06455', linestyle='--', label=h_line_label)
    plt.legend()
    plt.show()

def analyze_protein_gen_helper(protein_name, dfs, dfs_labels, clrs, key, y_label='', v_line_thresh=None, v_line_label=''):
    energy_points = None
    group_labels = []
    for df, label in zip(dfs, dfs_labels):
      pred_ddg = df[df['protein_name'] == protein_name + '.pdb'][key].values
      group_labels.extend([label] * pred_ddg.shape[0])
      energy_points = pred_ddg if (energy_points is None) else np.concatenate((energy_points, pred_ddg))

    data = pd.DataFrame({
    'Energy': np.array(energy_points),
    'Group': group_labels
    })
    plt.style.use('default')
    plt.figure(figsize=(8, 6))
    for label, clr in zip(dfs_labels, clrs):
      sns.kdeplot(data=data[data['Group'] == label], x='Energy', color=clr, label=label, fill=True, alpha=0.4, bw_adjust=0.8)

    if v_line_thresh is not None:
      plt.axvline(x=v_line_thresh, color='#E06455', linestyle='--', label=v_line_label)

    plt.xlabel(y_label, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='', loc='upper right', fontsize=10)
    
    plt.show()

def plot_reward_comparison(iterations, rewards, colors, linestyles, labels, title, reward_label):
  plt.style.use('default')
  fig, ax = plt.subplots()
  ax.grid(color='gray', linewidth=0.5)
  for i, reward in enumerate(rewards):
      ax.plot(iterations, reward, color=colors[i], label=labels[i], linestyle=linestyles[i])
  ax.set_ylabel(reward_label)
  ax.set_xlabel('Diffusion Iteration')
  ax.tick_params(axis='y', labelcolor='black')
  all_reward = np.concatenate(rewards)
  ax.set_ylim(min(all_reward), max(all_reward))
  ax.set_xlim(min(iterations), max(iterations))
  y_min = floor(min(all_reward) * 2) / 2
  y_max = ceil(max(all_reward) * 2) / 2
  ax.set_yticks(np.linspace(y_min, y_max, num=5))

  num_items = len(colors)
  ncol = num_items if num_items <= 3 else num_items // 2
  plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=ncol, frameon=False)
  plt.title(title, pad=40)
  plt.tight_layout()
  plt.show()
