import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import glob
import json
import matplotlib.pyplot as plt

# Path settings
json_dir = 'results'
pattern = os.path.join(json_dir, 'ablation_config_*.json')
json_files = sorted(glob.glob(pattern))
print(f"Found {len(json_files)} JSON files.")
# Load all results
results = []
for path in json_files:
    with open(path, 'r') as f:
        data = json.load(f)
    cfg = data['config']
    label = f"P{cfg['position']}_C{cfg['community']}_O{cfg['objects']}_E{cfg['edge_weight']}"
    results.append({
        'config': label,
        'gcn_train_loss': data.get('gcn_train_loss', []),
        'gcn_val_acc': data.get('gcn_val_acc', []),
        'gcn_test_acc': data.get('gcn_test_acc', None),
        'sage_train_loss': data.get('sage_train_loss', []),
        'sage_val_acc': data.get('sage_val_acc', []),
        'sage_test_acc': data.get('sage_test_acc', None),
        'gine_train_loss': data.get('gine_train_loss', []),
        'gine_val_acc': data.get('gine_val_acc', []),
        'gine_test_acc': data.get('gine_test_acc', None),
    })

# Ensure results directory
out_dir = os.path.join(json_dir, 'ab_results')
os.makedirs(out_dir, exist_ok=True)

models = ['gcn', 'sage', 'gine']

# Plot train loss and val acc for each model with horizontal legends
for model in models:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    loss_lines, loss_labels = [], []
    acc_lines, acc_labels = [], []

    for res in results:
        epochs = range(1, len(res[f'{model}_train_loss']) + 1)
        # plot loss
        line_loss, = ax1.plot(
            epochs,
            res[f'{model}_train_loss'],
            label=f"{res['config']} loss"
        )
        loss_lines.append(line_loss)
        loss_labels.append(f"{res['config']} loss")

        # plot validation accuracy
        if res[f'{model}_val_acc']:
            line_acc, = ax2.plot(
                epochs,
                res[f'{model}_val_acc'],
                '--',
                label=f"{res['config']} acc"
            )
            acc_lines.append(line_acc)
            acc_labels.append(f"{res['config']} acc")

    # axis labels
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax2.set_ylabel('Validation Accuracy')

    # # Horizontal legend for loss at top center
    # ax1.legend(
    #     loss_lines,
    #     loss_labels,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 1.3),
    #     ncol=5,
    #     title='Loss'
    # )
    # Horizontal legend for validation accuracy at bottom center
    ax2.legend(
        acc_lines+loss_lines,
        acc_labels+loss_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=5,
    )

    fig.suptitle(f'{model.upper()} Training Loss & Validation Accuracy')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, f'{model}_train_val_horizontal_legends.png'))
    plt.close(fig)


# Plot combined test accuracy for all models
configs = [res['config'] for res in results]
indices = np.arange(len(configs))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
# Bars for each model
bars_gcn = ax.bar(indices - bar_width, [res['gcn_test_acc'] for res in results],
                  width=bar_width, label='GCN')
bars_sage = ax.bar(indices, [res['sage_test_acc'] for res in results],
                   width=bar_width, label='GraphSAGE')
bars_gine = ax.bar(indices + bar_width, [res['gine_test_acc'] for res in results],
                   width=bar_width, label='GINE')

# Annotate each bar with its height
for bar in list(bars_gcn) + list(bars_sage) + list(bars_gine):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:.3f}',
        ha='center',
        va='bottom'
    )


ax.set_xlabel('Config')
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy per Config for All Models')
ax.set_xticks(indices)
ax.set_xticklabels(configs, rotation=45, ha='right')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'all_models_test_acc.png'))
plt.close(fig)

print(f"Plots saved in {out_dir}")