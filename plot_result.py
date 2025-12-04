import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os

# 配置
PRED_FILE = 'predictResult/cold_start_preds.npy'
LABEL_FILE = 'predictResult/cold_start_labels.npy'
SAVE_DIR = 'predictResult/plots'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 1. 加载数据
print("Loading results...")
y_pred = np.load(PRED_FILE)
y_true = np.load(LABEL_FILE)

print(f"Loaded {len(y_true)} samples.")

# 2. 设置绘图风格
sns.set(style="whitegrid")
plt.figure(figsize=(18, 5))

# --- 图1: 散点回归图 (Scatter Plot) ---
print("Plotting Scatter Plot...")
plt.subplot(1, 3, 1)
# 为了防止点太密集，随机采样 2000 个点来画散点，但回归线用全量数据
indices = np.random.choice(len(y_true), 2000, replace=False)
sns.regplot(x=y_true[indices], y=y_pred[indices], 
            scatter_kws={'alpha':0.3, 's':10, 'color':'#3498db'}, 
            line_kws={'color':'#e74c3c'})
plt.xlabel('True Frequency Label')
plt.ylabel('Predicted Score')
plt.title('True vs Predicted (Sampled Scatter)')

# --- 图2: ROC 曲线 (ROC Curve) ---
print("Plotting ROC Curve...")
plt.subplot(1, 3, 2)
# 二值化标签：大于0视为有副作用(Positive)，等于0视为无(Negative)
y_true_binary = (y_true > 0).astype(int)
fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# --- 图3: PR 曲线 (Precision-Recall Curve) ---
print("Plotting PR Curve...")
plt.subplot(1, 3, 3)
precision, recall, _ = precision_recall_curve(y_true_binary, y_pred)
pr_auc = average_precision_score(y_true_binary, y_pred)

plt.plot(recall, precision, color='#9b59b6', lw=2, label=f'PR curve (AUPR = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

# 保存
plt.tight_layout()
save_path = os.path.join(SAVE_DIR, 'cold_start_analysis.png')
plt.savefig(save_path, dpi=300)
print(f"所有图片已合并保存至: {save_path}")
plt.show()