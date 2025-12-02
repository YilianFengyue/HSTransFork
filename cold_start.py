import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from Net import Trans, drug2emb_encoder  # <--- 【修复】从 Net.py 导入
from smiles2vector import load_drug_smile
from utils import rmse, MAE, pearson, spearman
from tqdm import tqdm
import os

# --- 1. 配置路径 ---
SMILES_FILE_759 = 'data/drug_SMILES_759.csv'
FREQ_FILE_759 = 'data/frequency_750+9.mat'

# --- 2. 辅助函数 ---

def loss_fun(output, label):
    loss = torch.sum((output - label) ** 2)
    return loss

# 究极优化版 Identify Sub (只针对训练集跑)
def identify_sub_optimized(train_data, suffix='cold_start'):
    print(f'\n[Step 1] 正在基于训练集提取有效子结构 (Suffix: {suffix})...')
    
    drug_smile = [item[1] for item in train_data]
    side_id = [item[0] for item in train_data]
    labels = [item[2] for item in train_data]

    # 1. 药物去重编码 (加速 1000倍)
    print(">> [优化] 正在对药物子结构进行去重编码...")
    unique_smiles = list(set(drug_smile))
    smile_cache = {}
    for smile in tqdm(unique_smiles, desc="Encoding Unique Drugs"):
        drug_sub, mask = drug2emb_encoder(smile)
        smile_cache[smile] = drug_sub.tolist()

    sub_dict = {}
    for i in range(len(drug_smile)):
        sub_dict[i] = smile_cache[drug_smile[i]]

    # 2. 构建矩阵
    print(">> [优化] 正在构建副作用关联矩阵...")
    SE_sub = np.zeros((994, 2686))
    for j in tqdm(range(len(drug_smile)), desc="Building Matrix"):
        sideID = side_id[j]
        label = float(labels[j])
        if label > 0:
            for sub_k in sub_dict[j]:
                if sub_k == 0: continue
                SE_sub[int(sideID)][int(sub_k)] += label

    # 3. 计算频率 (Numpy加速)
    n = np.sum(SE_sub)
    SE_sum = np.sum(SE_sub, axis=1)
    SE_p = SE_sum / n
    Sub_sum = np.sum(SE_sub, axis=0)
    Sub_p = Sub_sum / n
    SE_sub_p = SE_sub / n

    print(">> [优化] 正在使用 Numpy 矩阵加速计算频率 R 值...")
    se_p_mat = SE_p[:, None]
    sub_p_mat = Sub_p[None, :]
    term1 = (se_p_mat * sub_p_mat / n) * (1 - se_p_mat) * (1 - sub_p_mat)
    term1[term1 <= 0] = 1e-10 
    numerator = SE_sub_p - (se_p_mat * sub_p_mat)
    denominator = np.sqrt(term1)
    freq = (numerator / denominator) + 1e-5

    # 4. 筛选 Top 50
    non_nan_values = freq[~np.isnan(freq)]
    percentile_95 = np.percentile(non_nan_values, 95)
    print(f"   95% 分位点: {percentile_95}")

    SE_sub_index = np.zeros((994, 50))
    for i in range(994):
        sorted_indices = np.argsort(freq[i])[::-1]
        filtered_indices = sorted_indices[freq[i][sorted_indices] > percentile_95]
        k_idx = 0
        for j_idx in filtered_indices:
            if k_idx < 50:
                SE_sub_index[i][k_idx] = j_idx
                k_idx += 1
            else: break

    # 保存文件
    if not os.path.exists('data/sub'): os.makedirs('data/sub')
    np.save(f"data/sub/SE_sub_index_50_{suffix}.npy", SE_sub_index)
    SE_sub_mask = SE_sub_index.copy()
    SE_sub_mask[SE_sub_mask > 0] = 1
    np.save(f"data/sub/SE_sub_mask_50_{suffix}.npy", SE_sub_mask)
    print("[Step 1] 提取完成！\n")

# 究极优化版 Dataset (内存全缓存 = 极速)
class Data_Encoder(torch.utils.data.Dataset):
    def __init__(self, df, suffix='cold_start'):
        self.df = df
        self.suffix = suffix 
        
        print(f"Loading Dataset ({len(df)} samples)...")
        # 1. 药物缓存 (内存)
        print("  1. Pre-encoding drugs to RAM...")
        self.drug_cache = {}
        unique_drugs = self.df['Drug_smile'].unique()
        for d in unique_drugs:
            self.drug_cache[d] = drug2emb_encoder(d)
            
        # 2. 副作用缓存 (内存) - 你的新要求
        print(f"  2. Pre-loading Side Effect Matrices to RAM...")
        self.SE_index_all = np.load(f"data/sub/SE_sub_index_50_{suffix}.npy").astype(int)
        self.SE_mask_all = np.load(f"data/sub/SE_sub_mask_50_{suffix}.npy")
        print("Dataset Ready!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        d = row['Drug_smile']
        s = int(row['SE_id'])
        label = row['Label']

        # 全程查表，无硬盘IO
        d_v, input_mask_d = self.drug_cache[d]
        s_v = self.SE_index_all[s, :]
        input_mask_s = self.SE_mask_all[s, :]
        
        return d_v, s_v, input_mask_d, input_mask_s, label

# 训练函数 (极速版)
def trainfun(model, device, train_loader, optimizer, epoch):
    model.train()
    avg_loss = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in loop:
        Drug = Drug.to(device)
        SE = SE.to(device)
        DrugMask = DrugMask.to(device)
        SEMsak = SEMsak.to(device)
        Label = torch.FloatTensor([int(item) for item in Label]).to(device)

        optimizer.zero_grad()
        out, _, _ = model(Drug, SE, DrugMask, SEMsak)
        loss = loss_fun(out.flatten(), Label)

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        loop.set_postfix(loss=loss.item())

    return sum(avg_loss) / len(avg_loss)

# 预测函数
def predict(model, device, test_loader):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    model.eval()
    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            Drug = Drug.to(device)
            SE = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.FloatTensor([int(item) for item in Label]).to(device)
            
            out, _, _ = model(Drug, SE, DrugMask, SEMsak)
            
            # 预测所有，不过滤
            location = torch.where(Label >= 0) 
            pred = out[location]
            label = Label[location]

            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# --- 3. 主流程 ---

def run_cold_start():
    # 参数设置
    EPOCHS = 5 # 建议跑 20 轮，反正现在速度快
    BATCH_SIZE = 128
    LR = 0.0001
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running Cold Start Experiment on {DEVICE}")

    # 1. 加载 759 数据
    print("Loading 759 Data...")
    df_smiles = pd.read_csv(SMILES_FILE_759, header=None) 
    all_smiles = df_smiles.iloc[:, 1].tolist()
    
    mat_data = sio.loadmat(FREQ_FILE_759)
    # 自动找 key
    matrix_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    freq_matrix = mat_data[matrix_key]
    
    # 2. 切分数据 (前9个Test，后750个Train)
    test_smiles = all_smiles[:9]
    test_matrix = freq_matrix[:9, :]
    
    train_smiles = all_smiles[9:]
    train_matrix = freq_matrix[9:, :]
    
    print(f"Train Drugs: {len(train_smiles)}, Test Drugs: {len(test_smiles)}")

    # 3. 构造训练集 list (用于 identify_sub)
    train_data_list = []
    for i in range(len(train_smiles)):
        for j in range(994):
            train_data_list.append([j, train_smiles[i], train_matrix[i, j]])
    
    # 4. 关键步骤：只用训练集运行 Identify Sub (生成 _cold_start.npy 文件)
    identify_sub_optimized(train_data_list, suffix='cold_start')

    # 5. 准备 Dataset 所需的 DataFrame
    print("Preparing DataFrames...")
    df_train = pd.DataFrame(train_data_list, columns=['SE_id', 'Drug_smile', 'Label'])
    
    # 为了加速训练，可以只训练 Label!=0 的数据 (作者原逻辑是做了采样的)
    # 但为了简单且不出错，全量训练也没问题，反正现在有内存加速
    # 如果想更快，可以加上：df_train = df_train[df_train['Label'] > 0]
    
    test_data_list = []
    for i in range(len(test_smiles)):
        for j in range(994):
            test_data_list.append([j, test_smiles[i], test_matrix[i, j]])
    df_test = pd.DataFrame(test_data_list, columns=['SE_id', 'Drug_smile', 'Label'])

    # 6. DataLoader
    train_dataset = Data_Encoder(df_train, suffix='cold_start')
    test_dataset = Data_Encoder(df_test, suffix='cold_start')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 7. 模型初始化
    model = Trans().to(DEVICE)
    model.device = DEVICE # 修复硬编码
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

    # 8. 训练循环
    for epoch in range(EPOCHS):
        loss = trainfun(model, DEVICE, train_loader, optimizer, epoch+1)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    # 9. 预测与评估
    print("Predicting Cold Start Drugs...")
    true_labels, pred_scores = predict(model, DEVICE, test_loader)
    
    # 指标计算
    try:
        p_val = pearson(true_labels, pred_scores)
        rmse_val = rmse(true_labels, pred_scores)
        mae_val = MAE(true_labels, pred_scores)
        sp_val = spearman(true_labels, pred_scores)
    except:
        p_val, rmse_val, mae_val, sp_val = 0,0,0,0

    print("\n=== Cold Start Experiment Results (9 New Drugs) ===")
    print(f"Pearson:  {p_val:.5f}")
    print(f"Spearman: {sp_val:.5f}")
    print(f"RMSE:     {rmse_val:.5f}")
    print(f"MAE:      {mae_val:.5f}")
    
    # 保存结果
    if not os.path.exists('predictResult'): os.makedirs('predictResult')
    np.save('predictResult/cold_start_preds.npy', pred_scores)
    np.save('predictResult/cold_start_labels.npy', true_labels)

if __name__ == '__main__':
    run_cold_start()