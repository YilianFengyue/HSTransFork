import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from Net import Trans, drug2emb_encoder
from utils import rmse, MAE, pearson, spearman
from tqdm import tqdm
import os
import random

# --- 1. 配置路径 ---
SMILES_FILE = 'data/drug_SMILES_750.csv'
FREQ_FILE = 'data/raw_frequency_750.mat'

# --- 2. 辅助函数 ---
def loss_fun(output, label):
    loss = torch.sum((output - label) ** 2)
    return loss

# === 【新增】采样函数 (复刻 main.py 逻辑) ===
def get_balanced_train_data(drug_indices, freq_matrix, all_smiles):
    """
    只针对训练集进行 1:1 正负样本采样
    """
    print(f">> Performing Balanced Sampling (1:1) on {len(drug_indices)} training drugs...")
    data_list = []
    
    pos_samples = []
    neg_samples = []
    
    # 1. 遍历训练集的药，收集所有正样本和负样本坐标
    # 为了加速，直接利用 numpy 索引
    # matrix subset:
    sub_matrix = freq_matrix[drug_indices, :]
    
    # 找到所有非0元素 (Pos)
    rows, cols = np.nonzero(sub_matrix)
    for r, c in zip(rows, cols):
        real_drug_idx = drug_indices[r]
        label = sub_matrix[r, c]
        pos_samples.append([c, all_smiles[real_drug_idx], label])
        
    # 找到所有0元素 (Neg)
    rows_neg, cols_neg = np.where(sub_matrix == 0)
    # 因为负样本太多，我们先存索引，稍后随机采
    neg_indices = list(range(len(rows_neg)))
    
    # 2. 随机采样负样本，数量等于正样本
    n_pos = len(pos_samples)
    if len(neg_indices) > n_pos:
        selected_neg_indices = random.sample(neg_indices, n_pos)
    else:
        selected_neg_indices = neg_indices # 负样本不够（不太可能）全取
        
    for idx in selected_neg_indices:
        r = rows_neg[idx]
        c = cols_neg[idx]
        real_drug_idx = drug_indices[r]
        neg_samples.append([c, all_smiles[real_drug_idx], 0.0]) # Label 0
        
    # 3. 合并并打乱
    final_data = pos_samples + neg_samples
    random.shuffle(final_data)
    
    print(f"   Positives: {len(pos_samples)}, Negatives: {len(neg_samples)}")
    print(f"   Total Balanced Training Samples: {len(final_data)}")
    
    # 转为 DataFrame
    return pd.DataFrame(final_data, columns=['SE_id', 'Drug_smile', 'Label'])

# --- 3. 核心功能函数 ---

def identify_sub_optimized(train_df, suffix='cold_start'):
    """
    基于 DataFrame 提取子结构
    """
    print(f'\n[Step 1] 正在基于训练集提取有效子结构 (Suffix: {suffix})...')
    
    drug_smile = train_df['Drug_smile'].tolist()
    side_id = train_df['SE_id'].tolist()
    labels = train_df['Label'].tolist()

    # 1. 药物去重编码
    print(">> [优化] 正在对药物子结构进行去重编码...")
    unique_smiles = list(set(drug_smile))
    smile_cache = {}
    for smile in tqdm(unique_smiles, desc="Encoding Unique Drugs"):
        drug_sub, mask = drug2emb_encoder(smile)
        smile_cache[smile] = drug_sub.tolist()

    sub_dict_list = [] # 对应 drug_smile 的顺序
    for s in drug_smile:
        sub_dict_list.append(smile_cache[s])

    # 2. 构建矩阵
    print(">> [优化] 正在构建副作用关联矩阵...")
    SE_sub = np.zeros((994, 2686))
    
    # 直接迭代加速
    for j in tqdm(range(len(drug_smile)), desc="Building Matrix"):
        label = float(labels[j])
        if label > 0:
            sideID = side_id[j]
            subs = sub_dict_list[j]
            for sub_k in subs:
                if sub_k == 0: continue
                SE_sub[int(sideID)][int(sub_k)] += label

    # 3. 计算频率
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

    if not os.path.exists('data/sub'): os.makedirs('data/sub')
    np.save(f"data/sub/SE_sub_index_50_{suffix}.npy", SE_sub_index)
    SE_sub_mask = SE_sub_index.copy()
    SE_sub_mask[SE_sub_mask > 0] = 1
    np.save(f"data/sub/SE_sub_mask_50_{suffix}.npy", SE_sub_mask)
    print("[Step 1] 提取完成！\n")

class Data_Encoder(torch.utils.data.Dataset):
    def __init__(self, df, suffix='cold_start'):
        self.df = df
        self.suffix = suffix 
        
        print(f"Loading Dataset ({len(df)} samples)...")
        print("  1. Pre-encoding drugs to RAM...")
        self.drug_cache = {}
        unique_drugs = self.df['Drug_smile'].unique()
        for d in unique_drugs:
            self.drug_cache[d] = drug2emb_encoder(d)
            
        print(f"  2. Pre-loading Side Effect Matrices (Suffix: {suffix})...")
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

        d_v, input_mask_d = self.drug_cache[d]
        s_v = self.SE_index_all[s, :]
        input_mask_s = self.SE_mask_all[s, :]
        
        return d_v, s_v, input_mask_d, input_mask_s, label

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
            
            location = torch.where(Label >= 0) 
            pred = out[location]
            label = Label[location]

            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# --- 4. 主逻辑 ---

def run_cold_start():
    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 0.0001
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running Cold Start Experiment on {DEVICE}")

    # 1. 加载数据
    print("Loading 750 Data...")
    df_smiles = pd.read_csv(SMILES_FILE, header=None) 
    all_smiles = df_smiles.iloc[:, 1].tolist()
    mat_data = sio.loadmat(FREQ_FILE)
    freq_matrix = mat_data['R']
    
    # 2. 手动切分 (90% Train, 10% Test)
    indices = list(range(len(all_smiles)))
    random.seed(42) 
    random.shuffle(indices)
    
    split_point = int(len(all_smiles) * 0.9)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    test_smiles = [all_smiles[i] for i in test_indices]
    test_matrix = freq_matrix[test_indices, :]
    
    print(f"Split: Train Drugs={len(train_indices)}, Test Drugs={len(test_indices)}")

    # 3. 准备数据 DataFrame
    # 训练集：使用平衡采样 (1:1) -> 速度快，逻辑对齐 main.py
    df_train = get_balanced_train_data(train_indices, freq_matrix, all_smiles)
    
    # 测试集：全量测试 (不采样) -> 严谨评估
    print("Preparing Test Data (Full)...")
    test_data_list = []
    for i in range(len(test_smiles)):
        for j in range(994):
            test_data_list.append([j, test_smiles[i], test_matrix[i, j]])
    df_test = pd.DataFrame(test_data_list, columns=['SE_id', 'Drug_smile', 'Label'])

    # 4. Identify Sub (只用训练集的 DataFrame)
    # 注意：这里传给 identify_sub 的是已经采样过的 balanced df，但这也没问题，
    # 或者为了更精准，可以用全量正样本算 sub。
    # 为了简化且不报错，我们直接用 df_train，因为它包含了所有正样本。
    identify_sub_optimized(df_train, suffix='cold_start_750')

    # 5. DataLoader
    train_dataset = Data_Encoder(df_train, suffix='cold_start_750')
    test_dataset = Data_Encoder(df_test, suffix='cold_start_750')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 模型
    model = Trans().to(DEVICE)
    model.device = DEVICE 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

    # 7. 训练
    for epoch in range(EPOCHS):
        loss = trainfun(model, DEVICE, train_loader, optimizer, epoch+1)
        print(f"Epoch {epoch+1} Loss: {loss:.4f}")

    # 8. 预测
    print("Predicting on Cold Start Test Set...")
    true_labels, pred_scores = predict(model, DEVICE, test_loader)
    
    try:
        p_val = pearson(true_labels, pred_scores)
        sp_val = spearman(true_labels, pred_scores)
        rmse_val = rmse(true_labels, pred_scores)
        mae_val = MAE(true_labels, pred_scores)
    except:
        p_val, sp_val, rmse_val, mae_val = 0,0,0,0

    print("\n=== Cold Start Experiment Results (75 New Drugs) ===")
    print(f"Pearson:  {p_val:.5f}")
    print(f"Spearman: {sp_val:.5f}")
    print(f"RMSE:     {rmse_val:.5f}")
    print(f"MAE:      {mae_val:.5f}")
    
    if not os.path.exists('predictResult'): os.makedirs('predictResult')
    np.save('predictResult/cold_start_preds.npy', pred_scores)
    np.save('predictResult/cold_start_labels.npy', true_labels)

if __name__ == '__main__':
    run_cold_start()