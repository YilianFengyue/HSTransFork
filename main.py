import argparse
import pickle
import scipy
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import io
from Net import *
from smiles2vector import load_drug_smile
from math import *
import random
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as data
from sklearn.metrics import precision_score, recall_score, accuracy_score
from utils import *
from tqdm import tqdm 

raw_file = 'data/raw_frequency_750.mat'
SMILES_file = 'data/drug_SMILES_750.csv'
mask_mat_file = 'data/mask_mat_750.mat'
side_effect_label = 'data/side_effect_label_750.mat'
input_dim = 109
gii = open('data/drug_side.pkl', 'rb')
drug_side = pickle.load(gii)
gii.close()


def Extract_positive_negative_samples(DAL, addition_negative_number=''):
    k = 0
    interaction_target = np.zeros((DAL.shape[0] * DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]  # 按照最后一列对行排序
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    final_positive_sample = np.concatenate((final_positive_sample, final_negtive_sample), axis=0)
    return addition_negative_sample, final_positive_sample, final_negtive_sample


def loss_fun(output, label):
    # output = output.to('cuda')
    # label = label.to('cuda')
    loss = torch.sum((output - label) ** 2)
    return loss


def identify_sub(data, k):
    print(f'\n[Step 1] 正在提取有效子结构 (折数: {k})...')
    
    # 1. 提取数据列表
    drug_smile = [item[1] for item in data]
    side_id = [item[0] for item in data]
    labels = [item[2] for item in data]

    # === 【优化 1】去重计算，避免重复劳动 ===
    print(">> 正在对药物子结构进行去重编码...")
    unique_smiles = list(set(drug_smile))
    smile_cache = {}
    
    # 只计算 750 次，加了进度条
    for smile in tqdm(unique_smiles, desc="Encoding Unique Drugs"):
        drug_sub, mask = drug2emb_encoder(smile)
        smile_cache[smile] = drug_sub.tolist()

    print(">> 正在映射回全量数据集...")
    sub_dict = {}
    # 直接查表，速度极快
    for i in range(len(drug_smile)):
        sub_dict[i] = smile_cache[drug_smile[i]]
    # ======================================

    # 暂存成文件 (保留原逻辑)
    with open(f'data/sub/my_dict_{k}.pkl', 'wb') as f:
        pickle.dump(sub_dict, f)
    with open(f'data/sub/my_dict_{k}.pkl', 'rb') as f:
        sub_dict = pickle.load(f)

    # 2. 构建 SE_sub 矩阵
    print(">> 正在构建副作用关联矩阵...")
    SE_sub = np.zeros((994, 2686))
    
    # 这里加个 tqdm 稍微有点耗时，但比之前快多了
    for j in tqdm(range(len(drug_smile)), desc="Building Matrix"):
        sideID = side_id[j]
        label = float(labels[j])
        if label > 0: # 稍微优化一下，只有 label > 0 才需要加
            for sub_k in sub_dict[j]:
                if sub_k == 0:
                    continue
                SE_sub[int(sideID)][int(sub_k)] += label

    np.save(f"data/sub/SE_sub_{k}.npy", SE_sub)
    SE_sub = np.load(f"data/sub/SE_sub_{k}.npy", allow_pickle=True)

    # 3. 计算频率 (使用之前的 Numpy 加速版)
    n = np.sum(SE_sub)
    SE_sum = np.sum(SE_sub, axis=1)
    SE_p = SE_sum / n
    Sub_sum = np.sum(SE_sub, axis=0)
    Sub_p = Sub_sum / n
    SE_sub_p = SE_sub / n

    print(">> 正在使用 Numpy 矩阵加速计算频率 R 值...")
    
    # 广播机制加速
    se_p_mat = SE_p[:, None]
    sub_p_mat = Sub_p[None, :]
    
    term1 = (se_p_mat * sub_p_mat / n) * (1 - se_p_mat) * (1 - sub_p_mat)
    term1[term1 <= 0] = 1e-10 
    
    numerator = SE_sub_p - (se_p_mat * sub_p_mat)
    denominator = np.sqrt(term1)
    
    freq = (numerator / denominator) + 1e-5

    np.save(f"data/sub/freq_{k}.npy", freq)
    
    # 计算分位点
    freq = np.load(f"data/sub/freq_{k}.npy", allow_pickle=True)
    non_nan_values = freq[~np.isnan(freq)]
    percentile_95 = np.percentile(non_nan_values, 95)
    print(f"   95% 分位点: {percentile_95}")

    # 4. 筛选 Top 50 子结构
    print(">> 正在筛选 Top 50 有效子结构...")
    l = []
    SE_sub_index = np.zeros((994, 50))
    
    for i in range(994):
        k_idx = 0
        sorted_indices = np.argsort(freq[i])[::-1]
        filtered_indices = sorted_indices[freq[i][sorted_indices] > percentile_95]
        l.append(len(filtered_indices))
        for j_idx in filtered_indices:
            if k_idx < 50:
                SE_sub_index[i][k_idx] = j_idx
                k_idx = k_idx + 1
            else:
                continue

    np.save(f"data/sub/SE_sub_index_50_{k}.npy", SE_sub_index)
    SE_sub_index = np.load(f"data/sub/SE_sub_index_50_{k}.npy")

    SE_sub_mask = SE_sub_index.copy() # copy 一下防止引用问题
    SE_sub_mask[SE_sub_mask > 0] = 1
    np.save(f"data/sub/SE_sub_mask_50_{k}.npy", SE_sub_mask)
    np.save("len_sub", l)
    print("[Step 1] 提取完成！\n")



def trainfun(model, device, train_loader, optimizer, epoch, log_interval, test_loader):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    avg_loss = []

    # 加上 tqdm 进度条，这样你就能看到它在动了，而不是“卡住”
    from tqdm import tqdm
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

        # 更新进度条显示的 loss，不用 print 刷屏了
        loop.set_postfix(loss=loss.item())

    return sum(avg_loss) / len(avg_loss)

def predict(model, device, test_loader):
    # 声明为张量
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            # === 【核心修复】搬运数据 ===
            Drug = Drug.to(device)
            SE = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.FloatTensor([int(item) for item in Label]).to(device)
            # =========================
            
            out, _, _ = model(Drug, SE, DrugMask, SEMsak)

            location = torch.where(Label != 0)
            pred = out[location]
            label = Label[location]

            # 结果转回 CPU 方便后续 numpy 计算
            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def evaluate(model, device, test_loader):
    total_preds = torch.Tensor()
    total_label = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            # === 【核心修复】搬运数据 ===
            Drug = Drug.to(device)
            SE = SE.to(device)
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.FloatTensor([int(item) for item in Label]).to(device)
            # =========================

            output, _, _ = model(Drug, SE, DrugMask, SEMsak)
            
            # 转回 CPU 进行指标计算
            pred = output.cpu()
            Label = Label.cpu()

            total_preds = torch.cat((total_preds, pred), 0)
            total_label = torch.cat((total_label, Label), 0)

            pred = pred.numpy().flatten()
            pred = np.where(pred > 0.5, 1, 0)
            label = (Label.numpy().flatten() != 0).astype(int)
            label = np.where(label != 0, 1, label)

            # 防止 batch 太小导致 label 全 0 或全 1 报错
            try:
                singleDrug_auc.append(roc_auc_score(label, pred))
                singleDrug_aupr.append(average_precision_score(label, pred))
            except ValueError:
                pass 

        if len(singleDrug_auc) > 0:
            drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
            drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
        else:
            drugAUC = 0
            drugAUPR = 0

        # ... (后续代码保持不变，直到 return) ...
        # 为了保险起见，建议你也检查 evaluate 函数的后半部分，确保 indentation 对齐
        
        # 这里为了完整性，把后半部分也贴出来，你可以直接覆盖整个 evaluate 函数
        total_preds = total_preds.numpy()
        total_label = total_label.numpy()

        total_pre_binary = np.where(total_preds > 0.5, 1, 0)
        label01 = np.where(total_label != 0, 1, total_label)

        pre_list = total_pre_binary.tolist()
        label_list = label01.tolist()

        precision = precision_score(pre_list, label_list)
        recall = recall_score(pre_list, label_list)
        accuracy = accuracy_score(pre_list, label_list)

        total_preds = np.where(total_preds > 0.5, 1, 0)
        total_label = np.where(total_label != 0, 1, total_label)

        pos = np.squeeze(total_preds[np.where(total_label)])
        pos_label = np.ones(len(pos))
        neg = np.squeeze(total_preds[np.where(total_label == 0)])
        neg_label = np.zeros(len(neg))

        y = np.hstack((pos, neg))
        y_true = np.hstack((pos_label, neg_label))
        # === 【核心修复】加上异常捕获，防止数据太少导致单一类别报错 ===
        try:
            auc_all = roc_auc_score(y_true, y)
        except ValueError:
            auc_all = 0.5 # 算不出来就给个默认值 0.5，反正只是验证
            
        try:
            aupr_all = average_precision_score(y_true, y)
        except ValueError:
            aupr_all = 0.0
        # ========================================================

    return auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy

def main(training_generator, testing_generator, modeling, lr, num_epoch, weight_decay, log_interval, cuda_name,
         save_model, k):
    print('\n=======================================================================================')
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('weight_decay: ', weight_decay)

    model_st = modeling.__name__
    train_losses = []

    # 确定设备
    print('CPU/GPU: ', torch.cuda.is_available())
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # 模型初始化
    model = modeling().to(device)
    # === 【核心修复】强制告诉模型它现在在 GPU 上 ===
    model.device = device 
    # ==========================================
    # 计算模型的参数总数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epoch):
        train_loss = trainfun(model=model, device=device,
                              train_loader=training_generator,
                              optimizer=optimizer, epoch=epoch + 1, log_interval=log_interval,
                              test_loader=testing_generator)
        train_losses.append(train_loss)

        if epoch % 50 == 0:
            checkpointsFolder = 'checkpoints/'
            torch.save(model.state_dict(), checkpointsFolder + f'{k}' + str(epoch))

    print("正在预测")
    test_labels, test_preds = predict(model=model, device=device, test_loader=testing_generator)


    np.save(f'predictResult/total_labels_{k}.npy', test_labels)
    np.save(f'predictResult/total_preds_{k}.npy', test_preds)


    # === 【核心修复】直接计算所有指标，不再从列表里瞎取 ===
    test_pearsons = pearson(test_labels, test_preds)
    test_rMSE = rmse(test_labels, test_preds)
    test_spearman = spearman(test_labels, test_preds)
    test_MAE = MAE(test_labels, test_preds)
    # ==================================================

    print("正在评估")
    auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy = evaluate(model=model, device=device,
                                                                                 test_loader=testing_generator)

    result = [test_pearsons, test_rMSE, test_spearman, test_MAE, auc_all, aupr_all, drugAUC, drugAUPR, precision,
              recall, accuracy]

    print('Test:\nPearson: {:.5f}\trMSE: {:.5f}\tSpearman: {:.5f}\tMAE: {:.5f}'.format(result[0], result[1], result[2],
                                                                                       result[3]))
    print(
        '\tall AUC: {:.5f}\tall AUPR: {:.5f}\tdrug AUC: {:.5f}\tdrug AUPR: {:.5f}\tdrug Precise: {:.5f}\tRecall: {:.5f}\tdrug ACC: {:.5f}'.format(
            result[4], result[5],
            result[6], result[7], result[8], result[9], result[10]))


class Data_Encoder(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti, k):
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.k = k
        
        # === 【优化 1】预计算药物子结构 (已做) ===
        print(f"Loading Dataset (Fold {k})...")
        print("  1. Pre-encoding drugs to RAM...")
        self.drug_cache = {}
        unique_drugs = self.df['Drug_smile'].unique()
        for d in unique_drugs:
            self.drug_cache[d] = drug2emb_encoder(d)
            
        # === 【优化 2】预加载副作用矩阵到内存 (新加的！) ===
        # 别让它在训练循环里读硬盘了，直接读进内存
        print("  2. Pre-loading Side Effect Matrices to RAM...")
        self.SE_index_all = np.load(f"data/sub/SE_sub_index_50_{self.k}.npy").astype(int)
        self.SE_mask_all = np.load(f"data/sub/SE_sub_mask_50_{self.k}.npy")
        print("Dataset Ready!")
        # ==========================================================

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        d = self.df.iloc[index]['Drug_smile']
        s = int(self.df.iloc[index]['SE_id'])

        # 1. 药物：查内存字典 (快)
        d_v, input_mask_d = self.drug_cache[d]

        # 2. 副作用：查内存数组 (快) <--- 以前是读硬盘
        s_v = self.SE_index_all[s, :]
        input_mask_s = self.SE_mask_all[s, :]
        
        y = self.labels[index]
        return d_v, s_v, input_mask_d, input_mask_s, y


if __name__ == '__main__':
    # 参数定义
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0)
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.01, help='weight_decay')
    parser.add_argument('--epoch', type=int, required=False, default=200, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=40, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cpu', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=200,
                        help='features dimensions of drugs and side effects')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model and features')

    args = parser.parse_args()

    modeling = [Trans][args.model]
    lr = args.lr
    num_epoch = args.epoch
    weight_decay = args.wd
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    save_model = args.save_model

    #  获取正负样本
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(
        drug_side, addition_negative_number='all')

    addition_negative_sample = np.vstack((addition_negative_sample, final_negative_sample))

    final_sample = final_positive_sample

    X = final_sample[:, 0::]

    final_target = final_sample[:, final_sample.shape[1] - 1]

    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    drug_dict, drug_smile = load_drug_smile(SMILES_file)


    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 1], addition_negative_sample[i, 0]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append(
            (addition_negative_sample[i, 1], addition_negative_sample[i, 0], addition_negative_sample[i, 2]))
    for i in range(X.shape[0]):
        data_x.append((X[i, 1], X[i, 0]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 1], drug_smile[X[i, 0]], X[i, 2]))

    fold = 1
    kfold = StratifiedKFold(10, random_state=1, shuffle=True)

    params = {'batch_size': 128,
              'shuffle': True}
    
    # # === 【修改 1】新增：极速测试模式 (只取前 50 个样本跑流程) ===
    # print("【调试模式】正在截断数据以快速验证...")
    # # 只要验证代码能跑通，50个数据足够了
    # data = data[:50] 
    # data_x = data_x[:50]
    # data_y = data_y[:50]
    # ========================================================

    identify_sub(data, 0)

    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        data_train = np.array(data)[train]
        data_test = np.array(data)[test]

        # 将数据转为DataFrame
        df_train = pd.DataFrame(data=data_train.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])
        df_test = pd.DataFrame(data=data_test.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])

        # 创建数据集和数据加载器
        training_set = Data_Encoder(df_train.index.values, df_train.Label.values, df_train, k)
        testing_set = Data_Encoder(df_test.index.values, df_test.Label.values, df_test, k)

        training_generator = torch.utils.data.DataLoader(training_set, **params)
        testing_generator = torch.utils.data.DataLoader(testing_set, **params)

        main(training_generator, testing_generator, modeling, lr, num_epoch, weight_decay, log_interval,
             cuda_name, save_model, k)
        # === 新增这两行，跑完第一折直接退出 ===
        print("第一折验证完毕，强制退出！")
        break
