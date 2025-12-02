import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder_MultipleLayers, Embeddings
import torch
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE


class Trans(torch.nn.Module):
    def __init__(self):
        super(Trans, self).__init__()


        self.device = 'cpu'

        # activation and regularization
        self.relu = nn.ReLU()

        input_dim_drug = 2586
        transformer_emb_size_drug = 200
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        # 嵌入编码层
        self.embDrug = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              transformer_dropout_rate)

        self.embSide = Embeddings(input_dim_drug,
                              transformer_emb_size_drug,
                              50,
                              transformer_dropout_rate)

        # Transformer层
        self.encoderDrug = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)

        self.encoderSide = Encoder_MultipleLayers(transformer_n_layer_drug,
                                              transformer_emb_size_drug,
                                              transformer_intermediate_size_drug,
                                              transformer_num_attention_heads_drug,
                                              transformer_attention_probs_dropout,
                                              transformer_hidden_dropout_rate)


        # 位置编码层
        self.position_embeddings = nn.Embedding(500, 200)


        self.dropout = 0.3

        self.decoder = nn.Sequential(
            nn.Linear(6912, 512),
            nn.ReLU(True),

            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),

            # output layer
            nn.Linear(32, 1)
        )

        self.icnn = nn.Conv2d(1, 3, 3, padding=0)

        self.CrossAttention = False

    def forward(self, Drug, SE, DrugMask, SEMsak):

        batch = Drug.size(0)

        # 子结构编码
        Drug = Drug.long().to(self.device)
        DrugMask = DrugMask.long().to(self.device)
        DrugMask = DrugMask.unsqueeze(1).unsqueeze(2)
        DrugMask = (1.0 - DrugMask) * -10000.0
        emb = self.embDrug(Drug)
        encoded_layers = self.encoderDrug(emb.float(), DrugMask.float(), False)
        x_d = encoded_layers

        # 副作用-子结构编码
        SE = SE.long().to(self.device)
        SEMsak = SEMsak.long().to(self.device)
        SEMsak = SEMsak.unsqueeze(1).unsqueeze(2)
        SEMsak = (1.0 - SEMsak) * -10000.0
        embE = self.embSide(SE)
        encoded_layers = self.encoderSide(embE.float(), SEMsak.float(), False)
        x_e = encoded_layers

        if self.CrossAttention:
            x_d, x_e = self.crossAttentionencoder([x_d.float(),x_e.float()], DrugMask.float(), True)


        # interaction
        d_aug = torch.unsqueeze(x_d, 2).repeat(1, 1, 50, 1)
        e_aug = torch.unsqueeze(x_e, 1).repeat(1, 50, 1, 1)


        i = d_aug * e_aug
        i_v = i.permute(0, 3, 1, 2)
        i_v = torch.sum(i_v, dim=1)
        i_v = torch.unsqueeze(i_v, 1)
        i_v = F.dropout(i_v, p=self.dropout)

        f = self.icnn(i_v)

        f = f.view(int(batch), -1)

        score = self.decoder(f)

        return score, Drug, SE


def drug2emb_encoder(smile):
    vocab_path = 'data/drug_codes_chembl_freq_1500.txt'
    sub_csv = pd.read_csv('data/subword_units_map_chembl_freq_1500.csv')

    # 初始化一个BPE编码器，该编码器可以用于将文本进行分词或编码
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    idx2word_d = sub_csv['index'].values  # 将所有的子结构列表给提取出来
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))  # 构造字典：让子结构与index一一对应

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # 将该smile的子结构找到对应的index，形成一个index的ndarray
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]  # 进行填充
        input_mask = [1] * max_d  # 构造mask（盖住填充部分）

    return i, np.asarray(input_mask)
