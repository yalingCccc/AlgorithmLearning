import numpy as np
import torch.nn as nn
from utils.models import get_model
from utils.normalization.l2_norm import L2_Normalization
from utils.layers.DinAttention import Din
from config import *

class FeedEncoder(nn.Module):
    def __init__(self):
        super(FeedEncoder, self).__init__()
        if args.feed_embeddings is not None:
            emb = np.load(args.feed_embeddings)
            self.feed_id_emb_layer = nn.Embedding.from_pretrained(torch.from_numpy(emb['emb1']).float(), freeze=False)
            self.w2v_feed_emb_layer = nn.Embedding.from_pretrained(torch.from_numpy(emb['emb2']).float(), freeze=False)
        else:
            self.feed_id_emb_layer = nn.Embedding(args.feed_vocab_size, embedding_dim=args.emb_size)
            self.w2v_feed_emb_layer = nn.Embedding(args.feed_vocab_size, embedding_dim=args.emb_size)
        self.fusion = None
        if args.feed_emb_fusion == 'dense':
            self.fusion = nn.Linear(2*args.emb_size, args.emb_size)

    def forward(self, feed_id):
        feed_id_embs = torch.cat([self.feed_id_emb_layer(feed_id), self.w2v_feed_emb_layer(feed_id)], dim=-1)
        out = self.fusion(feed_id_embs)
        return out


class FeedSeqEncoder(nn.Module):
    def __init__(self, feed_encoder):
        super(FeedSeqEncoder, self).__init__()
        self.feed_encoder = feed_encoder
        self.att = Din(args.emb_size_1, args.emb_size)

    def forward(self, query, seq, mask):
        query = query.unsqueeze(1)
        feed_emb_seq = self.feed_encoder(seq)
        output = self.att(query, feed_emb_seq, mask)
        return output


class Input(nn.Module):
    def __init__(self):
        super(Input, self).__init__()
        # user_emb
        self.user_emb_layer = nn.Embedding(args.vocab_size['userid'], embedding_dim=args.emb_size)
        self.feed_emb_layer = FeedEncoder()

        # feed_id att
        seq_feat_layers = {}
        for feat in SEQFEATURES:
            seq_feat_layers[feat] = FeedSeqEncoder(self.feed_emb_layer)
        self.seq_feat_layers = nn.ModuleDict(seq_feat_layers)

    def forward(self, inputs):
        input_list = []
        input_list.append(self.user_emb_layer(inputs[0]))
        feed_id_emb = self.feed_emb_layer(inputs[1])
        input_list.append(feed_id_emb)
        for feat in SEQFEATURES:
            idx = FEATURES.index(feat)
            input_list.append(self.seq_feat_layers[feat](feed_id_emb, inputs[idx], inputs[idx + 1]))
        return input_list

# DLRM
class DLRM(nn.Module):
    def __init__(self):
        super(DLRM, self).__init__()
        # Input网络
        self.inputLayer = Input()
        # 共享bottom网络
        self.mmoe = get_model('mmoe')(input_dim=args.mmoe_input_dim,
                         num_expert=args.num_expert,
                         num_task=args.num_task,
                         expert_dims=args.expert_dims,
                         use_top_layer=args.mmoe_mpl,
                         top_layer_dims=args.mmoe_mpl_dims,
                         activation=args.mmor_activation,
                         dropout=args.mmoe_dropout,
                         batch_norm=args.mmoe_batchnorm,
                         sigmoid=False)
        # 定义每个task的top网络

        self.batch_norm = nn.BatchNorm1d(args.emb_size)
        self.l2_layer = L2_Normalization()

    def forward(self, inputs):
        input_embs = self.inputLayer(inputs)
        emb_init = torch.cat(input_embs.copy(), dim=1)
        # 交互
        if args.l2:
            embs_normed = []
            for x in input_embs:
                x = self.batch_norm(x)
                x = self.l2_layer(x)
                embs_normed.append(x)
        else:
            embs_normed = input_embs
        rates = []
        for idx1, x in enumerate(embs_normed):
            for idx2 in range(idx1 + 1, len(embs_normed)):
                rates.append(x * embs_normed[idx2])
        emb_intered = torch.cat(rates, dim=-1)
        # mmoe
        mmoe_input = torch.cat([emb_init, emb_intered], dim=-1)
        mmoe_output = self.mmoe(mmoe_input)
        return mmoe_output
