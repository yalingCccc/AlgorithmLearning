import torch.nn as nn
from utils.models import get_model
from utils.normalization.l2_norm import L2_Normalization
from utils.layers.add_weight.DinAttention import Din
from config import *


class SideInformationLayer(nn.Module):
    def __init__(self, side_information):
        super(SideInformationLayer, self).__init__()
        side_information = {key: np.array(value) for key, value in side_information.items()}
        side_info_layers = {}
        for feat in SIDE_INFORMATIONS:
            if 'list' in feat:
                side_info_layers[feat] = torch.from_numpy(side_information[feat]).to(device)
            else:
                side_info_layers[feat] = torch.from_numpy(side_information[feat]).view(-1, 1).to(device)
        self.side_info_layers = nn.ModuleDict(side_info_layers)

    def forward(self, feedid, feature):
        return self.side_info_layers[feature][feedid]


class FeedEncoder(nn.Module):
    def __init__(self, pretrain_emb, side_information):
        super(FeedEncoder, self).__init__()
        vocab_size = hparams['vocab_size']
        emb_size = hparams['emb_size']
        # feed id
        if hparams['feed_pretrained'] is not None:
            self.feed_id_emb_layer = nn.Embedding.from_pretrained(
                torch.from_numpy(pretrain_emb['feed_emb']).float(), freeze=False,
                padding_idx=0)
            self.w2v_feed_emb_layer = nn.Embedding.from_pretrained(
                torch.from_numpy(pretrain_emb['feed_w2v_embeddings']).float(), freeze=False,
                padding_idx=0)
        else:
            self.feed_id_emb_layer = nn.Embedding(vocab_size['feedid'] + 1, embedding_dim=emb_size,
                                                  padding_idx=0)
            self.w2v_feed_emb_layer = nn.Embedding(vocab_size['feedid'] + 1, embedding_dim=emb_size,
                                                   padding_idx=0)
        dense_input_size = emb_size

        # side information
        side_layer = {}
        for feat in SIDE_INFORMATIONS:
            input_dim = vocab_size[feat]
            output_dim = emb_size
            if 'list' in feat:
                side_layer[feat] = SeqEncoder(nn.Embedding(input_dim, output_dim, padding_idx=0))
            else:
                side_layer[feat] = nn.Embedding(input_dim, output_dim, padding_idx=0)
            dense_input_size += output_dim
        self.get_side_info = SideInformationLayer(side_information)
        self.side_att = nn.ModuleDict(side_layer)

        # feed id interaction
        self.dense_1 = nn.Linear(2 * emb_size, emb_size)

        # feature interation
        if hparams['feed_emb_fusion'] == 'dense':
            self.dense_2 = nn.Linear(dense_input_size, emb_size)

    def forward(self, feed_id):
        # feed id emb
        feed_id_emb = [self.feed_id_emb_layer(feed_id), self.w2v_feed_emb_layer(feed_id)]
        feed_id_emb = torch.cat(feed_id_emb, dim=-1)
        if hparams['feed_emb_fusion']:
            feed_id_emb = self.dense_1(feed_id_emb)
        emb_list = [feed_id_emb]

        for feat in SIDE_INFORMATIONS:
            in_x = self.get_side_info(feed_id, feat)
            if 'list' in feat:
                mask = (in_x > 0).float()
                out = self.side_att[feat](feed_id_emb, in_x, mask)
            else:
                out = self.side_att[feat](in_x.long())
                out = out.squeeze(-2)
            emb_list.append(out)
        emb = torch.cat(emb_list, dim=-1)
        output = self.dense_2(emb)
        return output


class SeqEncoder(nn.Module):
    def __init__(self, id_encoder):
        super(SeqEncoder, self).__init__()
        self.feed_encoder = id_encoder
        self.att = Din(hparams['emb_size'])

    def forward(self, query, seq, mask):
        query = query.unsqueeze(1)
        feed_emb_seq = self.feed_encoder(seq)
        output = self.att(query, feed_emb_seq, mask)
        return output


class DinAttention(nn.Module):
    def __init__(self):
        super(DinAttention, self).__init__()
        self.att = Din(hparams['emb_size'])

    def forward(self, query, seq, mask):
        query = query.unsqueeze(1)
        output = self.att(query, seq, mask)
        return output


class ContextLayer(nn.Module):
    def __init__(self):
        super(ContextLayer, self).__init__()
        self.device_encoder = nn.Embedding(hparams['vocab_size']['device'], hparams['emb_size'])

    def forward(self, x):
        return self.device_encoder(x[0])


class Input(nn.Module):
    def __init__(self, pretrain_emb, side_information):
        super(Input, self).__init__()
        vocab_size = hparams['vocab_size']
        emb_size = hparams['emb_size']
        # user embedding
        if hparams['user_pretrained']:
            self.user_encoder = nn.Embedding.from_pretrained(
                torch.from_numpy(pretrain_emb['user_embeddings']).float(), freeze=False,
                padding_idx=0)
        else:
            self.user_encoder = nn.Embedding(vocab_size['userid'], embedding_dim=emb_size,
                                               padding_idx=0)

        # feed embedding
        self.feed_encoder = FeedEncoder(pretrain_emb, side_information)
        self.context_layer = ContextLayer()

        # hist action sequences attention
        seq_feat_layers = {}
        for feat in SEQ_FEATURES:
            seq_feat_layers[feat] = DinAttention()
        self.seq_feat_layers = nn.ModuleDict(seq_feat_layers)

    def forward(self, inputs):
        emb_list = []
        emb_list.append(self.user_encoder(inputs[FEATURES.index('userid')]))
        feed_id_emb = self.feed_encoder(inputs[FEATURES.index('feedid')])
        emb_list.append(feed_id_emb)
        if len(CONTEXT_FEATURES) > 0:
            start_idx = FEATURES.index(CONTEXT_FEATURES[0])
            end_idx = FEATURES.index(CONTEXT_FEATURES[-1]) + 1
            emb_list.append(self.context_layer(inputs[start_idx:end_idx]))
        if len(SEQ_FEATURES) > 0:
            for feat in SEQ_FEATURES:
                feat_idx = FEATURES.index(feat)
                hist_item_list = inputs[feat_idx]
                mask = (hist_item_list > 0).float()
                batch_size, seq_len = hist_item_list.shape
                hist_items = hist_item_list.view(batch_size * seq_len)
                hist_item_embs = self.feed_encoder(hist_items).view(batch_size, seq_len, -1)
                emb_list.append(self.seq_feat_layers[feat](feed_id_emb, hist_item_embs, mask))
        return emb_list


# DLRM
class DLRM(nn.Module):
    def __init__(self, pretrain_emb=None, side_information=None):
        super(DLRM, self).__init__()
        # Input网络
        self.inputLayer = Input(pretrain_emb, side_information)
        # 共享bottom网络
        self.mmoe = get_model('mmoe')(input_dim=hparams['mmoe_input_dim'],
                                      num_expert=hparams['num_expert'],
                                      num_task=hparams['num_task'],
                                      expert_dims=hparams['expert_dims'],
                                      use_top_layer=hparams['mmoe_mpl'],
                                      top_layer_dims=hparams['mmoe_mpl_dims'],
                                      activation=hparams['mmor_activation'],
                                      dropout=hparams['mmoe_dropout'],
                                      batch_norm=hparams['mmoe_batchnorm'],
                                      sigmoid=False)
        self.batch_norm = nn.BatchNorm1d(hparams['emb_size'])
        self.l2_layer = L2_Normalization()

    def forward(self, inputs):
        input_embs = self.inputLayer(inputs)
        emb_init = torch.cat(input_embs.copy(), dim=1)
        assert emb_init.shape[1] == hparams['input_dim']
        # 交互
        if hparams['l2']:
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
        # TODO：Multiple Dropout
        return mmoe_output
