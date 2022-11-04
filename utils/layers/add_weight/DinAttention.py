import torch
import torch.nn as nn
from utils.layers.MPL import LinearLayer

class AttentionUnitLayer(nn.Module):
    def __init__(self, emb_size, act_func='Dice', batch_norm=False, dropout=0):
        super(AttentionUnitLayer, self).__init__()
        self.emb_size = emb_size,
        self.act_func = act_func
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.mlp = LinearLayer(4 * emb_size,
                               dims=[2 * emb_size, emb_size],
                               act_func=act_func,
                               batch_norm=batch_norm,
                               dropout=dropout,
                               sigmoid=False)

    def forward(self, query_emb, hist_item_emb):
        # 扩展query_emb
        hist_len = hist_item_emb.size(1)
        exp_query_emb = torch.cat([query_emb for _ in range(hist_len)], dim=1)

        # query与历史行为交互
        input_mlp = torch.cat(
            [exp_query_emb, hist_item_emb, exp_query_emb - hist_item_emb, exp_query_emb * hist_item_emb], dim=2)
        (s1, s2, s3) = input_mlp.shape
        output = self.mlp(input_mlp.view(s1 * s2, -1))
        return output.view(s1, s2, -1)


class Din(nn.Module):
    def __init__(self, emb_size, act_func='Dice', batch_norm=False, dropout=0):
        super(Din, self).__init__()
        self.emb_size = emb_size
        self.act_func = act_func
        self.batch_norm = batch_norm
        self.dropout=dropout
        self.att_unit = AttentionUnitLayer(emb_size, act_func, batch_norm, dropout)

    def forward(self,
                query,  # (batch_size, 1, emb_size)
                keys,  # (batch_size, max_hist_len, emb_size)
                mask  # (batch_size, max_hist_len)
                ):
        output_att_unit = self.att_unit(query, keys)
        mask = torch.cat([mask.unsqueeze(dim=2) for _ in range(output_att_unit.size(2))], dim=2)
        output_pool_layer = keys.mul(output_att_unit.mul(mask)).sum(dim=1)
        return output_pool_layer
