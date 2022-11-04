import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()
        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x):
        # batch_size, feature_num, emb_dim
        B, N, D = x.shape

        # 生成转换矩阵并分多头,将特征分到多个头中交互提取不同的语义信息
        q = self.q(x)                              # (B, N, D)
        q = q.reshape(B, N, self.num_heads, -1)    # (B, N, H, D/H)
        q = q.permute(0, 2, 1, 3)                  # （B, H, N, D/H)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)

        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        return v
