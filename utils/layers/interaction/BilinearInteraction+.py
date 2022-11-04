import torch
import torch.nn as nn
from itertools import combinations

class BilinearInteraction1(nn.Module):
    def __init__(self, input_shape, bilinear_type="interaction", seed=1024):
        self.bilinear_type = bilinear_type
        self.seed = seed

        super(BilinearInteraction1, self).__init__()

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')
        embedding_size = int(input_shape[0][-1])  ## K

        if self.bilinear_type == "all":
            ## Field-All Type: W_list 的 shape 为 K * K
            self.W = nn.Parameter(data=torch.randn([embedding_size, embedding_size]), requires_grad=True)
        elif self.bilinear_type == "each":
            ## Field-Each Type: W 的 shape 为 F * K * K
            self.W_list = [nn.Parameter(data=torch.randn([embedding_size, embedding_size]), requires_grad=True)
                           for _ in range(len(input_shape) - 1)]
        elif self.bilinear_type == "interaction":
            ## Field-Interaction Type: W_list 的 shape 为 F*(F - 1)/2 * K * K
            self.W_list = [nn.Parameter(data=torch.randn([embedding_size, embedding_size]), requires_grad=True) for _, _
                           in combinations(range(len(input_shape)), 2)]
        else:
            raise NotImplementedError

    def forward(self, inputs):
        ## inputs = [e1, e2, ..., ef],
        ## 其中 ei 的大小为 [B, 1, K]
        if len(inputs[0].shape) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % len(inputs[0].shape))

        n = len(inputs)  # feature 的个数
        B, _, K = inputs[0].shape
        # 所有特征共用一套参数, f_i.dot(w) * f_j
        if self.bilinear_type == "all":
            vidots = [torch.matmul(inputs[i], self.W) for i in range(n)]
            p = [torch.matmul(vidots[i], inputs[j].view(B, K, 1)) for i, j in combinations(range(n), 2)]
        # 每个特征独享参数 f_i.dot(w_i) * f_j
        elif self.bilinear_type == "each":
            vidots = [torch.matmul(inputs[i], self.W_list[i]) for i in range(n - 1)]
            p = [torch.matmul(vidots[i], inputs[j].view(B, K, 1)) for i, j in combinations(range(n), 2)]
        # 特征交互 f_i.dot(w_ij) * f_j
        elif self.bilinear_type == "interaction":
            p = [torch.matmul(torch.matmul(v[0], w), v[1].view(B, K, 1)) for v, w in zip(combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=2).squeeze(1)


if __name__ == '__main__':
    v1 = torch.tensor([[0.5, 0.2, 0.3, 0.4],
                       [0.3, 0.5, 0.2, 0.1]]).view(2, 1, -1)
    v2 = torch.tensor([[0.5, 0.6, 0.7, 0.8],
                        [0.9, 0.3, 0.4, 0.23]]).view(2, 1, -1)
    bil = BilinearInteraction1(input_shape=[v1.shape, v2.shape], bilinear_type="interaction")
    print(bil([v1, v2]))
