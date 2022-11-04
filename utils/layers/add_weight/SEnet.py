import torch
import torch.nn as nn


class SENETLayer(nn.Module):
    """
      动态学习特征重要性
      SENETLayer used in FiBiNET.
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, input_shape, reduction_ratio=3, seed=1024):
        """
        :param input_shape: [feature_num, emb_size]
        :param reduction_ratio:
        :param seed:
        """
        self.input_shape = input_shape
        self.reduction_ratio = reduction_ratio
        self.seed = seed
        super(SENETLayer, self).__init__()

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')

        self.filed_size = len(input_shape)  ## F, 表示 Field 的数量
        self.embedding_size = input_shape[0][-1]  ## K, 表示 embedding 的大小
        reduction_size = max(1, self.filed_size // self.reduction_ratio)  ## r, 表示 reduction ratio

        ## W1, shape 为 (F, F/r)
        self.W_1 = nn.Parameter(data=torch.randn([self.filed_size, reduction_size]), requires_grad=True)
        ## W2, shape 为 (F/r, F)
        self.W_2 = nn.Parameter(data=torch.randn([reduction_size, self.filed_size]), requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        """
        :param inputs: [feat_emb1:(batch_size, 1, emb_size), ..., feat_embk]
        :return:
        """
        if len(inputs.shape[0]) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs[0].shape)))

        # 每一个样本的每一个特征向量聚合为一个分值
        inputs = torch.cat(inputs, dim=1)  ## [B, F, K]
        Z = torch.mean(inputs, dim=-1)  ## [B, F]

        # 引入参数，增加结构的普适性
        A_1 = self.relu(torch.matmul(Z, self.W_1))  ## [B, F/r]
        A_2 = self.relu(torch.matmul(A_1, self.W_2)).unsqueeze(dim=2)  ## [B, F, 1]

        # 特征加权
        V = torch.multiply(inputs, A_2)  ## [B, F, K]

        # 将输出处理为与输入同shape的向量列表，使本结构可以随意拔插
        return torch.split(V, self.filed_size, dim=1)
