import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from collections import Counter
import numpy as np
import random

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
from torch import optim
from torch.optim import lr_scheduler

random.seed(1224)
np.random.seed(1224)
torch.manual_seed(1224)


# 定义PyTorch模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        init_range = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size)
        self.out_embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, input_labels, pos_labels, neg_lables):  # loss function
        """
        :param input_labels: [batch_size]
        :param pos_labels: [batch_size, (window_size * 2)]
        :param neg_lables: [batch_size, (window_size * 2 * K)]
        :return: loss, [batch_size]
        """
        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels.squeeze()).view(batch_size, -1)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window_size * 2), embed_size]
        neg_embedding = self.out_embed(neg_lables)  # [batch_size, (window_size * 2 * K), embed_size]

        # unsqueeze()升维, squeeze()降维
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1], 第二个维度加1
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze().view(batch_size,
                                                                           -1)  # [batch_size, (window_size * 2)]
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze().view(batch_size,
                                                                            -1)  # [batch_size, (window_size * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):  # 取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()


class MyDataset(tud.Dataset):
    """
    为了使用Dataloader，我们需要定义以下两个function:
    - __len__(), 需要返回整个数据集中有多少item
    - __getitem__(), 根据给定的index返回一个item
    有了Dataloader后，可以轻松随机打乱整个数据集，拿到一个batch的数据等。
    """

    def __init__(self, sentences, neg_num, sample_list, window_size):
        super(MyDataset, self).__init__()
        self.sentences = sentences
        self.neg_num = neg_num
        self.sample_list = sample_list
        self.sample_total_num = len(sample_list)
        self.window_size = window_size

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):  # 返回数据用于训练
        half_window = self.window_size // 2
        s = [0] * half_window + self.sentences[idx] + [0] * half_window
        # 按window滑动取数
        center_words, pos_words, neg_words = [], [], []
        for index, value in enumerate(s[0:-self.window_size]):
            center = s[index+half_window]
            pos = s[index:index+self.window_size].remove(center)
            neg = []
            for _ in range(self.neg_num):
                while True:
                    id = np.random.randint(1, self.sample_total_num, 1)
                    word = self.sample_list[id]
                    if (word != center) and (word not in pos) and (word not in neg):
                        neg.append(word)
                        break
            center_words.append(torch.tensor(center))
            pos_words.append(torch.tensor(pos))
            neg_words.append(torch.tensor(neg))
        return center_words, pos_words, neg_words


class Word2Vec(object):
    def __init__(self, vocab_size, window=3, neg_samples=100, learning_rate=0.01, num_epoches=5, batch_size=32,
                 emb_size=64, min_count=5, b=0.75):
        self.vocab_size = vocab_size
        self.window = window
        self.neg_samples = neg_samples
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.min_count = min_count
        self.model = EmbeddingModel(vocab_size=vocab_size, embed_size=emb_size)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [1, 2, 3, 4], 0.1, last_epoch=4)
        self.b = b

    def fit(self, sentences):
        # 统计词频
        counter = Counter()
        for s in sentences:
            counter.update(s)
        word_dict = {key: counter[key] for key in counter.most_common(counter.__len__()) if
                     counter[key] >= self.min_count}

        # 编码
        word2idx = {key: index + 1 for index, key in enumerate(word_dict.keys())}  # 将0编号留出来作为填充
        word_dict = {word2idx[key]: value for key, value in word_dict.items()}
        sentences = [[word2idx[w] for w in s if w in word2idx] for s in sentences]

        # 负采样轮盘
        expose_score = [e^self.b for e in word_dict.values()]
        expose_sum = np.array(expose_score).sum()
        expose_rate = [e * 100 // expose_sum + 1 for e in expose_score]
        self.sample_list = []
        for e, rate in zip(word_dict, expose_rate):
            self.sample_list.extend([e] * rate)

        dataset = MyDataset(sentences, self.neg_samples, self.sample_list, self.window)
        dataLoader = tud.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epoches):
            # 训练模型
            for index, (input_labels, pos_labels, neg_labels) in enumerate(dataLoader):
                input_labels = input_labels.long()
                pos_labels = pos_labels.long()
                neg_labels = neg_labels.long()

                self.optimizer.zero_grad()
                loss = self.model(input_labels, pos_labels, neg_labels).mean()
                loss.backward()
                self.optimizer.step()

                if index % 10 == 0:
                    print(f'Epoch: {epoch}, MiniBatch: {index}, Loss: {loss.item()}')
            self.scheduler.step()
        self.vector = self.model.input_embedding()
