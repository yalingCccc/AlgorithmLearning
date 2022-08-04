import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from gensim.models import Word2Vec
from config import *
import os


def pca_content_embedding(feed_embeddings, n_components):
    pca = PCA(n_components=n_components)
    embeddings = np.array([np.fromstring(e, dtype=float, sep=' ') for e in tqdm(feed_embeddings['feed_embedding'])])
    embeddings = pca.fit_transform(embeddings)
    embeddings = sorted(embeddings, feed_embeddings['feedid'], reverse=False)
    return embeddings


def feed_w2v_embedding(user_action):
    # 数据处理
    users = pd.DataFrame(user_action[user_action['date_'] == args.test_day]['userid'], columns=['userid']).drop_duplicates(
        'userid')
    all_users = pd.DataFrame(user_action[user_action['date_'] < args.test_day]['userid'], columns=['userid']).drop_duplicates(
        'userid')
    drop_users = users.sample(users.shape[0] // 2)
    w2v_users = set(all_users['userid']) - set(drop_users['userid'])
    w2v_users = pd.DataFrame(list(w2v_users), columns=['userid'])
    w2v_df = user_action.merge(w2v_users, on='userid', how='inner')
    sentences = w2v_df.groupby('userid')['feedid'].agg(lambda x: [str(e) for e in x])
    model = Word2Vec(sentences, hs=1, window=128, min_count=1, vector_size=args.emb_size)
    model.wv.save_word2vec_format(os.path.join(args.output_dir, 'feed_w2v_embeddings.txt'), binary=False)


def user_w2v_embedding(user_action):
    user_action = user_action[user_action['date_']<=args.train_day]
    sentences = user_action.groupby('feedid')['userid'].agg(lambda x: [str(e) for e in x])
    model = Word2Vec(sentences, hs=1, window=128, min_count=1, vector_size=args.emb_size)
    model.wv.save_word2vec_format(os.path.join(args.output_dir, 'user_w2v_embeddings.txt'), binary=False)


def hist_act_seq(user, date_, user_action):
    hist_seq = user_action[(user_action['userid'] == user) & (user_action['date_'] < date_)]
    hist_seq = hist_seq.sort_values('date_', ascending=False).reset_index(drop=True)
    hist_seq = hist_seq['feedid'].tolist()
    max_len = args.max_hist_len
    if len(hist_seq) >= max_len:
        return "|".join([str(e) for e in hist_seq[0:max_len]]), "|".join([str(e) for e in [1] * max_len])
    else:
        hist_len = len(hist_seq)
        return "|".join([str(e) for e in hist_seq + [0] * (max_len - hist_len)]), "|".join([str(e) for e in [1] * hist_len + [0] * (max_len - hist_len)])


def hist_sequences(data, user_action):
    users = data[['userid', 'date_']].drop_duplicates()

    logger.info("交互历史序列")
    user_action_1 = user_action.copy()
    user_action_1['act'] = user_action[ALLACTIONS].sum(axis=1)
    user_action_2 = user_action_1[user_action_1['act'] > 0]
    seqs, masks = [], []
    for key, row in tqdm(users.iterrows()):
        seq, mask = hist_act_seq(row['userid'], row['date_'], user_action_1)
        seqs.append(seq)
        masks.append(mask)
    users['hist_acts_seq'] = seqs
    users['hist_acts_mask'] = masks

    logger.info("展现但未交互序列")
    user_action_2 = user_action_1[user_action_1['act'] == 0]
    seqs, masks = [], []
    for key, row in tqdm(users.iterrows()):
        seq, mask = hist_act_seq(row['userid'], row['date_'], user_action_2)
        seqs.append(seq)
        masks.append(mask)
    users['no_act_seq'] = seqs
    users['no_act_mask'] = masks

    logger.info("单行为历史序列")
    for act in ALLACTIONS:
        print(act)
        user_action_1 = user_action[user_action[act] == 1]
        seqs, masks = [], []
        for key, row in tqdm(users.iterrows()):
            seq, mask = hist_act_seq(row['userid'], row['date_'], user_action_1)
            seqs.append(seq)
            masks.append(mask)
        users['hist_%s_seq' % act] = seqs
        users['hist_%s_mask' % act] = masks

    logger.info("完成序列")
    feed_info = pd.read_csv(args.feed_info)
    user_action_1 = user_action.merge(feed_info[['feedid', 'videoplayseconds']], on='feedid', how='inner')
    user_action_1['finished'] = user_action_1['play'] / user_action_1['videoplayseconds']
    user_action = user_action_1[user_action_1['finished'] > 0.99]
    seqs, masks = [], []
    for key, row in tqdm(users.iterrows()):
        seq, mask = hist_act_seq(row['userid'], row['date_'], user_action)
        seqs.append(seq)
        masks.append(mask)
    users['finished_seq'] = seqs
    users['finished_mask'] = masks

    logger.info("未完成序列")
    seqs, masks = [], []
    user_action = user_action_1[user_action_1['finished'] < 0.01]
    for key, row in tqdm(users.iterrows()):
        seq, mask = hist_act_seq(row['userid'], row['date_'], user_action)
        seqs.append(seq)
        masks.append(mask)
    users['unfinished_seq'] = seqs
    users['unfinished_mask'] = masks

    logger.info("关联历史序列")
    data = data.merge(users, on=['userid', 'date_'], how='left')
    return data


def main():
    # embedding 降维
    feed_embeddings = pd.read_csv(args.feed_embedding)
    all_feed = feed_embeddings['feedid']
    feed_embeddings = pca_content_embedding(feed_embeddings, args.emb_size)
    np.save(os.path.join(args.output_dir, 'feed_embeddings.npy'), feed_embeddings)


    user_action = pd.read_csv(args.user_action)
    all_user = pd.DataFrame(user_action[user_action['date_'] < args.test_day]['userid'],
                             columns=['userid']).drop_duplicates('userid')

    # feed embedding
    feed_w2v_embedding(user_action)

    # TODO: user id embedding by word2vec

    # 历史行为序列
    user_action = pd.read_csv(args.user_action)[0:5000]
    train_data = user_action[user_action['date_'] <= args.train_day]
    train_data = hist_sequences(train_data, user_action)
    train_data.to_csv(os.path.join(args.output_path, 'train.csv'))
    test_data = user_action[user_action['date_'] == args.test_day]
    test_data = hist_sequences(test_data, user_action)
    test_data.to_csv(os.path.join(args.output_path, 'test.csv'))


if __name__ == '__main__':
    main()
