import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
from gensim.models import Word2Vec
from config import *
from utils.utils.pickle_pro import pickle_process
import os
import gc


def feed_info_process(feed_info, vocab_size):
    '''
    1、id特征自增1
    2、videoplayseconds处理为标量值，分为短视频、中长视频、长视频
    3、获得各特征的vocab size
    4、缺失值填充为0
    '''
    feat_list, fill_ids = {}, []
    feed_info = feed_info.set_index('feedid')
    for feat in SIDE_INFORMATIONS:
        if feat == 'videoplayseconds':
            feed_info.loc[feed_info[feat] <= 30, feat] = 1
            feed_info.loc[(feed_info[feat] > 30) & (feed_info[feat] <= 90), feat] = 2
            feed_info.loc[feed_info[feat] > 90, feat] = 3
            feat_pro = [feed_info.loc[idx, feat] if idx in feed_info.index else 0 for idx in
                        range(vocab_size['feedid'])]
            vocab_size[feat] = 4
        elif 'id' in feat:
            feat_pro = np.array([feed_info.loc[idx, feat] + 1 if idx in feed_info.index else 0 for idx in
                                 range(vocab_size['feedid'])])
            feat_pro[np.isnan(feat_pro)] = 0  # 空值处理
            vocab_size[feat] = np.max(feat_pro) + 1
        elif 'list' in feat:
            feat_pro, feat_max, maxlen = [], 0, 0
            for idx in range(vocab_size['feedid']):
                if idx in feed_info.index and isinstance(feed_info.loc[idx, feat], str):
                    tmp = [e + 1 for e in np.fromstring(feed_info.loc[idx, feat], dtype=int, sep=';')]
                    feat_pro.append(tmp)
                    feat_max = max(feat_max, np.max(tmp))
                    maxlen = max(maxlen, len(tmp))
                else:
                    feat_pro.append([])
            feat_pro = [e + [0] * (maxlen - len(e)) for e in feat_pro]
            vocab_size[feat] = feat_max + 1
        else:
            print("未定义处理方式的特征：%s" % feat)
            feat_pro = []
        feat_list[feat] = feat_pro
    return feat_list, vocab_size


def fill_embedding(embeddings, ids, vocab_size):
    embeddings = np.matrix(embeddings)
    avg_emb = embeddings.mean(axis=0)
    result = []
    for idx in range(vocab_size):
        if idx not in ids:
            result.append([idx, avg_emb])
    for idx, emb in zip(ids, embeddings):
        result.append([idx, emb])
    result = sorted(result, key=lambda x: x[0], reverse=False)
    result = np.concatenate([e[1] for e in result], axis=0)
    return result


def pca_content_embedding(feed_embeddings, n_components, vocab_size):
    pca = PCA(n_components=n_components)
    feed_embeddings = feed_embeddings.sort_values('feedid', ascending=True)
    embedding = np.array([np.fromstring(e, dtype=float, sep=' ') for e in tqdm(feed_embeddings['feed_embedding'])])
    embedding = pca.fit_transform(embedding)
    embedding = fill_embedding(embedding, feed_embeddings['feedid'], vocab_size)
    return embedding


def feed_w2v_embedding(user_action, vocab_size):
    # 数据处理
    all_users = np.unique(user_action['userid'])
    test_users = np.unique(user_action[user_action['date_'] == hparams['test_day']]['userid'])
    samp_ids = []
    test_user_num = len(test_users)
    while len(samp_ids) < test_user_num / 2:
        idx = np.random.randint(0, test_user_num)
        if idx not in samp_ids:
            samp_ids.append(idx)
    w2v_users = list(set(all_users) - set(samp_ids))
    w2v_users = pd.DataFrame(w2v_users, columns=['userid'])
    w2v_df = user_action.merge(w2v_users, on='userid', how='inner').sort_values('date_', ascending=True)
    sentences = w2v_df.groupby('userid')['feedid'].agg(lambda x: [str(e) for e in x])
    model = Word2Vec(sentences, hs=1, window=128, min_count=1, vector_size=hparams['emb_size'], seed=0)
    # model.wv.save_word2vec_format(os.path.join(hparams['output_dir, 'feed_w2v_embeddings.txt'), binary=False)
    embeddings, ids = [], []
    for idx in range(vocab_size):
        try:
            emb = model.wv[idx]
            embeddings.append(emb), ids.append(idx)
        except KeyError:
            continue
    embeddings = fill_embedding(np.array(embeddings), ids, vocab_size)
    return embeddings


def user_w2v_embedding(user_action, feed_embedding, vocab_size):
    user_action = user_action.sort_values('date_', ascending=True)
    sentences = user_action.groupby('userid')['feedid'].agg(list)
    embeddings, ids = [], []
    for idx in range(vocab_size):
        if idx in sentences.index:
            avg_emb = feed_embedding[sentences[idx], :].mean(axis=0)
            embeddings.append(avg_emb), ids.append(idx)
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = fill_embedding(embeddings, ids, vocab_size)
    return embeddings


def hist_act_seq(user, date_, user_action):
    hist_seq = user_action[(user_action['userid'] == user) & (user_action['date_'] < date_)]
    hist_seq = hist_seq.sort_values('date_', ascending=False).reset_index(drop=True)
    hist_seq = hist_seq['feedid'].tolist()
    max_len = hparams['max_hist_len']
    if len(hist_seq) >= max_len:
        return "|".join([str(e) for e in hist_seq[0:max_len]]), "|".join([str(e) for e in [1] * max_len])
    else:
        hist_len = len(hist_seq)
        return "|".join([str(e) for e in hist_seq + [0] * (max_len - hist_len)]), \
               "|".join([str(e) for e in [1] * hist_len + [0] * (max_len - hist_len)])


def hist_sequences(data, user_action):
    users = data[['userid', 'date_']].drop_duplicates()

    logger.info("交互历史序列")
    user_action_1 = user_action.copy()
    user_action_1['act'] = user_action[ALLACTIONS].sum(axis=1)
    user_action_2 = user_action_1[user_action_1['act'] > 0]
    seqs, masks = [], []
    for key, row in tqdm(users.iterrows()):
        seq, mask = hist_act_seq(row['userid'], row['date_'], user_action_2)
        seqs.append(seq)
        masks.append(mask)
    users['hist_acts_seq'] = seqs
    # users['hist_acts_mask'] = masks

    logger.info("展现但未交互序列")
    user_action_2 = user_action_1[user_action_1['act'] == 0]
    seqs, masks = [], []
    for key, row in tqdm(users.iterrows()):
        seq, mask = hist_act_seq(row['userid'], row['date_'], user_action_2)
        seqs.append(seq)
        masks.append(mask)
    users['no_act_seq'] = seqs
    # users['no_act_mask'] = masks

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
        # users['hist_%s_mask' % act] = masks

    logger.info("完成序列")
    feed_info = pd.read_csv(hparams['feed_info'])
    user_action_1 = user_action.merge(feed_info[['feedid', 'videoplayseconds']], on='feedid', how='inner')
    user_action_1['finished'] = user_action_1['play'] / user_action_1['videoplayseconds']
    user_action = user_action_1[user_action_1['finished'] > 0.99]
    seqs, masks = [], []
    for key, row in tqdm(users.iterrows()):
        seq, mask = hist_act_seq(row['userid'], row['date_'], user_action)
        seqs.append(seq)
        masks.append(mask)
    users['finished_seq'] = seqs
    # users['finished_mask'] = masks

    logger.info("未完成序列")
    seqs, masks = [], []
    user_action = user_action_1[user_action_1['finished'] < 0.01]
    for key, row in tqdm(users.iterrows()):
        seq, mask = hist_act_seq(row['userid'], row['date_'], user_action)
        seqs.append(seq)
        masks.append(mask)
    users['unfinished_seq'] = seqs
    # users['unfinished_mask'] = masks
    return users


def statis_feature(data, start_day, dim, t=7):
    data = data[(data['date_'] > day - t) & (data['date_'] < day)]
    data = data.groupby(dim)[ALLACTIONS].agg(['sum']).reset_index()
    return data


def main():
    vocab_size = {}

    # 所有id类特征加1，将0值留出作为填充值
    logger.info("读入数据。。。")
    feed_embeddings = pd.read_csv(hparams['feed_embeddings'])
    feed_embeddings['feedid'] = feed_embeddings['feedid'] + 1
    vocab_size['feedid'] = feed_embeddings['feedid'].max() + 1

    feed_info = pd.read_csv(hparams['feed_info'])
    feed_info, vocab_size = feed_info_process(feed_info, vocab_size)

    logger.info("embedding 降维。。。")
    feed_embeddings = pca_content_embedding(feed_embeddings, hparams['emb_size'], vocab_size['feedid'])

    logger.info("获取共现embedding。。。")
    if hparams['mode'] == 'offline_train':
        user_action = pd.read_csv(hparams['user_actions'])[0:5000]
    else:
        user_action = pd.read_csv(hparams['user_actions'])
    user_action['userid'] = user_action['userid'] + 1
    user_action['feedid'] = user_action['feedid'] + 1
    user_action['device'] = user_action['device'] + 1
    user_action = user_action.fillna(0)
    vocab_size['userid'] = user_action['userid'].max() + 1
    vocab_size['device'] = user_action['device'].max() + 1
    feed_w2v_embeddings = feed_w2v_embedding(user_action, vocab_size['feedid'])
    user_embeddings = user_w2v_embedding(user_action[user_action['date_'] <= hparams['train_day']],
                                         feed_w2v_embeddings, vocab_size['userid'])

    pickle_process('dump', os.path.join(hparams['output_dir'], 'pretrain_emb.pkl'),
                   data={'feed_emb': feed_embeddings,
                         'feed_w2v_embeddings': feed_w2v_embeddings,
                         'user_embeddings': user_embeddings})
    pickle_process('dump', os.path.join(hparams['output_dir'], 'vocab_size.pkl'), vocab_size)
    pickle_process('dump', os.path.join(hparams['output_dir'], 'side_info.pkl'), feed_info)
    print(vocab_size)
    del feed_embeddings
    del feed_w2v_embeddings
    del user_embeddings
    del feed_info
    gc.collect()

    # TODO：统计特征，统计前7天user、feed行为次数

    logger.info("划分数据集并提取context特征和序列特征。。。")
    train_data = user_action[user_action['date_'] == hparams['train_day']]
    train_seqs = hist_sequences(train_data, user_action)
    train_data = train_data.merge(train_seqs, on=['userid', 'date_'], how='left')
    user_context = statis_feature(user_action, hparams['train_day'], dim='userid', t=7)
    feed_context = statis_feature(user_action, hparams['train_day'], dim='feedid', t=7)
    train_data = train_data.merge(user_context, on=['userid'])
    train_data.to_csv(os.path.join(hparams['output_dir'], 'train.csv'), index=None)

    test_data = user_action[user_action['date_'] == hparams['test_day']]
    test_seqs = hist_sequences(test_data, user_action)
    test_data = test_data.merge(test_seqs, on=['userid', 'date_'], how='left')
    # 冷启动用户/物品
    test_data.loc[test_data['userid'] > vocab_size['userid'] - 1, 'userid'] = 0
    test_data.loc[test_data['feedid'] > vocab_size['feedid'] - 1, 'feedid'] = 0
    test_data.to_csv(os.path.join(hparams['output_dir'], 'test.csv'), index=None)


if __name__ == '__main__':
    main()
