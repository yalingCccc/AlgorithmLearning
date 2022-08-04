import pickle

def pickle_process(mode, dir, data=None):
    if mode == 'load':
        f = open(dir, 'rb')
        data = pickle.load(f)
        f.close()
        return data
    else:
        f = open(dir, 'wb')
        pickle.dump(data, f)
        f.close()