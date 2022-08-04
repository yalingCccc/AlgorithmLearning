from utils.models.MMoE import MMoE
from utils.models.W2V import Word2Vec

def get_model(model_name):
    model_name = model_name.lower()
    if model_name == 'mmoe':
        return MMoE
    elif model_name == 'w2v':
        return Word2Vec
    else:
        print("未找到模型")
        return None