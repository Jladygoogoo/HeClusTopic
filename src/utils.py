import wandb
import pynvml
import pickle
from datetime import datetime
from collections import Counter
from tqdm import tqdm
import torch
from gensim.corpora.mmcorpus import MmCorpus
from nltk.corpus import stopwords
from transformers import BertTokenizer
import os
import string
from nltk.tag import pos_tag
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics, manifold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


def create_dataset(dataset_dir, text_file="texts.txt", max_len=512):
    data_file = os.path.join(dataset_dir, "data.pkl")
    bow_file = os.path.join(dataset_dir, "bows.mm")
    if os.path.exists(data_file) and os.path.exists(bow_file):
        print_log("Loading encoded texts from {}".format(data_file))
        # data = torch.load(loader_file)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        bows = MmCorpus(bow_file)
        print_log("Loaeded.")
    else:
        print_log(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        vocab = tokenizer.get_vocab()
        inv_vocab = {k:v for v, k in vocab.items()}
        corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
        docs = []
        for doc in corpus.readlines():
            content = doc.strip()
            docs.append(content)

        print_log(f"Converting texts into tensors.")
        encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']

        print_log("Generate valid positions...")
        stop_words = set(stopwords.words('english'))
        filter_idx = []
        valid_tag = ["NOUN", "VERB", "ADJ"]
        for i in inv_vocab:
            token = inv_vocab[i]
            if len(token)<2 or token in stop_words or token.startswith('##') \
            or token in string.punctuation or token.startswith('[') \
            or pos_tag([token], tagset='universal')[0][-1] not in valid_tag:
                filter_idx.append(i)
        valid_pos = attention_masks.clone()
        for i in filter_idx:
            valid_pos[input_ids == i] = 0

        print_log("Constructing BOW into {}...".format(bow_file))
        bows = []
        for i in range(len(input_ids)):
            bow = {}
            for idx in input_ids[i]:
                if idx in bow:
                    bow[idx] += 1
                else:
                    bow[idx] = 1
            bows.append(list(bow.items()))            

        data = {"input_ids": input_ids, "attention_masks": attention_masks, "valid_pos": valid_pos, "filter_ids": filter_idx}
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        MmCorpus.serialize(bow_file, bows)            
    return data, bows


def get_device(id_=0):
    if not torch.cuda.is_available() or id_==-1:
        return torch.device("cpu")
    else:
        return torch.device("cuda:{}".format(id_))


def print_log(s, author="wnj"):
    print("[{}][{}] {}".format(datetime.now().strftime("%m-%d %H:%M:%S"), author, s))


def nvidia_info():
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    try:
        pynvml.nvmlInit()
        nvidia_dict["nvidia_version"] = pynvml.nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] =pynvml. nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": pynvml.nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{pynvml.nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": pynvml.nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except pynvml.NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    return nvidia_dict


def wandb_log(title, log_dict, use_wandb=True):
    '''
    Wrap wandb.log function.
    param: title: str like "train/mertic"
    param: log_dict: dict like {"f1": 0.981, "acc": 0.996}
    param: use_wandb: is use_wandb is False, then do nothing
    '''
    if use_wandb == True:
        new_log_dict = {"{}/{}".format(title, k): v for k,v in log_dict.items()}
        wandb.log(new_log_dict)


def check_gpu_mem_usedRate():
    info = nvidia_info()
    # print(info)
    used = info['gpus'][0]['used']
    tot = info['gpus'][0]['total']
    print(f"GPU0 used: {used}, tot: {tot}, 使用率：{used/tot}")


def freeze_parameters(parameters):
    for p in parameters:
        p.requires_grad = False

def unfreeze_parameters(parameters):
    for p in parameters:
        p.requires_grad = True

def calc_cosine_dist_torch(a, b, mode="m2m"):
    '''
    Calculate consine distance between two matrix.
    param: a: torch.Tensor, shape=(size_a, latent_dim)
    param: b: torch.Tensor, shape=(size_b, latent_dim)
    param: mode: "o2o"|"m2m"
    return: res, shape=(size_a, size_b)-"m2m", shape=(size_a)="o2o"
    '''
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    if mode == "m2m":
        res = torch.mm(a_norm, b_norm.transpose(0,1))    
    elif mode == "o2o":
        if a_norm.shape != b_norm.shape:
            raise ValueError("a.shape and b.shape should be identical under 'o2o' mode.")
        res = torch.sum(a_norm * b_norm, dim=1)
    return res


def assign_weight_from_freq(vocab_freq, device):
    q = torch.tensor([0.25, 0.5, 0.75]).to(device)
    ql = torch.quantile(vocab_freq, q) # shape=(3,)
    vocab_weight = torch.zeros_like(vocab_freq).to(device)
    vocab_weight[torch.where(vocab_freq<=ql[0])] = 0.1
    vocab_weight[torch.where(vocab_freq>ql[0])] = 0.25
    vocab_weight[torch.where(vocab_freq>ql[1])] = 0.5
    vocab_weight[torch.where(vocab_freq>ql[2])] = 1.0
    return vocab_weight


def tsne_vis(features, fig_save_path):
    # 建模
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=21) # tsne模型
    X_tsne = tsne.fit_transform(features)

    # 绘图
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    # plt.scatter(X_norm[:,0], X_norm[:,1], c=labels, cmap=plt.cm.tab20b)
    plt.scatter(X_norm[:,0], X_norm[:,1])

    plt.xticks([])
    plt.yticks([])
    # plt.show()
    plt.savefig(fig_save_path)





if __name__ == "__main__":
    check_gpu_mem_usedRate()