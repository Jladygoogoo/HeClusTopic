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


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


def create_dataset(dataset_dir, text_file="texts.txt", max_len=512):
    data_file = os.path.join(dataset_dir, "data.pkl")
    if os.path.exists(data_file):
        print_log("Loading encoded texts from {}".format(data_file))
        # data = torch.load(loader_file)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
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
        valid_pos = ["NOUN", "VERB", "ADJ"]
        for i in inv_vocab:
            token = inv_vocab[i]
            if token in stop_words or token.startswith('##') \
            or token in string.punctuation or token.startswith('[') \
            or pos_tag([token], tagset='universal')[0][-1] not in valid_pos:
                filter_idx.append(i)
        valid_pos = attention_masks.clone()
        for i in filter_idx:
            valid_pos[input_ids == i] = 0

        data = {"input_ids": input_ids, "attention_masks": attention_masks, "valid_pos": valid_pos, "filter_ids": filter_idx}
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    return data   


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



def check_gpu_mem_usedRate():
    info = nvidia_info()
    # print(info)
    used = info['gpus'][0]['used']
    tot = info['gpus'][0]['total']
    print(f"GPU0 used: {used}, tot: {tot}, 使用率：{used/tot}")


if __name__ == "__main__":
    check_gpu_mem_usedRate()