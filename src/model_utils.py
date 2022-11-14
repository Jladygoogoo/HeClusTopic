import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from model import HeClusTopicModel, AutoEncoder
import os
from tqdm import tqdm
import argparse
import utils
import numpy as np


class MyDataset(Dataset):
    def __init__(self, input_ids, attention_mask, valid_pos, bows, vocab_size) -> None:
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.valid_pos = valid_pos
        self.bows = bows
        self.vocab_size = vocab_size

    def __getitem__(self, idx):
        src_bow = self.bows[idx]
        one_hot_bow = torch.zeros(self.vocab_size)
        item = list(zip(*src_bow))
        one_hot_bow[list(item[0])] = torch.tensor(list(item[1])).float()
        # one_hot_bow[self.filter_ids] = 0        
        return self.input_ids[idx], self.attention_mask[idx], self.valid_pos[idx], one_hot_bow
    
    def __len__(self):
        return len(self.input_ids)


class HeClusTopicModelUtils:
    def __init__(self, config) -> None:
        self.device = utils.get_device(config.device)
        config.device = self.device
        utils.print_log("Use device: {}".format(self.device))
        self.config = config
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.vocab = tokenizer.get_vocab()        
        self.inv_vocab = {v:k for k, v in self.vocab.items()}
        self.model = None
        self.data_dir = "datasets/{}".format(self.config.dataset)
        self._load_dataset()

    def _load_dataset(self):
        self.data, bows = utils.create_dataset(self.data_dir)
        self.dataset = MyDataset(self.data["input_ids"], self.data["attention_masks"], self.data["valid_pos"], bows, len(self.vocab))

    def _init_model(self, model_path=None):
        self.model = HeClusTopicModel(self.config)
        if model_path is not None:
            self.model.load_state_dict(torch.load(self.model))

    def _pretrain_ae(self):
        utils.print_log("Pretraining AutoEncoder...")
        ae = self.model.ae
        ae_model_path = os.path.join(self.data_dir, "ae.pt")
        if os.path.exists(ae_model_path):
            utils.print_log("Found existed AutoEncoder model in: {}".format(ae_model_path))
            ae.load_state_dict(torch.load(os.path.join(self.data_dir, "ae.pt")))
        else:
            train_dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size)
            # freeze BERT parameters when pretraining Autoencoder
            utils.freeze_parameters(self.model.bert.parameters())
            self.model.bert.eval()
            parameters = filter(lambda p: p.requires_grad, ae.parameters())
            optimizer = torch.optim.Adam(parameters, self.config.lr)
            for epoch in range(self.config.n_pre_epochs):
                total_loss = 0
                for batch in tqdm(train_dataloader):
                    optimizer.zero_grad()
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    x = self.model.bert_encode(input_ids, attention_mask)
                    x_rec = ae(x)
                    loss = F.mse_loss(x, x_rec)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                utils.print_log(f"epoch {epoch}: loss = {total_loss / (self.config.batch_size):.4f}")
            torch.save(ae.state_dict(), ae_model_path)
            utils.print_log("Save pretrained AutoEncoder model to: {}".format(ae_model_path))
        utils.print_log("AutoEncoder pretrain finished.")

    
    def get_vocab_emb(self, return_freq=True):
        '''
        Get vocab's average embeddings based on current model.
        '''
        self.model.bert.eval()
        dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size)
        latent_embs = torch.zeros((len(self.vocab), self.config.latent_dim)).to(self.device)
        freq = torch.zeros(len(self.vocab), dtype=int).to(self.device)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                valid_pos = batch[2].to(self.device)
                latent_emb = self.model.get_latent_emb(input_ids, attention_mask, valid_pos)
                valid_ids = input_ids[valid_pos != 0]
                latent_embs.index_add_(0, valid_ids, latent_emb)
                freq.index_add_(0, valid_ids, torch.ones_like(valid_ids))
        latent_embs = latent_embs[freq > 0]
        freq = freq[freq > 0]
        latent_embs = latent_embs / freq.unsqueeze(-1)
        if return_freq == True:
            return latent_embs, freq
        return latent_embs


    def _pre_clustering(self):
        init_latent_emb_path = os.path.join(self.data_dir, "init_latent_emb.pt")
        if os.path.exists(init_latent_emb_path):
            utils.print_log(f"Loading initial latent embeddings from {init_latent_emb_path}.")
            latent_embs, freq = torch.load(init_latent_emb_path)
        else:
            latent_embs, freq = self.get_vocab_emb()
            utils.print_log(f"Saving initial embeddings to {init_latent_emb_path}.")
            torch.save((latent_embs, freq), init_latent_emb_path)
        utils.print_log(f"Running K-Means for initialization...")
        self.model.kmeans.init_cluster(latent_embs.numpy(), sample_weight=freq.numpy())


    def train(self):
        self._init_model()
        self.model.to(self.device)
        self._pretrain_ae()
        self._pre_clustering()

        utils.freeze_parameters(self.model.bert.parameters())
        bert_params_for_finetune = [p for n,p in self.model.bert.named_parameters() if "layer.11" in n]
        utils.unfreeze_parameters(bert_params_for_finetune)
        bert_params_ids = list(map(id, self.model.bert.parameters()))
        other_params = list(filter(lambda p:id(p) not in bert_params_ids, self.model.parameters()))

        train_dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size)
        optimizer = torch.optim.Adam([
            {"params": bert_params_for_finetune, "lr": 1e-5},
            {"params": other_params, "lr": self.config.lr}
        ])
        utils.print_log("Start training HeCluTopicModel...")
        batch_size = self.config.batch_size
        for epoch in range(self.config.n_epochs):
            total_loss = 0
            kmeans_loss = 0
            co_loss = 0
            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                valid_pos = batch[2].to(self.device)
                bow = batch[3].to(self.device)
                valid_ids_set, co_matrix, avg_latent_embs, cluster_ids = self.model(input_ids, attention_mask, valid_pos, bow)
                loss = self.model.get_loss(co_matrix, avg_latent_embs, cluster_ids)
                total_loss += loss["total_loss"].item()
                kmeans_loss += loss["kmeans_loss"].item()
                co_loss += loss["co_loss"].item()
                loss["total_loss"].backward()
                optimizer.step()              
            utils.print_log("Epoch-{}: total loss={:.4f} | kmeans loss={:.4f} | co loss={:.4f}".format(
                epoch, total_loss/batch_size, kmeans_loss/batch_size, co_loss/batch_size
            ))
            if (epoch+1) % 1 == 0:
                self.show_clusters()
        # torch.save(self.model.state_dict(), pretrained_path)
        # utils.print_log(f"Pretrained model saved to {pretrained_path}")


    def show_clusters(self):
        '''
        Show vocab clustering results.
        '''
        utils.print_log("Showing clusters results...")
        if self.model.kmeans.is_init == False:
            utils.print_log("None. Kmeans should be initiated first.")
        latent_embs, freq = self.get_vocab_emb()
        labels = self.model.kmeans.assign_cluster(latent_embs)
        print(labels.shape)
        label_2_vids = {}
        for i, label in enumerate(labels):
            if label in label_2_vids:
                label_2_vids[label].append(i)
            else:
                label_2_vids[label] = [i]
        for label, vids in label_2_vids.items():
            print("=========== cluster-{} ===========".format(label))
            tokens = [self.inv_vocab[vid] for vid in vids if vid not in self.data["filter_ids"]]
            print(tokens)

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', default='20news')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_clusters', default=100, type=int, help='number of topics')
    parser.add_argument('--k', default=10, type=int, help='number of top words to display per topic')
    parser.add_argument('--bert_dim', default=768, type=int, help='embedding dimention of pretrained language model')
    parser.add_argument('--latent_dim', default=256, type=int, help='latent embedding dimention')
    parser.add_argument('--n_epochs', default=20, type=int, help='number of epochs for clustering')
    parser.add_argument('--n_pre_epochs', default=20, type=int, help='number of epochs for pretraining autoencoder')
    # parser.add_argument('--kappa', default=10, type=float, help='concentration parameter kappa')
    # parser.add_argument('--hidden_dims', default='[500, 500, 1000, 100]', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cluster_weight', default=0.1, type=float, help='weight of clustering loss')
    parser.add_argument('--emb_weight', default=0.0, type=float, help='weight of document embedding reconstruct loss')
    parser.add_argument('--bow_weight', default=1.0, type=float, help='weight of document bow reconstruct loss')

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    he_clus_utiler = HeClusTopicModelUtils(args)
    
    if args.train:
        he_clus_utiler.train()
    # if args.do_inference:
    #     model_path = os.path.join("datasets", args.dataset, "model.pt")
    #     try:
    #         he_clus_utiler.model.load_state_dict(torch.load(model_path))
    #     except:
    #         print("No model found! Run clustering first!")
    #         exit(-1)
    #     he_clus_utiler.inference(topk=args.k, suffix=f"_final")