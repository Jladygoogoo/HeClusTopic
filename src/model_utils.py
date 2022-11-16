import wandb
from datetime import datetime
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
                    x_rec, _ = ae(x)
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
        return: valid_ids_final: valid token ids, len(valid_ids_final) <= len(self.vocab)
        return: latent_embs: valid token latent embeddings, shape=(len(valid_ids_final), latent_dim)
        return: freq: valid token freqencies, shape=(len(valid_ids_final),)
        '''
        self.model.bert.eval()
        dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size)
        latent_embs = torch.zeros((len(self.vocab), self.config.latent_dim)).to(self.device)
        freq = torch.zeros(len(self.vocab), dtype=int).to(self.device)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                valid_pos = batch[2].to(self.device)
                latent_emb = self.model.get_latent_emb(input_ids, attention_mask, valid_pos)
                valid_ids = input_ids[valid_pos != 0]
                latent_embs.index_add_(0, valid_ids, latent_emb)
                freq.index_add_(0, valid_ids, torch.ones_like(valid_ids))
        valid_ids_final = torch.where(freq>0)[0]
        latent_embs = latent_embs[freq > 0]
        freq = freq[freq > 0]
        latent_embs = latent_embs / freq.unsqueeze(-1)
        if return_freq == True:
            return valid_ids_final, latent_embs, freq
        return valid_ids_final, latent_embs


    def _pre_clustering(self):
        '''
        Clusters initialization
        '''
        init_latent_emb_path = os.path.join(self.data_dir, "init_latent_emb.pt")
        if os.path.exists(init_latent_emb_path):
            utils.print_log(f"Loading initial latent embeddings from {init_latent_emb_path}.")
            valid_ids_final, latent_embs, freq = torch.load(init_latent_emb_path)
        else:
            valid_ids_final, latent_embs, freq = self.get_vocab_emb()
            utils.print_log(f"Saving initial embeddings to {init_latent_emb_path}.")
            torch.save((valid_ids_final, latent_embs, freq), init_latent_emb_path)

        # initialize vocab frequencies
        self.vocab_freq = torch.zeros(len(self.vocab)).to(self.device)
        self.vocab_freq.index_add_(0, valid_ids_final.to(torch.int), freq.to(torch.float))
        self.model.vocab_weights = utils.assign_weight_from_freq(self.vocab_freq, self.device)

        utils.print_log(f"Running K-Means for initialization...")
        self.model.kmeans.init_cluster(latent_embs.cpu().numpy(), sample_weight=freq.cpu().numpy())


    def train(self):
        self._init_model()
        self.model.to(self.device)
        self._pretrain_ae()
        self._pre_clustering()
        self.show_clusters(save=True, suffix="init")

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
        if self.config.wandb == True:
            wandb.watch(models=(self.model.ae, self.model.bert.encoder.layer[11]), log_freq=1)
        batch_size = self.config.batch_size
        step = 0
        for epoch in range(self.config.n_epochs):
            total_loss = 0
            rec_loss = 0
            kmeans_loss = 0
            co_loss = 0
            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                valid_pos = batch[2].to(self.device)
                bow = batch[3].to(self.device)
                valid_ids_set, co_matrix, bert_embs, bert_embs_rec, avg_latent_embs, cluster_ids = self.model(input_ids, attention_mask, valid_pos, bow)
                loss = self.model.get_loss(valid_ids_set, bert_embs, bert_embs_rec, co_matrix, avg_latent_embs, cluster_ids)
                total_loss += loss["total_loss"].item()
                rec_loss += loss["rec_loss"].item()
                kmeans_loss += loss["kmeans_loss"].item()
                co_loss += loss["co_loss"].item()
                loss["total_loss"].backward()
                optimizer.step()
                if (step+1) % self.config.wandb_log_interval == 0:
                    utils.wandb_log(
                        "train/loss", 
                        {
                            "total_loss": loss["total_loss"].item(),
                            "rec_loss": loss["rec_loss"].item(),
                            "kmeans_loss": loss["kmeans_loss"].item(),
                            "co_loss": loss["co_loss"].item(),
                        },
                        self.config.wandb)
                step += 1
            utils.print_log("Epoch-{}: total loss={:.4f} | rec loss={:.4f} | kmeans loss={:.4f} | co loss={:.4f}".format(
                epoch, total_loss/batch_size, rec_loss/batch_size, kmeans_loss/batch_size, co_loss/batch_size
            ))
            if (epoch+1) % 1 == 0:
                self.show_clusters(save=True, suffix="{}".format(epoch+1))
        # torch.save(self.model.state_dict(), pretrained_path)
        # utils.print_log(f"Pretrained model saved to {pretrained_path}")


    def show_clusters(self, save=False, visual=True, suffix=""):
        '''
        Show and save vocab clustering results.
        param: mode: "print"|"save"|"all"
        '''
        if len(suffix)>0: 
            suffix = "_{}".format(suffix)

        utils.print_log("Showing clusters results...")
        if self.model.kmeans.is_init == False:
            utils.print_log("None. Kmeans should be initiated first.")

        valid_ids, latent_embs = self.get_vocab_emb(return_freq=False)
        labels_prob = self.model.kmeans.assign_cluster(latent_embs, mode="soft")
        _, topk_id_mtx = torch.topk(labels_prob.t(), dim=1, k=20)
        valid_ids = valid_ids.detach().cpu().numpy()
        topk_id_mtx = topk_id_mtx.detach().cpu().numpy()
        utils.print_log("Clusters size={}, valid vocab size={}".format(len(topk_id_mtx), len(valid_ids)))
        
        # display and save
        topic_tokens = []
        for i in range(len(topk_id_mtx)):
            topic_tokens.append([self.inv_vocab[int(valid_ids[id_])] for id_ in topk_id_mtx[i]])
        for i, tokens in enumerate(topic_tokens):
            print("topic-{}: {}".format(i, ','.join(tokens)))
        if save == True:
            save_path = "results/{}/topics{}.txt".format(self.config.dataset, suffix)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            with open(save_path, 'w', encoding='utf8') as f:
                for i, tokens in enumerate(topic_tokens):
                    f.write("topic-{}: {}\n".format(i, ','.join(tokens))) 

        # visualize
        if visual == True:
            # use t-sne to visualize latent embeddings
            utils.tsne_vis(latent_embs.detach().cpu().numpy(), 
                fig_save_path=os.path.join("results", self.config.dataset, "tsne{}.png".format(suffix)))
                
            

        

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
    parser.add_argument("--wandb_log_interval", default=50, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument("--wandb", action="store_true", help="whether to log at wandb")
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

    if args.wandb:
        wandb.init(project="HeClusTopicModel", name=datetime.now().strftime("%y-%m-%d %H:%M:%S"))

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