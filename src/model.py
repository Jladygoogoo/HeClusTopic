from transformers import BertPreTrainedModel, BertModel
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import utils

class KmeansBatch:
    def __init__(self, config):
        self.config = config
        self.n_features = config.latent_dim
        self.n_clusters = config.n_clusters
        self.cluster_centers = np.zeros((self.n_clusters, self.n_features))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.is_init = False

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=1)(
            delayed(utils._parallel_compute_distance)(X, self.cluster_centers[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, X, sample_weight=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters)
        model.fit(X)
        # model.fit(X, sample_weight=sample_weight)
        self.cluster_centers = torch.tensor(model.cluster_centers_).to(self.config.device)  # copy clusters
        self.is_init = True

    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        with torch.no_grad():
            n_samples = X.shape[0]
            for i in range(n_samples):
                self.count[cluster_idx] += 1
                eta = 1.0 / self.count[cluster_idx]
                updated_cluster = ((1 - eta) * self.cluster_centers[cluster_idx] +
                                eta * X[i])
                self.cluster_centers[cluster_idx] = updated_cluster

    def assign_cluster(self, X):
        """ Assign samples in `X` to clusters """
        cluster_centers = F.normalize(self.cluster_centers, dim=-1)
        sim = torch.matmul(X, cluster_centers.transpose(0,1))
        p = F.softmax(sim, dim=-1)
        return torch.argmax(p, axis=1)

        # detached_X = X.detach().cpu().numpy()
        # dis_mat = self._compute_dist(detached_X)



class AutoEncoder(nn.Module):
    ''' Use AutoEncoder to initialize HeClusTopicModel's encoder. '''
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.bert_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, config.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, config.bert_dim)
        )
    
    def get_latent_emb(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=-1)
        return z        

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=-1)
        x_rec = self.decoder(z)
        return x_rec    



class HeClusTopicModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.device = config.device
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)   
        self.kmeans = KmeansBatch(config)
        self.ae = AutoEncoder(self.config)    
        # set bert output layers
        n_layers = 4
        self.layers = list(range(-n_layers, 0))
    
    def bert_encode(
            self,
            input_ids:torch.Tensor, 
            attention_mask:torch.Tensor, 
        ):
        # TODO: BERT embedding calculation reconsider
        # bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        # specified_hidden_states = [bert_outputs.hidden_states[i] for i in self.layers]
        # specified_embeddings = torch.stack(specified_hidden_states, dim=0)
        # token_embeddings = torch.squeeze(torch.mean(specified_embeddings, dim=0))
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        token_embeddings = bert_outputs[0]
        return token_embeddings


    def forward(
            self, 
            input_ids:torch.Tensor, 
            attention_mask:torch.Tensor, 
            valid_pos:torch.Tensor,
            bow: torch.Tensor,
            train=True
        ):
        
        '''
        Get valid word ids, co-occur matrix, latent embeddings and cluster assignments.
        '''
        batch_size = len(input_ids)
        valid_mask = valid_pos != 0
        last_hidden_states = self.bert_encode(input_ids, attention_mask)
        latent_embs = self.ae.encoder(last_hidden_states[valid_mask]) # shape=(torch.sum(valid_pos), latent_dim)
        valid_ids =  input_ids[valid_mask] # shape=(torch.sum(valid_pos),)
        valid_ids_set = list(set(valid_ids.detach().cpu().numpy()))
        
        # calculate averaged latent embeddings
        avg_latent_embs = torch.zeros((len(valid_ids_set), self.config.latent_dim)).to(self.device)
        freq = torch.zeros(len(valid_ids_set), dtype=int).to(self.device)
        # map valid ids to position ids in avg_latent_embs
        valid_pos_ids = [valid_ids_set.index(int(idx.item())) for idx in valid_ids]
        valid_pos_ids = torch.Tensor(valid_pos_ids).to(torch.int).to(self.device)
        avg_latent_embs.index_add_(0, valid_pos_ids, latent_embs)
        freq.index_add_(0, valid_pos_ids, torch.ones_like(valid_ids))
        avg_latent_embs = avg_latent_embs[freq > 0]
        freq = freq[freq > 0]
        avg_latent_embs = avg_latent_embs / freq.unsqueeze(-1)  

        # construct co-occur matrix
        trans_matrix = torch.zeros(bow.shape[1], len(valid_ids_set)).to(self.device)
        for i, idx in enumerate(valid_ids_set):
            trans_matrix[idx, i] = 1
        bow_matrix = torch.mm(bow, trans_matrix)
        co_matrix = torch.zeros((len(valid_ids_set), len(valid_ids_set))).to(self.device)
        p_i = torch.sum(bow_matrix!=0, dim=0) / batch_size
        p_ij = torch.mm((bow_matrix.t()!=0).to(torch.float), (bow_matrix!=0).to(torch.float)) / batch_size
        co_matrix = p_ij / p_i.unsqueeze(dim=1) / p_i

        # cluster assignments
        cluster_ids = self.kmeans.assign_cluster(avg_latent_embs)
        # if on train mode, update cluster centers
        if train == True: 
            elem_count = torch.bincount(cluster_ids,
                                     minlength=self.config.n_clusters)
            for k in range(self.config.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.kmeans.update_cluster(avg_latent_embs[cluster_ids == k], k)

        return valid_ids_set, co_matrix, avg_latent_embs, cluster_ids


    def get_latent_emb(self, input_ids, attention_mask, valid_pos=None):
        last_hidden_states = self.bert_encode(input_ids, attention_mask)
        if valid_pos is not None:
            mask = valid_pos != 0
            latent_embs = self.ae.get_latent_emb(last_hidden_states[mask])
        else:
            latent_embs = self.ae.get_latent_emb(last_hidden_states)
        return latent_embs

    
    def get_loss(self, co_matrix, latent_embs, cluster_ids):
        '''
        param: co_matrix: batch token co-occur matrix, shape=(batch_token_size, batch_token_size)
        param: latent_embs: batch token embeddings, shape=(batch_token_size, latent_dim)
        param: cluster_ids: batch token cluster id, shape=(batch_token_size,)
        '''
        batch_size = len(latent_embs)
        # kmeans loss
        kmeans_loss = torch.tensor(0.).to(self.device)
        cluster_centers = self.kmeans.cluster_centers.to(self.device)
        for i in range(batch_size):
            diff_vec = latent_embs[i] - cluster_centers[cluster_ids[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            kmeans_loss += 0.5 * torch.squeeze(sample_dist_loss)  

        # co-occur loss
        weight_matrix = co_matrix / 10
        dist_matrix = utils.calc_cosine_dist_torch(latent_embs, latent_embs)
        co_loss = torch.sum(dist_matrix * weight_matrix / 2)

        total_loss = kmeans_loss + co_loss
    
        loss = {
            "kmeans_loss": kmeans_loss,
            "co_loss": co_loss,
            "total_loss": total_loss
        } 

        return loss
