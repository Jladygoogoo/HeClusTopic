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

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=1)(
            delayed(utils._parallel_compute_distance)(X, self.cluster_centers[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, X, sample_weight=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters, sample_weight=sample_weight)
        model.fit(X)
        self.cluster_centers = model.cluster_centers_  # copy clusters

    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.cluster_centers[cluster_idx] +
                               eta * X[i])
            self.cluster_centers[cluster_idx] = updated_cluster

    def assign_cluster(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)
        return np.argmin(dis_mat, axis=1)



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
        print(z.shape)
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
        self.bert = BertModel.from_pretrained("bert-base-uncased")        
        self.kmeans = KmeansBatch(config)
        self.ae = AutoEncoder(self.config)
    
    def bert_encode(
            self,
            input_ids:torch.Tensor, 
            attention_mask:torch.Tensor, 
        ):
        # TODO: BERT embedding calculation reconsider
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_states = bert_outputs[0]
        return last_hidden_states


    def forward(
            self, 
            input_ids:torch.Tensor, 
            attention_mask:torch.Tensor, 
            valid_pos:torch.Tensor,
            train=True
        ):
        
        '''
        Get valid word ids, co-occur matrix, latent embeddings and cluster assignments.
        '''
        batch_size = len(input_ids)
        valid_mask = valid_pos != 0
        last_hidden_states = self.bert_encode(input_ids, attention_mask)
        latent_embs = self.encoder(last_hidden_states[valid_pos])
        valid_ids =  input_ids[valid_mask]
        valid_ids_set = list(set(valid_ids.view(-1).detach().cpu().numpy()))
        
        # calculate averaged latent embeddings
        avg_latent_embs = torch.zeros((len(valid_ids_set), self.config.latent_dim)).to(self.device)
        freq = torch.zeros(len(valid_ids_set), dtype=int).to(self.device)
        # map valid ids to position ids in avg_latent_embs
        valid_pos_ids = []
        for ids in valid_ids.detach().cpu().numpy():
            valid_pos_ids.append(list(map(lambda id_: valid_ids_set.index(id_), ids)))
        valid_pos_ids = torch.Tensor(valid_pos_ids).dtype(int).to(self.device)
        avg_latent_embs.index_add_(0, valid_pos_ids, latent_embs)
        freq.index_add_(0, valid_pos_ids, torch.ones_like(valid_ids))
        avg_latent_embs = avg_latent_embs[freq > 0]
        freq = freq[freq > 0]
        avg_latent_embs = avg_latent_embs / freq.unsqueeze(-1)  

        # construct co-occur matrix
        bow_matrix = np.zeros((batch_size, len(valid_ids_set)))
        for i, ids in enumerate(valid_pos_ids):
            for id_ in ids:
                bow_matrix[i][id_.item()] += 1
        co_matrix = np.zeros((len(valid_ids_set), len(valid_ids_set)))
        for i in range(len(valid_ids_set)):
            p_i = np.sum(bow_matrix[:,i]!=0) / batch_size
            for j in range(i+1, len(valid_ids_set)):
                p_j = np.sum(bow_matrix[:,j]!=0) / batch_size
                p_ij = np.sum((bow_matrix[:, i] * bow_matrix[:, j])>0) / batch_size
                n_p_ij = p_ij / (p_i*p_j)
                co_matrix[i][j] = n_p_ij
                co_matrix[j][i] = n_p_ij

        # cluster assignments
        cluster_ids = self.kmeans.assign_cluster(avg_latent_embs)
        # if on train mode, update cluster centers
        if train == True: 
            elem_count = np.bincount(cluster_ids,
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
            latent_embs = self.ae.get_latent_emb(last_hidden_states[valid_pos])
        else:
            latent_embs = self.ae.get_latent_emb(last_hidden_states)
        return latent_embs

    
    def get_loss(self, co_matrix, latent_embs, cluster_ids):
        batch_size = len(latent_embs)
        # kmeans loss
        kmeans_loss = torch.tensor(0.).to(self.device)
        cluster_centers = torch.FloatTensor(self.kmeans.cluster_centers).to(self.device)
        for i in range(batch_size):
            diff_vec = latent_embs[i] - cluster_centers[cluster_ids[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            kmeans_loss += 0.5 * torch.squeeze(sample_dist_loss)  

        # co-occur loss
        # TODO
        loss = {
            "kmeans_loss": kmeans_loss,
            "total_loss": kmeans_loss
        } 

        return loss
