import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from tqdm import tqdm
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_model', type=str, default='model.pth')
    parser.add_argument('--output_h5ad', type=str, default='output.h5ad')
    return parser.parse_args()


class scVIVAE(nn.Module):
    def __init__(self, n_genes, n_batches, n_layers=2, n_latent=30, dropout_rate=0.01):
        super().__init__()
        layers = [nn.Linear(n_genes + n_batches, 128), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(n_layers-1):
            layers += [nn.Linear(128, 128), nn.ReLU(), nn.Dropout(dropout_rate)]
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(128, n_latent)
        self.fc_logvar = nn.Linear(128, n_latent)
        self.matrix = nn.Sequential(nn.Linear(n_latent, n_latent), nn.ReLU(), nn.Linear(n_latent, n_latent))
        self.decoder = nn.Sequential(
            nn.Linear(n_latent + n_batches, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_genes),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(n_genes, n_genes)
        
        
        self.fc_theta = nn.Linear(n_genes, n_genes)
        self.n_batches = n_batches

    def forward(self, x, batch):
        batch_onehot = F.one_hot(batch, num_classes=self.n_batches).float().to(x.device)
        h = torch.cat([x, batch_onehot], dim=1)
        q = self.encoder(h)
        mu = self.fc_mu(q)
        logvar = self.fc_logvar(q)
        matrix = self.matrix(mu)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        dec_in = torch.cat([z, batch_onehot], dim=1)
        recon = self.decoder(dec_in) ## px_rate
        recon_mean = F.softplus(self.fc_mean(recon))
        
        
        recon_theta = F.softplus(self.fc_theta(recon))


        return mu, logvar,recon_mean,recon_theta, matrix

def nb_loss(x, mean, theta):
    ll = (torch.lgamma(theta + x) - torch.lgamma(x+1) - torch.lgamma(theta)
          + theta*torch.log(theta+1e-8) + x*torch.log(mean+1e-8)
          - (theta + x)*torch.log(theta + mean+1e-8))
    return -ll.mean(dim=1).mean() 
    
def loss_fn(x, mu, logvar, px_rate, px_r,kl_weight = 1,pt = False):
    px_rate = torch.clamp(px_rate, min=1e-5)
    px_r = torch.clamp(px_r, min=1e-5)
    recons = nb_loss(x, px_rate, px_r)   ###
    kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp(), dim=1).mean()
    if pt == True:
        print(kl)
        print(x[0:2,930:950])
        print(px_rate[0:2,930:950])
    return recons + kl_weight * kl    ####

def triplet_margin_loss(mu, labels, logvar = None, margin=1.0, num_triplets=None):
    device = mu.device
    labels = labels.cpu().numpy()
    batch = mu.size(0)
    label_to_indices = {}
    for idx, l in enumerate(labels):
        label_to_indices.setdefault(l, []).append(idx)
    all_labels = list(label_to_indices.keys())
    triplet_indices = []
    for _ in range(num_triplets):
        anchor_label = np.random.choice(all_labels)
        pos_indices = label_to_indices[anchor_label]
        if len(pos_indices) < 2:
            continue
        i, j = np.random.choice(pos_indices, size=2, replace=False)
        neg_label = np.random.choice([l for l in all_labels if l != anchor_label])
        k = np.random.choice(label_to_indices[neg_label])
        triplet_indices.append((i, j, k))
    if not triplet_indices:
        return torch.tensor(0.0, device=device)
    anchors = torch.stack([mu[a] for a, b, c in triplet_indices])
    positives = torch.stack([mu[b] for a, b, c in triplet_indices])
    negatives = torch.stack([mu[c] for a, b, c in triplet_indices])
    tmloss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='sum')
    return tmloss(anchors, positives, negatives)

def prepare_tensor(data, args, max_cells_per_batch=50000): 
    

    X = data.layers["counts"]
    n_cells = X.shape[0]
    
    if n_cells > max_cells_per_batch and hasattr(X, "todense"):
        
        batches = []
        for i in range(0, n_cells, max_cells_per_batch):
            end_idx = min(i + max_cells_per_batch, n_cells)
            batch_X = X[i:end_idx]
            if hasattr(batch_X, "todense"):
                batch_X = batch_X.todense()
            batches.append(torch.tensor(batch_X, dtype=torch.float32))
        X = torch.cat(batches, dim=0)
    else:
        if hasattr(X, "todense"):
            X = X.todense()
        X = torch.tensor(X, dtype=torch.float32)
    
    batch = data.obs["Dataset"].astype('category').cat.codes.values
    batch = torch.tensor(batch, dtype=torch.long)
    labels = data.obs['Immune_type'].astype('category').cat.codes.values
    labels = torch.tensor(labels, dtype=torch.long)
    return X, batch, labels




def main():
    args = get_args()
    args.additional_layer = False 
    print("Loading AnnData file:", args.input)
    adata0 = sc.read_h5ad(args.input)
    print(adata0)
    #adata0.layers['counts'] = adata0.X.copy()
    #sc.pp.normalize_total(adata0, target_sum=1e4)
    #sc.pp.log1p(adata0)
    #adata0.layers['logNor'] = adata0.X.copy()
    sc.pp.highly_variable_genes(adata0, n_top_genes = 22000, batch_key = 'Dataset')
    hvg = adata0.var.index[adata0.var['highly_variable']]
    adata = adata0[:, hvg]
    print(adata.shape)
    print("Splitting data into training and test sets ...")
    n_obs = adata.n_obs
    indices = np.random.choice(n_obs, int(n_obs * 0.1), replace=False)
    test_data = adata[indices].copy()
    train_indices = np.setdiff1d(np.arange(n_obs), indices)
    train_data = adata[train_indices].copy()
    counts, batch_tensor, labels_tensor = prepare_tensor(train_data, args)
    n_genes = counts.shape[1]
    n_batches = int(batch_tensor.max().item()) + 1
    n_cells = counts.shape[0]
    train_size = int(n_cells * 0.9)
    val_size = n_cells - train_size
    dataset = TensorDataset(counts, batch_tensor, labels_tensor)
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)


    #device_count = torch.cuda.device_count()
    #print(f"Number of available GPUs: {device_count}")
    #device = [f"cuda:{i}" for i in range(device_count)]
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}.")
    model = scVIVAE(n_genes, n_batches, n_layers=4, n_latent=30, dropout_rate=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    max_epochs = 300
    early_stopping = True
    patience = 45
    best_val_loss = float('inf')
    patience_counter = 0
    vae_weight = 1e3
    margin = 0.2



    print("Starting training loop ...")
    for epoch in tqdm(range(max_epochs), desc='Epoch'):
        model.train()
        train_loss_sum = 0
        for x, batch, label in train_loader:
            x, batch, label = x.to(device), batch.to(device), label.to(device)
            optimizer.zero_grad()
            mu, logvar, recon_mean, recon_theta, matrix = model(x, batch)
            vae_loss = vae_weight * loss_fn(x, mu, logvar, recon_mean, recon_theta,kl_weight = 1e-4)
            if args.additional_layer:
                triplet_loss =  triplet_margin_loss(matrix, label, logvar, margin=margin, num_triplets=512)
            else:
                triplet_loss =  triplet_margin_loss(mu, label, logvar, margin=margin, num_triplets=512)
            loss =   vae_loss +triplet_loss #+
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        #printcorr(mu)
        #print(vae_loss,triplet_loss)
    #torch.save(model,'model.pth')
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, batch, label in val_loader:
                x, batch, label = x.to(device), batch.to(device), label.to(device)
                mu, logvar, recon_mean, recon_theta, matrix = model(x, batch)
                vae_loss = vae_weight * loss_fn(x, mu, logvar, recon_mean, recon_theta,kl_weight = 1e-4)
                triplet_loss =  triplet_margin_loss(mu, label, logvar, margin=margin, num_triplets=512)
                val_loss = vae_loss + triplet_loss
                val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if early_stopping and patience_counter > patience:
            print("Early stopping triggered.")
            break
            
    torch.save(model, './Integrate_clustering/model.pth')

    model.eval()
    counts, batch_tensor, _ = prepare_tensor(adata, args)
    all_latent = []
    all_pred = [] 
    with torch.no_grad():
        for i in range(0, counts.shape[0], 512):
            x_part = counts[i:i+512].to(device)
            batch_part = batch_tensor[i:i+512].to(device)
            mu, logvar, recon_mean,recon_theta, matrix = model(x_part, batch_part)
            if args.additional_layer:
                all_latent.append(mu.cpu())
                all_pred.append(matrix.cpu()) ##mu/matrix
            else:
                all_latent.append(mu.cpu())
    integrated_data = torch.cat(all_latent, dim=0).numpy()
#predicted_data = torch.cat(all_pred, dim=0).numpy()
    if args.additional_layer:
        savedata = ad.AnnData(X = integrated_data,obs = adata.obs)
        savedata.obsm['New_matrix'] = integrated_data
        print(f"Saving AnnData with matrix to {args.output_h5ad}")
        savedata.write('adata_integrate.h5ad')
        print("All done.")
    else:
        adata.obsm["New_embedding"] = integrated_data
        print(f"Saving AnnData with latent embedding to {args.output_h5ad}")
        adata.write('./Integrate_clustering/adata_integrate.h5ad')
        sc.pp.neighbors(adata, use_rep = 'New_embedding')
        sc.tl.umap(adata)
        sc.pl.umap(adata,
                   color=["Immune_type", 'Dataset', 'MRC1', 'CD3D','CD8A','FOXP3','TGFBR3','S100A9'],
                   ncols=2)
        print(adata.obsm['New_embedding'])
        plt.savefig('./Integrate_clustering/inte_new.png',  dpi=200)
        print("All done.")

if __name__ == "__main__":
    main()




