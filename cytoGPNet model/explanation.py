import os
import numpy as np
from scipy import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
from torchvision import transforms, datasets
from loaddata import CyTOF_Dataset
from loadmodel import simple_AE, GaussianProcessLayer, Attention_Layer, Simple_Classifier


####################

def setup_args():

    options = argparse.ArgumentParser()

    # data directory
    options.add_argument('-datadir', '--data-dir', action="store", dest="data_dir", default='./HEUvsUE')
    options.add_argument('-fold', action="store", dest="fold", default = 1, type=int)

    # save and directory options
    options.add_argument('-sd', '--save-dir', action="store", dest="save_dir", default='./cytoGPNet_output')
    options.add_argument('--save-freq', action="store", dest="save_freq", default=10, type=int)

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=1, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-4, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-2, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=100, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)


    # hyperparameters
    # options.add_argument('--alpha', action="store", default=1., type=float) # weight for classfication loss compared to discriminative loss
    # options.add_argument('--hidden-dims', action="store", dest="hidden_dims", default=4, type=int)
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=2, type=int) # size of dimension for latent space of autoencoder
    options.add_argument('--num-inducing-points', action="store", dest="num_inducing_points", default=100, type=int)


    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    return options.parse_args()

args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False

def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).float().sum().item()
    return correct

device = torch.device("cuda" if args.use_gpu else "cpu")

# ========== Load Data ==========
dataset = CyTOF_Dataset(datadir=args.datadir, name=args.filename, mode='test')
cyto_tensor = torch.from_numpy(dataset.data[1]).float()  # (N, C, T, M)
labels = torch.tensor(dataset.data[0]["label"].values).float()
patient_ids = dataset.data[0]["patient_id"].values
markerNames = dataset.data[2]

N, C, T, M = cyto_tensor.shape

# ========== Load Model Components ==========
autoencoder = simple_AE(input_dim=M, embed_dim=args.latent_dims).to(device)
autoencoder.load_state_dict(torch.load(os.path.join(args.save_dir, f"simpleAE_finetune_epoch{args.max_epochs}.pth")))
autoencoder.eval()

attention = Attention_Layer().to(device)
attention.load_state_dict(torch.load(os.path.join(args.save_dir, f"Attention_Layer_epoch{args.max_epochs}.pth")))
attention.eval()

classifier = Simple_Classifier(nz=1).to(device)
classifier.load_state_dict(torch.load(os.path.join(args.save_dir, f"Simple_Classifier_epoch{args.max_epochs}.pth")))
classifier.eval()

# Prepare GP
with torch.no_grad():
    flat = cyto_tensor.reshape(-1, M)[:args.num_inducing_points, :].to(device)
    z_induce = autoencoder.encoder(flat)

gp_layer = GaussianProcessLayer(
    input_dim=args.latent_dims,
    num_inducing_points=z_induce.size(0),
    inducing_points=z_induce.clone(),
    mean_inducing_points=z_induce.clone(),
    grid_bounds=[
        (z_induce[:, 0].min().item(), z_induce[:, 0].max().item()),
        (z_induce[:, 1].min().item(), z_induce[:, 1].max().item())
    ],
    likelihood_type='classification',
    using_ngd=True,
    using_ksi=False,
    using_ciq=False,
    using_sor=False,
    using_OrthogonallyDecouple=False
).to(device)
gp_layer.load_state_dict(torch.load(os.path.join(args.save_dir, f"GaussianProcessLayer_epoch{args.max_epochs}.pth")))
gp_layer.eval()

likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
likelihood.load_state_dict(torch.load(os.path.join(args.save_dir, f"BernoulliLikelihood_epoch{args.max_epochs}.pth")))
likelihood.eval()

# ========== Define Wrapper ==========
class CytoGPNetModel(torch.nn.Module):
    def __init__(self, ae, gp, attn, clf):
        super().__init__()
        self.ae = ae
        self.gp = gp
        self.attn = attn
        self.clf = clf

    def forward(self, x):
        B, T, M = x.shape
        z = self.ae.encoder(x.view(-1, M))  # (B*T, latent_dim)
        f_dist = self.gp(z)
        f_mean = f_dist.mean.view(B, T)
        pooled = self.attn(f_mean).unsqueeze(1)  # (B, 1)
        return self.clf(pooled)  # (B, 1)

full_model = CytoGPNetModel(autoencoder, gp_layer, attention, classifier).to(device)
full_model.eval()

# ========== Run BBMP ==========
bbmp = BBMPExp(model=full_model, num_markers=M, device=device)
mask_records = []

for i in tqdm(range(N), desc="Running BBMP"):
    x = cyto_tensor[i].permute(2, 0, 1).reshape(1, -1, M).to(device)  # (1, C*T, M)
    y_true = int(labels[i].item())
    pid = patient_ids[i]

    with torch.no_grad():
        y_pred = int(full_model(x).squeeze().item() > 0.5)

    mask_true = bbmp.explain(x, target_label=y_true)
    mask_error = bbmp.explain(x, target_label=y_pred)

    for m_idx in range(M):
        mask_records.append({
            "patient_id": pid,
            "marker_name": markerNames[m_idx],
            "mask_true": mask_true[m_idx].item(),
            "mask_error": mask_error[m_idx].item()
        })

# ========== Save CSV ==========
mask_df = pd.DataFrame(mask_records)
mask_df.to_csv(os.path.join(args.save_dir, f"BBMP_mask_scores_epoch{args.max_epochs}.csv"), index=False)
print("BBMP masks saved to CSV.")