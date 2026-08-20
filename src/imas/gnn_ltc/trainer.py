from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from .model import ClusterCommLTC


@dataclass
class GNNLTCConfig:
    hidden_dim: int = 128
    epochs: int = 300
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    teacher_weight: float = 1.0
    reconstruction_weight: float = 0.2
    smooth_weight: float = 0.2


def train_communication_pseudotime(features, weak_teacher, reconstruction_target,
                                   neighbor_index, neighbor_mask, edge_features,
                                   config: GNNLTCConfig = GNNLTCConfig(), device="cuda"):
    device=torch.device(device); x=torch.as_tensor(features,np.float32,device=device); teacher=torch.as_tensor(weak_teacher,np.float32,device=device)
    target=torch.as_tensor(reconstruction_target,np.float32,device=device); ni=torch.as_tensor(neighbor_index,device=device); nm=torch.as_tensor(neighbor_mask,device=device); ef=torch.as_tensor(edge_features,np.float32,device=device)
    model=ClusterCommLTC(x.shape[1],config.hidden_dim,target.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    for epoch in range(1,config.epochs+1):
        opt.zero_grad(); time,recon,tau=model(x,ni,nm,ef)
        loss=config.teacher_weight*F.mse_loss(time,teacher)+config.reconstruction_weight*F.mse_loss(recon,target)
        # graph-neighbor temporal smoothness
        neighbor_time=time[ni]; valid=nm.float(); smooth=((neighbor_time-time[:,None])**2*valid).sum()/(valid.sum()+1e-8)
        loss=loss+config.smooth_weight*smooth; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip); opt.step()
    model.eval()
    with torch.no_grad(): time,recon,tau=model(x,ni,nm,ef)
    return model, time.cpu().numpy(), tau.cpu().numpy(), recon.cpu().numpy()
