import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import GCNConv

class Encoder(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_layers: int = 2):
        super(Encoder, self).__init__()
        self.base_model = GCNConv
        self.activation = F.relu
        assert num_layers >= 2
        self.num_layers= num_layers
        self.conv = [GCNConv(dim_in, 2 * dim_out)]
        for _ in range(1, num_layers-1):
            self.conv.append(GCNConv(2 * dim_out, 2 * dim_out))
        self.conv.append(GCNConv(2 * dim_out, dim_out))
        self.conv = nn.ModuleList(self.conv)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.num_layers):
            x = self.activation(self.conv[i](x, edge_index))
        return x
    
  
    
class AGCLModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, n_hidden: int, n_proj_hidden: int,
                 tau: float = 0.5):
        super(AGCLModel, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc_layer1 = torch.nn.Linear(n_hidden, n_proj_hidden)
        self.fc_layer2 = torch.nn.Linear(n_proj_hidden, n_hidden)
      
        
    def forward(self, x: torch.Tensor,
                Adj: torch.Tensor) -> torch.Tensor:
       
        return self.encoder(x, Adj)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc_layer1(z))
        return self.fc_layer2(z)

    #Codes are modified from https://github.com/Shengyu-Feng/ARIEL
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        
            
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() 

        return ret




