import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):
    def __init__(self, embed_dim=64):
        """
        The 'God-view' Judge used ONLY during training (CTDE).
        """
        super(CentralizedCritic, self).__init__()
        
        # Takes the concatenated embeddings of ALL 3 bodies (3 * embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single Value scalar V(s)
        )

    def forward(self, all_embeddings):
        return self.net(all_embeddings)