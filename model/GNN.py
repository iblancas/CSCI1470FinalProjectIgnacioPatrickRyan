import torch
import torch.nn as nn

class PhysicsGNN(nn.Module):
    def __init__(self, node_in_dim=7, edge_in_dim=4, embed_dim=64):
        """
        Permutation-invariant Graph Neural Network.
        node_in_dim: 3(pos) + 3(vel) + 1(mass) = 7
        edge_in_dim: 3(rel_pos) + 1(distance) = 4
        """
        super(PhysicsGNN, self).__init__()
        
        # TODO: Define MLPs for Message Generation (phi_e)
        # TODO: Define MLPs for Node Update (phi_n)
        self.embed_dim = embed_dim

    def forward(self, nodes, edges):
        """
        Performs Message Passing.
        Returns a permutation-invariant embedding for each body.
        """
        # TODO: Implement message passing loop for K layers
        # Returns embeddings of shape (3 bodies, embed_dim)
        pass