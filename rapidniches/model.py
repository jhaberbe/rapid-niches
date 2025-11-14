import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

from nflows.flows import Flow
from nflows.transforms import MaskedAffineAutoregressiveTransform, RandomPermutation, CompositeTransform
from nflows.distributions import StandardNormal

class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels=16,
        hidden_channels=32,
        out_channels=16,
        num_layers=3,
        edge_dim=1,
        dropout=0.1,
        n_transforms=6
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Graph Attention Setup
        # Input projection onto larger graph representation.
        self.input_proj = nn.Linear(
            in_channels,
            hidden_channels
        )

        # Graph attention with edge attributes.
        self.convs = nn.ModuleList([
            TransformerConv(
                hidden_channels,
                hidden_channels,
                edge_dim=edge_dim,
                heads=4,
                concat=False,
                dropout=dropout
            ) 
            for _ in range(num_layers)
        ])

        # Layer Normalization
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_channels) 
                for _ in range(num_layers)
            ]
        )

        # Project hidden dimension to output dimension.
        self.output_proj = nn.Linear(
            hidden_channels,
            out_channels
        )

        # Normalizing flow to predict tokens over the latent space.
        self.flow = self._build_flow(out_channels, n_transforms, hidden_channels)

    def _build_flow(self, embedding_dim, n_transforms, hidden_dim):
        base_dist = StandardNormal([embedding_dim])

        # Normalizing flow, with random feature permutations.
        transforms = []
        for i in range(n_transforms):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=embedding_dim,
                    hidden_features=hidden_dim,
                    context_features=embedding_dim
                )
            )
            if i < n_transforms - 1:
                transforms.append(RandomPermutation(features=embedding_dim))

        return Flow(CompositeTransform(transforms), base_dist)

    def graph_forward(self, subgraph, distance_normalization: float = 100):
        # Ducks in a row
        x = subgraph.X
        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr
        batch_idx = subgraph.batch

        # Project our input to a larger hidden dimension.
        x = F.relu(self.input_proj(x))

        # FIXME: choice of 100 _is_ arbitrary, could be better.
        edge_attr = edge_attr.view(-1, 1).float() / distance_normalization

        # Graph Attention + Dropout + Layer Normalization.
        for conv, ln in zip(self.convs, self.layer_norms):
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = ln(x)

        # Project output
        x = self.output_proj(x)

        # Global mean pooling (may be useful to try different methods).
        return global_mean_pool(x, batch_idx)

    def flow_forward(self, next_token, context):
        # Token Probability
        return self.flow.log_prob(next_token, context=context)
