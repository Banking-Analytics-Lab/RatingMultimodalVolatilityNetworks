import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv, GINConv
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU
from decoder import Decoder
from typing import Union, Tuple, List, Optional


class GConvLSTM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            n_nodes: int,
            heads: int = 1,
            edge_dim: int = None,
    ):
        super(GConvLSTM, self).__init__()
        # Create a list of GConvLSTM layers
        self.gconv_lstms = nn.ModuleList(
            [GConvLSTM(in_channels if i == 0 else out_channels, out_channels, 5) for i in range(num_layers)])
        self.n_nodes = n_nodes
        self.hidden_dim = out_channels
        self.n_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = Decoder(out_channels, out_channels)
        self.node_embedding = nn.Embedding(n_nodes, in_channels)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

    def forward(self, edge_index, edge_weight, h0=None):
        h0s = self.init_hidden_states(self.device) if h0 is None else h0
        hidden_states = h0s
        x = self.node_embedding.weight.to(self.device)  # cuda()

        # Process input through each GConvLSTM layer
        current_input = x
        new_hidden_states = []

        for layer in range(self.n_layers):
            h, c = hidden_states[layer]
            h, c = self.gconv_lstms[layer](X=current_input, edge_index=edge_index, edge_weight=edge_weight, H=h, C=c)
            h = F.relu(h)
            new_hidden_states.append((h, c))
            current_input = h  # Output of the current layer is input to the next
        hidden_states = new_hidden_states

        # Get the output from the last hidden state of the last layer
        out = hidden_states[-1][0]  # Only the hidden state h is needed

        return out, hidden_states, []

    def init_hidden_states(self, device):
        return [
            (torch.ones(self.n_nodes, self.hidden_dim).to(device), torch.ones(self.n_nodes, self.hidden_dim).to(device))
            for _ in range(self.n_layers)]


class GConvGRU(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            n_nodes: int,
            heads: int = 1,
            edge_dim: int = None,
    ):
        super(GConvGRU, self).__init__()
        # Create a list of GConvGRU layers
        self.gconv_grus = nn.ModuleList(
            [GConvGRU(in_channels if i == 0 else out_channels, out_channels, 5) for i in range(n_layers)])
        self.n_nodes = n_nodes
        self.hidden_dim = out_channels
        self.n_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = Decoder(out_channels, out_channels)
        self.node_embedding = nn.Embedding(n_nodes, in_channels)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

    def forward(self, edge_index, edge_weight, h0=None):
        hidden_states = self.init_hidden_states(self.device) if h0 is None else h0

        x = self.node_embedding.weight.to(self.device)  # cuda()

        # Process input through each GConvGRU layer
        current_input = x
        new_hidden_states = []
        for layer in range(self.n_layers):
            h = hidden_states[layer]
            h = self.gconv_grus[layer](X=current_input, edge_index=edge_index, edge_weight=edge_weight, H=h)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)
            new_hidden_states.append(h)
            current_input = h  # Output of the current layer is input to the next

        hidden_states = new_hidden_states

        # Get the output from the last hidden state of the last layer
        out = hidden_states[-1]

        return out, hidden_states, []

    def init_hidden_states(self, device):
        return [torch.ones(self.n_nodes, self.hidden_dim).to(device) for _ in range(self.n_layers)]


class GATGRU(torch.nn.Module):
    r"""An implementation of the Graph Attention Network (GAT) Gated Recurrent Unit (GRU) Cell.
    For details on GAT, see the paper: "Graph Attention Networks" <https://arxiv.org/abs/1710.10903>.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        heads (int): Number of attention heads.
        concat (bool, optional): If True, concatenate the outputs of each attention head.
                                 Otherwise, average them. (default: True)
        negative_slope (float, optional): LeakyReLU angle of the negative slope. (default: 0.2)
        dropout (float, optional): Dropout probability for attention coefficients. (default: 0.0)
        add_self_loops (bool, optional): If True, add self-loops to the graph. (default: True)
        bias (bool, optional): If False, the layer will not learn an additive bias. (default: True)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            n_nodes: int,
            heads: int = 1,
            concat: bool = False,
            negative_slope: float = 0.2,
            dropout: float = 0.3,
            add_self_loops: bool = True,
            edge_dim: int = None,
            fill_value: Union[float, torch.Tensor, str] = 'mean',
            bias: bool = True,
            residual: bool = False,
    ):
        super(GATGRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = num_layers
        self.n_nodes = n_nodes
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.bias = bias

        self.node_embedding = nn.Embedding(n_nodes, in_channels)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gat_out_channels = out_channels if not concat else out_channels * heads

        self._create_parameters_and_layers()

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_channels)
            for _ in range(self.n_layers)
        ])

        self.input_norm = nn.LayerNorm(in_channels)

        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(out_channels)
            for _ in range(self.n_layers)
        ])

        self.skip_scale = nn.Parameter(torch.ones(1))

    def _create_update_gate_parameters_and_layers(self):
        self.gat_x_zs = nn.ModuleList([GATConv(
            in_channels=self.in_channels if i == 0 else self.out_channels,
            out_channels=self.gat_out_channels,
            heads=self.heads,
            concat=self.concat,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=self.add_self_loops,
            bias=self.bias,
        ) for i in range(self.n_layers)])

        self.gat_h_zs = nn.ModuleList([GATConv(
            in_channels=self.out_channels,
            out_channels=self.gat_out_channels,
            heads=self.heads,
            concat=self.concat,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=self.add_self_loops,
            bias=self.bias,
        ) for _ in range(self.n_layers)])

    def _create_reset_gate_parameters_and_layers(self):
        self.gat_x_rs = nn.ModuleList([GATConv(
            in_channels=self.in_channels if i == 0 else self.out_channels,
            out_channels=self.gat_out_channels,
            heads=self.heads,
            concat=self.concat,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=self.add_self_loops,
            bias=self.bias,
        ) for i in range(self.n_layers)])

        self.gat_h_rs = nn.ModuleList([GATConv(
            in_channels=self.out_channels,
            out_channels=self.gat_out_channels,
            heads=self.heads,
            concat=self.concat,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=self.add_self_loops,
            bias=self.bias,
        ) for _ in range(self.n_layers)])

    def _create_candidate_state_parameters_and_layers(self):
        self.gat_x_hs = nn.ModuleList([GATConv(
            in_channels=self.in_channels if i == 0 else self.out_channels,
            out_channels=self.gat_out_channels,
            heads=self.heads,
            concat=self.concat,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=self.add_self_loops,
            bias=self.bias,
        ) for i in range(self.n_layers)])

        self.gat_h_hs = nn.ModuleList([GATConv(
            in_channels=self.out_channels,
            out_channels=self.gat_out_channels,
            heads=self.heads,
            concat=self.concat,
            negative_slope=self.negative_slope,
            dropout=self.dropout,
            add_self_loops=self.add_self_loops,
            bias=self.bias,
        ) for _ in range(self.n_layers)])

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_attr, H, layer_idx):
        Z = self.gat_x_zs[layer_idx](X, edge_index, edge_attr)
        Z = Z + self.gat_h_zs[layer_idx](H, edge_index, edge_attr)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_attr, H, layer_idx):
        R = self.gat_x_rs[layer_idx](X, edge_index, edge_attr)
        R = R + self.gat_h_rs[layer_idx](H, edge_index, edge_attr)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_attr, H, R, layer_idx):
        H_tilde = self.gat_x_hs[layer_idx](X, edge_index, edge_attr)
        H_tilde = H_tilde + self.gat_h_hs[layer_idx](H * R, edge_index, edge_attr)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
            self,
            edge_index: torch.LongTensor,
            edge_attr: torch.FloatTensor = None,
            H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass.

        Args:
            X (torch.FloatTensor): Node features.
            edge_index (torch.LongTensor): Graph edge indices.
            edge_attr (torch.FloatTensor, optional): Edge features.
            H (torch.FloatTensor, optional): Hidden state matrix for all nodes.

        Returns:
            torch.FloatTensor: Updated hidden state matrix for all nodes.
        """
        X = self.node_embedding.weight.to(self.device)
        X = self.input_norm(X)

        # Store hidden states for each layer
        hidden_states = []

        # Current input to be passed through layers
        current_input = X

        # Process through each layer
        for layer in range(self.n_layers):
            # Store the input for skip connection
            layer_input = current_input

            # Initialize or get layer's hidden state
            layer_H = self._set_hidden_state(current_input, H[layer])

            # Calculate gates for current layer
            Z = self._calculate_update_gate(current_input, edge_index, edge_attr, layer_H, layer)
            R = self._calculate_reset_gate(current_input, edge_index, edge_attr, layer_H, layer)

            # Calculate candidate state and new hidden state
            H_tilde = self._calculate_candidate_state(current_input, edge_index, edge_attr, layer_H, R, layer)
            new_H = self._calculate_hidden_state(Z, layer_H, H_tilde)

            del Z, R, H_tilde  # Free memory explicitly
            torch.cuda.empty_cache()  # Clear unused memory

            # Apply layer normalization
            normalized_H = self.layer_norms[layer](new_H)

            # Add skip connection if not the first layer
            if layer > 0 and normalized_H.shape == layer_input.shape:
                normalized_H = normalized_H + self.skip_scale * layer_input
                normalized_H = self.skip_norms[layer](normalized_H)

            # Store the new hidden state
            hidden_states.append(normalized_H)

            # Update input for next layer
            current_input = normalized_H

        # Return final output and all hidden states
        return current_input, hidden_states, []

    def init_hidden_states(self, device):
        return [torch.ones(self.n_nodes, self.out_channels).to(device) for _ in range(self.n_layers)]


class GINGRU(torch.nn.Module):
    r"""An implementation of the Graph Isomorphism Network (GIN) Gated Recurrent Unit (GRU) Cell.
    For details on GIN, see the paper: "How Powerful are Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        num_layers (int): Number of GRU layers.
        n_nodes (int): Number of nodes in the graph.
        train_eps (bool, optional): If True, epsilon will be a trainable parameter. (default: False)
        eps (float, optional): Initial epsilon value. (default: 0.0)
        dropout (float, optional): Dropout probability. (default: 0.3)
        bias (bool, optional): If False, the layer will not learn an additive bias. (default: True)
        residual (bool, optional): If True, use residual connections. (default: False)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            n_nodes: int,
            train_eps: bool = False,
            eps: float = 0.0,
            dropout: float = 0.3,
            bias: bool = True,
            residual: bool = False,
            edge_dim: int = None,
            heads: int = 1,
    ):
        super(GINGRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = num_layers
        self.n_nodes = n_nodes
        self.train_eps = train_eps
        self.eps = eps
        self.dropout = dropout
        self.bias = bias
        self.residual = residual

        print("Number of nodes:", n_nodes)
        print("Output channels:", out_channels)

        self.node_embedding = nn.Embedding(n_nodes, in_channels)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._create_parameters_and_layers()

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_channels)
            for _ in range(self.n_layers)
        ])

        self.input_norm = nn.LayerNorm(in_channels)

        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(out_channels)
            for _ in range(self.n_layers)
        ])

        self.skip_scale = nn.Parameter(torch.ones(1))

    def _create_mlp(self, in_dim, out_dim):
        """Create a simple MLP for GINConv"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def _create_update_gate_parameters_and_layers(self):
        # Create MLPs for update gates
        self.gin_x_mlps_z = nn.ModuleList([
            self._create_mlp(
                self.in_channels if i == 0 else self.out_channels,
                self.out_channels
            ) for i in range(self.n_layers)
        ])

        self.gin_h_mlps_z = nn.ModuleList([
            self._create_mlp(self.out_channels, self.out_channels)
            for _ in range(self.n_layers)
        ])

        # Create GINConv layers for update gates
        self.gin_x_zs = nn.ModuleList([
            GINConv(
                nn=self.gin_x_mlps_z[i],
                eps=self.eps,
                train_eps=self.train_eps
            ) for i in range(self.n_layers)
        ])

        self.gin_h_zs = nn.ModuleList([
            GINConv(
                nn=self.gin_h_mlps_z[i],
                eps=self.eps,
                train_eps=self.train_eps
            ) for i in range(self.n_layers)
        ])

    def _create_reset_gate_parameters_and_layers(self):
        # Create MLPs for reset gates
        self.gin_x_mlps_r = nn.ModuleList([
            self._create_mlp(
                self.in_channels if i == 0 else self.out_channels,
                self.out_channels
            ) for i in range(self.n_layers)
        ])

        self.gin_h_mlps_r = nn.ModuleList([
            self._create_mlp(self.out_channels, self.out_channels)
            for _ in range(self.n_layers)
        ])

        # Create GINConv layers for reset gates
        self.gin_x_rs = nn.ModuleList([
            GINConv(
                nn=self.gin_x_mlps_r[i],
                eps=self.eps,
                train_eps=self.train_eps
            ) for i in range(self.n_layers)
        ])

        self.gin_h_rs = nn.ModuleList([
            GINConv(
                nn=self.gin_h_mlps_r[i],
                eps=self.eps,
                train_eps=self.train_eps
            ) for i in range(self.n_layers)
        ])

    def _create_candidate_state_parameters_and_layers(self):
        # Create MLPs for candidate states
        self.gin_x_mlps_h = nn.ModuleList([
            self._create_mlp(
                self.in_channels if i == 0 else self.out_channels,
                self.out_channels
            ) for i in range(self.n_layers)
        ])

        self.gin_h_mlps_h = nn.ModuleList([
            self._create_mlp(self.out_channels, self.out_channels)
            for _ in range(self.n_layers)
        ])

        # Create GINConv layers for candidate states
        self.gin_x_hs = nn.ModuleList([
            GINConv(
                nn=self.gin_x_mlps_h[i],
                eps=self.eps,
                train_eps=self.train_eps
            ) for i in range(self.n_layers)
        ])

        self.gin_h_hs = nn.ModuleList([
            GINConv(
                nn=self.gin_h_mlps_h[i],
                eps=self.eps,
                train_eps=self.train_eps
            ) for i in range(self.n_layers)
        ])

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_attr, H, layer_idx):
        Z = self.gin_x_zs[layer_idx](X, edge_index)
        Z = Z + self.gin_h_zs[layer_idx](H, edge_index)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_attr, H, layer_idx):
        R = self.gin_x_rs[layer_idx](X, edge_index)
        R = R + self.gin_h_rs[layer_idx](H, edge_index)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_attr, H, R, layer_idx):
        H_tilde = self.gin_x_hs[layer_idx](X, edge_index)
        H_tilde = H_tilde + self.gin_h_hs[layer_idx](H * R, edge_index)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
            self,
            edge_index: torch.LongTensor,
            edge_attr: torch.FloatTensor = None,
            H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass.

        Args:
            edge_index (torch.LongTensor): Graph edge indices.
            edge_attr (torch.FloatTensor, optional): Edge features (not used in GIN but kept for API consistency).
            H (torch.FloatTensor, optional): Hidden state matrix for all nodes.

        Returns:
            torch.FloatTensor: Tuple containing (final_output, all_hidden_states, [])
        """
        X = self.node_embedding.weight.to(self.device)
        X = self.input_norm(X)

        # Store hidden states for each layer
        hidden_states = []

        # Current input to be passed through layers
        current_input = X

        # Process through each layer
        for layer in range(self.n_layers):
            # Store the input for skip connection
            layer_input = current_input

            # Initialize or get layer's hidden state
            layer_H = self._set_hidden_state(current_input, H[layer] if H is not None else None)

            # Calculate gates for current layer
            Z = self._calculate_update_gate(current_input, edge_index, edge_attr, layer_H, layer)
            R = self._calculate_reset_gate(current_input, edge_index, edge_attr, layer_H, layer)

            # Calculate candidate state and new hidden state
            H_tilde = self._calculate_candidate_state(current_input, edge_index, edge_attr, layer_H, R, layer)
            new_H = self._calculate_hidden_state(Z, layer_H, H_tilde)

            del Z, R, H_tilde  # Free memory explicitly
            torch.cuda.empty_cache()  # Clear unused memory

            # Apply layer normalization
            normalized_H = self.layer_norms[layer](new_H)

            # Add skip connection if not the first layer and if residual is True
            if self.residual and layer > 0 and normalized_H.shape == layer_input.shape:
                normalized_H = normalized_H + self.skip_scale * layer_input
                normalized_H = self.skip_norms[layer](normalized_H)

            # Store the new hidden state
            hidden_states.append(normalized_H)

            # Update input for next layer
            current_input = normalized_H

        # Return final output and all hidden states
        return current_input, hidden_states, []

    def init_hidden_states(self, device):
        return [torch.ones(self.n_nodes, self.out_channels).to(device) for _ in range(self.n_layers)]

class GATLSTM(nn.Module):
    r"""An implementation of the Graph Attention Network (GAT) Long Short-Term Memory (LSTM) Cell.
    For details on GAT, see: "Graph Attention Networks" <https://arxiv.org/abs/1710.10903>.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features per node.
        num_layers (int): Number of stacked GAT-LSTM layers.
        n_nodes (int): Number of nodes in the fixed graph.
        heads (int): Number of attention heads.
        concat (bool): If True, concatenate heads; otherwise average.
        negative_slope (float): LeakyReLU negative slope.
        dropout (float): Dropout on attention coefficients.
        add_self_loops (bool): Add self-loops to graph.
        bias (bool): Learn bias in GATConv.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            n_nodes: int,
            heads: int = 1,
            concat: bool = False,
            negative_slope: float = 0.2,
            dropout: float = 0.3,
            add_self_loops: bool = True,
            bias: bool = True,
            edge_dim: int = None,
    ):
        super(GATLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = num_layers
        self.n_nodes = n_nodes
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.bias = bias

        # Learnable node embeddings
        self.node_embedding = nn.Embedding(n_nodes, in_channels)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Output dim for each head
        self.gat_out_channels = out_channels if not concat else out_channels * heads

        # Create GATConv modules for LSTM gates
        self._create_gate_parameters()

        # Normalization and skip connections
        self.input_norm = nn.LayerNorm(in_channels)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(self.n_layers)])
        self.skip_norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(self.n_layers)])
        self.skip_scale = nn.Parameter(torch.ones(1))

    def _create_gate_parameters(self):
        # Input gate
        self.gat_x_is = nn.ModuleList()
        self.gat_h_is = nn.ModuleList()
        # Forget gate
        self.gat_x_fs = nn.ModuleList()
        self.gat_h_fs = nn.ModuleList()
        # Output gate
        self.gat_x_os = nn.ModuleList()
        self.gat_h_os = nn.ModuleList()
        # Candidate cell
        self.gat_x_gs = nn.ModuleList()
        self.gat_h_gs = nn.ModuleList()

        for i in range(self.n_layers):
            in_dim = self.in_channels if i == 0 else self.out_channels
            # X-to-gates
            self.gat_x_is.append(GATConv(in_dim, self.gat_out_channels, heads=self.heads,
                                         concat=self.concat, negative_slope=self.negative_slope,
                                         dropout=self.dropout, add_self_loops=self.add_self_loops,
                                         bias=self.bias))
            self.gat_x_fs.append(GATConv(in_dim, self.gat_out_channels, heads=self.heads,
                                         concat=self.concat, negative_slope=self.negative_slope,
                                         dropout=self.dropout, add_self_loops=self.add_self_loops,
                                         bias=self.bias))
            self.gat_x_os.append(GATConv(in_dim, self.gat_out_channels, heads=self.heads,
                                         concat=self.concat, negative_slope=self.negative_slope,
                                         dropout=self.dropout, add_self_loops=self.add_self_loops,
                                         bias=self.bias))
            self.gat_x_gs.append(GATConv(in_dim, self.gat_out_channels, heads=self.heads,
                                         concat=self.concat, negative_slope=self.negative_slope,
                                         dropout=self.dropout, add_self_loops=self.add_self_loops,
                                         bias=self.bias))
            # H-to-gates
            self.gat_h_is.append(GATConv(self.out_channels, self.gat_out_channels, heads=self.heads,
                                         concat=self.concat, negative_slope=self.negative_slope,
                                         dropout=self.dropout, add_self_loops=self.add_self_loops,
                                         bias=self.bias))
            self.gat_h_fs.append(GATConv(self.out_channels, self.gat_out_channels, heads=self.heads,
                                         concat=self.concat, negative_slope=self.negative_slope,
                                         dropout=self.dropout, add_self_loops=self.add_self_loops,
                                         bias=self.bias))
            self.gat_h_os.append(GATConv(self.out_channels, self.gat_out_channels, heads=self.heads,
                                         concat=self.concat, negative_slope=self.negative_slope,
                                         dropout=self.dropout, add_self_loops=self.add_self_loops,
                                         bias=self.bias))
            self.gat_h_gs.append(GATConv(self.out_channels, self.gat_out_channels, heads=self.heads,
                                         concat=self.concat, negative_slope=self.negative_slope,
                                         dropout=self.dropout, add_self_loops=self.add_self_loops,
                                         bias=self.bias))

    def _init_states(self, ):
        # Initialize hidden and cell states for all layers
        H = [torch.zeros(self.n_nodes, self.out_channels).to(self.device)
             for _ in range(self.n_layers)]
        C = [torch.zeros(self.n_nodes, self.out_channels).to(self.device)
             for _ in range(self.n_layers)]
        return H, C

    def forward(self,
                edge_index: torch.LongTensor,
                edge_attr: torch.FloatTensor = None,
                states: Tuple[List[torch.Tensor], List[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            edge_index (LongTensor): Edge indices of the graph.
            edge_attr (FloatTensor, optional): Edge features.
            states (H, C) optional: previous hidden and cell states.

        Returns:
            current_input: Tensor of shape [n_nodes, out_channels]
            hidden_states: list of hidden states per layer
            cell_states: list of cell states per layer
        """
        # Input features from learnable embeddings
        X = self.node_embedding.weight.to(self.device)
        X = self.input_norm(X)

        print("Edge_attr shape:", edge_attr.shape)

        # Unpack or init states
        if states is None:
            H_prev, C_prev = self._init_states()
        else:
            H_prev, C_prev = states

        hidden_states = []
        cell_states = []
        current_input = X

        for layer in range(self.n_layers):
            layer_input = current_input
            H_l = H_prev[layer]
            C_l = C_prev[layer]

            # Compute gates
            i_t = torch.sigmoid(
                self.gat_x_is[layer](current_input, edge_index, edge_attr)
                + self.gat_h_is[layer](H_l, edge_index, edge_attr)
            )
            f_t = torch.sigmoid(
                self.gat_x_fs[layer](current_input, edge_index, edge_attr)
                + self.gat_h_fs[layer](H_l, edge_index, edge_attr)
            )
            g_t = torch.tanh(
                self.gat_x_gs[layer](current_input, edge_index, edge_attr)
                + self.gat_h_gs[layer](H_l, edge_index, edge_attr)
            )
            o_t = torch.sigmoid(
                self.gat_x_os[layer](current_input, edge_index, edge_attr)
                + self.gat_h_os[layer](H_l, edge_index, edge_attr)
            )

            # Update cell and hidden
            C_new = f_t * C_l + i_t * g_t
            H_new = o_t * torch.tanh(C_new)

            # Layer normalization
            H_norm = self.layer_norms[layer](H_new)
            # Skip connection
            if layer > 0 and H_norm.shape == layer_input.shape:
                H_norm = H_norm + self.skip_scale * layer_input
                H_norm = self.skip_norms[layer](H_norm)

            hidden_states.append(H_norm)
            cell_states.append(C_new)
            current_input = H_norm

        return current_input, hidden_states, cell_states

    def init_hidden_states(self, device):
        # return [torch.ones(self.n_nodes, self.out_channels).to(device) for _ in range(self.n_layers)]
        self.device = device
        return self._init_states()

class GINLSTM(nn.Module):
    r"""Graph-Isomorphism-Network (GIN) + LSTM cell for fixed-size graphs.
    Mirrors the structure of your GATLSTM, but swaps in GINConv gates.

    Args:
        in_channels (int): Dimensionality of node input embeddings.
        out_channels (int): Hidden size of each GIN-LSTM layer.
        num_layers (int): Number of stacked GIN-LSTM layers.
        n_nodes (int): Number of nodes in your graph (1401).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            n_nodes: int,
            edge_dim: int = None,
            heads: int = 1,
    ):
        super(GINLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = num_layers
        self.n_nodes = n_nodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # learnable node embeddings
        self.node_embedding = nn.Embedding(n_nodes, in_channels)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

        # for GIN we don't use multi-head, so out_dim == out_channels
        self.gin_out_channels = out_channels

        # build one GINConv per gate, per layer, for x→gate and h→gate
        self._create_gate_parameters()

        # layer norms + skip-connection norms
        self.input_norm = nn.LayerNorm(in_channels)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_channels) for _ in range(self.n_layers)
        ])
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(out_channels) for _ in range(self.n_layers)
        ])
        self.skip_scale = nn.Parameter(torch.ones(1))

    def _create_gate_parameters(self):
        # four gates: input, forget, output, candidate
        self.gin_x_is, self.gin_h_is = nn.ModuleList(), nn.ModuleList()
        self.gin_x_fs, self.gin_h_fs = nn.ModuleList(), nn.ModuleList()
        self.gin_x_os, self.gin_h_os = nn.ModuleList(), nn.ModuleList()
        self.gin_x_gs, self.gin_h_gs = nn.ModuleList(), nn.ModuleList()

        for layer in range(self.n_layers):
            in_dim = self.in_channels if layer == 0 else self.out_channels

            # helper to build a 2-layer MLP
            def make_mlp(input_dim):
                return nn.Sequential(
                    nn.Linear(input_dim, self.out_channels),
                    nn.ReLU(),
                    nn.Linear(self.out_channels, self.out_channels)
                )

            # x → gates
            self.gin_x_is.append(GINConv(make_mlp(in_dim)))
            self.gin_x_fs.append(GINConv(make_mlp(in_dim)))
            self.gin_x_os.append(GINConv(make_mlp(in_dim)))
            self.gin_x_gs.append(GINConv(make_mlp(in_dim)))

            # h → gates
            self.gin_h_is.append(GINConv(make_mlp(self.out_channels)))
            self.gin_h_fs.append(GINConv(make_mlp(self.out_channels)))
            self.gin_h_os.append(GINConv(make_mlp(self.out_channels)))
            self.gin_h_gs.append(GINConv(make_mlp(self.out_channels)))

    def _init_states(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        H = [
            torch.zeros(self.n_nodes, self.out_channels, device=self.device)
            for _ in range(self.n_layers)
        ]
        C = [
            torch.zeros(self.n_nodes, self.out_channels, device=self.device)
            for _ in range(self.n_layers)
        ]
        return H, C

    def forward(
            self,
            edge_index: torch.LongTensor,
            edge_attr: torch.Tensor = None,  # Unused by GINConv
            states: Tuple[List[torch.Tensor], List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # initial node features from embedding
        X = self.node_embedding.weight.to(self.device)
        X = self.input_norm(X)

        # init or unpack hidden/cell states
        if states is None:
            H_prev, C_prev = self._init_states()
        else:
            H_prev, C_prev = states

        hidden_states, cell_states = [], []
        current_input = X

        for layer in range(self.n_layers):
            H_l = H_prev[layer]
            C_l = C_prev[layer]

            # gating computations (no edge_attr support in vanilla GINConv)
            i_t = torch.sigmoid(
                self.gin_x_is[layer](current_input, edge_index)
                + self.gin_h_is[layer](H_l, edge_index)
            )
            f_t = torch.sigmoid(
                self.gin_x_fs[layer](current_input, edge_index)
                + self.gin_h_fs[layer](H_l, edge_index)
            )
            g_t = torch.tanh(
                self.gin_x_gs[layer](current_input, edge_index)
                + self.gin_h_gs[layer](H_l, edge_index)
            )
            o_t = torch.sigmoid(
                self.gin_x_os[layer](current_input, edge_index)
                + self.gin_h_os[layer](H_l, edge_index)
            )

            # LSTM update
            C_new = f_t * C_l + i_t * g_t
            H_new = o_t * torch.tanh(C_new)

            # layer norm + optional skip
            H_norm = self.layer_norms[layer](H_new)
            if layer > 0 and H_norm.shape == current_input.shape:
                H_norm = H_norm + self.skip_scale * current_input
                H_norm = self.skip_norms[layer](H_norm)

            hidden_states.append(H_norm)
            cell_states.append(C_new)
            current_input = H_norm

        return current_input, hidden_states, cell_states

    def init_hidden_states(self, device: torch.device = None):
        if device is not None:
            self.device = device
        return self._init_states()


class GINELSTM(nn.Module):
    r"""Graph-Isomorphism-Network (GIN) + LSTM cell for fixed-size graphs,
    using GINEConv to incorporate edge attributes in message passing.

    Args:
        in_channels (int): Dimensionality of node input embeddings.
        out_channels (int): Hidden size of each GIN-LSTM layer.
        num_layers (int): Number of stacked GIN-LSTM layers.
        n_nodes (int): Number of nodes in your graph.
        edge_dim (int, optional): Dimensionality of edge attribute vectors.
            If None, requires edge_attr dim == node feature dim.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            n_nodes: int,
            edge_dim: Optional[int] = None,
            heads: int = 1,
    ):
        super(GINELSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = num_layers
        self.n_nodes = n_nodes
        self.edge_dim = edge_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # learnable node embeddings
        self.node_embedding = nn.Embedding(n_nodes, in_channels)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

        # build one GINEConv per gate, per layer, for x→gate and h→gate
        self._create_gate_parameters()

        # layer norms + skip-connection norms
        self.input_norm = nn.LayerNorm(in_channels)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_channels) for _ in range(self.n_layers)
        ])
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(out_channels) for _ in range(self.n_layers)
        ])
        self.skip_scale = nn.Parameter(torch.ones(1))

    def _create_gate_parameters(self):
        # four gates: input, forget, output, candidate
        self.gin_x_is, self.gin_h_is = nn.ModuleList(), nn.ModuleList()
        self.gin_x_fs, self.gin_h_fs = nn.ModuleList(), nn.ModuleList()
        self.gin_x_os, self.gin_h_os = nn.ModuleList(), nn.ModuleList()
        self.gin_x_gs, self.gin_h_gs = nn.ModuleList(), nn.ModuleList()

        for layer in range(self.n_layers):
            in_dim = self.in_channels if layer == 0 else self.out_channels

            def make_mlp(input_dim: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(input_dim, self.out_channels),
                    nn.ReLU(),
                    nn.Linear(self.out_channels, self.out_channels)
                )

            # x -> gates
            self.gin_x_is.append(GINEConv(make_mlp(in_dim), edge_dim=self.edge_dim))
            self.gin_x_fs.append(GINEConv(make_mlp(in_dim), edge_dim=self.edge_dim))
            self.gin_x_os.append(GINEConv(make_mlp(in_dim), edge_dim=self.edge_dim))
            self.gin_x_gs.append(GINEConv(make_mlp(in_dim), edge_dim=self.edge_dim))

            # h -> gates
            self.gin_h_is.append(GINEConv(make_mlp(self.out_channels), edge_dim=self.edge_dim))
            self.gin_h_fs.append(GINEConv(make_mlp(self.out_channels), edge_dim=self.edge_dim))
            self.gin_h_os.append(GINEConv(make_mlp(self.out_channels), edge_dim=self.edge_dim))
            self.gin_h_gs.append(GINEConv(make_mlp(self.out_channels), edge_dim=self.edge_dim))

    def _init_states(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        H = [
            torch.zeros(self.n_nodes, self.out_channels, device=self.device)
            for _ in range(self.n_layers)
        ]
        C = [
            torch.zeros(self.n_nodes, self.out_channels, device=self.device)
            for _ in range(self.n_layers)
        ]
        return H, C

    def forward(
            self,
            edge_index: torch.LongTensor,
            edge_attr: torch.Tensor,
            states: Tuple[List[torch.Tensor], List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # require edge attributes
        assert edge_attr is not None, "edge_attr must be provided for GINEConv"

        # initial node features from embedding
        X = self.node_embedding.weight.to(self.device)
        X = self.input_norm(X)

        # init or unpack hidden/cell states
        if states is None:
            H_prev, C_prev = self._init_states()
        else:
            H_prev, C_prev = states

        hidden_states, cell_states = [], []
        current_input = X

        for layer in range(self.n_layers):
            H_l = H_prev[layer]
            C_l = C_prev[layer]

            # gating computations now include edge_attr
            i_t = torch.sigmoid(
                self.gin_x_is[layer](current_input, edge_index, edge_attr)
                + self.gin_h_is[layer](H_l, edge_index, edge_attr)
            )
            f_t = torch.sigmoid(
                self.gin_x_fs[layer](current_input, edge_index, edge_attr)
                + self.gin_h_fs[layer](H_l, edge_index, edge_attr)
            )
            g_t = torch.tanh(
                self.gin_x_gs[layer](current_input, edge_index, edge_attr)
                + self.gin_h_gs[layer](H_l, edge_index, edge_attr)
            )
            o_t = torch.sigmoid(
                self.gin_x_os[layer](current_input, edge_index, edge_attr)
                + self.gin_h_os[layer](H_l, edge_index, edge_attr)
            )

            # LSTM update
            C_new = f_t * C_l + i_t * g_t
            H_new = o_t * torch.tanh(C_new)

            # layer norm + optional skip
            H_norm = self.layer_norms[layer](H_new)

            if layer > 0 and H_norm.shape == current_input.shape:
                H_norm = H_norm + self.skip_scale * current_input
                H_norm = self.skip_norms[layer](H_norm)

            hidden_states.append(H_norm)
            cell_states.append(C_new)
            current_input = H_norm

        return current_input, hidden_states, cell_states

    def init_hidden_states(self, device: torch.device = None):
        if device is not None:
            self.device = device
        return self._init_states()

class GINEGRU(nn.Module):
    r"""Graph‐Isomorphism‐Network (GIN) + GRU cell for fixed‐size graphs,
    using GINEConv to incorporate edge attributes in message passing.

    Args:
        in_channels (int): Dimensionality of node input embeddings.
        out_channels (int): Hidden size of each GIN‐GRU layer.
        num_layers (int): Number of stacked GIN‐GRU layers.
        n_nodes (int): Number of nodes in your graph.
        edge_dim (int, optional): Dimensionality of edge attribute vectors.
            If None, requires edge_attr dim == node feature dim.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int,
            n_nodes: int,
            edge_dim: Optional[int] = None,
            heads: int = 1,
    ):
        super(GINEGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = num_layers
        self.n_nodes = n_nodes
        self.edge_dim = edge_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # learnable node embeddings
        self.node_embedding = nn.Embedding(n_nodes, in_channels)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

        # build one GINEConv per gate (update, reset, candidate), per layer, for x→gate and h→gate
        self._create_gate_parameters()

        # layer norms + skip-connection norms
        self.input_norm = nn.LayerNorm(in_channels)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_channels) for _ in range(self.n_layers)
        ])
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(out_channels) for _ in range(self.n_layers)
        ])
        self.skip_scale = nn.Parameter(torch.ones(1))

    def _create_gate_parameters(self):
        # three gates: update (z), reset (r), candidate (g)
        self.gin_x_zs, self.gin_h_zs = nn.ModuleList(), nn.ModuleList()
        self.gin_x_rs, self.gin_h_rs = nn.ModuleList(), nn.ModuleList()
        self.gin_x_gs, self.gin_h_gs = nn.ModuleList(), nn.ModuleList()

        for layer in range(self.n_layers):
            in_dim = self.in_channels if layer == 0 else self.out_channels

            def make_mlp(input_dim: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(input_dim, self.out_channels),
                    nn.ReLU(),
                    nn.Linear(self.out_channels, self.out_channels)
                )

            # x → update, reset, candidate
            self.gin_x_zs.append(GINEConv(make_mlp(in_dim), edge_dim=self.edge_dim))
            self.gin_x_rs.append(GINEConv(make_mlp(in_dim), edge_dim=self.edge_dim))
            self.gin_x_gs.append(GINEConv(make_mlp(in_dim), edge_dim=self.edge_dim))

            # h → update, reset, candidate
            self.gin_h_zs.append(GINEConv(make_mlp(self.out_channels), edge_dim=self.edge_dim))
            self.gin_h_rs.append(GINEConv(make_mlp(self.out_channels), edge_dim=self.edge_dim))
            self.gin_h_gs.append(GINEConv(make_mlp(self.out_channels), edge_dim=self.edge_dim))

    def _init_states(self) -> List[torch.Tensor]:
        H = [
            torch.zeros(self.n_nodes, self.out_channels, device=self.device)
            for _ in range(self.n_layers)
        ]
        return H

    def forward(
            self,
            edge_index: torch.LongTensor,
            edge_attr: torch.Tensor,
            states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # require edge attributes
        assert edge_attr is not None, "edge_attr must be provided for GINEConv"

        # initial node features from embedding
        X = self.node_embedding.weight.to(self.device)
        X = self.input_norm(X)

        # init or unpack hidden states
        if states is None:
            H_prev = self._init_states()
        else:
            H_prev = states

        hidden_states = []
        current_input = X

        for layer in range(self.n_layers):
            H_l = H_prev[layer]

            # update gate
            z_t = torch.sigmoid(
                self.gin_x_zs[layer](current_input, edge_index, edge_attr)
                + self.gin_h_zs[layer](H_l, edge_index, edge_attr)
            )
            # reset gate
            r_t = torch.sigmoid(
                self.gin_x_rs[layer](current_input, edge_index, edge_attr)
                + self.gin_h_rs[layer](H_l, edge_index, edge_attr)
            )
            # candidate hidden
            g_t = torch.tanh(
                self.gin_x_gs[layer](current_input, edge_index, edge_attr)
                + self.gin_h_gs[layer](r_t * H_l, edge_index, edge_attr)
            )

            # GRU update
            H_new = (1 - z_t) * H_l + z_t * g_t

            # layer norm + optional skip
            H_norm = self.layer_norms[layer](H_new)
            if layer > 0 and H_norm.shape == current_input.shape:
                H_norm = H_norm + self.skip_scale * current_input
                H_norm = self.skip_norms[layer](H_norm)

            hidden_states.append(H_norm)
            current_input = H_norm

        return current_input, hidden_states, []

    def init_hidden_states(self, device: torch.device = None) -> List[torch.Tensor]:
        if device is not None:
            self.device = device
        return self._init_states()
