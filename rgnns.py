import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GConvGRU
from typing import Tuple
from decoder import Decoder

def get_hidden(h):
    if isinstance(h, Tuple):
        return h[0].clone().detach(), h[1].clone().detach()
    else:
        return h.clone().detach()

class GConvLSTMModel2(nn.Module):
    """ For Uni-modal training/inference """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_nodes):
        super(GConvLSTMModel2, self).__init__()
        self.gconv_lstms = nn.ModuleList([GConvLSTM(input_dim if i == 0 else hidden_dim, hidden_dim, 5) for i in range(n_layers)])
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = Decoder(hidden_dim, output_dim)
        self.node_embedding = nn.Embedding(n_nodes, input_dim)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

    def forward(self, edge_index, edge_weight, h0=None):
        h0s = self.init__hidd(self.device) if h0 is None else h0
        hidden_states = h0s
        x = self.node_embedding.weight.cuda()
        window_size = len(edge_index)

        # Process input through each GConvLSTM layer
        for i in range(window_size):
            current_input = x
            new_hidden_states = []
            for layer in range(self.n_layers):
                h, c = hidden_states[layer]
                h, c = self.gconv_lstms[layer](X=current_input, edge_index=edge_index[i].cuda(), edge_weight=edge_weight[i].cuda(), H=h, C=c)
                h = F.relu(h)
                new_hidden_states.append((h, c))
                current_input = h  # Output of the current layer is input to the next

            hidden_states = new_hidden_states

        # Get the output from the last hidden state of the last layer
        last_hidden = hidden_states[-1][0]  # Only the hidden state h is needed
        out = self.decoder(last_hidden)
        out = F.log_softmax(out, dim=1)

        return out, hidden_states, []

    def init__hidd(self, device):
        return [(torch.ones(self.n_nodes, self.hidden_dim).to(device), torch.ones(self.n_nodes, self.hidden_dim).to(device)) for _ in range(self.n_layers)]

class GConvLSTMModel(nn.Module):
    """ For Multi-modal training/inference """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_nodes):
        super(GConvLSTMModel, self).__init__()
        self.gconv_lstms = nn.ModuleList([GConvLSTM(input_dim if i == 0 else hidden_dim, hidden_dim, 5) for i in range(n_layers)])
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = Decoder(hidden_dim, output_dim)
        self.node_embedding = nn.Embedding(n_nodes, input_dim)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

    def forward(self, edge_index, edge_weight, h0=None):
        h0s = self.init__hidd(self.device) if h0 is None else h0
        hidden_states = h0s
        x = self.node_embedding.weight.cuda()

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

    def init__hidd(self, device):
        return [(torch.ones(self.n_nodes, self.hidden_dim).to(device), torch.ones(self.n_nodes, self.hidden_dim).to(device)) for _ in range(self.n_layers)]


class GConvGRUModel(nn.Module):
    """ For Multi-modal training/inference """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_nodes):
        super(GConvGRUModel, self).__init__()
        # Create a list of GConvGRU layers
        self.gconv_grus = nn.ModuleList([GConvGRU(input_dim if i == 0 else hidden_dim, hidden_dim, 5) for i in range(n_layers)])
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = Decoder(hidden_dim, output_dim)
        self.node_embedding = nn.Embedding(n_nodes, input_dim)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

    def forward(self, edge_index, edge_weight, h0=None):
        h0s = self.init__hidd(self.device) if h0 is None else h0
        hidden_states = h0s
        x = self.node_embedding.weight.cuda()

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

    def init__hidd(self, device):
        return [torch.ones(self.n_nodes, self.hidden_dim).to(device) for _ in range(self.n_layers)]

class GConvGRUModel2(nn.Module):
    """ For Uni-modal training/inference """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_nodes):
        super(GConvGRUModel2, self).__init__()
        # Create a list of GConvGRU layers
        self.gconv_grus = nn.ModuleList([GConvGRU(input_dim if i == 0 else hidden_dim, hidden_dim, 5) for i in range(n_layers)])
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = Decoder(hidden_dim, output_dim)
        self.node_embedding = nn.Embedding(n_nodes, input_dim)
        nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)

    def forward(self, edge_index, edge_weight, h0=None):
        h0s = self.init__hidd(self.device) if h0 is None else h0
        hidden_states = h0s
        x = self.node_embedding.weight.cuda()
        window_size = len(edge_index)

        # Process input through each GConvGRU layer
        for i in range(window_size):
            current_input = x
            new_hidden_states = []
            for layer in range(self.n_layers):
                h = hidden_states[layer]
                h = self.gconv_grus[layer](X=current_input, edge_index=edge_index[i].cuda(), edge_weight=edge_weight[i].cuda(), H=h)
                h = F.relu(h)
                new_hidden_states.append(h)
                current_input = h  # Output of the current layer is input to the next
            hidden_states = new_hidden_states

        # Get the output from the last hidden state of the last layer
        last_hidden = hidden_states[-1]
        out = self.decoder(last_hidden)
        out = F.log_softmax(out, dim=1)

        return out, hidden_states, []

    def init__hidd(self, device):
        return [torch.ones(self.n_nodes, self.hidden_dim).to(device) for _ in range(self.n_layers)]