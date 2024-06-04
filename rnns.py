import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_nodes):
        super(LSTMModel, self).__init__()
        c_0 = nn.LSTMCell(input_dim, hidden_dim)
        self.cells = nn.ModuleList([nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(n_layers-1)])
        self.cells.insert(0, c_0)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_nodes = n_nodes

    def forward(self, inps, h0_list):
        prev_h, _ = self.cells[0](inps, h0_list[0])
        h_list = [(prev_h, _)]

        for i, (l, h_c) in enumerate(zip(self.cells, h0_list)):
            if i == 0: continue
            (prev_h, c) = l(prev_h, h_c)
            h_list.append((prev_h, c))

        return h_list

    def init__hidd(self, device):
        return [(torch.ones(self.n_nodes, self.hidden_dim).to(device), torch.ones(self.n_nodes, self.hidden_dim).to(device)) for _ in range(self.n_layers)]

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_nodes) -> None:
        super(GRUModel, self).__init__()
        c_0 = nn.GRUCell(input_dim, hidden_dim)
        self.cells = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(n_layers-1)])
        self.cells.insert(0, c_0)
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes

    def forward(self, inps, h0_list):
        prev_h = self.cells[0](inps, h0_list[0])

        h_list = [prev_h]

        for i, (l, h_c) in enumerate(zip(self.cells, h0_list)):
            if i == 0: continue
            prev_h = l(prev_h, h_c)
            h_list.append((prev_h))

        return h_list

    def init__hidd(self, device):
        h0s = [torch.ones(self.n_nodes, self.hidden_dim).to(device) for _ in range(self.n_layers)]

        return h0s

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_nodes, n_heads=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_heads = n_heads

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # Transformer expects input in the format (seq_length, batch_size, feature_dim)
        src = self.embedding(src) # Embed input sequence
        src = src.permute(1, 0, 2) # Permute to match Transformer's input format

        # Apply Transformer Encoder
        output = self.transformer_encoder(src)

        return output