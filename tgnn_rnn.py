from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from rgnns import GConvLSTM, GATLSTM, GINLSTM, GINELSTM
from rnns import LSTMModel
from decoder import Decoder

def get_hidden(h):
    if isinstance(h, Tuple):
        return h[0].clone().detach(), h[1].clone().detach()
    else:
        return h.clone().detach()

class TGNN_RNN(nn.Module):
    def __init__(self, num_features, rnn_hidden_dim, gnn_hidden_dim, num_classes, num_gnn_layers, num_rnn_layers, edge_dim, num_heads, num_nodes, tgnn_model=GConvLSTM, rnn_model=LSTMModel):
        super(TGNN_RNN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.rnn_hidden_dim = rnn_hidden_dim

        self.TGNN = tgnn_model(
            in_channels=num_features,
            out_channels=gnn_hidden_dim,
            n_nodes=num_nodes,
            heads=num_heads,
            edge_dim=edge_dim,
            num_layers=num_gnn_layers
        )

        self.RNN = rnn_model(
            input_dim=num_features,
            hidden_dim=rnn_hidden_dim,
            n_layers=num_rnn_layers,
            n_nodes=num_nodes
        )

        self.decoder = Decoder(gnn_hidden_dim + rnn_hidden_dim, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data, h0_n=None, h0_g=None, cell_states_g=None):
        h0s_n = self.RNN.init_hidden_states(self.device) if h0_n is None else h0_n

        if isinstance(self.TGNN, (GATLSTM, GINLSTM, GINELSTM)):
            h0s_g, cell_states_g = self.TGNN.init_hidden_states(self.device)
        else:
            h0s_g = self.TGNN.init_hidden_states(self.device) if h0_g is None else h0_g

        hidden_states_n = h0s_n
        hidden_states_g = h0s_g

        window_size = len(data)
        gnn_output = None
        attn_scores = []

        for i in range(window_size):
            edge_attr = data[i].edge_attr.to(self.device)

            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)     # turns [num_edges] → [num_edges, 1]

            if isinstance(self.TGNN, (GATLSTM, GINLSTM, GINELSTM)):
                gnn_output, hidden_states_g, cell_states_g = self.TGNN(data[i].edge_index.to(self.device), edge_attr, (hidden_states_g, cell_states_g))
            else:
                gnn_output, hidden_states_g, _ = self.TGNN(data[i].edge_index.to(self.device), edge_attr, hidden_states_g)

            hidden_states_n = self.RNN(data[i].x.to(self.device), hidden_states_n)

            if i == 0:
                h0_n = [get_hidden(hidden_states_n[0])]

        last_h_n = hidden_states_n[-1]
        rnn_output = last_h_n[0] if type(last_h_n) is tuple else last_h_n

        fused_output = torch.cat((gnn_output, rnn_output), dim=1)
        output = self.decoder(fused_output)
        output = F.log_softmax(output, dim=1)

        return output, h0_n, hidden_states_g, attn_scores, cell_states_g