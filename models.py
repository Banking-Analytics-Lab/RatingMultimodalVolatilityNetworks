from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from rgnns import GConvLSTMModel, GConvGRUModel
from rnns import LSTMModel, GRUModel, TransformerModel
from decoder import Decoder
import numpy as np

def get_hidden(h):
    if isinstance(h, Tuple):
        return h[0].clone().detach(), h[1].clone().detach()
    else:
        return h.clone().detach()

class RGNN_RNN(nn.Module):
    """ Multi-modal """

    def __init__(self, num_features, rnn_hidden_dim, gnn_hidden_dim, num_classes, num_gnn_layers, num_rnn_layers, edge_dim, num_heads, num_nodes, rgnn_model=GConvLSTMModel, rnn_model=LSTMModel):
        super(RGNN_RNN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.rnn_hidden_dim = rnn_hidden_dim

        self.RGNN = rgnn_model(num_features, gnn_hidden_dim, gnn_hidden_dim, num_gnn_layers, num_nodes)
        self.RNN = rnn_model(num_features, rnn_hidden_dim, num_rnn_layers, num_nodes)
        self.decoder = Decoder(gnn_hidden_dim + rnn_hidden_dim, num_classes)

        self.device = torch.device("cuda:0")

    def forward(self, data, h0_n=None, h0_g=None):
        h0s_n = self.RNN.init__hidd(self.device) if h0_n is None else h0_n
        h0s_g = self.GNN.init__hidd(self.device) if h0_g is None else h0_g

        hidden_states_n = h0s_n
        hidden_states_g = h0s_g

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        window_size = len(x)
        gnn_output = None

        if type(self.RNN) is TransformerModel:
            for i in range(window_size):
                gnn_output, hidden_states_g, _ = self.RGNN(edge_index[i].cuda(), edge_attr[i].cuda(), hidden_states_g)

            x = [x[i].cpu() for i in range(window_size)]
            x_tensor = torch.tensor(np.stack(x), dtype=torch.float32).cuda()
            transformer_output = self.RNN(x_tensor, hidden_states_n)
            # Average the transformer output across the second dimension (batch dimension)
            rnn_output = transformer_output.mean(dim=1)  # Now shape [318, hidden_dim]
        else:
            for i in range(window_size):
                gnn_output, hidden_states_g, _ = self.GNN(edge_index[i].cuda(), edge_attr[i].cuda(), hidden_states_g)

                hidden_states_n = self.RNN(x[i].cuda(), hidden_states_n)

                if i == 0:
                    h0_n = [get_hidden(hidden_states_n[0])]

            last_h_n = hidden_states_n[-1]
            rnn_output = last_h_n[0] if type(last_h_n) is tuple else last_h_n

        fused_output = torch.cat((gnn_output, rnn_output), dim=1)

        output = self.decoder(fused_output)
        output = F.log_softmax(output, dim=1)

        return output, h0_n, hidden_states_g


class RNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_rnn_layers, num_nodes, rnn_model=LSTMModel):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.RNN = rnn_model(num_features, hidden_dim, num_rnn_layers, num_nodes)
        self.decoder = Decoder(hidden_dim, num_classes)
        self.device = torch.device("cuda:0")

    def forward(self, data, h0=None):
        h0s = self.RNN.init__hidd(self.device) if h0 is None else h0
        hidden_states = h0s

        x = data.x
        window_size = len(x)

        if type(self.RNN) is TransformerModel:
            x = [x[i].cpu() for i in range(window_size)]
            x_tensor = torch.tensor(np.stack(x), dtype=torch.float32).cuda()
            rnn_output = self.RNN(x_tensor, hidden_states)
            # Average the transformer output across the second dimension (batch dimension)
            rnn_output = rnn_output.mean(dim=1)
        else:
            for i in range(window_size):
                hidden_states = self.RNN(x[i].cuda(), hidden_states)

                if i == 0:
                    h0 = [get_hidden(hidden_states[0])]

            last_h = hidden_states[-1]
            rnn_output = last_h[0] if type(last_h) is tuple else last_h

        output = self.decoder(rnn_output)
        output = F.log_softmax(output, dim=1)

        return output, h0