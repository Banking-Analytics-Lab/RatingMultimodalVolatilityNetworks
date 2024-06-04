import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, hidden_dim, target_size=1):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc2 = nn.Linear(int(hidden_dim/2), target_size)

    def forward(self, inp):
        h = self.fc1(inp)
        h = F.relu(h)
        h = self.fc2(h)
        return h