import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


# Creating a class for AE
# n.num of  ==> 9 ==> 28*28
class Autoencoder(nn.Module):
    def __init__(self, n_input, nz):
        super(AE, self).__init__()
        self.n_input = n_input
        self.nz = nz


        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, nz)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = nn.Sequential(
            nn.Linear(nz, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, n_input),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Attention_Layer(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.w = nn.Linear(
            in_features=n_feats,
            out_features=n_feats
        )

    def forward(self, X):
        w = self.w(X)
        output = F.softmax(torch.mul(X, w), dim=1)
        return output

def avg_pool(data, input_lens: Optional[torch.LongTensor] = None):
    """
    A 1d avg pool for variable length of data
    Args:
        data: of dim (batch, seq_len, hidden_size)
        input_lens: Optional long tensor of dim (batch,) that represents the
            original lengths without padding. Tokens past these lengths will not
            be included in the average.

    Returns:
        Tensor (batch, hidden_size)

    """
    if input_lens is not None:
        return torch.stack([
            torch.sum(data[i, :l, :], dim=0) / l for i, l in enumerate(input_lens)
        ])
    else:
        return torch.sum(data, dim=1) / float(data.shape[1])

class Simple_Classifier(nn.Module):
    def __init__(self, nz, n_out = 1):
        super(Simple_Classifier, self).__init__()
        self.nz = nz
        self.n_out = n_out

        self.net = nn.Linear(nz, n_out)

    def forward(self, x):
        if nz == 1:
            return torch.sigmoid(x)
        else:
            return torch.sigmoid(self.net(x))