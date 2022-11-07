import math
import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import datasets


class SingleDensityNetwork(nn.Module):
    def __init__(self, n_input, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_input, affine=False),
            nn.Linear(n_input, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 2)
        )

    def forward(self, x):
        x = self.net(x)
        mu = x[:, 0]            # first value = mu
        log_sigma = x[:, 1]     # second value = sigma -> immediate log
        sigma = torch.exp(
            torch.clip(log_sigma, -20, 20)      # because of floating point precision
        )

        return torch.distributions.Normal(mu, sigma)      # create pytorch dist


num_steps = 5000
batch_size = 256

device = "cuda"         # if pc not known check if cuda available

data = datasets.fetch_california_housing()
X = torch.FloatTensor(data["data"]).to(device)
y = torch.FloatTensor(data["target"]).to(device)

model = SingleDensityNetwork(X.shape[-1], 4).to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)

for _ in range(num_steps):
    batch_idx = np.random.choice(len(X), batch_size)
    dist = model(X[batch_idx])

    loss = -dist.log_prob(y[batch_idx]).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()

print(dist)


#def mdm(points):

 #   return ...


#if __name__ == '__main__':
 #   dataset = datasets.fetch_california_housing()
 #   print(dataset)
