import pandas as pd
import torch
from torch import tensor, full

from segment import energy

raw = pd.read_csv('data.txt', sep='\t', names=['x', 'y', 'z'])
pos = [tensor(group.values) for _, group in raw.groupby('x')]
print(len(pos))
print([x.shape for x in pos])
E = energy(pos)
act = [full((len(a), len(b)), 0.1, requires_grad=True) for a, b in zip(pos, pos[1:])]
print(len(act))
print([v.shape for v in act])

T = 1000.

for i in range(20):
    e = E(act)
    print(e.item())
    e.backward()
    for j in range(len(act)):
        act[j] = torch.sigmoid(- act[j].grad / T).clone().detach().requires_grad_(True)

print(act[0])
