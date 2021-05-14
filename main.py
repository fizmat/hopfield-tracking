import torch
from torch import tensor

from segment import energy

float_ = torch.float32

layer1 = tensor([[0, 0], [0, 1], [0, 2]], dtype=float_)
layer2 = tensor([[1, 0], [1, 2]], dtype=float_)
layer3 = tensor([[2, 0], [2, 1], [2, 2]], dtype=float_)

E = energy([layer1, layer2, layer3])

v1 = torch.full((3, 2), 0.5, dtype=float_, requires_grad=True)
v2 = torch.full((2, 3), 0.5, dtype=float_, requires_grad=True)

T = 3

for i in range(20):
    e = E([v1, v2])
    print(e.item())
    e.backward()
    v1 = torch.sigmoid(- v1.grad / T).clone().detach().requires_grad_(True)
    v2 = torch.sigmoid(- v2.grad / T).clone().detach().requires_grad_(True)

print(v1)
