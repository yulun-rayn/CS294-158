import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, batch_norm=True, act="relu", final_act=None):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[i], sizes[i+1]),

                nn.BatchNorm1d(sizes[i+1])
                    if batch_norm and i < len(sizes) - 2 else
                None,

                nn.ReLU()
                    if act == "relu" and i < len(sizes) - 2 else
                nn.LeakyReLU(0.2) 
                    if act == "leaky_relu" and i < len(sizes) - 2 else
                None
            ]
        if final_act is None:
            pass
        elif final_act == "relu":
            layers += [nn.ReLU()]
        elif final_act == "tanh":
            layers += [nn.Tanh()]
        elif final_act == "sigmoid":
            layers += [nn.Sigmoid()]
        elif final_act == "softmax":
            layers += [nn.Softmax(dim=-1)]
        else:
            raise ValueError("final_act not recognized")

        layers = [l for l in layers if l is not None]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
