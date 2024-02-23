import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_epoch(model, train_loader, optimizer, grad_clip=None):
    train_epoch_losses = []
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        train_epoch_losses.append(loss.item())
    return train_epoch_losses

def test(model, test_loader):
    test_epoch_losses = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            loss = model.loss(data)

            test_epoch_losses.append(loss.item())
    return sum(test_epoch_losses)/len(test_epoch_losses)

def train(model, train_loader, test_loader,
          lr=1e-3, epochs=10, grad_clip=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = [test(model, test_loader)]
    for epoch in range(epochs):
        train_epoch_losses = train_epoch(model, train_loader, optimizer, grad_clip=grad_clip)
        train_losses.extend(train_epoch_losses)

        test_epoch_loss = test(model, test_loader)
        test_losses.append(test_epoch_loss)
        print(f"Train Epoch: {epoch+1} \tTest Loss: {test_epoch_loss}")

    return train_losses, test_losses


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


"""
GPU wrappers
"""

_use_gpu = False
device = None
_gpu_id = 0


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs).to(torch_device)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def get_numpy(tensor):
    return tensor.to("cpu").detach().numpy()

def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)

def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)

def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)

def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)

def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)

def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)

def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)
