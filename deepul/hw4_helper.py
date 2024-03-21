import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.datasets import make_swiss_roll

from .model_util import VAE
from .pytorch_util import cosine_lr_lambda
from .utils import (
    save_training_plot,
    savefig,
    show_samples
)


def train_epoch(model, train_loader, optimizer, scheduler,
                grad_clip=None, supervised=False):
    model.train()

    train_losses = []
    for batch in train_loader:
        if supervised:
            x, y = batch
            x = x.float().to(model.device)
            y = y.long().to(model.device)
            loss = model.loss(x, y=y)
        else:
            x, y = batch, None
            x = x.float().to(model.device)
            loss = model.loss(x)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())
    return train_losses

def test(model, data_loader, supervised=False):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            if supervised:
                x, y = batch
                x = x.float().to(model.device)
                y = y.long().to(model.device)
                loss = model.loss(x, y=y)
            else:
                x, y = batch, None
                x = x.float().to(model.device)
                loss = model.loss(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()

def train(model, train_loader, test_loader,
          lr=1e-3, epochs=10, grad_clip=None,
          warmup_steps=100, cos_decay=False,
          supervised=False, quiet=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=cosine_lr_lambda(total_steps, warmup_steps, cos_decay)
    )

    train_losses = []
    test_losses = [test(model, test_loader, supervised=supervised)]
    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                                 grad_clip=grad_clip, supervised=supervised)
        train_losses.extend(train_loss)
        test_loss = test(model, test_loader, supervised=supervised)
        test_losses.append(test_loss)
        if not quiet:
            print(f'Epoch {epoch}, Test loss {test_loss:.4f}')

    return train_losses, test_losses


######################
##### Question 1 #####
######################


def q1_data(n=100000):
    x, _ = make_swiss_roll(n, noise=0.5)
    x = x[:, [0, 2]]
    return x.astype('float32')


def visualize_q1_dataset():
    data = q1_data()
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def save_multi_scatter_2d(data: np.ndarray) -> None:
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            axs[i, j].scatter(data[i * 3 + j, :, 0], data[i * 3 + j, :, 1])
    plt.title("Q1 Samples")
    savefig("hw4_latex/figures/q1_samples.png")


def q1_save_results(fn):
    train_data = q1_data(n=100000)
    test_data = q1_data(n=10000)
    train_losses, test_losses, samples = fn(train_data, test_data)

    print(f"Final Test Loss: {test_losses[-1]:.4f}")

    save_training_plot(
        train_losses,
        test_losses,
        f"Q1 Train Plot",
        f"hw4_latex/figures/q1_train_plot.png"
    )

    save_multi_scatter_2d(samples)
    

######################
##### Question 2 #####
######################

def load_q2_data():
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    test_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=False)
    return train_data, test_data


def visualize_q2_data():
    train_data, _ = load_q2_data()
    imgs = train_data.data[:100]
    show_samples(imgs, title=f'CIFAR-10 Samples')


def q2_save_results(fn):
    train_data, test_data = load_q2_data()
    train_data = train_data.data / 255.0
    test_data = test_data.data / 255.0
    train_losses, test_losses, samples = fn(train_data, test_data)
    
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        "Q2 Train Plot",
        "hw4_latex/figures/q2_train_plot.png"
    ) 

    samples = samples.reshape(-1, *samples.shape[2:])
    show_samples(samples * 255.0, fname="hw4_latex/figures/q2_samples.png", title=f"Q2 CIFAR-10 generated samples")


######################
##### Question 3 #####
######################

def load_q3_data():
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    test_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=False)
    return train_data, test_data


def load_pretrain_vae(data_dir="data/"):
    vae = VAE()
    vae.load_state_dict(torch.load(os.path.join(data_dir, f"vae_cifar10.pth")))
    vae.eval()
    return vae


def visualize_q3_data():
    train_data, _ = load_q3_data()
    imgs = train_data.data[:100]
    labels = train_data.targets[:100]
    show_samples(imgs, title=f'CIFAR-10 Samples')
    print('Labels:\n', np.reshape(labels, (10, -1)))

    
def q3a_save_results(fn, save_dir="data/"):
    train_data, _ = load_q3_data()
    train_images = train_data.data / 255.0
    idxs = torch.randint(0, len(train_images), (1000,))
    images = train_images[idxs]
    vae = load_pretrain_vae()

    recons, scale_factor = fn(images, vae)
    recons = recons.reshape(-1, *recons.shape[2:])
    show_samples(recons * 255.0, fname="hw4_latex/figures/q3_a_reconstructions.png", title=f"Q3(a) CIFAR-10 VAE Reconstructions")
    print(f"Scale factor: {scale_factor:.4f}")
    with open(os.path.join(save_dir, f"q3_scale_factor.txt"), 'w') as f:
        f.write('%f' % scale_factor)


def q3b_save_results(fn, save_dir="data/"):
    train_data, test_data = load_q3_data()
    with open(os.path.join(save_dir, f"q3_scale_factor.txt"), 'r') as f:
        scale_factor = float(f.readline())

    train_images = train_data.data / 255.0
    train_labels = np.array(train_data.targets, dtype=np.int32)
    test_images = test_data.data / 255.0
    test_labels = np.array(test_data.targets, dtype=np.int32)
    vae = load_pretrain_vae()
    train_losses, test_losses, samples, model = fn(train_images, train_labels, test_images, test_labels, vae, scale_factor)
    torch.save(model.state_dict(), os.path.join(save_dir, f"q3.pt"))

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        "Q3(b) Train Plot",
        "hw4_latex/figures/q3_b_train_plot.png"
    ) 

    samples = samples.reshape(-1, *samples.shape[2:])
    show_samples(samples * 255.0, fname=f"hw4_latex/figures/q3_b_samples.png", title=f"Q3(b) CIFAR-10 generated samples")


def q3c_save_results(fn, model, save_dir="data/"):
    with open(os.path.join(save_dir, f"q3_scale_factor.txt"), 'r') as f:
        scale_factor = float(f.readline())

    vae = load_pretrain_vae()
    samples = fn(model, vae, scale_factor)

    cfg_values = [1.0, 3.0, 5.0, 7.5]
    for i in range(4):
        cfg_val = cfg_values[i]
        s = samples[i]
        s = s.reshape(-1, *s.shape[2:])
        show_samples(s * 255.0, fname=f"hw4_latex/figures/q3_c_samples_cfg{cfg_val}.png", title=f"Q3(c) CIFAR-10 generated samples (CFG {cfg_val})")
