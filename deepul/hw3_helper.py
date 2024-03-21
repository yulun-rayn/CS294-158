import os
import sys
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms as transforms

from PIL import Image as PILImage
import scipy.ndimage
import cv2

import deepul.pytorch_util as ptu
from .model_util import GoogLeNet
from .utils import (
    savefig,
    show_samples,
)


def train_epoch(model, train_loader, g_optimizer, d_optimizer, n_steps=1, g_scheduler=None, d_scheduler=None, weight_clipping=None):
    model.train()

    g_losses, d_losses = dict(), dict()
    for i, x in enumerate(train_loader):
        x = x.to(ptu.device).float()
        d_loss, d_loss_dict = model.loss_discriminator(x)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        if weight_clipping is not None:
            for param in model.discriminator.parameters():
                param.data.clamp_(-weight_clipping, weight_clipping)

        for k in d_loss_dict.keys():
            if k not in d_losses:
                d_losses[k] = []
            d_losses[k].append(d_loss_dict[k])

        if i % n_steps == 0:  # generator step
            g_loss, g_loss_dict = model.loss_generator(x)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            if g_scheduler is not None:
                g_scheduler.step()
            if d_scheduler is not None:
                d_scheduler.step()

            for k in g_loss_dict.keys():
                if k not in g_losses:
                    g_losses[k] = []
                g_losses[k].append(g_loss_dict[k])
    return {**g_losses, **d_losses}

def test(model, data_loader):
    model.eval()
    total_losses = dict()
    with torch.no_grad():
        for x in data_loader:
            x = x.to(ptu.device).float()
            _, loss_dict = model.loss(x)
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v * x.shape[0]

        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
    return total_losses

def train(model, train_loader, test_loader=None, lr=1e-3, betas=(0, 0.9), epochs=10, n_steps=1,
          lr_schedule=None, weight_clipping=None, save_state_dict=None, save_dir="data/"):
    if save_state_dict is not None:
        os.makedirs(save_dir, exist_ok=True)

    g_optimizer = optim.Adam(model.generator.parameters(), lr=lr, betas=betas)
    d_optimizer = optim.Adam(model.discriminator.parameters(), lr=lr, betas=betas)

    if lr_schedule is None:
        g_scheduler = None
        d_scheduler = None
    elif lr_schedule == "linear":
        g_iterations = epochs * len(train_loader) // n_steps
        g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer,
            lambda i: (g_iterations - i) / g_iterations)
        d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer,
            lambda i: (g_iterations - i) / g_iterations)
    else:
        g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_schedule)
        d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_schedule)

    model.to(ptu.device)
    train_losses = dict()
    if test_loader is not None:
        test_losses = dict((k, [v]) for k, v in test(model, test_loader).items())
    for epoch in tqdm(range(1, epochs+1), desc='Epoch', leave=False):
        train_loss = train_epoch(model, train_loader,
            g_optimizer, d_optimizer, n_steps=n_steps,
            g_scheduler=g_scheduler, d_scheduler=d_scheduler,
            weight_clipping=weight_clipping)
        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
            train_losses[k].extend(train_loss[k])

        if test_loader is not None:
            test_loss = test(model, test_loader)
            for k in test_loss.keys():
                test_losses[k].append(test_loss[k])

        if save_state_dict is not None:
            if (save_state_dict == "all") or (epoch in save_state_dict):
                torch.save(model.state_dict(), os.path.join(save_dir, f"model-epoch{epoch}.pt"))

    if test_loader is None:
        return train_losses
    else:
        return train_losses, test_losses


def plot_gan_training(losses, title, fname):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    savefig(fname)


######################
##### Question 1 #####
######################

def q1_data(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1.5, scale=0.35, size=(n//2,))
    gaussian2 = np.random.normal(loc=0.2, scale=0.6, size=(n//2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data -1

def visualize_q1_dataset():
    data = q1_data()
    plt.hist(data, bins=50, alpha=0.7, label='train data')
    plt.legend()
    plt.show()


def q1_gan_plot(data, samples, xs, ys, title, fname):
    plt.figure()
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
    plt.hist(data, bins=50, density=True, alpha=0.7, label='real')

    plt.plot(xs, ys, label='discrim')
    plt.legend()
    plt.title(title)
    savefig(fname)

def q1_save_results(part, fn):
    data = q1_data()
    xs = np.linspace(-1, 1, 1000)
    losses, samples1, ys1, samples_end, ys_end = fn(data, xs)

    # loss plot
    plot_gan_training(losses, 'Q1{} Losses'.format(part), 'hw3_latex/figures/q1{}_losses.png'.format(part))

    # samples
    q1_gan_plot(data, samples1, xs, ys1, 'Q1{} Epoch 1'.format(part), 'hw3_latex/figures/q1{}_epoch1.png'.format(part))
    q1_gan_plot(data, samples_end, xs, ys_end, 'Q1{} Final'.format(part), 'hw3_latex/figures/q1{}_final.png'.format(part))

######################
##### Question 2 #####
######################

def calculate_is(samples, bs=100):
    assert (type(samples[0]) == np.ndarray)
    assert (len(samples[0].shape) == 3)

    model = GoogLeNet().to(ptu.device)
    model.load_state_dict(torch.load("./data/classifier.pt"))
    softmax = nn.Sequential(model, nn.Softmax(dim=1))

    softmax.eval()
    with torch.no_grad():
        preds = []
        n_batches = int(math.ceil(float(len(samples)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = ptu.FloatTensor(samples[(i * bs):min((i + 1) * bs, len(samples))])
            pred = ptu.get_numpy(softmax(inp))
            preds.append(pred)
    preds = np.concatenate(preds, 0)
    kl = preds * (np.log(preds) - np.log(np.expand_dims(np.mean(preds, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    return np.exp(kl)

def load_q2_data():
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(),
                                              download=True, train=True)
    return train_data

def visualize_q2_data():
    train_data = load_q2_data()
    imgs = train_data.data[:100]
    show_samples(imgs, title=f'CIFAR-10 Samples')

def q2_save_results(fn):
    train_data = load_q2_data()
    train_data = train_data.data / 255.0
    train_losses, samples = fn(train_data)

    print("Inception score:", calculate_is(samples.transpose([0, 3, 1, 2])))
    plot_gan_training(train_losses, 'Q2 Losses', 'hw3_latex/figures/q2_losses.png')
    show_samples(samples[:100] * 255.0, fname='hw3_latex/figures/q2_samples.png', title=f'CIFAR-10 generated samples')

######################
##### Question 3 #####
######################

def load_q3_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform).data.transpose((0, 3, 1, 2)) / 255.0
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform).data.transpose((0, 3, 1, 2)) / 255.0
    return train_data, test_data


def visualize_q3_data():
    train_data, _ = load_q3_data()
    imgs = train_data.data[:100]
    show_samples(imgs.reshape([100, 28, 28, 1]) * 255.0, title='CIFAR10 samples')

def save_plot(
    train_losses: np.ndarray, test_losses: np.ndarray, title: str, fname: str
) -> None:
    plt.figure()
    if test_losses is None:
        plt.plot(train_losses, label="train")
        plt.xlabel("Iteration")
    else:
        n_epochs = len(test_losses) - 1
        x_train = np.linspace(0, n_epochs, len(train_losses))
        x_test = np.arange(n_epochs + 1)

        plt.plot(x_train, train_losses, label="train")
        plt.plot(x_test, test_losses, label="test")
        plt.xlabel("Epoch")
    plt.legend()
    plt.title(title)
    plt.ylabel("loss")
    savefig(fname)


def q3_save_results(fn, part):
    train_data, test_data = load_q3_data()
    gan_losses, lpips_losses, l2_train_losses, l2_val_losses, recon_show = fn(train_data, test_data, test_data[:100])

    plot_gan_training(gan_losses, f'Q3{part} Discriminator Losses', f'hw3_latex/figures/q3{part}_gan_losses.png')
    save_plot(l2_train_losses, l2_val_losses, f'Q3{part} L2 Losses', f'hw3_latex/figures/q3{part}_l2_losses.png')
    save_plot(lpips_losses, None, f'Q3{part} LPIPS Losses', f'hw3_latex/figures/q3{part}_lpips_losses.png')
    show_samples(test_data[:100].transpose(0, 2, 3, 1) * 255.0, nrow=20, fname=f'hw3_latex/figures/q3{part}_data_samples.png', title=f'Q3{part} CIFAR10 val samples')
    show_samples(recon_show * 255.0, nrow=20, fname=f'hw3_latex/figures/q3{part}_reconstructions.png', title=f'Q3{part} VQGAN reconstructions')
    print('final_val_reconstruction_loss:', l2_val_losses[-1])

######################
##### Question 4 #####
######################

def get_colored_mnist(data, data_dir="data/"):
    # from https://www.wouterbulten.nl/blog/tech/getting-started-with-gans-2-colorful-mnist/
    # Read Lena image
    lena = PILImage.open(os.path.join(data_dir, "lena.jpg"))

    # Resize
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in data])

    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)

    # Make binary
    batch_binary = (batch_rgb > 0.5)

    batch = np.zeros((data.shape[0], 28, 28, 3))

    for i in range(data.shape[0]):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        image = np.asarray(image) / 255.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]

        batch[i] = cv2.resize(image, (0, 0), fx=28 / 64, fy=28 / 64, interpolation=cv2.INTER_AREA)
    return batch.transpose(0, 3, 1, 2)

def _load_q4_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_data, test_data

def load_q4_data():
    train, _ = _load_q4_data()
    mnist = np.array(train.data.reshape(-1, 28, 28, 1) / 255.0)
    colored_mnist = get_colored_mnist(mnist)
    return mnist.transpose(0, 3, 1, 2), colored_mnist

def visualize_cyclegan_datasets():
    mnist, colored_mnist = load_q4_data()
    mnist, colored_mnist = mnist[:100], colored_mnist[:100]
    show_samples(mnist.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')
    show_samples(colored_mnist.transpose([0, 2, 3, 1]) * 255.0, title=f'Colored MNIST samples')

def q4_save_results(fn):
    mnist, cmnist = load_q4_data()

    m1, c1, m2, c2, m3, c3 = fn(mnist, cmnist)
    m1, m2, m3 = m1.repeat(3, axis=3), m2.repeat(3, axis=3), m3.repeat(3, axis=3)
    mnist_reconstructions = np.concatenate([m1, c1, m2], axis=0)
    colored_mnist_reconstructions = np.concatenate([c2, m3, c3], axis=0)

    show_samples(mnist_reconstructions * 255.0, nrow=20,
                 fname='hw3_latex/figures/q4_mnist.png',
                 title=f'Source domain: MNIST')
    show_samples(colored_mnist_reconstructions * 255.0, nrow=20,
                 fname='hw3_latex/figures/q4_colored_mnist.png',
                 title=f'Source domain: Colored MNIST')
