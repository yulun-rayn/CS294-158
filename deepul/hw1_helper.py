import json
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import torch

from deepul.models.vqvae import VQVAE
from deepul.models.text_encoder import CharEncoder, WordEncoder

from .utils import (
    load_colored_mnist_text,
    load_pickled_data,
    load_text_data,
    save_distribution_1d,
    save_distribution_2d,
    save_text_to_plot,
    save_timing_plot,
    save_training_plot,
    savefig,
    show_samples,
)


# Question 1
def q1_sample_data_1():
    count = 1000
    rand = np.random.RandomState(0)
    samples = 0.2 + 0.2 * rand.randn(count)
    data = np.digitize(samples, np.linspace(0.0, 1.0, 20))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data


def q1_sample_data_2():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.4 + 0.05 * rand.randn(count)
    b = 0.5 + 0.10 * rand.randn(count)
    c = 0.7 + 0.02 * rand.randn(count)
    mask = np.random.randint(0, 3, size=count)
    samples = np.clip(a * (mask == 0) + b * (mask == 1) + c * (mask == 2), 0.0, 1.0)

    data = np.digitize(samples, np.linspace(0.0, 1.0, 100))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data


def visualize_q1_data(dset_type):
    if dset_type == 1:
        train_data, test_data = q1_sample_data_1()
        d = 20
    elif dset_type == 2:
        train_data, test_data = q1_sample_data_2()
        d = 100
    else:
        raise Exception("Invalid dset_type:", dset_type)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Train Data")
    ax1.hist(train_data, bins=np.arange(d) - 0.5, density=True)
    ax1.set_xlabel("x")
    ax2.set_title("Test Data")
    ax2.hist(test_data, bins=np.arange(d) - 0.5, density=True)
    print(f"Dataset {dset_type}")
    plt.show()


def q1_save_results(dset_type, part, fn):
    if dset_type == 1:
        train_data, test_data = q1_sample_data_1()
        d = 20
    elif dset_type == 2:
        train_data, test_data = q1_sample_data_2()
        d = 100
    else:
        raise Exception("Invalid dset_type:", dset_type)

    train_losses, test_losses, distribution = fn(train_data, test_data, dset_type, d)
    assert np.allclose(
        np.sum(distribution), 1
    ), f"Distribution sums to {np.sum(distribution)} != 1"

    print(f"Final Test Loss: {test_losses[-1]:.4f}")

    save_training_plot(
        train_losses,
        test_losses,
        f"Q1({part}) Dataset {dset_type} Train Plot",
        f"hw1_latex/figures/q1_{part}_dset{dset_type}_train_plot.png",
    )
    save_distribution_1d(
        train_data,
        distribution,
        f"Q1({part}) Dataset {dset_type} Learned Distribution",
        f"hw1_latex/figures/q1_{part}_dset{dset_type}_learned_dist.png",
    )


# Question 2
def q2ab_save_results(dset_type, part, fn, data_dir="data/", save_dir="data/"):
    if part == "a":
        dataset_suffix = ""
        channel = 1
    elif part == "b":
        dataset_suffix = "_colored"
        channel = 3
    else:
        raise Exception("Invalid part:", part, "Must be 'a' or 'b'")

    if dset_type == 1:
        train_data, test_data = load_pickled_data(
            join(data_dir, f"shapes{dataset_suffix}.pkl")
        )
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(
            join(data_dir, f"mnist{dataset_suffix}.pkl")
        )
    else:
        raise Exception()

    train_losses, test_losses, samples, model = fn(train_data, test_data, dset_type)
    torch.save(model.state_dict(), join(save_dir, f"q2-{dset_type}-{part}.pkl"))
    samples = samples.astype("float32") / channel * 255

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q2({part}) Dataset {dset_type} Train Plot",
        f"hw1_latex/figures/q2_{part}_dset{dset_type}_train_plot.png",
    )
    show_samples(samples, f"hw1_latex/figures/q2_{part}_dset{dset_type}_samples.png")

    return model


def visualize_q2a_data(dset_type, data_dir="data/"):
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "shapes.pkl"))
        name = "Shape"
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist.pkl"))
        name = "MNIST"
    else:
        raise Exception("Invalid dset type:", dset_type)

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype("float32") / 1 * 255
    show_samples(images, title=f"{name} Samples")


def visualize_q2b_data(dset_type, data_dir="data/"):
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "shapes_colored.pkl"))
        name = "Colored Shape"
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist_colored.pkl"))
        name = "Colored MNIST"
    else:
        raise Exception("Invalid dset type:", dset_type)

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype("float32") / 3 * 255
    show_samples(images, title=f"{name} Samples")


# Question 3
def q3ab_save_results(dset_type, part, fn, data_dir="data/", save_dir="data/"):
    if part == "a":
        dataset_suffix = ""
        channel = 1
    elif part == "b":
        dataset_suffix = "_colored"
        channel = 3
    else:
        raise Exception("Invalid part:", part, "Must be 'a' or 'b'")

    if dset_type == 1:
        train_data, test_data = load_pickled_data(
            join(data_dir, f"shapes{dataset_suffix}.pkl")
        )
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(
            join(data_dir, f"mnist{dataset_suffix}.pkl")
        )
    else:
        raise Exception()

    train_losses, test_losses, samples, model = fn(train_data, test_data, dset_type)
    torch.save(model.state_dict(), join(save_dir, f"q3-{dset_type}-{part}.pkl"))
    samples = samples.astype("float32") / channel * 255

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q3({part}) Dataset {dset_type} Train Plot",
        f"hw1_latex/figures/q3_{part}_dset{dset_type}_train_plot.png",
    )
    show_samples(samples, f"hw1_latex/figures/q3_{part}_dset{dset_type}_samples.png")

    return model


def q3c_save_results(dset_type, fn, model):
    if dset_type != 1 and dset_type != 2:
        raise Exception()

    (
        time_list_no_cache,
        time_list_with_cache,
        samples_no_cache,
        samples_with_cache,
    ) = fn(model)
    samples_no_cache = samples_no_cache.astype("float32") / 3 * 255
    samples_with_cache = samples_with_cache.astype("float32") / 3 * 255

    save_timing_plot(
        time_list_no_cache,
        time_list_with_cache,
        "Q3(c) Timing Plot",
        f"hw1_latex/figures/q3_c_dset{dset_type}_timing_plot.png",
        time1_label="no cache",
        time2_label="with cache",
    )
    show_samples(samples_no_cache, f"hw1_latex/figures/q3_c_no_cache_dset{dset_type}_samples.png")
    show_samples(
        samples_with_cache, f"hw1_latex/figures/q3_c_with_cache_dset{dset_type}_samples.png"
    )


# Question 4


# load vqvae mode
def load_pretrain_vqvae(name: str, data_dir: str = "data/"):
    loaded_args = torch.load(join(data_dir, f"vqvae_args_{name}_ft" + ".pth"))
    vqvae = VQVAE(**loaded_args)
    vqvae.load_state_dict(torch.load(join(data_dir, f"vqvae_{name}_ft" + ".pth")))
    return vqvae


def q4a_save_results(dset_type, fn, data_dir="data/"):
    if dset_type == 1:
        #  @ load colored mnist
        train_data, _ = load_pickled_data(join(data_dir, "mnist_colored.pkl"))
        vqvae = load_pretrain_vqvae("colored_mnist")
    elif dset_type == 2:
        train_data, _, _, _ = load_colored_mnist_text(
            join(data_dir, "colored_mnist_with_text.pkl")
        )
        vqvae = load_pretrain_vqvae("colored_mnist_2")
    else:
        raise Exception()

    # get two images
    images = train_data[:2]
    post_decoded_images = fn(images, vqvae)
    stacked_images = np.concatenate([images, post_decoded_images], axis=0)

    vq_images = stacked_images.astype("float32") / 3 * 255
    show_samples(vq_images, f"hw1_latex/figures/q4_a_dset{dset_type}_samples.png", nrow=2)


def q4b_save_results(dset_type, fn, data_dir="data/", save_dir="data/"):
    if dset_type == 1:
        #  @ load colored mnist
        train_data, test_data = load_pickled_data(join(data_dir, "mnist_colored.pkl"))
        vqvae = load_pretrain_vqvae("colored_mnist")
    elif dset_type == 2:
        train_data, test_data, _, _ = load_colored_mnist_text(
            join(data_dir, "colored_mnist_with_text.pkl")
        )
        vqvae = load_pretrain_vqvae("colored_mnist_2")
    else:
        raise Exception()

    train_losses, test_losses, samples, model = fn(train_data, test_data, dset_type, vqvae)
    torch.save(model.state_dict(), join(save_dir, f"q4-{dset_type}-b.pkl"))
    samples = samples.astype("float32") / 3 * 255

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Qb(a) Dataset {dset_type} Train Plot",
        f"hw1_latex/figures/q4_b_dset{dset_type}_train_plot.png",
    )
    show_samples(samples, f"hw1_latex/figures/q4_b_dset{dset_type}_samples.png")

    return model


# Question 5
def visualize_q5_data(data_dir="data/"):
    train_data, _ = load_text_data(join(data_dir, "poetry.pkl"))

    # randomly sample 4 sentences
    idx = np.random.choice(len(train_data), size=4, replace=False)
    for idx, i in enumerate(idx):
        print(f"Sample {idx+1}")
        print(train_data[i])
        print("-" * 80 + "\n")


def q5a_save_results(fn, data_dir="data/", save_dir="data/"):
    train_data, test_data = load_text_data(join(data_dir, "poetry.pkl"))
    encoder = CharEncoder(train_data+test_data)
    (
        train_losses,
        test_losses,
        text_samples,
        model
    ) = fn(
        train_data,
        test_data,
        encoder
    )
    torch.save(model.state_dict(), join(save_dir, f"q5-a.pkl"))

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q5(a) Dataset Poetry Train Plot",
        f"hw1_latex/figures/q5_a_train_plot.png",
    )
    save_text_to_plot(text_samples, f"hw1_latex/figures/q5_a_samples.png")

    return model


# Question 6
def visualize_q6_data(data_dir="data/"):
    """
    Visualize samples from the colored MNIST dataset.

    Parameters:
    data (list): The colored MNIST dataset.
    num_samples (int): Number of samples to display (default is 9).
    """
    num_samples = 9
    train_data, _, train_labels, _ = load_colored_mnist_text(
        join(data_dir, "colored_mnist_with_text.pkl")
    )
    # get 9 random samples
    idx = np.random.choice(len(train_data), size=num_samples, replace=False)

    images = train_data[idx]
    labels = [train_labels[i] for i in idx]
    packed_samples = list(zip(images, labels))
    plot_q6a_samples(packed_samples)


def plot_q6a_samples(samples_img_txt_tuples, filename=None, fig_title=None):
    assert len(samples_img_txt_tuples) == 9
    # unzip into list of images and labels
    images = np.stack([tup[0] for tup in samples_img_txt_tuples])
    labels = [tup[1] for tup in samples_img_txt_tuples]
    images = np.floor(images.astype("float32") / 3 * 255).astype(int)
    labels = [labels[i] for i in range(len(labels))]
    plt.figure(figsize=(6, 6))
    for i in range(9):
        img = images[i]
        label = labels[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(label, fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    if fig_title is not None:
        plt.suptitle(fig_title, fontsize=10)

    if filename is None:
        plt.show()
    else:
        savefig(filename)


def q6a_save_results(fn, data_dir="data/"):
    train_data, test_data, train_labels, test_labels = load_colored_mnist_text(
        join(data_dir, "colored_mnist_with_text.pkl")
    )
    encoder = WordEncoder(train_labels+test_labels)
    vqvae = load_pretrain_vqvae("colored_mnist_2")
    # extract out the images only
    img_test_prompt = test_data[:9]  # get first 9 samples
    text_test_prompt = test_labels[:9]  # get first 9 samples
    (
        train_losses,
        test_losses,
        samples_from_image,
        samples_from_text,
        samples_unconditional,
    ) = fn(
        train_data,
        test_data,
        train_labels,
        test_labels,
        img_test_prompt,
        text_test_prompt,
        vqvae,
        encoder,
        n_samples=9
    )

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q6(a) Train Plot",
        f"hw1_latex/figures/q6_a_train_plot.png",
    )
    plot_q6a_samples(
        samples_from_image,
        f"hw1_latex/figures/q6_a_samples_img_conditioned.png",
        fig_title="Image Conditioned Samples",
    )
    plot_q6a_samples(
        samples_from_text,
        f"hw1_latex/figures/q6_a_samples_text_conditioned.png",
        fig_title="Text Conditioned Samples",
    )
    plot_q6a_samples(
        samples_unconditional,
        f"hw1_latex/figures/q6_a_samples_unconditional.png",
        fig_title="Unconditional Samples",
    )
