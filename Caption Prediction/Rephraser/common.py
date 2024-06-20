import os
import matplotlib.pyplot as plt
import numpy as np

"""
Read file to obtain its raw string contents.
"""
def read_file(src_directory, filename, encoding=None, log=True):
    if log: print("Reading file:", filename)
    with open(os.path.join(src_directory, filename), mode='r', encoding=encoding) as file:
        content = file.read().split("\n")
    return content

"""
Write a given text.
"""
def write(text, path=None, filename=None, write_back=True, encoding='utf-8'):
    # Write preprocessing results back to file.
    if write_back:
        with open(os.path.join(path, filename), mode='w', encoding=encoding) as file:
            file.write(text)

    return text

"""
Find captions' maximum length in Mimic-CXR medical images dataset.
"""
def min_length(captions):
    return np.min(np.array([len(caption) for caption in captions]))

"""
Find captions' maximum length in Mimic-CXR medical images dataset.
"""
def max_length(captions):
    return np.max(np.array([len(caption) for caption in captions]))

"""
Find captions' average length in Mimic-CXR medical images dataset.
"""
def avg_length(captions):
    return np.average(np.array([len(caption) for caption in captions]))

def plot(loss, color, label='BERTScore',  yLabel='BERTScore'):
    num_epochs = len(loss)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, loss, color, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(yLabel)
    plt.legend(loc=4)

def BERTScore_plots(BS_validation, BS_development, directory=None):
    # Plotting validation scores per epoch
    plt.subplot(1, 2, 1)
    plot(BS_validation, 'b-', "Validation")

    # Plotting development scores per epoch
    plt.subplot(1, 2, 1)
    plot(BS_development, 'r-', "Development")

    if directory is not None:
        plt.savefig(directory)

    plt.show()


def BERTScore_plot(BS_development, directory=None):
    # Plotting development scores per epoch
    plt.subplot(1, 1, 1)
    plot(BS_development, 'r-', "Development")

    if directory is not None:
        plt.savefig(directory)

    plt.show()


def plot_loss_1(loss_train, directory=None):
    epochs = range(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, 'r', label='Training set')
    plt.title('LM loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if directory is not None:
        plt.savefig(directory)

    plt.show()


def loss_plots(loss_train, loss_validation, loss_development, directory=None):
    # Plotting training scores per epoch
    plt.subplot(1, 3, 1)
    plot(loss_train, 'g-', "Training", yLabel='Loss')

    # Plotting validation scores per epoch
    plt.subplot(1, 3, 2)
    plot(loss_validation, 'b-', "Validation", yLabel='Loss')

    # Plotting development scores per epoch
    plt.subplot(1, 3, 3)
    plot(loss_development, 'r-', "Development", yLabel='Loss')

    if directory is not None:
        plt.savefig(directory)

    plt.show()
