import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import torchvision

def _show_comparison_graph(x1s, y1s, label1, x2s, y2s, label2, xlabel=None, ylabel=None, title=None, ylim=None):
    fig, axs = plt.subplots(1, 1)
    axs.plot(x1s, y1s, label=label1)
    axs.plot(x2s, y2s, label=label2)
    if xlabel is not None:
        axs.set_xlabel(xlabel)
    if ylabel is not None:
        axs.set_ylabel(ylabel)
    if title is not None:
        axs.set_title(title)
    if ylim is not None:
        axs.set_ylim(ylim)
    axs.legend()
    plt.show()

def show_learning_progress(losses, accs, ylim=None):
    _show_comparison_graph(np.arange(len(losses['train'])),
                          losses['train'], 'train',
                          np.arange(len(losses['val'])),
                          losses['val'], 'val',
                          title = 'Loss',
                          ylim = ylim)
    _show_comparison_graph(np.arange(len(accs['train'])),
                          accs['train'], 'train',
                          np.arange(len(accs['val'])),
                          accs['val'], 'val',
                          title = 'Accuracy',
                          ylim = ylim)

def show_data_histogram(root_dir, title, show=True, weight = []):
    labels_list = []
    dirs = sorted(os.listdir(root_dir))
    labels = dirs
    data_nums = []
    
    for i, d in enumerate(dirs):
        num = len(os.listdir("{}/{}".format(root_dir, d)))
        if any(weight) == False:
            data_nums.append(num)
        else:
            data_nums.append(num * weight[i])

        for i in range(num):
            labels_list.append(d)
    
    if show == True:
        fig = plt.figure()
        fig.set_size_inches(4, 2)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_title(title)
        ax.bar(labels, data_nums)
        plt.show()
    return labels_list

def show_frequency_histogram(data_dir):
    _ = show_data_histogram('{}/{}'.format(data_dir, 'train'), 'val data')
    _ = show_data_histogram('{}/{}'.format(data_dir, 'val'), 'val data')

def _imshow(inp, title=None):
    plt.figure(figsize=(20,10))
    inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_augmented_images(dataloader, class_names):
    inputs, classes = next(iter(dataloader))
    # print(inputs, classes)
    out = torchvision.utils.make_grid(inputs)
    _imshow(out, title=[class_names[x] for x in classes])