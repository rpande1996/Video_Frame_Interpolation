import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(exp_dir, save):
    with open(os.path.join(exp_dir, 'stats.pickle'), 'rb') as handle:
        stats = pickle.load(handle)
        handle.close()
    with open(os.path.join(exp_dir, 'hyperparams.pickle'), 'rb') as handle:
        hyperparams = pickle.load(handle)
        handle.close()
    print("Experiment settings:\n{}".format(hyperparams))
    epoch_interval = hyperparams['eval_every']

    train_g_loss = [i[0] * 100 for i in stats['train_loss']]
    val_g_loss = [i[0] * 100 for i in stats['val_loss']]
    length = len(stats['train_loss'])
    epochs = [epoch_interval * i + 1 for i in range(length)]
    print(val_g_loss)
    _, axes = plt.subplots(1, 2)
    axes[0].set_title('Train Loss vs Epoch')
    axes[1].set_title('Val Loss vs Epoch')
    axes[0].plot(epochs, train_g_loss)
    axes[1].plot(epochs, val_g_loss)
    axes[0].set_xticks(np.arange(5,length,5))
    axes[1].set_xticks(np.arange(5,length,5))
    if save:
        plt.savefig('../plots/plot_{}_{}.png'.format(hyperparams['lr'], hyperparams['batch_size']))

    plt.show()

def getlatestexp(exp_dir):
    names = os.listdir(exp_dir)
    names.sort()
    exp = names[-1]

    return exp_dir + exp + '/'

if __name__ == '__main__':
    exp_dir = '../models/details/'
    new_path = getlatestexp(exp_dir)
    save = True

    plot_stats(new_path, save)