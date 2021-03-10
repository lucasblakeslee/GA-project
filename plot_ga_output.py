#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

def main():
    if len(sys.argv) == 2 and sys.argv[1] in ('--help', '-h'):
        print(f'usage: {sys.argv[0]} [GA-gen-info_pid#.out ...]')
        print('with no arguments, process all GA-gen-info_pid*.out in this directory')
        sys.exit(0)
    if len(sys.argv) > 1:
        flist = sys.argv[1:]
    else:
        flist = glob.glob('GA-gen-info_pid*.out')
    for runfile in flist:
        plot_runfile(runfile)
    plt.show()

def plot_runfile(fname):
    print(f'# plotting GA run from file {fname}')
    gens, max_fit, avg_fit, elite_avg_fit, entropy = load_file(fname)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(gens, max_fit, gens, avg_fit, gens, elite_avg_fit)
    axs[0].set_xlim(0, gens.size - 1)
    axs[0].set_xlabel('generation')
    axs[0].set_ylabel('fitness (max,avg,elite)')

    axs[1].plot(gens, entropy)
    axs[1].set_xlim(0, gens.size - 1)
    axs[1].set_ylabel('entropy')

    fig.tight_layout()


def load_file(fname):
    # gens, mf, af, eaf, entropy = np.loadtxt(fname, usecols=[1,5,6,7,8])
    data = np.loadtxt(fname, usecols=[1,5,6,7,8])
    print(data)
    print(data.shape)
    print(data[0])
    print(data[0].shape)
    gens, mf, af, eaf, entropy = np.transpose(data)
    return gens, mf, af, eaf, entropy

if __name__ == '__main__':
    main()