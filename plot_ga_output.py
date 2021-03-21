#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        fig, axtxt, axs = prepare_plots(runfile)
        plot_runfile(runfile, fig, axs)
        t = "This is a really long string that I'd rather have wrapped so that it"\
            " doesn't go outside of the figure, but if it's long enough it will go"\
            " off the top or bottom!"
        axtxt.text(0, 0, t, ha='left', rotation=0, wrap=True)
    plt.show()

def prepare_plots(fname):
    # fig, axs = plt.subplots(3, 2)
    fig = plt.figure(constrained_layout = True)
    gs = gridspec.GridSpec(3, 2)
    axtxt = fig.add_subplot(gs[:, -1])
    axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    print(axtxt)
    print(axs)
    suptitle = fig.suptitle(f'GA output file {fname}', fontsize='x-large')
    suptitle.set_y(0.98)
    return fig, axtxt, axs
    

def plot_runfile(fname, fig, axs):
    print(f'# plotting GA run from file {fname}')
    gens, fittest, max_fit, avg_fit, elite_avg_fit, entropy = load_file(fname)
    axs[0].plot(gens, max_fit, gens, avg_fit, gens, elite_avg_fit)
    axs[0].set_xlim(0, gens.size - 1)
    axs[0].set_xlabel('generation')
    axs[0].set_ylabel('fitness (max,avg,elite)')

    axs[1].plot(gens, entropy)
    axs[1].set_xlim(0, gens.size - 1)
    axs[1].set_ylabel('entropy')

    axs[2].plot(gens, fittest)
    axs[2].set_xlim(0, gens.size - 1)
    axs[2].set_ylabel('position')


    fig.subplots_adjust(top=0.65)
    fig.tight_layout()
    fig.savefig(f'{fname}.pdf')

def load_file(fname):
    # gens, mf, af, eaf, entropy = np.loadtxt(fname, usecols=[1,5,6,7,8])
    data = np.loadtxt(fname, usecols=[1,3,5,6,7,8])
    # print(data)
    # print(data.shape)
    # print(data[0])
    # print(data[0].shape)
    gens, fittest, mf, af, eaf, entropy = np.transpose(data)
    return gens, fittest, mf, af, eaf, entropy

if __name__ == '__main__':
    main()
