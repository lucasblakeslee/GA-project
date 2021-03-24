#! /usr/bin/env python3

import math
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

max_gen = 3000
show_logscale_panel = False

def main():
    if len(sys.argv) == 2 and sys.argv[1] in ('--help', '-h'):
        print(f'usage: {sys.argv[0]} [GA-gen-info_pid#.out ...]')
        print('with no arguments, process all GA-gen-info_pid*.out in this directory')
        sys.exit(0)
    if len(sys.argv) > 1:
        flist = sys.argv[1:]
    else:
        flist = sorted(glob.glob('GA-gen-info_pid*.out'))
    for runfile in flist:
        fig, axtxt, axs, axslog = prepare_plots(runfile)
        present_metadata(runfile, axtxt)
        plot_runfile(runfile, fig, axs, axslog)
        show_generations(runfile)
    plt.show()

def present_metadata(fname, ax):
    """Loads metadata from a file and shows it in an ax object."""
    metadata = load_LEM(fname)
    n_lines = len(metadata.keys())
    font0 = FontProperties()
    font = font0.copy()
    font.set_family('monospace')
    font.set_size('xx-small')
    for i, key in enumerate(metadata.keys()):
        ax.text(0.02, 1.01 - (i+1)/n_lines, key + ': ' + metadata[key], ha='left',
                fontproperties=font)

def prepare_plots(fname):
    # fig = plt.figure(constrained_layout = True)
    fig = plt.figure(figsize=(9.6, 5.0))
    n_panels = 3 if show_logscale_panel else 2
    gs = gridspec.GridSpec(3, n_panels)
    axtxt = fig.add_subplot(gs[:, -1])
    axtxt.xaxis.set_visible(False)
    axtxt.yaxis.set_visible(False)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axslog = None
    if show_logscale_panel:
        axslog = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    suptitle = fig.suptitle(f'GA output file {fname}', fontsize='x-large')
    suptitle.set_y(0.98)
    return fig, axtxt, axs, axslog
    

def plot_runfile(fname, fig, axs, axslog):
    print(f'# plotting GA run from file {fname}')
    gens, fittest, max_fit, avg_fit, elite_avg_fit, entropy = load_file(fname)
    if not show_logscale_panel:
        metadata_panel_index = 2
    else:
        metadata_panel_index = 1

    axs[0].plot(gens, max_fit, gens, avg_fit, gens, elite_avg_fit)
    axs[0].set_xlim(0, gens.size - 1)
    axs[0].set_xlabel('generation')
    axs[0].set_ylabel('fitness (max,avg,elite)')
    if axslog:
        axslog[0].set_yscale('log')
        axslog[0].plot(gens, max_fit, gens, avg_fit, gens, elite_avg_fit)
        axslog[0].set_xlim(0, gens.size - 1)
        axslog[0].set_xlabel('generation')
        axslog[0].set_ylabel('fitness (max,avg,elite)')
    
    axs[1].plot(gens, entropy)
    axs[1].set_xlim(0, gens.size - 1)
    axs[1].set_ylabel('entropy')
    if axslog:
        axslog[1].set_yscale('log')
        axslog[1].plot(gens, entropy)
        axslog[1].set_xlim(0, gens.size - 1)
        axslog[1].set_ylabel('entropy')

    axs[metadata_panel_index].plot(gens, fittest)
    axs[metadata_panel_index].set_xlim(0, gens.size - 1)
    axs[metadata_panel_index].set_ylabel('position')
    if axslog:
        axslog[metadata_panel_index].set_yscale('log')
        axslog[metadata_panel_index].plot(gens, fittest)
        axslog[metadata_panel_index].set_xlim(0, gens.size - 1)
        axslog[metadata_panel_index].set_ylabel('position')


    fig.subplots_adjust(top=0.65)
    fig.tight_layout()
    fig.savefig(f'{fname}.pdf')

def load_file(fname):
    data = np.loadtxt(fname, usecols=[1,3,5,6,7,8])
    gens, fittest, mf, af, eaf, entropy = np.transpose(data)
    if max_gen:
        return (gens[:max_gen], fittest[:max_gen], mf[:max_gen],
                af[:max_gen], eaf[:max_gen], entropy[:max_gen])
    return gens, fittest, mf, af, eaf, entropy

def load_LEM(fname):
    """Loads the LEM (low effort metadata) from file fname."""
    LEM_dict = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line[:2] != '##':
                return LEM_dict
            line = line.strip()
            key, val = line[2:].split(':', 1)
            assert(not key in LEM_dict)
            LEM_dict[key.strip()] = val.strip()


def show_generations(runfile):
    filebase = runfile[len('GA-gen-info_'):-len('.out')]
    print('filebase:', filebase)
    gen_fname = 'gen-dump_' + filebase + '.out'
    
    


if __name__ == '__main__':
    main()
