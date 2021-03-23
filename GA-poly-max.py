#!/usr/bin/env python3

"""Evolves a genetic algorithm that looks for the max of a polynomial
by flipping bits in the ieee binary representation of a floating point
number.
"""

import pandas as pd
import numpy as np
import random
import struct
from struct import pack, unpack
import math
from math import sin, exp
import sys
import os
import shortuuid
import numpy as np
from scipy.stats import lognorm

verbose = 2

# how much do we shift the exponentials in the sin */+ exp fitness
# functions
shift = 87500.3

# don't change these functions here; change them in main()
global initial_func, mate_func, fit_func, entropy_func
initial_func = mate_func = fit_func = entropy_func = None

## parameters of the run
# set run_seed to an int to force a seed (and get a reproducible run),
# or None to have a different random sequence eacth time
random_seed = 123456
n_pop = 4000
assert(n_pop % 2 == 0)
# define an "oscillation scale" which is the typical distance between
# the sin() peaks (i.e. the period!) this is then used to define
# "mutation rate" and entropy bin sizes for drift mutation
oscillation_scale = 80*math.pi
mutation_rate = 0.03
mutation_rate_drift = oscillation_scale / 2.0
lognormal_mean = 0.1
lognormal_sigma = 3.0
num_parents_mating = n_pop // 2
assert(num_parents_mating % 2 == 0)
num_generations = 1400
# add to read

def main():
    global initial_func, mate_func, fit_func, entropy_func
    # Change these functions here
    initial_func = make_initial_rnd_0 # make_initial_all_85000, make_initial_random, ...
    mate_func = mate_drift_lognormal # mate_bitflip, mate_drift, mate_drift_lognormal
    fit_func = fit_flattop_on_gaussian # fit_sin_on_gaussian, fit_flattop_on_gaussian, 
    entropy_func = calc_entropy_drift # calc_entropy_bits, calc_entropy_drift, ...

    print_fit_func_for_plot(f'fit-func_pid{os.getpid()}.out')
    # NOTE: you can force the seed to get reproducible runs.
    # comment the random.seed() call to get different runs each time.
    gen_fname = get_gen_info_fname()
    print(f'# output going to file {gen_fname}')
    print_metadata(gen_fname, gen_fname)
    random.seed(random_seed)
    np.random.seed(random_seed)
    population = make_initial_pop(n_pop)
    # data = np.empty(10000)
    for gen in range(num_generations):
        new_pop = advance_one_generation(gen, population)
        population = new_pop
    return


def make_histogram(df, num_generations):
    plt.figure()
    for i in range(num_generations):
        data = df.iloc[i, 7]
        plt.hist(data, density=False, bins=40)
        plt.ylabel('no. of individuals with this occupancy')
        plt.xlabel('occupancy')

def fit_vs_dude(df):
    df[["max_dude", "max_fit"]].plot(x="max_dude")

def advance_one_generation(gen, pop):
    """Calculates fitness for each person and return them in a list."""
    fit_list = calc_pop_fitness(pop)
    pop, fit_list = sort_pop_by_fitness(pop, fit_list)
    max_index = np.argmax(fit_list) # best dude's index
    max_dude = pop[max_index] # best dude
    max_fit = fit_list[max_index] # best dude's fitness
    avg_fit = np.mean(fit_list)
    elite_pop = sorted(fit_list, reverse=True)
    elite_pop = elite_pop[:len(elite_pop) // 2]
    elite_avg_fit = np.mean(elite_pop)
    entropy, occupancy = calc_entropy(gen, elite_pop)
    print_pop_stats(gen, pop, fit_list)
    if verbose >= 1:
        dump_pop(gen, pop, fit_list)
    # occupancy_dataframe.append({"occupancy" : sorted(occupancy, reverse=True)[:20]})
    # Selecting the best parents in the population for mating.
    new_pop = select_pool(pop, fit_list, num_parents_mating)
    return new_pop

def make_pop_stats_line(gen, pop, fit_list):
    """Print useful bits of info about the current population list."""
    # print("fitness_list:", fit_list)
    # The best result in the current iteration.
    max_index = np.argmax(fit_list) # best dude's index
    max_dude = pop[max_index] # best dude
    max_fit = fit_list[max_index]        # best dude's fitness
    avg_fit = np.mean(fit_list)
    elite_pop = sorted(fit_list, reverse=True)
    elite_pop = elite_pop[:len(elite_pop) // 2]
    elite_avg_fit = np.mean(elite_pop)
    entropy, occupancy = calc_entropy(gen, elite_pop)
    # print(fit_list)
    # print("best_result:", max_index, max_dude, max_fit)
    line_to_print = (f'max_dude_fit:   {gen}   {max_index}   {max_dude:20.26g}'
                     + f'   {float_to_bin(max_dude)}   {max_fit:20.26g}'
                     + f'   {elite_avg_fit}   {avg_fit}    {entropy}    {sorted(occupancy, reverse=True)[:20]}')
    return line_to_print

def print_pop_stats(gen, pop, fit_list):
    line_to_print = make_pop_stats_line(gen, pop, fit_list)
    if verbose >= 1:
        print(line_to_print)
        sys.stdout.flush()
    with open(get_gen_info_fname(), 'a') as f:
        f.write(line_to_print + '\n')
        f.flush()


def dump_pop(gen, pop, fit_list):
    return
    print_pop_stats(gen, pop, fit_list)
    with open(get_gen_info_fname(), 'a') as f:
        f.write(line_to_print + '\n')
        f.flush()

def get_gen_info_fname():
    return f'GA-gen-info_pid{os.getpid()}.out'

def calc_pop_fitness(pop):
    """Calculates the fitnesses of each member of a population and returns
    a list with all of them.  This list has the order of the members
    of the population."""
    fit_list = []
    for dude in pop:
        dude_fitness = calc_fitness(dude)
        fit_list.append(dude_fitness)
    return np.array(fit_list)

def calc_fitness(x):
    """evaluate the polynomial at the given point - for now that's our
    fitness function.  sometimes we also multiply it by a wide
    gaussian envelope to avoid fitness values that are too extreme
    """
    if not math.isfinite(x):
        return -sys.float_info.max
    # return fit_sin_on_gaussian(x)
    # return fit_flattop_on_gaussian(x)
    return fit_func(x)

def fit_sin_on_gaussian(x):
    """A fitness function that looks like a sin() riding on top of a
    gaussian."""
    # fit = math.sin((x-400)/20) * (1 + 10*math.exp(-(x-400)**2/100000.0))
    xp = x - shift
    fit = (sin(2*math.pi*xp/oscillation_scale) * exp(-xp**2/20000)
           + 100*exp(-xp**2/1000000)
           + 2*sin(2*math.pi*xp/oscillation_scale))
    if not math.isfinite(fit):
        return -sys.float_info.max
    return fit

def fit_flattop_on_gaussian(x):
    """A flat top fitness function that looks like a square wave riding on
    top of a gaussian.  This should allow for drifting at constant
    fitness and more interesting fitness plateaus."""
    # first adjust for the shift
    xp = x - shift
    # then adjust for keeping a flat region
    xm = xp - xp % (2*oscillation_scale)
    exp_base = 300*np.exp(-xm**2/2000000.0)
    flattop_max = exp_base + 10*np.exp(-xm**2/2000000.0)
    flattop_min = exp_base - 10*np.exp(-xm**2/2000000.0)
    # fit = 100*exp(-xp**2/1000000) + 2*sin(2*math.pi*xp/oscillation_oscillation_scale)
    # now try to figure out a square wave-ish thing that rides on top
    # of the gaussian
    if xp % oscillation_scale < oscillation_scale/2:
        fit = flattop_min
    else:
        fit = flattop_max
    if not math.isfinite(fit):
        return -sys.float_info.max
    return fit
    

def select_pool(pop, fit_list, num_parents_mating):
    """Take the population, pick out the top half of the parents, and
    mate them to form a population of children.  Then pool together the
    elite half of the parents together with the children to form the
    new population, and return that new population."""
    #find the indices
    sorted_parent_indices = np.argsort(fit_list)
    top_half_parent_indices = sorted_parent_indices[-num_parents_mating:]
    top_half_parents = pop[tuple([top_half_parent_indices])]
    child_pop = []
    child_pop = []
    for i in range(num_parents_mating // 2):
        p1 = top_half_parents[2*i]
        p2 = top_half_parents[2*i + 1]
        ## for now we have this set up so you can choose your mating
        ## function to be mate_bitflip() or mate_drift()
        # c1, c2 = mate_bitflip(p1, p2)
        # c1, c2 = mate_drift(p1, p2)
        c1, c2 = mate_func(p1, p2)
        child_pop.append(c1)
        child_pop.append(c2)
    new_pop = np.concatenate((child_pop, top_half_parents))
    return new_pop

def mate_bitflip(p1, p2):
    """Mate two parents and get two children; mutate the children by
    flipping bits in the ieee binary representation of their floating
    point values."""
    c1, c2 = crossover(p1, p2)
    if random.random() < mutation_rate:  # a small % of the time make a bit flip
        placeholder_1 = c1 + 0
        placeholder_2 = c2 + 0
        pos_1 = random.randint(0, 31)
        c1 = bitflip(placeholder_1, pos_1)
        pos_2 = random.randint(0, 31)
        c2 = bitflip(placeholder_2, pos_2)
    return c1, c2

def calc_genetic_distance(p1, p2):
    a = float_to_bin(p1)
    b = float_to_bin(p2)
    array_p1 = bytearray(a, 'utf-8')
    array_p2 = bytearray(b, 'utf-8')
    genetic_distance = np.sum(np.bitwise_xor(array_p1,array_p2))
    return genetic_distance


def calc_entropy(gen, pop):
    entropy, occupancy = entropy_func(gen, pop)
    return entropy, occupancy


def calc_entropy_bits(gen, pop):
    """Calculate the shannon entropy for this population by applying the
    Shannon formula using occupancy of each state in the population."""
    # used_pop tracks the population members that we actually use.
    # this excludes all NAN values.
    n_pop_finite = 0
    n_pop_bad = 0
    pop_unique = list(set(pop))
    # now collapse all NANs (numbers that satisfy math.isnan(x)) into
    # the single math.nan value.  this way the pop_unique.index() call
    # will work correctly
    pop_unique = [x for x in pop_unique if math.isfinite(x)]
    n_species = len(pop_unique)
    occupancy = [0]*n_species
    bad_strings = {}
    # our approach to calculating entropy is to form an "occupancy"
    # histogram: how many population members are in each possible bit
    # configuration.  Then we use Shannon's -Sum(p_i*log(p_i))
    # formula, where p_i is the occupancy fraction for each state.
    for member in pop:
        # find the index of this member in the list of species.  we
        # use a special treatment for NANs, since comparing them is
        # not well defined (there are several values that are NAN).
        # to understand this try the following:
        # In [22]: x = bin_to_float('01111111110000000000000000000000')
        # In [23]: y = bin_to_float('01111111110000000000010000000000')
        # In [24]: x == y
        # Out[24]: False
        # In [25]: math.isnan(x)
        # Out[25]: True
        # In [26]: math.isnan(y)
        # Out[26]: True
        # print('ABOUT:', member, pop_unique)
        # if math.isnan(member):
        #     occ_index = pop_unique.index(math.nan)
        # else:
        #     occ_index = pop_unique.index(member)
        if math.isfinite(member):
            occ_index = pop_unique.index(member)
            occupancy[occ_index] += 1
            n_pop_finite += 1
        else:
            n_pop_bad += 1
        
    shannon_entropy = occupancy_list2entropy(occupancy, pop)
    return shannon_entropy, occupancy

def calc_entropy_drift(gen, pop):
    """Entropy calculation that is appropriate for drift mutations.  The
    main difference is that instead of assuming unique population
    members, we instead put the population into bins -- i.e. we define
    the species to be linearly spaced bins in the population's range
    of values.  Then we use occupancy levels in those bins to
    calculate entropy."""
    n_bins = 2*len(pop)
    pop_spread = 200*1000
    pmin, pmax = shift - pop_spread/2, shift + pop_spread/2
    this_center = pmin + (pmax - pmin)/2.0
    bin_width = pop_spread / n_bins
    occupancy = [0]*n_bins
    occupancy[0] = count_pop_in_bin(pop, -sys.float_info.max, pmin + bin_width)
    occupancy[n_bins-1] = count_pop_in_bin(pop, pmax - bin_width, sys.float_info.max)
    for i in range(1, n_bins-1):
        x_left_edge = pmin + i * bin_width
        x_right_edge = pmin + (i+1) * bin_width
        occupancy[i] = count_pop_in_bin(pop, x_left_edge, x_right_edge)
    entropy = occupancy_list2entropy(occupancy, pop)
    return entropy, occupancy


def calc_entropy_drift_adapt(gen, pop):
    """Entropy calculation that is appropriate for drift mutations.  The
    main difference is that instead of assuming unique population
    members, we instead put the population into bins -- i.e. we define
    the species to be linearly spaced bins in the population's range
    of values.  Then we use occupancy levels in those bins to
    calculate entropy."""
    n_bins = 2*len(pop)
    pmin = np.min(pop)
    pmax = np.max(pop)
    this_pop_spread = pmax - pmin
    this_center = pmin + (pmax - pmin)/2.0
    canonical_pop_spread = 60 * oscillation_scale
    canonical_pmin = this_center - canonical_pop_spread / 2.0
    canonical_pmax = this_center + canonical_pop_spread / 2.0
    if verbose >= 2:
        print(f'#POP_SPREAD: {pmin}   {pmax}   {this_center}   {this_pop_spread}'
              + f'   {canonical_pop_spread}   {oscillation_scale}'
              + f'   {this_pop_spread / oscillation_scale}')
    bin_width = canonical_pop_spread / n_bins
    occupancy = [0]*n_bins
    occupancy[0] = count_pop_in_bin(pop, -sys.float_info.max, canonical_pmin + bin_width)
    occupancy[n_bins-1] = count_pop_in_bin(pop, canonical_pmax - bin_width, sys.float_info.max)
    for i in range(1, n_bins-1):
        x_left_edge = canonical_pmin + i * bin_width
        x_right_edge = canonical_pmin + (i+1) * bin_width
        occupancy[i] = count_pop_in_bin(pop, x_left_edge, x_right_edge)
    entropy = occupancy_list2entropy(occupancy, pop)
    return entropy, occupancy


def occupancy_list2entropy(occ_list, pop):
    """Underlying formula for entropy: this is the Shannon formula that
    takes occupancy probability and divides by the log thereof."""
    shannon_entropy = 0
    for i in range(len(occ_list)):
        # FIXME: I should study if I should divide by n_pop or
        # n_pop_finite.  i.e. do I use fraction of full population, or
        # fraction of finite population?
        if occ_list[i] != 0:
            prob = occ_list[i] / len(pop)
            individual_surprise = -prob * math.log(prob)
            shannon_entropy += individual_surprise
    # # now handle the bad numbers, those that are not finite.  we
    # # pretend that they are all different from each other (see
    # # Tolstoy), and add n_pop_bad * (-(1/n_pop) * log(1/n_pop))
    # prob = 1.0 / n_pop
    # individual_bad_surprise = -prob * math.log(prob)
    # shannon_entropy += n_pop_bad * individual_bad_surprise

    return shannon_entropy


def count_pop_in_bin(pop, left, right):
    total = 0
    for member in pop:
        if member >= left and member < right:
            total += 1
    return total


def mate_drift(p1, p2):
    """Mate two parents and get two children; do mutations by drifting the
    floating point value."""
    c1 = (p1 + p2) / 2.0
    c1 += (random.random() - 0.5) * mutation_rate_drift
    c2 = (p1 + p2) / 2.0
    c2 += (random.random() - 0.5) * mutation_rate_drift
    if random.random() < 0.005:   # small % of the time make shift much bigger
        c1 += (random.random() - 0.5) * 10*mutation_rate_drift
        c2 += (random.random() - 0.5) * 10*mutation_rate_drift
    return c1, c2

def mate_drift_lognormal(p1, p2):
    direction = 1 if random.random() < 0.5 else -1
    c1 = (p1 + p2) / 2.0
    c2 = (p1 + p2) / 2.0
    if random.random() < mutation_rate:
        drift1 = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_sigma)
        c1 += drift1
        if verbose >= 2:
            print(f'#drift1: {drift1} -> {c1}')
    if random.random() < mutation_rate:
        drift2 = np.random.lognormal(mean=lognormal_mean, sigma=lognormal_sigma)
        c2 += drift2
        if verbose >= 2:
            print(f'#drift2: {drift2} -> {c2}')
    return c1, c2


def bitflip(x, pos):
    fs = pack('f',x)
    bval = list(unpack('BBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew=unpack('f',fs)
    return fnew[0]

def float_to_bin(num):
    """Given a float, return a string with the individual bits"""
    result = format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')
    return result

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

def crossover(p1, p2):
    """Breed two parents into two children with crossover.  Organisms are
    floats, and we can think of them as sequences of 32 bits in ieee
    format."""
    crossover_point = random.randint(0, 31)
    p1bits = list(float_to_bin(p1))
    p2bits = list(float_to_bin(p2))
    c1bits = [0]*32
    c2bits = [0]*32
    for k in range(0, crossover_point):
        c1bits[k] = p1bits[k]
        c2bits[k] = p2bits[k]
    for k in range(crossover_point, 32):
        c1bits[k] = p2bits[k]
        c2bits[k] = p1bits[k]
        
    c1 = bin_to_float(''.join(c1bits))
    c2 = bin_to_float(''.join(c2bits))
    return c1, c2

def mutation(offspring_crossover, num_mutations=1):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        #        The random value to be added to the gene.
        mutation_index = np.random.randint(0,6)
        random_value = np.random.uniform(-1.0, 1.0, 1)
        while (offspring_crossover[idx, mutation_index] + random_value) > 4 or (offspring_crossover[idx, mutation_index] + random_value) < -4:
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, mutation_index] = offspring_crossover[idx, mutation_index] + random_value
        return offspring_crossover

def print_pop(gen, pop):
    print(f'# population at generation {gen}')
    # print(pop)
    for i, member in enumerate(pop):
        bit_str = float_to_bin(member)
        print(i, bit_str, '%20.12f' % member)

def run_tests():
    # test the bitflip routine on a given number
    x = 2.4
    y = bitflip(x, 12)
    print(x, y)
    print(float_to_bin(x))
    print(float_to_bin(y))
    xbits = list(float_to_bin(x))
    print('bitlist:', xbits)
    xstr = ''.join(xbits)
    print('from_bitlist_to_str:', xstr)
    x_after_processing = bin_to_float(xstr)
    print('from_str_to_float:', x_after_processing)

def print_metadata(fname, gen_fname):
    """Print some information about the run using the "low effort
    metadata" approach, described at
    https://programmingforresearch.wordpress.com/2020/06/07/low-effort-metadata-lem/
    """
    from datetime import datetime
    my_date = datetime.now()
    dt_str = my_date.isoformat()
    meta_str = (f"""##COMMENT: Genetic algorithm run
##COMMAND_LINE: {sys.argv.__str__()}
##RUN_DATETIME: {dt_str}
##OUTPUT_FILENAME: {gen_fname}
##VERBOSE: {verbose}
##SHIFT: {shift}
##random_seed: {random_seed}
##MATE_FUNCTION: {mate_func.__name__}
##INITIAL_FUNCTION: {initial_func.__name__}
##FITNESS_FUNCTION: {fit_func.__name__}
##ENTROPY_FUNCTION: {entropy_func.__name__}
##MUTATION_RATE: {mutation_rate}
##MUTATION_RATE_DRIFT: {mutation_rate_drift}
##LOGNORMAL_MEAN: {lognormal_mean}
##LOGNORMAL_SIGMA: {lognormal_sigma}
##OSCILLATION_SCALE: {oscillation_scale}
##N_POP: {n_pop}
##N_GENERATIONS: {num_generations}
##COLUMN0: constant string max_dude_fit:
##COLUMN1: generation number
##COLUMN2: index of max fitness
##COLUMN3: fittest member
##COLUMN4: fittest member bitstring
##COLUMN5: highest fitness
##COLUMN6: elite average fitness
##COLUMN7: average fitness
##COLUMN8: population entropy
""")
    if len(fname) == 0:         # no file: print to stdout
        sys.stdout.write(meta_str)
        sys.stdout.flush()
    else:
        with open(fname, 'a') as f:
            f.write(meta_str)

def print_fit_func_for_plot(fname):
    x = shift - 5000
    print_metadata(fname, fname)
    with open(fname, 'a') as f:
        while x <= shift + 5000:
            f.write(f'fit_plot:   {x}   {calc_fitness(x)}\n')
            x += 0.5
    print(f'# wrote function info to {fname}')


def sort_pop_by_fitness(pop, fit_list):
    """Use the fitness list to sort the population."""
    # zipped_lists = zip(fit_list, pop) # put fit_list first
    # sorted_zip_lists = sorted(zipped_lists)
    # sorted_pop = [element for _, element in sorted_zip_lists]
    # sorted_fit_list = sorted(fit_list)
    # # print(sorted_pop)
    # return sorted_pop, sorted_fit_list
    fit_list_inds_sorted = fit_list.argsort()
    fit_list_sorted = fit_list[tuple([fit_list_inds_sorted])]
    pop_sorted = pop[tuple([fit_list_inds_sorted])]
    return pop_sorted, fit_list_sorted

def make_initial_pop(n_pop):
    # population = np.zeros(n_pop)
    # for i in range(n_pop):
    #     bitlist = [str(random.randint(0,1)) for i in range(32)]
    #     bitstr = ''.join(bitlist)
    #     x = bin_to_float(bitstr)
    #     print(bitstr, x)
    #     population[i] = x
    # population = np.random.uniform(low=70000.0, high=85000.0, size=n_pop)
    # population = np.random.uniform(low=80000.0, high=80000.0, size=n_pop)
    # population = np.random.uniform(low=-10000002,
    #                                high=-10000001,
    #                                size=n_pop)

    # population = np.full(n_pop, -1000.0)
    # population = np.full(n_pop, 85000.0)
    # population = np.random.uniform(low=-200000.0000000001, 
    #                                # high=100000.0000000002,
    #                                high=-100000.0000000002,
    #                                size=n_pop)
    population = initial_func(n_pop)
    return population

def make_initial_rnd_85k(n_pop):
    pop = np.random.uniform(low=80000.0, high=90000.0, size=n_pop)
    return pop

def make_initial_rnd_80k(n_pop):
    pop = np.random.uniform(low=70000.0, high=90000.0, size=n_pop)
    return pop

def make_initial_rnd_0(n_pop):
    pop = np.random.uniform(low=-10000.0, high=10000.0, size=n_pop)
    return pop

def make_initial_all_zero(n_pop):
    pop = np.full(n_pop, 0.0);
    return pop

def make_initial_all_80k(n_pop):
    pop = np.full(n_pop, 80000.0);
    return pop

def make_initial_all_85k(n_pop):
    pop = np.full(n_pop, 85000.0);
    return pop

def make_initial_all_minus1k(n_pop):
    pop = np.full(n_pop, -1000.0);
    return pop

# run_tests()
if __name__ == '__main__':
    main()
