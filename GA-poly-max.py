
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
from sys import exit
import sys
import os
import shortuuid


import matplotlib.pyplot as plt

# how much do we shift the exponentials in the sin */+ exp fitness
# functions
shift = 87500.3

#creates unique name for the output file
random_name = shortuuid.ShortUUID().random(length=5)
#creates said file
#open("GA-gen-info_pid{}.out".format(random_name), "x")

# polynomial coefficients.  it has to be even degree (i.e. list has to
# have odd length), and last (most significant) coefficient has to be
# negative so that it has a peak
poly_coefs = [2.8, 1, 1, -3.6, 3.8, 1.6, -0.3]

# Number of the polynomial coefs we are looking to optimize.
n_poly_coefs = len(poly_coefs)


#datarame
header = {
    "gen" : [],
    "max_index" : [],
    "max_dude" : [],
    "float_max_dude" : [],
    "elite_avg_fit" : [],
    "avg_fit" : [],
    "entropy" : [],
    "occupancy" : []
}

#df = pd.DataFrame(data = info, index=num_generations, columns=["max_index", "max_dude", "float_max_dude", "elite_avg_fit", "avg_fit", "entropy", "occupancy"] )

       #''' (f'max_dude_fit:   {gen}   {max_index}   {max_dude:20.26g}'+ f'   {float_to_bin(max_dude)}   {max_fit:20.26g}' + f'   {elite_avg_fit}   {avg_fit}    {entropy}    {sorted(occupancy, reverse=True)[:20]}')'''
         

#


## parameters of the run
# set run_seed to an int to force a seed (and get a reproducible run),
# or None to have a different random sequence eacth time
random_seed = 123456
n_pop = 1000
assert(n_pop % 2 == 0)
mutation_rate = 0.14
num_parents_mating = n_pop // 2
assert(num_parents_mating % 2 == 0)
num_generations = 1000
global data_template
data_template= []



def main():
#    print_poly_for_plot(f'fit-func_pid{os.getpid()}.out')
    # NOTE: you can force the seed to get reproducible runs.  comment
    # the random.seed() call to get different runs each time.
    random.seed(random_seed)
    np.random.seed(random_seed)
    population = make_initial_pop(n_pop)
    #print_pop(0, population)
    #print_pop(0, population)
    #print_metadata(gen_fname)
    data = np.empty(10000)
    for gen in range(num_generations):
        new_pop = advance_one_generation(gen, population)
        population = new_pop
    df = pd.DataFrame(data_template, index=range(num_generations))    
    df.to_csv(r'c:\GA-project\data_{}.txt'.format(random_name), header=None, index=None, sep='\t', mode='a')
    print('c:\GA-project\data_{}.txt'.format(random_name))
        #df = (data = data_template, index=num_generations, columns=["max_fit", "max_index", "max_dude", "float_max_dude", "elite_avg_fit", "avg_fit", "entropy", "occupancy"])
        
    # plt.title(Genetic Algorithm Developement)
    # ax1.set_ylabel("Fitness")
    # ax2.set_ylabel("Entropy")
    # plt.autoscale(enable=True, axis='both', tight=False)
    # plt.plot(genfname: 1, 2)
    # #print(f'# wrote fitness plot data to fit_func_pid{os.getpid()}.out')
    #print(f'# wrote generation information to {get_gen_info_fname()}')
    #print('# you could make plots with:')
    # for interactive in (True, False):
    #     cmd = ("gnuplot"
    #            + (f""" -e 'set terminal pdf' """
    #               + f""" -e 'set output "{gen_fname}.pdf"' """
    #               if not interactive else "")
    #            + f" -e 'set grid'"
    #            # + f" -e 'set multiplot layout 2, 1'"
    #            + f""" -e 'set title "EvolutionPID{os.getpid()}"' """
    #            # + f""" -e 'set logscale y' """
    #            + f""" -e 'set ytics autofreq' """
    #            + f""" -e 'set ylabel "fitness"' """
    #            + f""" -e 'set y2tics autofreq' """
    #            + f""" -e 'set y2label "entropy"' """
    #            + f''' -e "plot '{gen_fname}' using 2:6 with lines lw 2 axes x1y1 title 'top fitness' '''
    #            + f'''     , '{gen_fname}' using 2:7 with lines lw 2 axes x1y2 title 'elite avg fitness' '''
    #            + f'''     , '{gen_fname}' using 2:9 with lines lw 2 axes x1y2 title 'entropy' '''
    #            + f'"'
    #            + (f""" -e "pause -1" """ if interactive else "")
    #            )
    #     print(f'# {("NON" if not interactive else "")} interactive')
    #     print(cmd)
    #     os.system(cmd + ' &')
    # os.system(f'evince {gen_fname}.pdf &')
    # print('# and you can plot the function with:')
    # func_fname = f'fit-func_pid{os.getpid()}.out'
    # for interactive in (True, False):
    #     cmd = ("gnuplot"
    #            + (f""" -e 'set terminal pdf' """
    #               + f""" -e 'set output "{func_fname}.pdf"' """
    #               if not interactive else "")
    #            + f" -e 'set grid'"
    #            + f""" -e 'set title "FunctionPID{os.getpid()}"' """
    #            # + f""" -e 'set logscale y' """
    #            # + f""" -e 'set ytics autofreq' """
    #            + f""" -e 'set xlabel "x"' """
    #            + f""" -e 'set ylabel "fitness(x)"' """
    #            # + f""" -e 'set y2tics autofreq' """
    #            # + f""" -e 'set y2label "entropy"' """
    #            + f''' -e "plot '{func_fname}' using 2:3 with lines lw 2 title 'fitness'" '''
    #            + (f""" -e "pause -1" """ if interactive else "")
    #            )
    #     print(f'# {("NON" if not interactive else "")} interactive')
    #     print(cmd)
    # os.system(cmd + ' &')
    # os.system(f'evince {func_fname}.pdf &')


    

def advance_one_generation(gen, pop):
    """Calculates fitness for each person and return them in a list."""
    fit_list = calc_pop_fitness(pop)
    # print('================ BEFORE ================')
    # print(fit_list)
    # print(pop)
    pop, fit_list = sort_pop_by_fitness(pop, fit_list)
    # print('================ AFTER ================')
    # print(fit_list)
    # print(pop)
    max_index = np.argmax(fit_list) # best dude's index
    max_dude = pop[max_index] # best dude
    max_fit = fit_list[max_index]        # best dude's fitness
    avg_fit = np.mean(fit_list)
    elite_pop = sorted(fit_list, reverse=True)
    elite_pop = elite_pop[:len(elite_pop) // 2]
    elite_avg_fit = np.mean(elite_pop)
    entropy, occupancy = calc_entropy(gen, pop)
    # global data_template
    data_template.append({"max_fit" : max_fit, "max_dude" : max_index, "float_max_dude" : float_to_bin(max_dude), "elite_avg_fit" : elite_avg_fit, "avg_fit" : avg_fit, "entropy" : entropy, "occupancy" : sorted(occupancy, reverse=True)[:20]})
    # Selecting the best parents in the population for mating.
    new_pop = select_pool(pop, fit_list, num_parents_mating)
    return new_pop

def print_pop_stats(gen, pop, fit_list):
    """Print useful bits of info about the current population list."""
    # print("fitness_list:", fit_list)
    # The best result in the current iteration.
    # max_index = np.argmax(fit_list) # best dude's index
    # max_dude = pop[max_index] # best dude
    # max_fit = fit_list[max_index]        # best dude's fitness
    # avg_fit = np.mean(fit_list)
    # elite_pop = sorted(fit_list, reverse=True)
    # elite_pop = elite_pop[:len(elite_pop) // 2]
    # elite_avg_fit = np.mean(elite_pop)
    # entropy, occupancy = calc_entropy(gen, pop)
    # # global data_template
    # # data_template= []
    # # data_template.append({"max_fit" : max_fit, "max_dude" : max_index, "float_max_dude" : float_to_bin(max_dude), "elite_avg_fit" : elite_avg_fit, "avg_fit" : avg_fit, "entropy" : entropy, "occupancy" : sorted(occupancy, reverse=True)[:20]})
    # #data_template.append({"max_fit" : max_fit})
    #data_template.append({"max_dude" : max_index})
    #data_template.append({"float_max_dude" : float_to_bin(max_dude)})
    #data_template.append({"elite_avg_fit" : elite_avg_fit})
    #data_template.append({"avg_fit" : avg_fit})
    #data_template.append({"entropy" : entropy})
    #data_template.append({"occupancy" : sorted(occupancy, reverse=True)[:20]})
    #print(data_template)
    
    # print(fit_list)
    # print("best_result:", max_index, max_dude, max_fit)
    #line_to_print = (f'max_dude_fit:   {gen}   {max_index}   {max_dude:20.26g}'
     #                + f'   {float_to_bin(max_dude)}   {max_fit:20.26g}'
     #                + f'   {elite_avg_fit}   {avg_fit}    {entropy}    {sorted(occupancy, reverse=True)[:20]}')
#    print(line_to_print + '   ')
    #with open(get_gen_info_fname(), 'a') as f:
     #   f.write(line_to_print + '\n')

    
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
    # fit = 0
    # for i in range(n_poly_coefs):
    #     fit += poly_coefs[i] * x**i
    # # now multiply by a gaussian envelope
    # # fit = fit * math.exp(-x**2 / 100.0)
    # fit = math.cos(x-40)+10*math.exp(-(x-40)**2/5000)
    if not math.isfinite(x):
        return -sys.float_info.max
    # fit = math.sin((x-400)/20) * (1 + 10*math.exp(-(x-400)**2/100000.0))
    xp = x - shift
    fit = sin(xp/20) * exp(-xp**2/20000) + 100*exp(-xp**2/1000000) + 2*sin(xp/20)
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
    # print('top_half_parents:', top_half_parents)
    child_pop = []
    # for i in range(num_parents_mating // 2):
    #     #print('i:', i, num_parents_mating)
    #     print('top_half_parents:', top_half_parents)
    #     print(' ')
    child_pop = []
    for i in range(num_parents_mating // 2):
#        print('i:', i, num_parents_mating)
        p1 = top_half_parents[2*i]
        p2 = top_half_parents[2*i + 1]
        ## for now we have this set up so you can choose your mating
        ## function to be mate_bitflip() or mate_drift()
        c1, c2 = mate_bitflip(p1, p2)
        # c1, c2 = mate_drift(p1, p2)
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
    """Calculate the shannon entropy for this population by applying the
    Shannon formula using occupancy of each state in the population."""
    # used_pop tracks the population members that we actually use.
    # this excludes all NAN values.
    n_pop_finite = 0
    n_pop_bad = 0
    # print('pop:')
    # print_pop(gen, pop)
    pop_unique = list(set(pop))
    # now collapse all NANs (numbers that satisfy math.isnan(x)) into
    # the single math.nan value.  this way the pop_unique.index() call
    # will work correctly
    # print('pop_unique:')
    # print_pop(gen, pop_unique)
    pop_unique = [x for x in pop_unique if math.isfinite(x)]
    # print('pop_unique:')
    # print_pop(gen, pop_unique)
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
            # occ_index = pop_unique.index(member)
        if math.isfinite(member):
            occ_index = pop_unique.index(member)
            occupancy[occ_index] += 1
            n_pop_finite += 1
        else:
            n_pop_bad += 1
        
    shannon_entropy = 0
    for i in range(len(occupancy)):
        # FIXME: I should study if I should divide by n_pop or
        # n_pop_finite.  i.e. do I use fraction of full population, or
        # fraction of finite population?
        prob = occupancy[i] / n_pop
        individual_surprise = -prob * math.log(prob)
        shannon_entropy += individual_surprise
    # now handle the bad numbers, those that are not finite.  we
    # pretend that they are all different from each other (see
    # Tolstoy), and add n_pop_bad * (-(1/n_pop) * log(1/n_pop))
    prob = 1.0 / n_pop
    individual_bad_surprise = -prob * math.log(prob)
    shannon_entropy += n_pop_bad * individual_bad_surprise

    # print_pop(gen, pop)
    # # print(pop)
    return shannon_entropy, occupancy


def mate_drift(p1, p2):
    """Mate two parents and get two children; do mutations by drifting the
    floating point value."""
    c1 = (p1 + p2) / 2.0
    c1 += (random.random() - 0.5) * mutation_rate
    c2 = (p1 + p2) / 2.0
    # c2 += (random.random() - 0.5) * mutation_rate
    c2 += np.random.lognormal(mutation_rate * 100)
    if random.random() < 0.1:   # 10% of the time make shift 100 times bigger
        c1 += (random.random() - 0.5) * 100.0 * mutation_rate
        c2 += (random.random() - 0.5) * 100.0 * mutation_rate
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


def make_plots(best_outputs):
    # Iteration vs. Fitness
    import matplotlib.pyplot
    matplotlib.pyplot.plot(best_outputs)
    matplotlib.pyplot.xlabel("Iteration")
    matplotlib.pyplot.ylabel("Fitness")
    matplotlib.pyplot.show()

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

def print_metadata(fname):
    """Print some information about the run using the "low effort
    metadata" approach, described at
    https://programmingforresearch.wordpress.com/2020/06/07/low-effort-metadata-lem/
    """
    from datetime import datetime
    my_date = datetime.now()
    dt_str = my_date.isoformat()
    with open(fname, 'a') as f:
        f.write(f"""##COMMENT: Genetic algorithm run
##COMMAND_LINE: {sys.argv.__str__()}
##RUN_DATETIME: {dt_str}
##SHIFT: {shift}
##POLY_COEFS: {poly_coefs.__str__()}
##random_seed: {random_seed}
##MUTATION_RATE: {mutation_rate}
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
##COLUMN8: population entroypy
""")

def print_poly_for_plot(fname):
    x = shift - 1000
    with open(fname, 'w') as f:
        while x <= shift + 1000:
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
    population = np.random.uniform(low=-1000.0, high=1000.0, size=n_pop)
    # population = np.random.uniform(low=-10000002,
    #                                high=-10000001,
    #                                size=n_pop)

    # population = np.full(n_pop, -1000.0)
    # population = np.random.uniform(low=-200000.0000000001, 
    #                                # high=100000.0000000002,
    #                                high=-100000.0000000002,
    #                                size=n_pop)
    return population


# run_tests()
if __name__ == '__main__':
    main()    
# plt.plot(outputs)
# plt.xlabel('gen')
# plt.ylabel('fit')
# plt.show()


"""
insert an analysis function that dumps out some more
statistical quantities
define a stability difference, the extent to which
genetic difference leads to fitness difference
track lineage and make phylogenetic trees
because we know everything we can save everything to
a file and hopefully have enough information to track
evolution
we picked random coefficients for our polynomial, one
of the results of which is that all the action happens
pretty close to zero but if you plot a small polynomial
you don't have a clear insight about where the zeroes
are, but if you plotted something like:
(x-1000)*(x+20000)*(x+1e6)*(x+1e8)
you have to have a very large range [-1e8:1e8] and then
you can see all the zero crossings and all the interesting
regions will be in drastically varying places
for genetic distance it might be prudent to differentiate
between the mantissa, signficand, and sign (two dudes with
a different sign will be more distant from each other than
two with the last bit flipped)
"""
