#!/usr/bin/env python3

import numpy as np
import random
import struct
from struct import pack, unpack

import math
from sys import exit
import sys

#

# Inputs
poly_coefs = [2.8, 1, 1, -3.6, 3.8, 1.6, -0.3]

# Number of the polynomial coefs we are looking to optimize.
n_poly_coefs = len(poly_coefs)


# mating pool size and population size

# number of chromosomes, or number of population members
n_pop = 100
assert(n_pop % 2 == 0)
num_parents_mating = n_pop // 2
assert(num_parents_mating % 2 == 0)

best_outputs = []
num_generations = 1000

def main():
    population = np.random.uniform(low=-100000.0, high=100000.0, size=n_pop)
    # print_pop(0, population)
    print_metadata()
    for gen in range(num_generations):
        new_pop = advance_one_generation(gen, population)
        population = new_pop


def advance_one_generation(gen, pop):
    """Calculates fitness for each person and return them in a list."""
    fit_list = calc_pop_fitness(pop)
    print_pop_stats(gen, pop, fit_list)
    # Selecting the best parents in the population for mating.
    new_pop = select_pool(pop, fit_list, num_parents_mating)
    return new_pop

def print_pop_stats(gen, pop, fit_list):
    """Print useful bits of info about the current population list."""
    # print("fitness_list:", fit_list)
    # The best result in the current iteration.
    max_index = np.argmax(fit_list) # best dude's index
    max_dude = pop[max_index] # best dude
    max_fit = fit_list[max_index]        # best dude's fitness
    avg_fit = np.mean(fit_list)
    elite_avg_fit = np.mean(fit_list[:len(fit_list)//2])
    # print(fit_list)
    # print("best_result:", max_index, max_dude, max_fit)
    print(f'max_dude_fit:   {max_index}   {max_dude}   {float_to_bin(max_dude)}'
          + f'   {max_fit}   {elite_avg_fit}   {avg_fit}')


def calc_pop_fitness(pop):
    """Calculates the fitnesses of each member of a population and returns
    a list with all of them."""
    fit_list = []
    for dude in pop:
        dude_fitness = calc_fitness(dude)
        if math.isnan(dude_fitness):
            dude_fitness = -sys.float_info.max
        fit_list.append(dude_fitness)
    return fit_list


def calc_fitness(x):
    """evaluate the polynomial at the given point - for now that's our
    fitness function"""
    fit = 0
    for i in range(n_poly_coefs):
        fit += poly_coefs[i] * x**i
    return fit


def select_pool(pop, fit_list, num_parents_mating):
    """Take the population, pick out the top half of the parents, and
    mate them to form a population of children.  Then pool together the
    elite half of the parents together with the children to form the
    new population, and return that new population."""
    #find the indices
    sorted_parent_indices = np.argsort(fit_list)
    top_half_parent_indices = sorted_parent_indices[-num_parents_mating:]
    top_half_parents = pop[[top_half_parent_indices]]
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
    if random.random() < 0.02:   # 10% of the time make a big shift
        placeholder_1 = c1 + 0
        placeholder_2 = c2 + 0
        pos_1 = random.randint(0, 31)
        c1 = bitflip(placeholder_1, pos_1)
        pos_2 = random.randint(0, 31)
        c2 = bitflip(placeholder_2, pos_2)
    return c1, c2


def mate_drift(p1, p2):
    """Mate two parents and get two children; do mutations by drifting the
    floating point value."""
    c1 = (p1 + p2) / 2.0
    c1 += (random.random() - 0.5) * 0.1
    c2 = (p1 + p2) / 2.0
    c2 += (random.random() - 0.5) * 0.1
    if random.random() < 0.1:   # 10% of the time make a big shift
        c1 += (random.random() - 0.5) * 100.0
        c2 += (random.random() - 0.5) * 100.0
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
    print(pop)

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

def print_metadata():
    """Print some information about the run using the "low effort
    metadata" approach."""
    from datetime import datetime
    my_date = datetime.now()
    dt_str = my_date.isoformat()
    print(f"""##COMMENT: Genetic algorithm run
##COMMAND_LINE: {sys.argv.__str__()}
##RUN_DATETIME: {dt_str}
##POLY_COEFS: {poly_coefs.__str__()}
##N_POP: {n_pop}
##N_GENERATIONS: {num_generations}
##COLUMN0: constant string max_dude_fit:
##COLUMN1: index of max fitness
##COLUMN2: fittest member
##COLUMN3: fittest member bitstring
##COLUMN4: highest fitness
##COLUMN5: elite average fitness
##COLUMN6: average fitness""")


# run_tests()
if __name__ == '__main__':
    main()
