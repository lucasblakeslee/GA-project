#! /user/bin/env python3

import numpy as np
import random
import struct
from struct import pack, unpack

import math
from sys import exit

#

# Inputs
poly_coefs = [2.8, 1, 1, -3.6, 3.8, 1.6, -0.3]

# Number of the polynomial coefs we are looking to optimize.
n_poly_coefs = len(poly_coefs)


#

#mating pool size and population size

n_pop = 8                       # number of chromosomes, or number of population members
assert(n_pop % 2 == 0)
num_parents_mating = n_pop // 2
assert(num_parents_mating % 2 == 0)

best_outputs = []
num_generations = 1000

def main():
    new_population = np.random.uniform(low=-1000.0, high=1000.0, size=n_pop)
    for generation in range(num_generations):
        print("Generation: ", generation)
        # Calculates fitness for each person and put them in a list of
        # fitnesses
        fit_list = []
        for dude in new_population:
            dude_fitness = calc_fitness(dude)
            fit_list.append(dude_fitness)
        print("Fitness:")
        print(fit_list)
        """this needs to be refined, right now it's only checking whether the first item in the fit list is a NaN -- I tried to use the all() function but I got an error saying bools aren't iterable"""
        if (math.isnan(fit_list[1]) == True):
            print("something has gone horribly wrong due to hubris and wanton negligence (the program is returning NaNs)")
            exit()
        else:
            pass
        # The best result in the current iteration.
        max_index = np.argmax(fit_list)
        max_dude = new_population[max_index]
        max_fit = fit_list[max_index]
        print(fit_list)
        print("Best result : ", max_index, max_dude, max_fit)
        print()
        # Selecting the best parents in the population for mating.
        new_population = select_pool(new_population, fit_list, num_parents_mating)
        print('==== new_pop ====')
        print(new_population)
        print()
        print('max_ind_fit:', max_dude, '  ', max_fit)
        print("Best result: ", max_index, max_dude, max_fit)

        # Selecting the best parents in the population for mating.
        new_population = select_pool(new_population, fit_list,
                                     num_parents_mating)

   


#

def calc_fitness(x):
    #Sum of products between each input and corresponding weight.
    # return fitness
    # evaluate the polynomial at the given point
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
    print()
    print('top_half_parents:', top_half_parents)
    print()
    child_pop = []
    for i in range(num_parents_mating // 2):
        #print('i:', i, num_parents_mating)
        print('top_half_parents:', top_half_parents)
        print(' ')
    child_pop = []
    for i in range(num_parents_mating // 2):
#        print('i:', i, num_parents_mating)
        p1 = top_half_parents[2*i]
        p2 = top_half_parents[2*i + 1]
        c1, c2 = mate(p1, p2)
        child_pop.append(c1)
        child_pop.append(c2)
    new_pop = np.concatenate((child_pop, top_half_parents))
    return new_pop

def mate(p1, p2):
    """Mate two parents and get two children."""
    c1 = (p1 + p2) / 2.0
    c1 += (random.random() - 0.5) * 0.1
    c2 = (p1 + p2) / 2.0
    c2 += (random.random() - 0.5) * 0.1
    if random.random() < 0.1:   # 10% of the time make a big shift
        placeholder_1 = c1 + 0
        placeholder_2 = c2 + 0
        pos_1 = random.randint(1, 31)
        c1 = bitflip(placeholder_1, pos_1)
        pos_2 = random.randint(1, 31)
        c2 = bitflip(placeholder_2, pos_2)
    return c1, c2

def bitflip(x,pos):
    fs = pack('f',x)
    bval = list(unpack('BBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew=unpack('f',fs)
    return fnew[0]

# def float_to_bin(num):
#     return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

# def bin_to_float(binary):
#     return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]



"""
====================HAS NO IMPACT, NEEDS TO BE INCORPORATED===========================

def crossover(parents, offspring_size):
   offspring = np.empty(offspring_size)
   crossover_point = np.uint8(offspring_size[1]/2)

   for k in range(offspring_size[0]):
#        Index of the first parent to mate.
       parent1_idx = k%parents.shape[0]
#        Index of the second parent to mate.
       parent2_idx = (k+1)%parents.shape[0]
#        The new offspring will have its first half of its genes taken from the first parent.
       offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
#        The new offspring will have its second half of its genes taken from the second parent.
       offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
       return offspring

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
"""
main()

###############################
# Iteration vs. Fitness

#import matplotlib.pyplot
#matplotlib.pyplot.plot(best_outputs)
#matplotlib.pyplot.xlabel("Iteration")
#matplotlib.pyplot.ylabel("Fitness")
#matplotlib.pyplot.show()

#>>>>>>> 3745daad59442e2f2eab895314a63d73f9a6b525
