#! /usr/bin/env python3

##  g e n e r a l   f l o w
#
#population
#v
#fitness calculation
#v
#mating pool
#v
#parents selection
#v
#mating - crossover
#       - mutation
#v
#offspring
#v
#>->->->->->->->->->->-> population


#######################
#
#goal is to maximize this
#    y = a1x1+a2x2+a3x3+a4x4+a5x5+a6x6
#   (x1,x2,x3,x4,x5,x6)=(2.8,-4.2,3.8,5.6,-11,-6.8) (arbitrary values)
#    What are the best values for a1-a6
#
#######################

import numpy
import random

######################


################################################################
# Inputs
poly_coefs = [2.8, 1, 1, -3.6, 3.8, 1.6, -0.3]

# Number of the polynomial coefs we are looking to optimize.
n_poly_coefs = len(poly_coefs)

####################
#
#mating pool size and population size

n_pop = 8                       # number of chromosomes, or number of population members
assert(n_pop % 2 == 0)
num_parents_mating = n_pop // 2
assert(num_parents_mating % 2 == 0)

###############################

best_outputs = []
num_generations = 1000

def main():
    new_population = numpy.random.uniform(low=-1000.0, high=1000.0, 
                                          size=n_pop)
    print(new_population)
    for generation in range(num_generations):
        print("Generation : ", generation)
        # Calculates fitness for each person and put them in a list of
        # fitnesses
        fit_list = []
        for dude in new_population:
            dude_fitness = calc_fitness(dude)
            fit_list.append(dude_fitness)
        print("Fitness")
        print(fit_list)
        
        # best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
        # The best result in the current iteration.
        max_index = numpy.argmax(fit_list)
        max_dude = new_population[max_index]
        max_fit = fit_list[max_index]
        print("Best result : ", max_index, max_dude, max_fit)

        # Selecting the best parents in the population for mating.
        new_population = select_pool(new_population, fit_list,
                                     num_parents_mating)
        # print("Parents")
        # print(parents)

        # # Generating next generation using crossover.
        # offspring_crossover = crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        # print("Crossover")
        # print(offspring_crossover)

        # # Adding variations to the offspring using mutation.
        # offspring_mutation = mutation(offspring_crossover, num_mutations=2)
        # print("Mutation")
        # print(offspring_mutation)

        # # new population based on the parents and offspring.
        # new_population[0:parents.shape[0], :] = parents
        # new_population[parents.shape[0]:, :] = offspring_mutation

        # # Getting the best solution after iterating all generations.
        # #fitness is calculated for each solution in the final generation.
        # fitness = cal_fitness(equation_inputs, new_population)
        # # index of solution corresponding to the best fitness.
        # best_match_idx = numpy.where(fitness == numpy.max(fitness))

        # print("Best solution : ", new_population[best_match_idx, :])
        # print("Best solution fitness : ", fitness[best_match_idx])
        print('==== new_pop ====')
        print(new_population)
        print('max_ind_fit:', max_dude, '   ', max_fit)



def calc_fitness(x):
    #Fitness value of each solution
    #Sum of products between each input and corresponding weight.
    # fitness = numpy.sum(pop*equation_inputs, axis=1)
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
    sorted_parent_indices = numpy.argsort(fit_list)
    top_half_parent_indices = sorted_parent_indices[-num_parents_mating:]
    top_half_parents = pop[[top_half_parent_indices]]
    print('top_half_parents:', top_half_parents)
    child_pop = []
    for i in range(num_parents_mating // 2):
        print('i:', i, num_parents_mating)
        p1 = top_half_parents[2*i]
        p2 = top_half_parents[2*i + 1]
        c1, c2 = mate(p1, p2)
        child_pop.append(c1)
        child_pop.append(c2)
    new_pop = numpy.concatenate((child_pop, top_half_parents))
    return new_pop

def mate(p1, p2):
    """Mate two parents and get two children."""
    c1 = (p1 + p2) / 2.0
    c1 += (random.random() - 0.5) * 0.1
    c2 = (p1 + p2) / 2.0
    c2 += (random.random() - 0.5) * 0.1
    if random.random() < 0.1:   # 10% of the time make a big shift
        c1 += (random.random() - 0.5) * 10.0
        c2 += (random.random() - 0.5) * 10.0
    return c1, c2


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
 # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        mutation_index = numpy.random.randint(0,6)
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        while (offspring_crossover[idx, mutation_index] + random_value) > 4 or (offspring_crossover[idx, mutation_index] + random_value) < -4:
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, mutation_index] = offspring_crossover[idx, mutation_index] + random_value
    return offspring_crossover


main()

###############################
# Iteration vs. Fitness

# import matplotlib.pyplot
# matplotlib.pyplot.plot(best_outputs)
# matplotlib.pyplot.xlabel("Iteration")
# matplotlib.pyplot.ylabel("Fitness")
# matplotlib.pyplot.show()
