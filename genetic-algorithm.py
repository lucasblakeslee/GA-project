#! /usr/bin/env python3

"""

goal is to maximize this
    y = a1x1+a2x2+a3x3+a4x4+a5x5+a6x6
    (x1,x2,x3,x4,x5,x6)=(2.8,-4.2,3.8,5.6,-11,-6.8) (arbitrary values)
    What are the best values for a1-a6

"""

import numpy

# Inputs
equation_inputs = [2.8,-4.2,3.8,5.6,-11,-6.8]

# Number of the weights we are looking to optimize.
num_weights = len(equation_inputs)

#mating pool size and population size

sol_per_pop = 8
num_parents_mating = 4

pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosomes, chromosomes have num_weights genes.
#Initial population.
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)

###############################

best_outputs = []
num_generations = 1000

def main():
    for generation in range(num_generations):
        print("Generation : ", generation)
        # Calculates fitness
        fitness = cal_fitness(equation_inputs, new_population)
        print("Fitness")
        print(fitness)

        best_outputs.append(numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))
        # The best result in the current iteration.
        print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

        # Selecting the best parents in the population for mating.
        parents = select_pool(new_population, fitness, 
                                      num_parents_mating)
        print("Parents")
        print(parents)

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))
        print("Crossover")
        print(offspring_crossover)

        # Adding variations to the offspring using mutation.
        offspring_mutation = mutation(offspring_crossover, num_mutations=2)
        print("Mutation")
        print(offspring_mutation)

        # new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        # Getting the best solution after iterating all generations.
        #fitness is calculated for each solution in the final generation.
        fitness = cal_fitness(equation_inputs, new_population)
        # index of solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))

        print("Best solution : ", new_population[best_match_idx, :])
        print("Best solution fitness : ", fitness[best_match_idx])



def cal_fitness(equation_inputs, pop):
    #Fitness value of each solution
    #Sum of products between each input & corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

def select_pool(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

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

import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
