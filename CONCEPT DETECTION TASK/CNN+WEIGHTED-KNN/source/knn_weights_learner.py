import numpy as np
from tqdm import tqdm
import utilities as utils

class GeneticWeightsSearch:
    def __init__(self, k, threshold, train_data, val_data, tags, ids, sorted_sims):
        self.k = k
        self.threshold = threshold
        self.train_data = train_data
        self.val_data = val_data
        self.tags = tags
        self.sorted_sims = sorted_sims
        self.ids = ids

    def calculate_fitness(self, weights):
        predictions = dict()
        img_subset = list(self.val_data)[:100]
        for img in img_subset:
            top_k = self.sorted_sims[img][-self.k:]
            top_k = np.flip(top_k)  # largest to smallest
            
            f = np.zeros(shape=len(self.tags))
            for i, nn_index in enumerate(top_k):
                nn_tags = self.train_data[self.ids[nn_index]].split(';')
                presence = np.array([int(t in nn_tags) for t in self.tags])

                f = f + (weights[i] * presence)
            
            f = f / sum(weights)

            tags_list = list()
            for i in np.argwhere(f >= self.threshold).flatten():
                tags_list.append(self.tags[i])
            
            predictions[img] = ';'.join(tags_list)
        
        return utils.evaluate_f1({k: self.val_data[k] for k in img_subset}, 
                                 predictions, test=False)  # F1 score over val data
    
    def initialize_population(self, population_size, values_range=(0., 1.)):
        population = list()
        for _ in range(population_size):
            individual = np.random.uniform(low=values_range[0], high=values_range[1],
                                           size=self.k)
            individual = np.sort(individual)[::-1]  # descending weights in [0, 1].
            individual /= individual.sum()  # sum to 1.
            population.append(individual)
        return np.array(population)
    
    def mutate(self, input_weights, mutation_rate):
        mutated = np.copy(input_weights)

        # for i in range(len(input_weights)):
        #     if np.random.rand() < mutation_rate:
        #         # Generate a small random change
        #         mutation_value = np.random.uniform(-0.1, 0.1)
        #         mutated[i] += mutation_value

        #         # Ensure the mutated value remains within [0, 1]
        #         mutated[i] = max(0, min(1, mutated[i]))
        
        mutation_values = np.random.uniform(-0.1, 0.1, size=len(input_weights))
        mutation_mask = np.random.rand(len(input_weights)) < mutation_rate
    
        # Apply mutations
        mutated += mutation_values * mutation_mask
    
        # Ensure mutated values remain within [0, 1]
        mutated = np.clip(mutated, 0, 1)
        
        # Ensure monotonicity constraint
        for i in range(1, len(input_weights)):
            mutated[i] = min(mutated[i], mutated[i-1])
        
        mutated /= mutated.sum()

        return mutated
    
    def crossover(self, parent1, parent2):
        m = len(parent1)
        crossover_point = np.random.randint(1, m)
    
        # Create children chromosomes
        child1 = np.zeros(m)
        child2 = np.zeros(m)

        # for i in range(m):
        #     if i < crossover_point:
        #         child1[i] = parent1[i]
        #         child2[i] = parent2[i]
        #     else:
        #         child1[i] = parent2[i]
        #         child2[i] = parent1[i]

        child1[:crossover_point] = parent1[:crossover_point]
        child1[crossover_point:] = parent2[crossover_point:]

        child2[:crossover_point] = parent2[:crossover_point]
        child2[crossover_point:] = parent1[crossover_point:]
        
        # for i in range(1, m):
        #     child1[i] = min(child1[i], child1[i-1])
        #     child2[i] = min(child2[i], child2[i-1])

        child1 = np.minimum(np.maximum.accumulate(child1[::-1])[::-1], child1)
        child2 = np.minimum(np.maximum.accumulate(child2[::-1])[::-1], child2)
        
        child1 /= child1.sum()
        child2 /= child1.sum()

        return child1, child2

    def optimize_weights(self, population_size, num_generations, crossover_rate, mutation_rate):
        population = self.initialize_population(population_size, )

        for _ in tqdm(range(num_generations)):
            population_values = [self.calculate_fitness(individual) for individual in population]
            # print(population_values)
            fitness_proportions = population_values / np.sum(population_values)
            # print(fitness_proportions)
            parents_indices = np.random.choice(np.arange(population_size), replace=True,
                                               size=(population_size // 2, 2), 
                                               p=fitness_proportions)
            
            new_population = list()

            for idx1, idx2 in parents_indices:
                if np.random.rand() < crossover_rate:
                    offspring1, offspring2 = self.crossover(population[idx1], population[idx2])
                else:
                    offspring1, offspring2 = population[idx1], population[idx2]
                offspring1 = self.mutate(offspring1, mutation_rate, )
                offspring2 = self.mutate(offspring2, mutation_rate, )
                new_population.extend([offspring1, offspring2])
            
            # population_values = [self.calculate_fitness(individual) for individual in new_population]
            # population_with_values = list(zip(population, population_values))
            # sorted_population = sorted(population_with_values, key=lambda x: x[1], reverse=True)  # descending.
            # population = [ind for ind, _ in sorted_population[:population_size]]
            population = new_population[:population_size]
        
        
        best_solution = max(population, key=lambda x: self.calculate_fitness(x))
        best_value = self.calculate_fitness(best_solution)
    
        return best_solution, best_value

        




