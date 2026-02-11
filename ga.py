import random
from typing import List, Dict, Tuple, Callable

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
POPULATION_SIZE = 100
MAX_GENERATION = 1000
CROSSOVER_RATE = 0.9
MUTATE_RATE = 0.05
TOURNAMENT_SIZE = 10


def encode_chrarr(pheno):
    return list(pheno)

def decode_chrarr(geno):
    return "".join(geno)

def generate_chromosome(chromosome_length: int, alphabet: str) -> List[str]:
    return [random.choice(alphabet) for _ in range(chromosome_length)]

def fitness(chromosome: List[str], target: List[str]) -> float:
    score = 0
    for gene, target_gene in zip(chromosome, target):
        score += abs(ord(gene) - ord(target_gene))
    return 1 / (1 + score)


def sort_population(population: List[List[str]], target: List[str], fitness: Callable) -> List[List[str]]:
    return sorted(population, key=lambda chromosome: fitness(chromosome, target))

def tournament(population: List[List[str]], target: List[str], sample_size: int, fitness: Callable):
    sub_population = random.sample(population, k=sample_size)
    sub_population = sort_population(sub_population, target, fitness)
    return sub_population[-1], sub_population[-2]

def recombine(chromosome1: List[str], chromosome2: List[str], cross_over_index: int, crossover_rand: float, crossover_rate: float) -> Tuple[List[str], List[str]]:
    if crossover_rand >= crossover_rate:
        return chromosome1, chromosome2
    child1 = chromosome1[:cross_over_index] + chromosome2[cross_over_index:]
    child2 = chromosome2[:cross_over_index] + chromosome1[cross_over_index:]  # crossover 2 list of strings
    return child1, child2


def mutate(chromosome: List[str], mutate_rand: float, mutate_rate: float) -> List[str]:
    index = random.randint(0, len(chromosome) - 1)
    if mutate_rand < mutate_rate:
        chromosome[index] = random.choice(ALPHABET)
    return chromosome

def breed(parent1, parent2, crossover_rate, mutate_rate):
    child1, child2 = recombine(parent1, parent2, random.randint(0, len(parent1) - 1), random.uniform(0, 1), crossover_rate)
    child1 = mutate(child1, random.uniform(0, 1), mutate_rate)
    child2 = mutate(child2, random.uniform(0, 1), mutate_rate)
    return child1, child2


def describe_chromosome(chromosome: List[str], target: List[str], fitness: Callable) -> Dict:
    return {"genotype": chromosome, "fitness": fitness(chromosome, target), "phenotype": decode_chrarr(chromosome)}


def genetic_algorithm(target_str: str):

    target: List[str] = encode_chrarr(target_str) # encode
    population: List[List[str]] = [generate_chromosome(len(target), ALPHABET) for _ in range(POPULATION_SIZE)]

    for generation in range(MAX_GENERATION):
        next_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = tournament(population, target, TOURNAMENT_SIZE, fitness)
            if parent1 == target:
                return describe_chromosome(parent1, target, fitness)
            if parent2 == target:
                return describe_chromosome(parent2, target, fitness)

            child1, child2 = breed(parent1, parent2, CROSSOVER_RATE, MUTATE_RATE)
            next_population.extend([child1, child2])

        population = next_population
        if generation % 10 == 0:
            chrom = describe_chromosome(sort_population(population, target, fitness)[-1], target, fitness)
            print(f"{generation:5} {chrom}")

    solution = describe_chromosome(sort_population(population, target, fitness)[-1], target, fitness)
    return solution


target_string = "this is so much fun"
result = genetic_algorithm(target_string)
print(result)
