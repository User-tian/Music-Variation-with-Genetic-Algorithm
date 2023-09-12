from random import choices, randint, randrange, random, sample
from typing import List, Optional, Callable, Tuple
from copy import deepcopy
from utils import int_from_bits
Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]


def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)

def edit_genome(score_genome: Genome, length: int, num_mutations) -> Genome:
    score_genome_cpy = deepcopy(score_genome)
    for _ in range(num_mutations):
        i = randint(0, length - 1)
        score_genome_cpy[i]  = score_genome_cpy[i] ^ 1
    return score_genome_cpy



def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def measure_level_crossover(a: Genome, b: Genome, num_bars: int, num_notes: int, BITS_PER_NOTE: int):
    p = randint(0, num_bars - 1)
    return a[0:p * num_notes * BITS_PER_NOTE] + b[p * num_notes * BITS_PER_NOTE:], b[0:p * num_notes * BITS_PER_NOTE] + a[p * num_notes * BITS_PER_NOTE:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome

def intelligent_mutation(genome: Genome, num: int = 1, probability: float = 0.5, num_bars: int = 4, BITS_PER_NOTE: int = 5) -> Genome:
    lst_of_printers = []
    lst_of_going_to_change_notes = []
    p = 1 
    if p == 0:# down an octave
        for i in range(len(genome) / BITS_PER_NOTE / num_bars - 1):
            note_now = 0
            note_after = 0
            for a in range(BITS_PER_NOTE):
                note_now += genome[i * BITS_PER_NOTE + a] * (2 ** a)
                note_after += genome[(i + 1) * BITS_PER_NOTE + a] * (2 ** a)
            if abs(note_after - note_now) > 9:
                    random_degree = randint(-3,3)
                    possibility = random()
                    if possibility <= probability:
                        note_after = note_now + random_degree
                        bits = int_from_bits(note_after)
                        genome[(i + 1) * BITS_PER_NOTE: (i + 2) * BITS_PER_NOTE] = bits

    elif p == 1: # repeat one of the bars
        possibility = random()
        if possibility <= probability:
            n = randint(0,num_bars - 2)
            tmp_lst = genome[int(len(genome) / num_bars * n) : int(len(genome) / num_bars * (n + 1))]
            genome[int(len(genome) / num_bars * (n + 1)) : int(len(genome) / num_bars * (n + 2))] = tmp_lst
    return genome
def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return sample(
        population=generate_weighted_distribution(population, fitness_func),
        k=2
    )


def generate_weighted_distribution(population: Population, fitness_func: FitnessFunc) -> Population:
    result = []

    for gene in population:
        result += [gene] * int(fitness_func(gene)+1)

    return result


def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    population = populate_func()

    i = 0
    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i
