import time
import music21 as m21
from typing import List, Dict
from random import sample, shuffle
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from genetic import generate_genome, Genome, intelligent_mutation, measure_level_crossover, single_point_crossover, mutation, edit_genome
from utils import append_chord_and_information_to_score, chord_measure, initialize_genome_with_pause_pattern, rhythm_measure, pitches_measure, expand_to_uniform, selection_pair, int_from_bits,Scale, decode_info, rhythm_measure_nocombine, pitches_measure_nocombine
from evaluation import bonus_for_overall_smoothness, bonus_for_phrase_clique, compute_chord_fitness, penalty_for_detune, penalty_for_pauses, similarity_rate_pitch_fitness, penalty_for_smoothness, print_similarity, print_score, variation_similarity_fitness, ending_dolmisol

import argparse

def bits_from_int(integer: int) -> list:
    str_bin = bin(integer)
    Lst = []
    for i in reversed(str_bin):
        if i != 'b':
            Lst.append(int(i))
        else:
            break
    if len(Lst) < BITS_PER_NOTE:
        for i in range(BITS_PER_NOTE - len(Lst)):
            Lst.append(0)
    return Lst

def sort_by_pitch(stream):
    '''
    utility function to calculate pitches for different octaves

    Args:
    stream: m21 stream object that type(stream[idx]) == m21.measure.Measure

    Returns:

    my_dict_sorted: calculating notes in octaves. The result may look like this: {5: 23, 4: 14}
    '''
    lst = []
    for measure in stream.getElementsByClass('Measure'):
        for note in measure.getElementsByClass('Note'):
            lst.append(note.octave)
    my_dict = {}
    # 统计直方图相当于是
    for i in lst:
        if i in my_dict:
            my_dict[i] += 1
        else:
            my_dict[i] = 1
    my_dict_sorted = sorted(my_dict.items(), key = lambda x : x[1], reverse = True)
    return my_dict_sorted

def genome_to_score(genome: Genome, num_bars: int, num_notes: int, num_steps: int,pauses: int, key: str, scale: str, root: int):
    '''
    utility function to transform a genome to m21 score

    Args:
    genome: current genome generated. size (num_bars * num_notes * BITS_PER_NOTE)
    num_of_bars:
    num_notes:
    num_steps:
    pauses:
    key:
    scale:
    root:

    

    Returns:
    stream: m21 stream object that type(stream[idx]) == m21.measure.Measure
    '''

    # 每BITS_PER_NOTE的二进制组合在一起形成一个数
    notes = [genome[i * BITS_PER_NOTE:i * BITS_PER_NOTE + BITS_PER_NOTE] for i in range(num_bars * num_notes)]
    # 最小的量化单位
    note_length = 4 / float(num_notes)
    
    # 用Utils中的Scale Clas进行对调内音初始化
    scl = Scale(key = key, modality = scale, base_octave = root, expand = pow(2, BITS_PER_NOTE - 1) // 7)
    # 通过对genome的解析形成下面的字典
    melody = {
        "notes": [],
        # "velocity": [],
        "beat": []
    }
    # 获取可产生的pitch列表
    scl_pitchlist = scl.pitch_list()

    for note in notes:
        # 十进制转二进制
        integer = int_from_bits(note)

        
        lucky_number = pow(2, BITS_PER_NOTE) - 1
        # -1 represent pause(typical value)
        if integer == lucky_number:
            # if we have -1 before in the dict melody, than we increase the length of the previous pause instead of adding another pause
            if len(melody["notes"]) > 0 and melody["notes"][-1] == -1:
                melody["beat"][-1] += note_length
            else:
                melody["notes"] += [-1]
                # melody["velocity"] += [0]
                melody["beat"] += [note_length]
        else:
            # use mod to make sure the idx afterwards can be recognized as the idx of scl.pitch_list
            pitch = scl_pitchlist[(integer) % len(scl_pitchlist)]
            # same note appeared consecutively 2 times, then only one note will be played and the duration doubled
            if len(melody["notes"]) > 0 and melody["notes"][-1] == pitch:
                melody["beat"][-1] += note_length
            else:
                melody["notes"] += [pitch]
                # melody["velocity"] += [127] 
                melody["beat"] += [note_length]

    stream = m21.stream.Stream()
    measure = m21.stream.Measure(0)
    measure.append(m21.meter.TimeSignature('4/4'))
    counter = 0
    Flag = 0
    for idx, note in enumerate(melody["notes"]):
        if note == -1:
            tmp = m21.note.Rest()
        else:
            tmp = m21.note.Note()
            tmp.pitch.midi = note

        if counter < 4 and counter + melody["beat"][idx] > 4:
            remain = counter + melody["beat"][idx] - 4
            now = melody["beat"][idx] - remain
            tmp.duration.quarterLength = now
            counter = remain
            measure.append(tmp)
            stream.append(measure)
            measure = m21.stream.Measure(Flag + 1)
            Flag += 1
            tmp1 = m21.note.Note()
            tmp1.pitch.midi = note
            tmp1.duration.quarterLength = remain
            measure.append(tmp1)
        elif counter  < 4 and counter + melody["beat"][idx] <= 4:
            tmp.duration.quarterLength = melody["beat"][idx]
            counter  = counter + melody["beat"][idx]
            measure.append(tmp)

         # flag indicates which measure we are at.
        if counter == 4 and Flag < num_bars - 1:
            stream.append(measure)
            measure = m21.stream.Measure(Flag + 1)
            Flag += 1
            counter = 0
        elif counter == 4 and Flag == num_bars - 1:
            stream.append(measure)
            counter = 0
            break
    return stream

def score_to_genome(score, num_bars: int, num_notes: int, num_steps: int,pauses: int, key: str, scale: str, root: int, chord_progression = None):
    '''
    utility function to transform m21 score to genome

    Args:
    score: m21 score
    num_of_bars:
    num_notes:
    num_steps:
    pauses:
    key:
    scale:
    root:

    Returns:
    genome: size (num_bars * num_notes * BITS_PER_NOTE)
    '''
    # step1: 用rhythm和pitch把score变成两个列表
    rhythm_score = []
    pitches_score = []
    for measure in score.getElementsByClass('Measure'):
        rhythm_Measure = rhythm_measure(measure)
        pitches_Measure = pitches_measure(measure)
        rhythm_expanded_measure, pitches_expanded_measure = expand_to_uniform(rhythm_Measure, pitches_Measure)
        # rhythm_score += rhythm_expanded_measure
        pitches_score += pitches_expanded_measure
    # step2: transform according to the scl information
    scl = Scale(key = key, modality = scale, base_octave = root, expand = pow(2, BITS_PER_NOTE - 1) // 7)
    scl_pitchlist = scl.pitch_list()
    integers = []
    genome = []
    lucky_number = pow(2, BITS_PER_NOTE) - 1
    for element in pitches_score:
        if element == -1:
            integers.append(lucky_number)
        else:
            integer = scl_pitchlist.index(element) # interger 的范围应该是两个八度的0～13
            integers.append(integer)
    # end of step2: got ints
    # step3: binary sequence
    for element in integers:
        genome += bits_from_int(element)
    return genome

def fitness(genome: Genome, score_obs, num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int, bpm: int, threshold, chord_progression = None, structural_information = None):
    '''
    fitness function to evaluate the quality of the generated measures

    Args:
    genome: current genome generated. size (num_bars * num_notes * BITS_PER_NOTE)
    num_of_bars:
    num_notes:
    num_steps:
    pauses:
    key:
    scale:
    root:
    bpm: int
    threshold: the maximum similarity of two scores, as a hyper-parameter
    chord_progression: the chord progression given as a hyper-parameter

    Returns:
    fitness score: contains different bonuses and penalties.
    '''
    # transform genome to score to impose evaluation on scores
    score_syn = genome_to_score(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)
    similarity_score = 0

 
    similarity_score = variation_similarity_fitness(score_syn, score_obs)

    
    # calculation of penalty if the melody is not smooth
    penalty_smoothness = penalty_for_smoothness(score_syn)

    # calculation of penalty if the melody has a lot of pauses
    penalty_pause = penalty_for_pauses(score_syn, pauses)

    bonus_smoothness = bonus_for_overall_smoothness(score_syn)

    bonus_clique = bonus_for_phrase_clique(score_syn, num_notes)
    # # calculation of penalty if the melody has a lot of notes off chord.(detune)
    penalty_chord = 0
    for idx, measure in enumerate(score_obs.getElementsByClass('Measure')):
        if chord_progression == None:
            rhythm, chord = chord_measure(measure)

            if chord != []: 
                rhythm_expand, chord_expand = expand_to_uniform(rhythm, chord)
            else:
                rhythm_expand = rhythm
                chord_expand = chord
        else:
            # here imposes the constraint for the type of chord progression
            assert type(chord_progression) == list or type(chord_progression) == tuple, "the type of chord_progression ought to be tuple or list, with idx=0 for rhythm(offset) and idx=1 for chord(m21.harmony.chordsymbol)"
            if chord_progression[idx][0] != [] and chord_progression[idx][1] != []:
                rhythm_expand, chord_expand = expand_to_uniform(chord_progression[idx][0], chord_progression[idx][1])# chord_progression 希望是一个二维的列表，第一维代表小节数，第二维代表rhythm or chord
                penalty_chord += penalty_for_detune(measure, rhythm_expand, chord_expand)
    
    # ending_note_requirements
    ending_score = 0
    [do_rate, re_rate, mi_rate, fa_rate, sol_rate, la_rate, ti_rate] = ending_dolmisol(score_syn, key, scale, root)
    if structural_information["dst"][1] == 1:
        ending_score += la_rate + re_rate
    elif structural_information["dst"][1] == 2:
        ending_score += mi_rate + sol_rate + la_rate
    elif structural_information["dst"][1] == 3:
        ending_score += re_rate + fa_rate
    elif structural_information["dst"][1] == 4:
        ending_score += do_rate

    weight = 10
    # the total score can be combined with the weighted average of the fitness above 
    return weight * (0.5 * similarity_score - 0.3 * penalty_smoothness - 0.3 * penalty_chord + 0 * bonus_clique + 0 * bonus_smoothness + 0.2 * ending_score)

def main(stream, num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int, population_size: int, num_mutations: int, mutation_probability: float, bpm: int, threshold: float, chord_progression = None, pause_template = None, pitchrhythm_template = None, structural_information = None):

    '''
    the main function to achieve the generative task of generating variation with genetic algorithm

    Args:
    stream: the input score
    num_bars: int
    num_notes: int
    num_steps: int
    pauses: bool
    key: str
    scale: str
    root: int
    population_size: int
    num_mutations: int
    mutation_probability: float
    bpm: int
    threshold: the maximum similarity of two scores, as a hyper-parameter
    chord_progression: the chord progression given as a hyper-parameter. Default to be None
    structural_information: the structural information of the input score. dict type, containing src's information and dest's information
    Returns:
    No return
    '''
    start_time = time.time()
    origin_genome = score_to_genome(stream, num_bars, num_notes, num_steps, pauses, key, scale, root)
    population = [edit_genome(origin_genome, num_bars * num_notes * BITS_PER_NOTE, num_mutations) for _ in range(population_size)]
    population_id = 0
    # write an origin for comparison
    origin = genome_to_score(population[0], num_bars, num_notes, num_steps, pauses, key, scale, root)
    origin = append_chord_and_information_to_score(origin, chord_progression, key, scale, bpm)
    origin.write('musicxml', 'origin.musicxml')
    for i in tqdm(range(num_generations)):
        shuffle(population)
        # calculate the fitness of each genome 
        population_fitness = [[genome, fitness(genome, stream, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm, threshold, chord_progression, structural_information)] for genome in population]
        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)

        sum_exp_fitness = sum(np.exp(x[1]) for x in population_fitness)
        fitness_list = []
        for lst in sorted_population_fitness:
            lst[1] = np.exp(lst[1]) / sum_exp_fitness
            fitness_list.append(lst[1])
        
        population = [e[0] for e in sorted_population_fitness]

        # elites
        next_generation = population[0:3]

        for j in range(int(len(population) / 2) - 1):
            # the process of genetic algorithm
            parents = selection_pair(population, fitness_list)
            offspring_a, offspring_b = measure_level_crossover(parents[0], parents[1], num_bars, num_notes, BITS_PER_NOTE)
            offspring_a = mutation(offspring_a, num=num_mutations, probability=mutation_probability)
            offspring_b = mutation(offspring_b, num=num_mutations, probability=mutation_probability)
            # offspring_a = intelligent_mutation(offspring_a, num=num_mutations, probability=mutation_probability, num_bars= num_bars, BITS_PER_NOTE = BITS_PER_NOTE)
            # offspring_b = intelligent_mutation(offspring_b, num=num_mutations, probability=mutation_probability, num_bars= num_bars, BITS_PER_NOTE = BITS_PER_NOTE)
            next_generation += [offspring_a, offspring_b]
        population = next_generation
        population_id += 1
    # the best score
    end_time = time.time()
    # generate the top 3 scores
    for j in range(4):
        score = genome_to_score(population[j], num_bars, num_notes, num_steps, pauses, key, scale, root)

        print_similarity(score, stream, num_bars, chord_progression)
        print_score(score, num_notes)
        score = append_chord_and_information_to_score(score, chord_progression, key, scale, bpm)
        score.write('musicxml', f'results/best{j}.musicxml')

    print("origin:")
    print_similarity(stream, stream, num_bars, chord_progression)
    print_score(stream, num_notes)
    print(f'total time: {end_time - start_time}s')


def pitch_change(stream, key, scale, root):
    scl = Scale(key = key, modality = scale, base_octave = root, expand = pow(2, BITS_PER_NOTE - 1) // 7)
    scl_lst = scl.pitch_list()
    measure_choice = sample(range(len(stream.getElementsByClass('Measure'))) , 1)
    measure_select = stream.getElementsByClass('Measure')[measure_choice[0]]
    pitch_choice = sample(range(len(measure_select.getElementsByClass('Note'))), 1)
    pitch_select = measure_select.getElementsByClass('Note')[pitch_choice[0]]
    pitch_select.pitch.midi = scl_lst[sample(range(len(scl_lst)), 1)[0]]
    new_measure = deepcopy(measure_select)

    new_stream_variation = m21.stream.Part()
    for i in range(len(stream.getElementsByClass('Measure'))):
        if i == measure_choice[0]:
            new_stream_variation.append(new_measure)
        else:
            new_stream_variation.append(stream.getElementsByClass('Measure')[i])
    return new_stream_variation

def just_pitch_change(stream, num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int, population_size: int, num_mutations: int, mutation_probability: float, bpm: int, threshold: float, chord_progression = None, pause_template = None, pitchrhythm_template = None, structural_information = None):
    start_time = time.time()
    excellent_lst = []
    fitness_lst = [-100]
    stream_variation = deepcopy(stream)
    population = [pitch_change(stream_variation, key, scale, root) for i in range(population_size)]
    population_id = 0
    for i in tqdm(range(num_generations)):
        shuffle(population)
        # calculate the fitness of each genome 
        population_fitness = [[score, fitness(score_to_genome(score, num_bars, num_notes, num_steps, pauses, key, scale, root, chord_progression), stream, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm, threshold, chord_progression, structural_information)] for score in population]
        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)
        
        sum_exp_fitness = sum(np.exp(x[1]) for x in population_fitness)
        fitness_list = []
        for lst in sorted_population_fitness:
            lst[1] = np.exp(lst[1]) / sum_exp_fitness
            fitness_list.append(lst[1])
        
        population = [e[0] for e in sorted_population_fitness]
        next_generation = population[0:3]

        for j in range(int(len(population) / 2) - 1):
            # the process of genetic algorithm
            parents = selection_pair(population, fitness_list)
            offspring_a, offspring_b = measure_level_crossover(parents[0], parents[1], num_bars, num_notes, BITS_PER_NOTE)
            if type(offspring_a) == list:
                offspring_a_stream = m21.stream.Part()
                for i in range(len(offspring_a)):
                    offspring_a_stream.append(offspring_a[i])
                offspring_a = offspring_a_stream
            if type(offspring_b) == list:
                offspring_b_stream = m21.stream.Part()
                for i in range(len(offspring_b)):
                    offspring_b_stream.append(offspring_b[i])
                offspring_b = offspring_b_stream
            next_generation += [offspring_a, offspring_b]
        population = next_generation
        population_id += 1
    # the best score
    end_time = time.time()
    for j in range(4):
        print_similarity(population[i], stream, num_bars, chord_progression)
        print_score(population[i], num_notes)
        population[i] = append_chord_and_information_to_score(population[i], chord_progression, key, scale, bpm)
        population[i].write('musicxml', f'results/best{j}.musicxml')

    
if __name__ == '__main__':
    '''
    global variables(一些可控制的全局变量)
    '''
    BITS_PER_NOTE = 5
    pauses = True
    '''
    add args by command line
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bars', type=int, default=2, help='num of bars in the file')
    parser.add_argument('--num_notes', type=int, default=16, help='num of notes in the bar')
    parser.add_argument('--num_steps', type=int, default=1, help='num of notes simultaneously hit in the music')
    parser.add_argument('--bpm', type=int, default=80, help='bpm of the music')
    parser.add_argument('--population_size', type=int, default=10, help='population size of genetic algorithm')
    parser.add_argument('--num_generations', type=int, default=250, help='num of generations in genetic algorithm')
    parser.add_argument('--num_mutations', type=int, default=5, help='num of mutations in one genome')
    parser.add_argument('--mutation_probability', type=float, default=0.4, help='mutation probability in genetic algorithm')
    parser.add_argument('--threshold_similarity', type=float, default=0.8, help='the threshold of similarity between two scores')
    parser.add_argument('--parse_info_from_score', action='store_true', help='whether to parse key, scale and root from the input score')
    parser.add_argument('--key', type=str, default='C', help='key of the music')
    parser.add_argument('--scale', type=str, default='major', help='scale of the music')
    parser.add_argument('--root', type=int, default=3, help='root octave of the music')
    parser.add_argument('--structural_information_period', type=str, nargs='+', default=None, help='structural information of period of both source and destination')
    parser.add_argument('--structural_information_phrase', type=int, nargs='+', default=None, help='structural information of phrase number of both source and destination')
    parser.add_argument('--give_chord_progression', action='store_true', help='whether to give chord progression')
    parser.add_argument('--chord_progression', type=str, default=None, help='chord progression of the music')

    parser.add_argument('--give_pause_template', action='store_true', help='whether to give pause template')
    parser.add_argument('--pause_template', type=str, default=None, help='pause template of the music')
    parser.add_argument('--give_pitchrhythm_template', action='store_true', help='whether to give pitchrhythm template')
    parser.add_argument('--pitchrhythm_template', type=str, default=None, help='pitchrhythm template of the music')

    parser.add_argument('--structural_information', type=str, default=None, help='structural information of the music')
    parser.add_argument('--just_pitch_change', action='store_true', help='whether to just change the pitch of the variation')
    parser.add_argument('--file_path', type=str, default='./data/country road motif2.musicxml', help='file path of the input score')

    args = parser.parse_args()
    '''
    process variables
    '''
    if args.give_chord_progression:
        chord_progression_raw = eval(args.chord_progression)
        chord_progression = []
        for onset, chord in chord_progression_raw:
            for i in range(len(chord)):
                if type(chord[i]) == str:
                    chord[i] = m21.harmony.ChordSymbol(chord[i])
            chord_progression.append([onset, chord])
    if args.give_pause_template:
        pause_template_raw = eval(args.pause_template)
        pause_template = dict()
        pause_template['offsets'] = pause_template_raw[0]
        pause_template['durations'] = pause_template_raw[1]
    if args.give_pitchrhythm_template:
        pitchrhythm_template_raw = eval(args.pitchrhythm_template)
        pitchrhythm_template = dict()
        pitchrhythm_template['offsets'] = pitchrhythm_template_raw[0]
        pitchrhythm_template['durations'] = pitchrhythm_template_raw[1]

    num_bars = args.num_bars
    num_notes = args.num_notes
    num_steps = args.num_steps
    bpm = args.bpm
    population_size = args.population_size
    num_generations = args.num_generations
    num_mutations = args.num_mutations
    mutation_probability = args.mutation_probability
    threshold_similarity = args.threshold_similarity
    parse_info_from_score = args.parse_info_from_score

    score = m21.converter.parse(args.file_path)
    stream = m21.stream.Part()
    for i in range(len(score)):
        if type(score[i]) == m21.stream.Part:
            for j in range(len(score[i])):
                if type(score[i][j]) == m21.stream.Measure:
                    if len(stream) < num_bars:
                        stream.append(score[i][j])
    if parse_info_from_score:
        key, scale, root = decode_info(stream)
    else:
        key = args.key
        scale = args.scale
        root = args.root

    give_pause_template = args.give_pause_template
    pause_template = args.pause_template
    give_pitchrhythm_template = args.give_pitchrhythm_template
    pitchrhythm_template = args.pitchrhythm_template

    si = {"src":['chorous', 2], "dst":['chorous', 4]}
    si["src"][0] = args.structural_information_period[0]
    si["dst"][0] = args.structural_information_period[1]
    si["src"][1] = args.structural_information_phrase[0]
    si["dst"][1] = args.structural_information_phrase[1]

    if args.just_pitch_change:
        just_pitch_change(stream, num_bars, num_notes, num_steps, pauses, key, scale, root, population_size, num_mutations, mutation_probability, bpm, threshold_similarity, chord_progression, pause_template, pitchrhythm_template, structural_information=si)
    else:
        main(stream, num_bars, num_notes, num_steps, pauses, key, scale, root, population_size, num_mutations, mutation_probability, bpm, threshold_similarity, chord_progression, pause_template, pitchrhythm_template, structural_information=si)