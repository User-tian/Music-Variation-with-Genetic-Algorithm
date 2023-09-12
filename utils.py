from cmath import nan
from copy import deepcopy
import music21 as m21
import numpy as np
from random import choices
from typing import List

def decode_info(stream: m21.stream.Part):
    first_measure = stream[0]
    num_bars = len(stream)
    Key_info = first_measure.getElementsByClass('KeySignature')[0]
    mode, tonic = Key_info.asKey().mode, Key_info.asKey().tonic
    pitches = []
    root = 4
    for measure in stream.getElementsByClass('Measure'):
        pitches += pitches_measure(measure)
    for iter in range(100):
        if -1 in pitches:
            pitches.remove(-1) # remove rests
        else:
            break
    if pitches != []:
        Minpitch = min(pitches)
        for idx in [-2,-1,0,1]:
            if tonic.midi + idx * 12 <= Minpitch:
                root = 4 + idx
            
    return tonic, mode, root


def initialize_genome_with_pause_pattern(num_bars, num_notes, BITS_PER_NOTE, pause_dict = None, pitchrhythm_dict = None):
    '''
    initialize the genome with given pause_pattern and rhythm pattern(the pause_pattern is thought to be somewhat expressive but rhythm pattern failed to have great performance)

    Args:
    num_bars:
    num_notes: the resolution of the score
    BITS_PER_NOTE: the bits needed to represent a note

    pause_dict: the pause pattern(offsets and durations): tuple
    pitchrhythm_dict: the rhythm pattern(offsets and durations): tuple

    Returns:
    the constraint genome
    '''
    no_constraint_genome = choices([0, 1], k=num_bars * num_notes * BITS_PER_NOTE)
    pause_constraint_genome = deepcopy(no_constraint_genome)
    if pause_dict != None:
        for idx, offset in enumerate(pause_dict["offsets"]):
            location = offset / (4.0 / num_notes) * BITS_PER_NOTE
            duration = pause_dict["durations"][idx]
            for i in range(int(location), int(location + duration / (4.0 / num_notes) * BITS_PER_NOTE)):
                # here assumes that luck_number is 31
                pause_constraint_genome[i] = 1
    # pitchrhythm_dict: 给定一个beat pattern，每次换音就最高位从0到1 但是目前效果不好
    rhythm_constraint_genome = deepcopy(pause_constraint_genome)
    if pitchrhythm_dict != None:
        for idx, offset in enumerate(pitchrhythm_dict["offsets"]):
            location = offset / (4.0 / num_notes) * BITS_PER_NOTE
            duration = pitchrhythm_dict["durations"][idx]
            for i in range(int(location), int(location + duration / (4.0 / num_notes) * BITS_PER_NOTE)):
                if idx % 2 == 0 or (idx % 2 == 1 and (i % BITS_PER_NOTE) != 0):
                    rhythm_constraint_genome[i] = 0
                else:
                    rhythm_constraint_genome[i] = 1
    return rhythm_constraint_genome

def rhythm_measure_nocombine(measure: m21.stream.Measure):

    '''
    the list of offsets of a measure

    Args:
    measure: m21.stream.Measure

    Returns:
    The list of offsets of the notes in a certain measure
    '''
    rhythm = []

    cmp = -2
    for note in measure:
        if type(note) == m21.note.Note or type(note) == m21.note.Rest:
            rhythm.append(note.offset)
    return rhythm

def pitches_measure_nocombine(measure: m21.stream.Measure):
    '''
    the list of pitches of a measure

    Args:
    measure: m21.stream.Measure

    Returns:
    The list of pitches of the notes in a certain measure
    '''
    pitches = []

    cmp = -2
    for note in measure:
        if type(note) == m21.note.Note:
            pitches.append(note.pitch.midi)
        elif type(note) == m21.note.Rest:
            pitches.append(-1)
    return pitches

def rhythm_measure(measure: m21.stream.Measure):
    '''
    the list of offsets of a measure

    Args:
    measure: m21.stream.Measure

    Returns:
    The list of offsets of the notes in a certain measure
    '''
    rhythm = []

    cmp = -2
    for note in measure:
        if type(note) == m21.note.Note or type(note) == m21.note.Rest:
            rhythm.append(note.offset)
            if type(note) == m21.note.Note:
                if len(rhythm) >= 2 and cmp == note.pitch.midi:
                    rhythm.pop()
                cmp = note.pitch.midi
            else:
                if len(rhythm) >= 2 and cmp == -1:
                    rhythm.pop()
                cmp = -1
    return rhythm

def pitches_measure(measure: m21.stream.Measure):
    '''
    the list of pitches(midi) of a measure

    Args:
    measure: m21.stream.Measure

    Returns:
    The list of pitches(midi) of the notes in a certain measure
    '''
    pitches = []

    for note in measure:
        if type(note) == m21.note.Note:
            pitches.append(note.pitch.midi)
        elif type(note) == m21.note.Rest:
            pitches.append(-1)
        if len(pitches) >= 2 and pitches[-2] == pitches[-1]:
            pitches.pop()
    return pitches

def duration_measure(measure: m21.stream.Measure):
    '''
    the list of durations of a measure

    Args:
    measure: m21.stream.Measure

    Returns:
    The list of durations of the notes in a certain measure
    '''
    duration_list= []
    for note in measure:
        if type(note) == m21.note.Rest or type(note) == m21.note.Note: 
            duration_list.append(note.duration.quarterLength)
    return duration_list

def notes_and_rests_score(score_syn):
    '''
    compute the notes and rests in a score
    Args:
    score_syn: the synthesized score

    Returns:
    the notes_and_rests list
    '''
    notes_and_rests_lst = []
    for measure in score_syn.getElementsByClass('Measure'):
        for note in measure:
            if type(note) == m21.note.Rest() or type(note) == m21.note.Note():
                notes_and_rests_lst.append(note)
    return notes_and_rests_lst

def extract_chord_progression(score):
    '''
    to extract chord progression from the score(chord symbol)

    Args:
    score: the score provided(with chord symbol)

    Returns:
    rhythm_harmony: list: the rhythm of harmony
    chord_harmony: list: the chord of harmony(type of the element of list is m21.harmony.chordsymbol)
    '''
    rhythm_harmony = []
    chord_harmony = []
    for measure in score.getElementsByClass('Measure'):
        for harmony in measure.getElementsByClass('Harmony'):
            rhythm_harmony.append(harmony.offset + measure.offset)# 带上小节的
            chord_harmony.append(harmony)
    return rhythm_harmony, chord_harmony

def chord_measure(measure: m21.stream.Measure):
    '''
    to extract chord progression from the measure(chord symbol)

    Args:
    measure: the measure provided(with chord symbol)

    Returns:
    rhythm_harmony: list: the rhythm of harmony
    chord_harmony: list: the chord of harmony(type of the element of list is m21.harmony.chordsymbol)
    '''
    rhythm_harmony = []
    chord_harmony = []
    for harmony in measure.getElementsByClass('Harmony'):
            rhythm_harmony.append(harmony.offset)
            chord_harmony.append(harmony)
    return rhythm_harmony, chord_harmony

def kurtosis(lst):
    '''
    math equation to compute the kurtosis(the fourth moment of the distribution) of the certain list

    Args:
    lst: list with numbers

    Returns:
    the kurtosis of the list(between 0 and 1)
    '''
    average = np.mean(lst)
    n = len(lst)
    if n > 3: 
        k1 = n * (n + 1) * (n - 1) / (n - 2) / (n - 3)
        k2 = 3 * (n - 1) ** 2 / (n - 2) / (n - 3)
        m2 = 0
        m4 = 0
        for i in lst:
            m4 += (i - average) ** 4
            m2 += (i - average) ** 2
        Kurtosis = k1 * m4 / m2 ** 2 - k2
        if Kurtosis != nan:
            return (1.0 / (1 + np.exp(-Kurtosis)))
    return 0

def expand_to_uniform(rhythm: list, pitches: list):
    '''
    utility function to expand the rhythm and pitches to uniform status(with the smallest division of sixteenth note)

    Args:
    rhythm: list
    pitches: list(maybe chord)

    Returns:
    rhythm_expanded:
    pitches_expanded:
    '''
    rhythm_expanded = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75]
    pitches_expanded = []
    counter = 0
    duration = 0
    for counter in range(len(rhythm)):
        if counter + 1 < len(pitches):
            duration = rhythm[counter + 1] - rhythm[counter]
            for i in range(int(duration / 0.25)):
                pitches_expanded.append(pitches[counter])
        counter += 1
    duration = 4 - rhythm[-1]

    for i in range(int(duration / 0.25)):
        pitches_expanded.append(pitches[counter - 1])
    return rhythm_expanded, pitches_expanded

def selection_pair(population, fitness_list):
    '''
    select pair in the population with the probability of fitness

    Args:
    population:list: the population of the genetic algorithm
    fitness_list: the fitness list of the population genome

    Returns:
    [population[choice1], population[choice2]]: the list of the selected pair
    '''
    np_population = np.array(range(len(population)))
    np_fitness_list = np.array(fitness_list)
    choice1 = np.random.choice(np_population, p = np_fitness_list)
    choice2 = np.random.choice(np_population, p = np_fitness_list)
    return [population[choice1], population[choice2]]

# 服务于genome_to_score函数
def int_from_bits(bits: List[int]) -> int:
    '''
    convert bits to int(binary to decimal)
    '''
    return int(sum([bit*pow(2, index) for index, bit in enumerate(bits)]))


class Scale:
    def __init__(self, key, modality, base_octave, expand):
        p1 = m21.pitch.Pitch(key)
        self.key = p1.pitchClass
        assert modality == "major" or modality == "minor", "we have only implemented major and minor scale! "
        self.modality = modality
        assert base_octave <= 7 and base_octave >= 0 , "the range of an octave is between 0 and 6! "
        self.base_octave = base_octave
        self.expand = expand
    
    def pitch_list(self):
        self.base_scale = np.zeros(7)
        self.scales = []
        self.pitchlst= []
        if self.modality == "major":
            self.base_scale = np.array([0, 2, 4, 5, 7, 9, 11])
        elif self.modality == "minor":
            self.base_scale = np.array([0, 2, 3, 5, 7, 8, 10])
        for i in range(self.expand):
            self.scales.append(self.base_octave + i)
        for element in self.scales:
            self.pitchlst += [x + element * 12 + 12 + self.key for x in self.base_scale]
        return self.pitchlst


def append_chord_and_information_to_score(score, chord_progression, key, scale, bpm):
    '''
    utility function to add useful information to the score

    Args:
    score:
    chord_progression:
    key: 
    scale: 
    bpm: 

    Returns:
    score: the score that has added the information
    '''
    score.removeByClass('KeySignature')
    if chord_progression != None:
        i = 0
        for element in chord_progression:
            for idx, rhythm in enumerate(element[0]):# element[0]是rhythm，element[1]是chord
                for j in range(len(score[i])):
                    if type(score[i][j]) == m21.harmony.ChordSymbol and score[i][j].offset == rhythm:
                        score[i].remove(score[i][j])
                score[i].removeByClass('ChordSymbol')
                score[i].insert(rhythm, element[1][idx])

            i += 1

    # add bpm:
    score[0].insert(m21.tempo.MetronomeMark(number=80, referent='quarter'))
    # add key signature
    KEY = m21.key.Key(key, scale)
    score.insert(m21.key.KeySignature(KEY.sharps))
    for i in score[0].getElementsByClass('KeySignature'):
        i.sharps = KEY.sharps
    return score