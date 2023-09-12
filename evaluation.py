from multiprocessing.sharedctypes import Value
import music21 as m21
import numpy as np

from utils import chord_measure, duration_measure, kurtosis, notes_and_rests_score, rhythm_measure, pitches_measure, expand_to_uniform, rhythm_measure_nocombine, pitches_measure_nocombine, Scale
from random import sample, uniform

'''
Evaluation: stores the function of different components of fitness score
'''

def similarity_rate_pitch_fitness(measure1 : m21.stream.Measure, measure2 : m21.stream.Measure):
    '''
    Utility function of computing the similarity_rate of pitches for fitness function in main

    Args:
    measure1 : m21.stream.Measure——syn
    measure2 : m21.stream.Measure——obs

    Returns:
    similarity_rate: int
    '''
    measure1_rhythm = rhythm_measure(measure1)
    measure2_rhythm = rhythm_measure(measure2)
    measure1_pitches = pitches_measure(measure1)
    measure2_pitches = pitches_measure(measure2)
    measure1_num_of_notes = len(measure1_rhythm)
    measure2_num_of_notes = len(measure2_rhythm)

    measure1_expanded_rhythm, measure1_expanded_pitches = expand_to_uniform(measure1_rhythm, measure1_pitches)
    measure2_expanded_rhythm, measure2_expanded_pitches = expand_to_uniform(measure2_rhythm, measure2_pitches)


    similarity_pitch_rate = sum(measure1_expanded_pitches[x] == measure2_expanded_pitches[x] for x in range(16))
    # the soft similarity rate: sum the difference between generated pitches and the observed pitches
    # similarity_soft_pitch_rate = sum(abs(measure1_expanded_pitches[x] - measure2_expanded_pitches[x]) for x in range(measure2_num_of_notes))

    # the punish of generating unexpected pauses
    similarity_pitch_rate -= 10 * sum(measure2_expanded_pitches[x] == -1 and measure1_expanded_pitches[x] != -1 for x in range(16))

    # rhythm_rate calculation
    similarity_rhythm_rate = sum(measure1_rhythm[x] in measure2_rhythm for x in range(measure1_num_of_notes))

    # rhythm_bonus: do not have pauses in the end of a measure
    rhythm_bonus = sum(x not in measure1_rhythm for x in np.linspace(2, 3.75, 8))

    # rhythm_length_bonus: for simple melody there is supposed to have few rhythm(offsets)
    rhythm_length_bonus = 0
    if len(measure1_rhythm) < 10:
        rhythm_length_bonus = 3
    elif len(measure1_rhythm) < 8:
        rhythm_length_bonus = 5

    # beat_bonus: wish that more notes can be settled on the strong beats
    beat_bonus = 1.5 * sum(x in [0.0, 1.0, 2.0, 3.0] for x in measure1_rhythm)  # 重音

    # return the weighted average of similarity .16 means num_of_notes
    return (similarity_pitch_rate + 0 * similarity_rhythm_rate + 0 * rhythm_bonus + 0 * rhythm_length_bonus + 0 * beat_bonus) / 16

def similarity_pitch_rate(measure1 : m21.stream.Measure, measure2 : m21.stream.Measure):
    '''
    Utility function of simply simply simply computing the similarity_rate of pitches

    Args:
    measure1 : m21.stream.Measure——syn
    measure2 : m21.stream.Measure——obs

    Returns:
    similarity_rate: int
    '''
    measure1_rhythm = rhythm_measure(measure1)
    measure2_rhythm = rhythm_measure(measure2)
    measure1_pitches = pitches_measure(measure1)
    measure2_pitches = pitches_measure(measure2)
    measure1_num_of_notes = len(measure1_rhythm)
    measure2_num_of_notes = len(measure2_rhythm)

    measure1_expanded_rhythm, measure1_expanded_pitches = expand_to_uniform(measure1_rhythm, measure1_pitches)
    measure2_expanded_rhythm, measure2_expanded_pitches = expand_to_uniform(measure2_rhythm, measure2_pitches)


    similarity_pitch_rate = sum(measure1_expanded_pitches[x] == measure2_expanded_pitches[x] for x in range(len(measure1_expanded_rhythm))) / len(measure1_expanded_rhythm)
    return similarity_pitch_rate

def variation_similarity_fitness(score_syn, score_obs):
    '''
    variation_similarity: tend to think of higher similarity in the first half of the score, and maybe some variation in the last half of the score
    '''
    # finding the first half of the score(not quite easy)
    num_bars = len(score_obs.getElementsByClass('Measure'))
    rhythms_syn = []
    pitches_syn = []
    pitches_syn_expand = []
    for idx, measure in enumerate(score_syn.getElementsByClass('Measure')):
        rhythms_syn +=[rhythm_measure_nocombine(measure)[i]+idx * 4.0 for i in range(len(rhythm_measure_nocombine(measure)))]
        pitches_syn += pitches_measure_nocombine(measure)
        pitches_syn_expand += expand_to_uniform(rhythm_measure(measure), pitches_measure(measure))[1]
    rhythms_obs = []
    pitches_obs = []
    pitches_obs_expand = []
    for idx, measure in enumerate(score_obs.getElementsByClass('Measure')):
        rhythms_obs +=[rhythm_measure_nocombine(measure)[i]+idx * 4.0 for i in range(len(rhythm_measure_nocombine(measure)))]
        pitches_obs_expand += expand_to_uniform(rhythm_measure(measure), pitches_measure(measure))[1]
        pitches_obs += pitches_measure_nocombine(measure)
    
    boundary_idx = min(int(len(rhythms_syn) / 4 * 3), int(len(rhythms_obs) / 4 * 3))
    first_rhythm_similarity = 5*sum(rhythms_obs[x] == rhythms_syn[x] for x in range(boundary_idx))
    last_rhythm_similarity = sum(rhythms_obs[x] == rhythms_syn[x] for x in range(boundary_idx))
    rhythm_similarity = (first_rhythm_similarity + last_rhythm_similarity) / max(len(rhythms_obs), len(rhythms_syn))

    boundary_idx_pitches = int(len(pitches_obs_expand) / 4 * 3)
    first_pitch_similarity = 5*sum(pitches_obs_expand[x] == pitches_syn_expand[x] for x in range(boundary_idx_pitches))

    last_pitch_similarity = sum(pitches_obs_expand[x] == pitches_syn_expand[x] for x in range(boundary_idx_pitches, len(pitches_obs_expand)))
    pitch_similarity = (first_pitch_similarity + last_pitch_similarity) / len(pitches_obs_expand)
    return (3*rhythm_similarity + pitch_similarity) / 4 
    # first_half_similarity = 3*sum(pitches_obs[x] == pitches_syn[x] for x in range(boundary_idx))
    # last_half_similarity = sum(pitches_obs[x] == pitches_syn[x] for x in range(boundary_idx, min(len(pitches_obs), len(pitches_syn))))
    # return (first_half_similarity + last_half_similarity) / max(len(pitches_obs), len(pitches_syn))




def penalty_for_smoothness(score_syn):
    '''
    Utility function to compute the penalty for melody that is not smooth

    Args:
    score_syn: the synthesized score by genetic algorithm

    Return s:
    the score for penalty.
    '''

    note_get = []
    count = 0
    # get all notes 
    for measure in score_syn.getElementsByClass('Measure'):
        for note in measure.getElementsByClass('Note'):
            note_get.append(note)
    # calculate the neighbor pitch difference of score
    for idx in range(1, len(note_get)):
        if abs(note_get[idx].pitch.midi - note_get[idx - 1].pitch.midi) > 10:
            count += 20
        elif abs(note_get[idx].pitch.midi - note_get[idx - 1].pitch.midi) == 0:
            count -= 3
        elif abs(note_get[idx].pitch.midi - note_get[idx - 1].pitch.midi) > 7:
            count += 5
    
    return count / 16 

def bonus_for_overall_smoothness(score_syn):
    '''
    the usage of the Kurtosis(the fourth moment of the distribution of the samples) to evaluate the overall smoothness of the notes

    Args:
    score_syn: the synthesized m21 score

    Returns:
    the kurtosis of the note pitch list
    '''
    # the bigger the kurtosis value, the more concentrated of the distribution of pitches
    note_get = []
    count = 0
    # get all notes.pitch.midi
    for measure in score_syn.getElementsByClass('Measure'):
        for note in measure.getElementsByClass('Note'):
            note_get.append(note.pitch.midi)
    note_get_np = np.array(note_get)
    return kurtosis(note_get)

def bonus_for_phrase_clique(score_syn, num_notes):
    '''
    to calculate the bonus for any phrase clique in the score
    which contains:
    value1_duration: the overall duration of a clique
    value2_lessnotes: encourage the behavior of the repition of the same pitch(in other words, encourage high duration)
    value3_longrest: encourage long rest after the clique

    '''
    rhythm_score = []
    pitches_score = []
    duration_score = []
    rhythm_score_expanded = []
    pitches_score_expanded = []
    for idx, measure in enumerate(score_syn.getElementsByClass('Measure')):
        rhythm_Measure = rhythm_measure(measure)
        pitches_Measure = pitches_measure(measure)
        duration_Measure = duration_measure(measure)
        rhythm_expanded_measure, pitches_expanded_measure = expand_to_uniform(rhythm_Measure, pitches_Measure)
        rhythm_Measure = [x + idx * 4.0 for x in rhythm_Measure]
        rhythm_expanded_measure = [x + idx * 4.0 for x in rhythm_expanded_measure]

        rhythm_score_expanded += rhythm_expanded_measure
        pitches_score_expanded += pitches_expanded_measure
        
        rhythm_score += rhythm_Measure
        pitches_score += pitches_Measure
        duration_score += duration_Measure
    
    score_length = len(score_syn.getElementsByClass('Measure')) * 4.0
    measure_num = score_length / 4.0
    i = 0
    count_for_cliques = 0
    bonus = 0
    rhythm_clique = []
    pitches_clique = []
    duration_clique = []
    while(i < len(rhythm_score)): 
        if pitches_score[i] != -1 and i != len(rhythm_score) - 1:
            rhythm_clique.append(rhythm_score[i])
            pitches_clique.append(pitches_score[i])
            # count_len_for_clique += 1
        else: # long duration and the melody is not scattered, than the score is high.
            count_for_cliques += 1
            # value1: overall duration need to be long (like a measure and a half)
            # value2: don't want to have too many sixteenth note
            value1_duration = 0
            value2_lessnotes = 0
            if rhythm_clique != []:
                rhythm_clique.append(rhythm_score[i])
                value1_duration = (rhythm_clique[-1] - rhythm_clique[0]) / score_length 

                duration_clique = [rhythm_clique[x] - rhythm_clique[x - 1] for x in range(1, len(rhythm_clique))]
                value2_lessnotes = sum((x - (4.0 / num_notes)) / (4.0 / num_notes)  for x in duration_clique) / len(duration_clique)
                # value2_lessnotes = len(rhythm_clique)
                if i < len(rhythm_score) - 1:
                    value3_longrest = rhythm_score[i + 1] - rhythm_score[i]
                else:
                    value3_longrest = score_length - rhythm_score[i]
                # the hard constrait failed to reward
                bonus += int((value1_duration > 4.0 / (num_notes) or value1_duration < 8.0 / (num_notes)))
                bonus += value2_lessnotes
                bonus += value3_longrest
            rhythm_clique = []
            pitches_clique = []
        i += 1
    # few cliques won't win the reward
    if count_for_cliques < measure_num / 2:
        return 0
    return bonus / count_for_cliques

def penalty_for_pauses(score_syn, pauses):
    '''
    Utility function to compute the penalty for melody that has too much pauses

    Args:
    score_syn: the synthesized score by genetic algorithm
    pauses: the hyperparameter whether it has pauses

    Returns:
    the score for penalty.
    '''
    if pauses:
        count = 0
        for measure in score_syn.getElementsByClass('Measure'):
            measure_rhythm = rhythm_measure(measure)
            measure_pitches = pitches_measure(measure)
            # when facing a rest, we don't expect it to be sixteenth note length(with the left and the right non-rest)
            for idx in range(len(measure_pitches)):
                if idx > 0 and idx < (len(measure_pitches) - 1):
                    if measure_pitches[idx - 1] > 0 and measure_pitches[idx + 1] > 0:
                        count += 2
        return count / 16
    return 0

def pause_rate(score_syn, num_bars):
    '''
    compute the pause_rate for the score

    Args:
    score_syn: the synthesized music21 score
    num_bars: the number of bars of the score_syn

    Returns:
    the pause rate: float(between 0 and 1)

    '''
    rhythm_pause = []
    duration_pause = []
    for measure in score_syn.getElementsByClass('Measure'):
        for rest in measure.getElementsByClass('Rest'):
            rhythm_pause.append(rest.offset)
            duration_pause.append(rest.duration.quarterLength)
    Sum = sum(duration_pause)
    return Sum / num_bars / 4

def penalty_for_detune(measure, rhythm, chord):
    '''
    Utility function to compute the penalty for melody that failed to follow the chord progression

    Args:
    measure: the synthesized measure in score by genetic algorithm
    pauses: the hyperparameter whether it has pauses

    Returns:
    the score for penalty.
    '''
    count = 0
    if chord != []:
        for note in measure.getElementsByClass('Note'):
            # class1: root
            if note.pitch.pitchClass == chord[rhythm.index(note.offset)].root().pitchClass:
                count -= 5
            # class2: notes in chord
            elif note.pitch.pitchClass in chord[rhythm.index(note.offset)].pitchClasses:
                count -= 3
            # class3: notes in octave
            else:
                count += 2
    return count / 16

def print_similarity(score_syn, score_obs, num_bars, chord_progression = None):
    '''
    print the pitch_similarity, rhythm_similarity and chord_fitness on the terminal

    Args:
    score_syn: the synthesized music21 score
    score_obs: the original score
    num_bars: the number of bars of the score_syn
    chord_progression: the given chord_progression(tuple)

    Returns:
    no return
    '''
    pitch_similarity = 0
    for i in range(num_bars):
        pitch_similarity += similarity_pitch_rate(score_syn.getElementsByClass('Measure')[i], score_obs.getElementsByClass('Measure')[i])
 
    pitch_similarity = pitch_similarity / num_bars

    rhythm_similarity = 0
    for i in range(num_bars):
        rhythm_syn = rhythm_measure(score_syn.getElementsByClass('Measure')[i])
        rhythm_obs = rhythm_measure(score_obs.getElementsByClass('Measure')[i])

        rhythm_similarity += sum(rhythm_syn[i] in rhythm_obs for i in range(len(rhythm_syn))) / max(len(rhythm_syn), len(rhythm_obs))
    rhythm_similarity /= num_bars

    print(f"pitch_similarity = {pitch_similarity}, rhythm_similarity = {rhythm_similarity}, chord_fitness = {compute_chord_fitness(score_syn, chord_progression)}")

def print_score(score_syn, num_notes):
    '''
    print the bonus scores for generation

    Args:
    score_syn: the synthesized music21 score
    num_notes: the number of notes in a measure(the resolution of the score)

    Returns:
    No returns
    '''
    print(f"bonus_for_overall_smoothness:{bonus_for_overall_smoothness(score_syn)}, " + f"bonus_for_phrase_clique:{bonus_for_phrase_clique(score_syn, num_notes)}")

def compute_chord_fitness(score_syn, chord_progression = None):
    '''

    utility function for computing the chord fitness
    Args:
    score_syn: the synthesized music21 score
    chord_progression: the given chord_progression: tuple

    Returns:
    chord fitness score: float(between 0 and 1)
    '''
    chord_fitness = 0

    # step1: extract chord progression
    for idx, measure in enumerate(score_syn.getElementsByClass('Measure')):
        if chord_progression == None:
            rhythm, chord = chord_measure(measure)
            if chord != []:
                rhythm_expand, chord_expand = expand_to_uniform(rhythm, chord)
            else:
                rhythm_expand = rhythm
                chord_expand = chord
        else:
            # here imposes the constraint for the type of chord progression
            assert type(chord_progression) == tuple or type(chord_progression) == list, "the type of chord_progression ought to be tuple, with idx=0 for rhythm(offset) and idx=1 for chord(m21.harmony.chordsymbol)"
            rhythm_expand, chord_expand = expand_to_uniform(chord_progression[idx][0], chord_progression[idx][1])
        pitch_measure = pitches_measure(score_syn.getElementsByClass('Measure')[idx])
        rhythm_expand = rhythm_measure(score_syn.getElementsByClass('Measure')[idx])
        rhythm_expand, pitch_expand = expand_to_uniform(rhythm_expand, pitch_measure)
        # step2: compute chord_fitness
        if chord_expand != []:
            for index, pitch in enumerate(pitch_expand):
                pitch_object = m21.pitch.Pitch()
                pitch_object.midi = pitch
                chord_fitness += (pitch_object.pitchClass in chord_expand[index].pitchClasses)
    chord_fitness /= (len(score_syn)* len(rhythm_expand))
    return chord_fitness

def horizontal_distance(score_syn, start_measure: int):
    if start_measure >= len(score_syn) - 1:
        sum = 0
        for i in range(len(score_syn) - 1):
            sum += abs(score_syn[i].getElementsByClass('Note')[-1].pitch.midi - score_syn[i + 1].getElementsByClass('Note')[0].pitch.midi)
        return sum
    else:
        sum = abs(abs(score_syn[start_measure].getElementsByClass('Note')[-1].pitch.midi - score_syn[start_measure + 1].getElementsByClass('Note')[0].pitch.midi))
        return sum

def ending_dolmisol(score_syn, key, scale, root):
    rhythms_syn = []
    pitches_syn = []
    pitches_syn_expand = []
    rhythms_syn_expand = []
    for idx, measure in enumerate(score_syn.getElementsByClass('Measure')):
        rhythms_syn +=[rhythm_measure_nocombine(measure)[i]+idx * 4.0 for i in range(len(rhythm_measure_nocombine(measure)))]
        pitches_syn += pitches_measure_nocombine(measure)
        rhythms_syn_expand += [expand_to_uniform(rhythm_measure(measure), pitches_measure(measure))[0][i]+idx * 4.0 for i in range(len(expand_to_uniform(rhythm_measure(measure), pitches_measure(measure))[0]))]
        pitches_syn_expand += expand_to_uniform(rhythm_measure(measure), pitches_measure(measure))[1]
    ending_note_offset = rhythms_syn[-1]
    for idx, element in reversed(list(enumerate(rhythms_syn))):
        if pitches_syn[idx] == -1:
            continue
        else:
            ending_note_offset = element
            break

    standard_ending = len(score_syn.getElementsByClass('Measure')) * 4.0-2.0
    ending_note_offset_idx = rhythms_syn_expand.index(ending_note_offset)
    standard_ending_note_offset_idx = rhythms_syn_expand.index(standard_ending)


    scl = Scale(key = key, modality = scale, base_octave = root, expand = pow(2, 5 - 1) // 7) # the 5 here is BITS_PER_NOTE
    pitch_lst = scl.pitch_list()
   
    dol_rate = sum((pitch_lst[0] == pitches_syn_expand[x] or pitch_lst[7] == pitches_syn_expand[x]) for x in range(min(ending_note_offset_idx, standard_ending_note_offset_idx), len(pitches_syn_expand)))
    re_rate = sum((pitch_lst[1] == pitches_syn_expand[x] or pitch_lst[8] == pitches_syn_expand[x]) for x in range(min(ending_note_offset_idx, standard_ending_note_offset_idx), len(pitches_syn_expand)))
    mi_rate = sum((pitch_lst[2] == pitches_syn_expand[x] or pitch_lst[9] == pitches_syn_expand[x]) for x in range(min(ending_note_offset_idx, standard_ending_note_offset_idx), len(pitches_syn_expand)))
    fa_rate = sum((pitch_lst[3] == pitches_syn_expand[x] or pitch_lst[10] == pitches_syn_expand[x]) for x in range(min(ending_note_offset_idx, standard_ending_note_offset_idx), len(pitches_syn_expand)))
    sol_rate = sum((pitch_lst[4] == pitches_syn_expand[x] or pitch_lst[11] == pitches_syn_expand[x]) for x in range(min(ending_note_offset_idx, standard_ending_note_offset_idx), len(pitches_syn_expand)))
    la_rate = sum((pitch_lst[5] == pitches_syn_expand[x] or pitch_lst[12] == pitches_syn_expand[x]) for x in range(min(ending_note_offset_idx, standard_ending_note_offset_idx), len(pitches_syn_expand)))
    xi_rate = sum((pitch_lst[6] == pitches_syn_expand[x] or pitch_lst[13] == pitches_syn_expand[x]) for x in range(min(ending_note_offset_idx, standard_ending_note_offset_idx), len(pitches_syn_expand)))
    return dol_rate, re_rate, mi_rate, fa_rate, sol_rate, la_rate, xi_rate
