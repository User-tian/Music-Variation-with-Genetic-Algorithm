# One effective music variation
python GAMV.py --file_path ./data/country\ road\ motif2.musicxml --num_bars 3 --num_notes 16 --num_steps 1\
    --bpm 80 --population_size 10 --num_generations 250 --num_mutations 5 --mutation_probability 0.4 --threshold_similarity 0.8\
    --parse_info_from_score --structural_information_period chrous chrous --structural_information_phrase 2 4 \
    --give_chord_progression --chord_progression "[[[0.0], ['E']], [[0.0], ['D']], [[0.0], ['A']] ]"  --just_pitch_change
    
# show all the argparse types
python GAMV.py --file_path ./data/country\ road\ motif2.musicxml --num_bars 3 --num_notes 16 --num_steps 1\
    --bpm 80 --population_size 10 --num_generations 250 --num_mutations 5 --mutation_probability 0.4 --threshold_similarity 0.8\
    --parse_info_from_score --structural_information_period chrous chrous --structural_information_phrase 2 4 \
    --give_chord_progression --chord_progression "[[[0.0], ['E']], [[0.0], ['D']], [[0.0], ['A']] ]" \
    --give_pause_template --pause_template "[7.0, 13.0], [1.0, 3.0]" --give_pitchrhythm_template --pitchrhythm_template "[0.0, 0.5, 1.0, 1.5, 2.0, 3.5, 4.0, 5.0, 6.0, 8.0, 8.5, 9.0, 9.5, 10.0, 11.5, 12.0], [0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.5, 0.5, 1.0]"

# another music variations
python GAMV.py --file_path ./data/MELONS_example_chord.musicxml --num_bars 4 --num_notes 16 --num_steps 1\
    --bpm 80 --population_size 10 --num_generations 250 --num_mutations 5 --mutation_probability 0.4 --threshold_similarity 0.8\
    --key C --root 4 --scale major --structural_information_period verse verse --structural_information_phrase 1 3 \
    --give_chord_progression --chord_progression "[[[0.0], ['Am']], [[0.0,2.0], ['G','Am']], [[0.0], ['F']],[[0.0],['Cmaj7']] ]" 