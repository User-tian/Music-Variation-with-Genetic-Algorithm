# Music-Variation-with-Genetic-Algorithm

## Introduction
Pop music melody variation with genetic algorithm

input: music phrase(2-8 bars); output: music phrase(corresponding number of bars)

## Algorithm Details
- Representation
  - Information to encode: Pitch + duration
  - Mapping: index of scale in 2 octaves ànumbers (2^𝐵𝐼𝑇𝑆)
  - Numbers $\rightarrow$ binary representation
- Operations
  - Measure-level crossover
    - Exchange in measure-level——capable in longer variation like more than 4 measures.
  - Intelligent Mutation
    - Genome-level mutation has no musical meaning, can be improved to
      - Move a note up/down an octave to improve smoothness
      - Repeat a measure

- Fitness functions——similar & melodic:
  - **Similarity pitch** rate: the similarity of pitch between the synthesized score and the given score
  - Bonus for **phrase clique**: bonus for excellent melody cliques(duration, less notes(8th notes > 16th notes), long pauses)
  - Bonus for **overall smoothness**: the Kurtosis(the fourth moment of the distribution of the samples) to evaluate the overall smoothness of the notes
  - Penalty for **detune**: the overall correspondence between the melody and the chord progression
  - Endingnote: To revise the variation according to structural information(naïve structural information, such as [verse 4]or [chorus 2])——to add some consistency in the changing process
## Usage

This code is run in an **Python 3.9** environment.

To install relevant packages, run:
```bash
pip install -r requirements.txt
```

Several running command examples is provided in **./run.sh** file, covering all the available inputs.

