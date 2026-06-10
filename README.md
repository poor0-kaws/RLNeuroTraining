# RL Neuro Training

RL Neuro Training is a neuroevolution project for teaching robot controllers to
solve physical tasks.

The first task is Panda robot pick-and-place inside a MuJoCo / Robosuite
simulation.

In plain English:

1. Make many robot brains.
2. Let each brain control the robot once.
3. Score each robot by how well it behaved.
4. Keep the best brains.
5. Copy and slightly mutate them.
6. Repeat until better behavior appears.

This is different from gradient-based reinforcement learning. There is no value
function, policy gradient, replay buffer, or backpropagation through the task.
The system improves through selection pressure: better genomes survive more
often.

## Current Status

The full first training loop is built:

- fitness function
- genome encoding
- population initializer
- simulator evaluator
- selector
- reproducer
- visualizer
- logger
- trainer
- Robosuite adapter
- training CLI
- champion replay CLI
- resume-from-population support
- unit tests

The current implementation uses NumPy for the evolutionary loop. DEAP was part
of the original project idea, but it is not required by the current code yet.

## Big Picture

The project is split into small pieces on purpose. Each file has one job.

```text
                          one full training run

    +-------------------+
    | initial population|
    |  many genomes     |
    +---------+---------+
              |
              v
    +-------------------+       +----------------------+
    | evaluator         | ----> | Robosuite / MuJoCo   |
    | run each genome   |       | Panda PickPlace env  |
    +---------+---------+       +----------+-----------+
              |                            |
              | states from each episode   |
              v                            |
    +-------------------+                  |
    | fitness function  | <----------------+
    | score behavior    |
    +---------+---------+
              |
              v
    +-------------------+
    | logger/visualizer |
    | save progress     |
    +---------+---------+
              |
              v
    +-------------------+
    | selector          |
    | keep top genomes  |
    +---------+---------+
              |
              v
    +-------------------+
    | reproducer        |
    | elites + mutation |
    +---------+---------+
              |
              v
      next generation
```

Think of one genome as a long list of numbers. Those numbers are reshaped into
neural network weights. The network reads the robot state and outputs an action.

```text
flat genome
    |
    | decode
    v
+-------------------+     +-------------------+     +-------------------+
| input -> hidden   | --> | hidden activation | --> | hidden -> output  |
| weights + bias    |     | tanh              |     | weights + bias    |
+-------------------+     +-------------------+     +-------------------+
                                                        |
                                                        v
                                              robot action values
```

## Fitness Function

The fitness function defines what "good behavior" means.

For pick-and-place, the robot gets credit for five stages:

```text
reach object -> grasp object -> lift object -> move toward target -> place object
```

The default weights are intentionally strict:

| Stage | Weight | Meaning |
| --- | ---: | --- |
| Reaching | 0.05 | Is the gripper close to the object? |
| Grasping | 0.10 | Is the object held by the gripper? |
| Lifting | 0.15 | Has the object been lifted above the table? |
| Moving | 0.20 | Did the object move closer to the target after lifting? |
| Placing | 0.50 | Is the object near the target and stable? |

Placement is the most important part. It is split into:

| Placement Part | Share | Meaning |
| --- | ---: | --- |
| Accuracy | 0.60 | How close is the object to the target at the end? |
| Stability | 0.40 | Did the object stay near the target without moving too fast? |

This means a robot can still get partial credit before it fully solves the task.
That matters because early generations are usually bad. Partial rewards give
evolution something useful to select from.

## Project Layout

```text
rl_neuro_training/
  fitness.py            scores one pick-and-place episode
  genome.py             turns flat genomes into neural network weights
  population.py         creates generation zero
  evaluator.py          runs genomes through a simulator
  selector.py           chooses the best genomes
  reproducer.py         creates elites and mutated children
  visualizer.py         creates progress summaries and SVG graphs
  logger.py             writes generation stats to CSV
  trainer.py            connects everything into one training loop
  robosuite_adapter.py  adapts Robosuite to the evaluator interface

scripts/
  train_pick_place.py   command-line training program
  replay_champion.py    command-line champion replay program

tests/
  test_*.py             unit tests for every major part

runs/
  pick_place/           generated training outputs
```

## Install

Use the Python environment where you want to run MuJoCo and Robosuite.

This project has been tested with:

- Python 3.13
- NumPy 1.26.4
- Robosuite 1.5.2
- MuJoCo 3.5.0

```bash
python -m pip install numpy robosuite mujoco
```

For live MuJoCo rendering on macOS, use `mjpython` instead of normal `python`.
Robosuite installs `mjpython` with MuJoCo.

Check the main dependencies:

```bash
python -c "import robosuite, mujoco, numpy; print(robosuite.__version__, mujoco.__version__, numpy.__version__)"
```

Run the tests:

```bash
python -m unittest discover -s tests
```

## Quick Smoke Test

This tiny run is only meant to prove the pipeline works. It is too small to
teach good behavior.

```bash
python scripts/train_pick_place.py \
  --generations 1 \
  --population-size 2 \
  --hidden-size 2 \
  --max-steps 1 \
  --survivor-count 1 \
  --elite-count 1 \
  --output-dir runs/smoke
```

Replay the saved champion:

```bash
python scripts/replay_champion.py \
  --champion-genome runs/smoke/champion_genome.npy
```

The replay script automatically looks for `run_config.json` next to the genome,
so you usually do not need to pass `--hidden-size`, `--max-steps`, or
`--action-mode` again.

## Real Training

Start with something moderate:

```bash
python scripts/train_pick_place.py \
  --generations 50 \
  --population-size 50 \
  --max-steps 200 \
  --output-dir runs/pick_place
```

For live rendering:

```bash
mjpython scripts/train_pick_place.py \
  --render \
  --generations 10 \
  --population-size 20 \
  --max-steps 200 \
  --output-dir runs/pick_place
```

Rendering is useful for watching behavior, but it is slower. Headless training
is better when you want many generations.

## Replay A Champion

Headless replay:

```bash
python scripts/replay_champion.py \
  --champion-genome runs/pick_place/champion_genome.npy
```

Live replay:

```bash
mjpython scripts/replay_champion.py \
  --render \
  --champion-genome runs/pick_place/champion_genome.npy
```

## Resume Training

Training saves the final population as `final_population.npy`.

You can continue from it:

```bash
python scripts/train_pick_place.py \
  --resume-population runs/pick_place/final_population.npy \
  --generations 50 \
  --population-size 50 \
  --max-steps 200 \
  --output-dir runs/pick_place_resume
```

Important: keep the same network shape when resuming. That means the same
`--hidden-size` and `--action-mode` as the run that produced the population.

## Training Outputs

Each training run writes these files:

| File | Purpose |
| --- | --- |
| `training_log.csv` | One row per generation with best, average, median, and worst fitness. |
| `champion_genome.npy` | The best genome found across the full run. |
| `champion_metadata.json` | Human-readable score report for the champion. |
| `final_population.npy` | The final generation, useful for resume training. |
| `generation_fitness_scores.npy` | Raw fitness scores for each generation. |
| `fitness_progress.svg` | A graph of best, average, worst, and running fitness. |
| `run_config.json` | The settings needed to replay the champion correctly. |

The artifact flow looks like this:

```text
training run
    |
    +-- champion_genome.npy --------+
    |                               |
    +-- run_config.json ------------+--> replay_champion.py
    |
    +-- final_population.npy ----------> resume training
    |
    +-- training_log.csv
    +-- champion_metadata.json
    +-- generation_fitness_scores.npy
    +-- fitness_progress.svg ----------> inspect progress
```

## Main CLI Options

Training:

| Option | Meaning |
| --- | --- |
| `--generations` | How many generations to evaluate. |
| `--population-size` | How many robot brains exist per generation. |
| `--hidden-size` | Hidden neurons in the controller network. |
| `--max-steps` | Maximum simulator steps per robot attempt. |
| `--action-mode` | `end_effector` or `joint`. |
| `--survivor-fraction` | Fraction of top genomes kept as parents. |
| `--survivor-count` | Exact number of survivors. Overrides fraction. |
| `--elite-count` | Best genomes copied unchanged into the next generation. |
| `--mutation-rate` | Chance that each gene gets changed. |
| `--mutation-strength` | Typical size of each mutation. |
| `--seed` | Seed for population and reproduction randomness. |
| `--env-seed` | Optional Robosuite environment seed. |
| `--object-type` | PickPlace object, such as `can`, `milk`, `bread`, or `cereal`. |
| `--output-dir` | Where run artifacts are saved. |
| `--resume-population` | Path to a previous `final_population.npy`. |
| `--render` | Show live MuJoCo rendering. |

Replay:

| Option | Meaning |
| --- | --- |
| `--champion-genome` | Path to the saved champion genome. |
| `--run-config` | Optional explicit path to `run_config.json`. |
| `--render` | Show live MuJoCo rendering. |
| `--hidden-size`, `--max-steps`, `--action-mode` | Manual overrides if no run config exists. |

## How One Generation Works

This is the exact logic of the trainer:

```text
for each generation:
    evaluate every genome
    calculate fitness scores
    log generation stats
    remember the best genome ever seen

    if this is not the final generation:
        select the top genomes
        copy a few elites unchanged
        fill the rest with mutated survivor copies
```

There is no magic hidden inside the trainer. It only connects the parts.

## Testing

Run all tests:

```bash
python -m unittest discover -s tests
```

The tests cover:

- fitness scoring
- genome encoding and decoding
- population creation
- evaluator behavior
- selector behavior
- reproducer behavior
- logger output
- visualizer output
- trainer loop
- Robosuite adapter behavior
- training script helpers
- replay script helpers

## Troubleshooting

### Robosuite warnings about `robosuite_models`

You may see warnings like:

```text
Could not import robosuite_models
Could not load the mink-based whole-body IK
```

These warnings are not usually blocking for the Panda PickPlace task. They are
about optional Robosuite robot/model features.

### Live rendering does not open on macOS

Use `mjpython`:

```bash
mjpython scripts/replay_champion.py \
  --render \
  --champion-genome runs/pick_place/champion_genome.npy
```

### Replay says the genome length is wrong

The saved genome length depends on the network shape. Use the `run_config.json`
from the same training run, or pass the same `--hidden-size` and `--action-mode`
that were used during training.

### Training seems slow

That is expected. Each genome must run a physics simulation. Increase settings
slowly:

1. First run a smoke test.
2. Then increase `--max-steps`.
3. Then increase `--population-size`.
4. Then increase `--generations`.

## Development Notes

The code is intentionally simple and readable:

- Each module has one responsibility.
- Data is passed with small dataclasses.
- NumPy arrays represent genomes and populations.
- The evaluator only depends on a tiny simulator interface.
- Robosuite is isolated inside `robosuite_adapter.py`.

That isolation matters. It means the fitness function, selector, reproducer, and
trainer can be tested without opening MuJoCo.
