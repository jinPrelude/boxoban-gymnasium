# boxoban-gym

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-orange.svg)](https://gymnasium.farama.org/)

A high-throughput [Gymnasium](https://gymnasium.farama.org/) environment for the [DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels) puzzle dataset. Boxoban is a simplified variant of Sokoban, the classic box-pushing puzzle game, designed for reinforcement learning research.

<p align="center">
  <img src="assets/demo1.gif" width="30%" />
  <img src="assets/demo2.gif" width="30%" />
  <img src="assets/demo3.gif" width="30%" />
</p>

## Features

- **High Throughput**: Optimized for fast simulation with minimal overhead, enabling efficient RL training
- **Native Gymnasium Support**: Works seamlessly with `gym.make_vec()` for both sync and async vectorization
- **Multiple Difficulty Levels**: Includes easy, medium, hard, and unfiltered level sets with train/valid/test splits
- **Compact Observation**: 10×10×3 RGB uint8 observations for efficient batch processing
- **Interactive Play Mode**: Play levels manually with pygame to understand the puzzle mechanics

## Installation

This repository uses the [boxoban-levels](https://github.com/google-deepmind/boxoban-levels) dataset as a git submodule. **You must clone with `--recursive`**:

```bash
git clone --recursive https://github.com/YOUR_USERNAME/boxoban-gym.git
cd boxoban-gym
pip install -e .
```

If you already cloned without `--recursive`, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

### Optional: pygame for interactive play

```bash
pip install pygame
```

## Play Interactively

You can play Boxoban directly using the included `play.py` script:

```bash
python play.py --level-set hard
```

**Controls:**
- **Arrow keys / WASD**: Move the player
- **R**: Restart current level
- **N**: Next level
- **Q / ESC**: Quit

**Options:**
```bash
python play.py --help

# Examples
python play.py --level-set medium --split valid
python play.py --level-set hard --level-idx 42
python play.py --tile-size 48 --fps 30
```

## Quick Start

```python
import gymnasium as gym
import boxoban  # registers environment IDs on import

env = gym.make("Boxoban-hard-v0", disable_env_checker=True)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(0)
```

### Vectorized Environments

For high-throughput training, use Gymnasium's built-in vectorization:

```python
vec_env = gym.make_vec(
    "Boxoban-medium-train-v0",
    num_envs=128,
    vectorization_mode="sync",  # or "async"
    disable_env_checker=True,
)
```

## Registered Environment IDs

| Environment ID | Level Set | Split | Levels |
|---------------|-----------|-------|--------|
| `Boxoban-hard-v0` | hard | - | 3,332 |
| `Boxoban-medium-train-v0` | medium | train | 450,000 |
| `Boxoban-medium-valid-v0` | medium | valid | 5,000 |
| `Boxoban-unfiltered-train-v0` | unfiltered | train | 900,000 |
| `Boxoban-unfiltered-valid-v0` | unfiltered | valid | 10,000 |
| `Boxoban-unfiltered-test-v0` | unfiltered | test | 1,000 |

## Environment Details

### Observation Space
- **Shape**: `(10, 10, 3)` — 10×10 grid with RGB channels
- **Dtype**: `np.uint8`

### Action Space
- **Type**: `Discrete(4)`
- **Actions**: `0=up`, `1=down`, `2=left`, `3=right`

### Rewards
| Event | Reward |
|-------|--------|
| Each step | `-0.1` |
| Box pushed onto target | `+1.0` |
| Box pushed off target | `-1.0` |
| All boxes on targets (solved) | `+10.0` |

### Episode Termination
- **Terminated**: All boxes are on targets (puzzle solved)
- **Truncated**: Maximum steps reached (default: 120)

## Dataset Path Resolution

The environment locates level files using this priority:

1. `level_root` argument passed to `gym.make(...)`
2. `BOXOBAN_LEVELS_DIR` environment variable
3. `./boxoban-levels` (current working directory)
4. Repository-relative `boxoban-levels` (editable install workflow)

## Benchmarks

Measure throughput on your hardware:

```bash
python benchmarks/throughput.py --env-id Boxoban-medium-train-v0 --num-envs 128 --steps 2000
```

Example output:
```
single env-steps/s: 85,000
sync   env-steps/s: 420,000
async  env-steps/s: 380,000
```

## Acknowledgments

- Level data from [DeepMind Boxoban Levels](https://github.com/google-deepmind/boxoban-levels) (Apache 2.0 License)
- Original Sokoban game by Hiroyuki Imabayashi (1981)

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
