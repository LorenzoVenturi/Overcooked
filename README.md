# MAPPO and MAA2C comparison and intra-algorithm evaluation

This project has been developed during the summer of 2025 for the Autonomous and Adaptive Systems course at the University of Bologna. We present a systematic comparison between two state of the art on-policy multi-agent reinforcement learning algorithms, MAPPO and MAA2C, within the Overcooked-AI benchmark. By implementing both algorithms from scratch under a shared CTDE framework, we ensured a fair and transparent evaluation across three axes: absolute performance, generalization, and an interesting and potentially key method that we could define as cross-algorithm coordination, to evaluate robustness.

## Gameplay Demo

![Gameplay Demo](rec.mov)

### All the experiments are inside MARL_comparison.ipynb.

## Quick Setup

1. **Prerequisites**
   - **Python 3.10** (required for overcooked-ai compatibility - versions 3.11+ will not work)
   
   **Installing Python 3.10:**
   
   **Mac users:**
   ```bash
   # Using Homebrew
   brew install python@3.10
   
   # Create virtual environment with Python 3.10
   python3.10 -m venv overcooked-env
   source overcooked-env/bin/activate
   ```
   
   **All platforms:**
   ```bash
   conda create -n overcooked python=3.10 -y
   conda activate overcooked
   ```

2. **Clone repository**
   ```bash
   git clone https://github.com/LorenzoVenturi/Overcooked.git
   cd Overcooked
   git submodule update --init --recursive
   ```

3. **Install Python dependencies**
   ```bash
   # Core libraries for this project
   pip install torch pygame numpy matplotlib seaborn
   
   # Install the overcooked_ai environment
   cd overcooked_ai
   pip install -e .
   cd ..
   ```

4. **Verify installation**
   ```bash
   # Test that everything works
   python utils/simple_render.py --list-models
   ```

## Training Agents

Train new models from scratch using the terminal:

```bash
# List available layouts for training
python utils/train_agent.py --list-layouts

# Train MAA2C on single layout
python utils/train_agent.py --algorithm MAA2C --model-name my_cramped_model --layout cramped_room --batches 100

# Train MAPPO on multiple layouts
python utils/train_agent.py --algorithm MAPPO --model-name my_multi_model --layout cramped_room --layout coordination_ring --batches 200

# Train on all layouts (generalization)
python utils/train_agent.py --algorithm MAA2C --model-name my_generalized_model --all-layouts --batches 150

# Advanced training with custom parameters
python utils/train_agent.py --algorithm MAPPO --model-name advanced_model --layout forced_coordination --batches 300 --episodes-per-batch 15 --iters 12
```

### Training Parameters
- `--algorithm` or `-a`: `MAA2C` or `MAPPO`
- `--model-name` or `-m`: Name for your trained model
- `--layout` or `-l`: Layout(s) to train on (repeatable)
- `--all-layouts`: Train on all available layouts
- `--batches` or `-b`: Number of training batches (default: 100)
- `--episodes-per-batch` or `-e`: Episodes per batch (default: 10)
- `--iters` or `-i`: MAPPO iterations (default: 10)
- `--device` or `-d`: `cpu` or `cuda` (default: cpu)

## Testing existing trained models

Test your models or existing pre-trained ones:

```bash
# List available models and layouts
python utils/simple_render.py --list-models

# Test pre-trained models (already available)
python utils/simple_render.py --algorithm MAPPO --model cramped_room_selfplay --layout cramped_room
python utils/simple_render.py --algorithm MAA2C --model asymmetric_advantages_selfplay --layout asymmetric_advantages

# Test your own trained model (after training with train_agent.py)
python utils/simple_render.py --algorithm MAA2C --model my_cramped_model --layout cramped_room
```

## Rendering Options

### Required Arguments (for rendering)
- `--algorithm` or `-a`: `MAA2C` or `MAPPO`
- `--model` or `-m`: Model name to load  
- `--layout` or `-l`: Layout to render

### Optional Arguments
- `--episodes` or `-e`: Number of episodes (default: 1)
- `--frame-rate` or `-f`: Frame rate (default: 10)
- `--device` or `-d`: `cpu` or `cuda` (default: cpu)
- `--list-models`: List all available models and layouts

## Examples

**List models:**
```bash
python utils/simple_render.py --list-models
```

**Basic rendering (using pre-trained models):**
```bash
python utils/simple_render.py -a MAA2C -m cramped_room_selfplay -l cramped_room
python utils/simple_render.py -a MAPPO -m coordination_ring_selfplay -l coordination_ring
```

**Multiple episodes:**
```bash
python utils/simple_render.py -a MAA2C -m multi_layout_generalized5 -l forced_coordination -e 3 -f 15
```

## Available Models and Layouts

### MAA2C Models
`cramped_room_selfplay`, `coordination_ring_selfplay`, `forced_coordination_selfplay`, `asymmetric_advantages_selfplay`, `multi_layout_generalized1-5`

### MAPPO Models  
`cramped_room_selfplay`, `coordination_ring_selfplay`, `forced_coordination_selfplay`, `asymmetric_advantages_selfplay`

### Layouts
`cramped_room`, `coordination_ring`, `forced_coordination`, `asymmetric_advantages`

## Workflow Examples

**Complete Training → Testing Workflow:**
```bash
# 1. Train a new model
python utils/train_agent.py --algorithm MAA2C --model-name my_custom_model --layout cramped_room --batches 150

# 2. Test the trained model
python utils/simple_render.py --algorithm MAA2C --model my_custom_model --layout cramped_room --episodes 5

# 3. Compare with pre-trained model
python utils/simple_render.py --algorithm MAA2C --model cramped_room_selfplay --layout cramped_room --episodes 5
```

**Multi-layout Generalization:**
```bash
# Train on multiple layouts for better generalization
python utils/train_agent.py --algorithm MAPPO --model-name generalized_mappo --all-layouts --batches 200

# Test generalization on different layouts
python utils/simple_render.py --algorithm MAPPO --model generalized_mappo --layout cramped_room
python utils/simple_render.py --algorithm MAPPO --model generalized_mappo --layout coordination_ring
```