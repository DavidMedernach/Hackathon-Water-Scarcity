# Water Management Simulation

This repository implements an agent-based evolutionary game to study water resource allocation under varying environmental and policy scenarios. It models multiple actors who decide whether to cooperate or defect in water usage, tracks ecological and economic impacts, and provides tools for running simulations and visualizing results.

## Usage

### Single Scenario (Notebook)
Open `single_scenario.ipynb` to run and customize one simulation interactively.

### Multi-Scenario Analysis (Notebook)
Open `multi_scenarios.ipynb` for comparative visualizations and advanced analysis.

## Installation

You can install dependencies either via `pip` (using `requirements.txt`) or using Poetry.

### Option 1: Using Poetry (recommended)

1. **Install Poetry**  
   Follow instructions at https://python-poetry.org/docs/#installation

2. **Clone the repository**  
   ```bash
   git clone https://github.com/DavidMedernach/Hackathon-Water-Scarcity.git
   cd Hackathon-Water-Scarcity
   ```

3. **Install dependencies**  
   ```bash
   poetry install
   ```

4. **Activate virtual environment**  
   ```bash
   poetry shell
   ```

### Option 2: Using pip

1. **Clone the repository**  
   ```bash
   git clone https://github.com/DavidMedernach/Hackathon-Water-Scarcity.git
   cd Hackathon-Water-Scarcity
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

> **Note:** We recommend committing `poetry.lock` for reproducibility, but you can regenerate it from `pyproject.toml` if needed.

## Project Structure

```
├── parameters/
│   ├── data.csv               # Real riverflow time series
│   └── scenarios/             # YAML parameter files for scenarios
├── src/
│   ├── core.py                # Main WaterManagementSimulation class
│   ├── actors.py              # ActorManager: decision-making & learning
│   ├── water_allocation.py    # WaterAllocator: pumping & quota logic
│   ├── ecology.py             # EcologyManager: flow & impact calculations
│   ├── utils.py               # Helper functions (e.g., YAML loader)
│   ├── plot_analysis.py       # Time-series plots for individual runs
│   ├── scenarios.py           # Script to batch-run scenarios
│   └── plot_multi_analysis.py # Impact trade-off & correlation plots
├── single_scenario.ipynb      # Interactive demo for one scenario
├── multi_scenarios.ipynb      # Comparative analysis notebook
├── pyproject.toml             # Poetry project configuration
├── poetry.lock                # Locked dependency versions (recommended)
├── requirements.txt           # Python dependencies (alternative)
└── README.md                  # Project overview and usage guide
```
