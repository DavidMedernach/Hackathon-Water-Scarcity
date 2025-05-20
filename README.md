# Hackathon on Water Scarcity 2025 - Baseline Model Repository

This repository offers a streamlined toolkit for training baseline models to predict river discharge and generating submission files for the [Hackathon on Water Scarcity 2025](https://www.codabench.org/competitions/4335). The baseline model forecasts water discharge at 52 evaluation stations using meteorological data, soil composition, hydrographic catchment characteristics, and other relevant features. You're encouraged to explore alternative modeling approaches or tailor models per station, provided your final submission conforms to the Codabench format.

## Installation

### Using Poetry

```bash
# Clone the repository
git clone https://github.com/DavidMedernach/Hackathon-Water-Scarcity.git
cd Hackathon-Water-Scarcity

# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

### Using requirements.txt

```bash
pip install -r requirements.txt
```

## Data

- **Download:**  
  Acquire the dataset from [Zenodo](https://zenodo.org/records/14826458).  
- **Setup:**  
  Unzip and place the dataset folder in the repository root.

## Notebook Workflow

0. **Preprocessing**  
   - `01 - Data Preprocessing.ipynb`  
   - `02 - Feature Engineering.ipynb`  

1. **Model Training & Submission**  
   - `03 - Modelisation.ipynb`  
   - `04 - Prediction Computation.ipynb`  

2. **Exploration & Analysis**  
   - `05 - Performance Comparison.ipynb`  
   - `06 - Single Model Optimisation.ipynb`  

## Submission

After executing the notebooks, package your predictions in `data/evaluation/predictions.zip` and submit via [Codabench](https://www.codabench.org/competitions/4335).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
