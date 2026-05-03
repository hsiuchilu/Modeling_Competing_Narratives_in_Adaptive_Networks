# Disinformation Resilience Simulation

These are the simulated models for the paper: **"Modeling Disinformation Resilience: How Social Pressure and Network Dynamics Drive and Sustain the Spread of False Narratives."** *(Details about the experiments and the paper’s content will be disclosed upon acceptance.)*

## The structure

* `model.py`: The script contains the code for one round of diffusion simulation and the required mathematical formulas.
* `execution.py`: The script contains the execution logic of the model and the output of the simulation. It assigns the parameters for a single run given the user settings and calls the simulation functions from `model.py`.
* `Data_Processing_and_Plotting/`: A dedicated directory containing the scripts required to process the raw simulation data and reproduce all figures presented in the manuscript.
    * `aggregate_data.py`: Processes the massive raw `.npy` trajectory files, performs regression fittings, and aggregates the results into lightweight `.csv` matrices.
    * `Figure_Generation.ipynb`: A Jupyter Notebook that reads the aggregated `.csv` files to instantly generate the final figures.
    * `data/`: A sub-directory storing the pre-computed aggregated `.csv` datasets.

## Usages & Replication Pipeline

Our replication pipeline is structured into three distinct stages to accommodate both deep methodological audits and quick figure reproduction.

### Step 1: Data Generation (Simulation)
You can run `execution.py` to generate the raw simulation data. 
* The default run of `execution.py` will output the single run result for $w=0.1$, $\alpha=0.05$, $b1=3$, $b2=3$, $n\_mali\_act\_1=0$, $n\_mali\_act\_0=0$, $c=0.2$.
* For other parameter settings across the entire parameter space, please modify the `params` dictionary list at the `simulate_information_cascade` function.

> **Note on Data Availability:** Running the full manuscript simulations generates thousands of raw trajectory files (`.npy`), requiring significant computational time and storage. To save reviewers' time, we have provided the pre-computed aggregated data for the following steps.

### Step 2: Data Aggregation (Processing)
Navigate to the `Data_Processing_and_Plotting/` directory and run `aggregate_data.py`. This script processes the raw `.npy` outputs generated in Step 1, calculates network metrics, and outputs lightweight aggregated matrices (in `.csv` format) into the `Data_Processing_and_Plotting/data/` folder.
> *Reviewers who only wish to verify the final visual outputs can skip Step 1 and Step 2, as the output of this script is already provided.*

### Step 3: Visualization (Quick Reproduction)
For maximum convenience, all figures in the manuscript can be reproduced instantly without running any heavy computations. 
Simply open and run `Figure_Generation.ipynb` located in the `Data_Processing_and_Plotting/` folder. This notebook reads the pre-computed `.csv` files and generates all main text figures (Figures 1, 4, 5, 6, and the phase diagrams) exactly as they appear in the manuscript. You can also view the rendered notebook directly on GitHub.

---

### Important Note on Figure 1 (Random Sampling)
Please note that panels (b) through (e) in Figure 1 display individual trajectory examples randomly sampled from the simulation runs. In the original manuscript submission, a fixed random seed was not explicitly set prior to this specific sampling step. 

To ensure strict future reproducibility, we have implemented a fixed `np.random.seed()` within the provided Jupyter Notebook. As a result, the specific trajectory examples rendered by the notebook may differ slightly from the static image in the published manuscript, but the statistical fits, overall patterns, and main conclusions remain completely identical.
