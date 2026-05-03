# Replication Package: Modeling Competing Narratives in Adaptive Networks

This repository contains the code and data necessary to reproduce the findings and figures from the manuscript: **"Modeling Competing Narratives in Adaptive Networks: How Social Pressure and Network Dynamics Drive Tipping and Persistence"** (*Humanities and Social Sciences Communications*).

## Repository Structure

To ensure both deep methodological transparency and ease of review, this repository is organized into core scripts for simulation, data processing, and visualization.

* `model.py`: Defines the core agent-based network model and network dynamics.
* `execution.py`: The main script to run the full simulations across all parameter spaces.
* `aggregate_data.py`: Processes the raw simulation outputs, performs statistical curve fitting, and aggregates the results into lightweight matrices.
* `Figure_Generation.ipynb`: A Jupyter Notebook containing all the code needed to instantly reproduce the figures as they appear in the manuscript.
* `data/`: A directory containing the pre-computed, aggregated `.csv` datasets required for generating the figures.

## Requirements

The code is written in Python 3. To run the scripts and the Jupyter Notebook, the following standard packages are required:
* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `seaborn`
* `jupyter` (for viewing/running the notebook)

## Replication Instructions

We have structured the replication pipeline into three distinct stages to accommodate both comprehensive methodological audits and quick figure reproduction.

### Step 1: Data Generation (Simulation)
**Code:** `model.py` and `execution.py`

To generate the raw data from scratch, researchers can run `execution.py`. This script executes the full network models across thousands of parameter combinations.
> **Note on Data Availability:** Due to the massive storage size of the raw trajectory files (thousands of `.npy` files), they are not hosted directly in this repository. Running this step from scratch requires significant computational time (several days) and local storage.

### Step 2: Data Aggregation (Processing)
**Code:** `aggregate_data.py`

This script processes the raw `.npy` outputs generated in Step 1. It performs the generalized logit regression fittings, calculates metrics (e.g., modularity, assortativity, tipping points), and outputs lightweight aggregated matrices (in `.csv` format) into the `data/` folder.
> *Reviewers who only wish to verify the final visual outputs can skip Step 1 and Step 2, as the output of this script is already provided in the `data/` folder.*

### Step 3: Visualization (Quick Reproduction)
**Code:** `Figure_Generation.ipynb`

For maximum convenience, all figures in the manuscript can be reproduced instantly without running any heavy computations. Simply open and run `Figure_Generation.ipynb`. This notebook reads the pre-computed `.csv` files from the `data/` directory and generates Figure 1, 4, 5, 6, and the complex phase diagrams. You can also view the rendered notebook and figures directly on GitHub.

---

## Important Notes on Reproducibility

**A Note on Reproducing Figure 1 (Panels b-e):**
Please note that the four smaller subplots (panels b, c, d, and e) in Figure 1 display randomly selected individual trajectories from the simulation runs to illustrate system states. In the original manuscript submission, a fixed random seed was not explicitly set prior to this specific sampling step. 

As a result, when executing the plotting script, the specific trajectory examples shown in these subpanels may differ slightly from the static image in the published manuscript. However, we have now implemented a fixed `np.random.seed()` in the provided notebook to ensure strict future reproducibility. The overarching patterns, the statistical fits, and the main conclusions drawn from these representative trajectories remain completely consistent with the manuscript.