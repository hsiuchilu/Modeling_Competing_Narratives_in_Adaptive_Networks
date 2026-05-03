# Modeling Competing Narratives in Adaptive Networks

This repository provides the simulation code, data-processing scripts, aggregated outputs, and figure-generation notebook for the manuscript:

**Modeling Competing Narratives in Adaptive Networks: How Social Pressure and Network Dynamics Drive Tipping and Persistence**

The model studies how two competing narratives diffuse through an adaptive social network. Agents hold varying levels of conviction toward two narratives, update their conviction through local social feedback, accept and propagate narratives under time-varying collective attention, and rewire social ties based on disagreement. The model also includes committed seeders, or **zealots**, who persistently promote their assigned narrative.

The framework is designed to capture generic dynamics of competing narrative diffusion under social reinforcement, adaptive exposure, and shifting collective attention. It does not directly model a “true versus false” contest, credibility judgments, platform ranking algorithms, or correction mechanisms. Disinformation campaigns are treated as an application context in which strategic seeding and identity-consistent reinforcement may be especially salient.

---

## Repository Structure

```text
.
├── model.py
├── execution.py
├── Data Processing and Plotting/
│   ├── aggregate_data.py
│   ├── Plotting.ipynb
│   └── data/
└── README.md
```

### Core simulation files

- `model.py`  
  Contains the core agent-based model functions for one narrative diffusion event. This includes the opinion-climate calculation, narrative adoption probability, disconnection probability, rewiring logic, and the main diffusion procedure.

- `execution.py`  
  Provides the execution logic for running repeated simulations under specified parameter settings. It initializes the network and conviction matrix, assigns zealots, calls functions from `model.py`, and records simulation outputs.

### Data-processing and plotting files

- `Data Processing and Plotting/aggregate_data.py`  
  Documents the processing pipeline used to aggregate raw simulation outputs into lightweight `.csv` files. These aggregated files are used for figure generation and statistical summaries.

- `Data Processing and Plotting/Plotting.ipynb`  
  Generates the manuscript figures from the provided aggregated `.csv` files.

- `Data Processing and Plotting/data/`  
  Contains pre-computed aggregated data used by `Plotting.ipynb` to reproduce the main figures without rerunning the full simulation pipeline.

---

## Model Overview

The model simulates a population of agents embedded in an undirected adaptive network. Each agent has a conviction matrix:

```text
Q[i, 0] = conviction toward narrative A
Q[i, 1] = conviction toward narrative B
```

At each propagation event:

1. A focal narrative is selected.
2. A spreader shares the narrative with neighboring agents.
3. Neighboring agents decide whether to accept and propagate the narrative.
4. The spreader updates their conviction based on the local opinion climate.
5. Agents who reject the narrative may disconnect from the spreader.
6. Disconnected agents immediately form a new tie, either through a friend-of-friend mechanism or random reconnection.
7. The diffusion event continues until no new agents remain to propagate the focal narrative.

The simulation tracks narrative adoption/exposure, network modularity, assortativity in conviction differences, zealot degree, and degree-distribution statistics.

---

## Key Parameters

| Parameter | Meaning | Default / Range |
|---|---|---|
| `N` | Number of agents | 500 in the manuscript simulations |
| `alpha` / `α` | Agent sensitivity to social pressure | `[0, 1]` |
| `w` | Network adaptability; controls the overall speed of disconnection and rewiring | `[0, 1]` |
| `b1` | Hardness/softness parameter in the narrative adoption function; used to model collective attention | default = 3.5 |
| `b2` | Hardness/softness parameter in the disconnection function | default = 1 |
| `h` | Homophily parameter for friend-of-friend reconnection | default = 2 |
| `c` | Probability of reconnecting to a friend of a friend rather than a random agent | default = 0.2 |
| `gamma` / `γ` | Ratio of zealots | default = 2% unless otherwise specified |
| `tau` / `τ` | Exposure/adoption rate; proportion of agents exposed to or accepting the focal narrative | `[0, 1]` |

---

## Replication Pipeline

The replication workflow has three stages. Readers who only want to reproduce the manuscript figures can begin directly with **Step 3**, because aggregated `.csv` files are already provided.

### Step 1: Run Simulations

Run `execution.py` to generate raw simulation outputs.

```bash
python execution.py
```

The default run executes a simulation under the parameter settings specified in `execution.py`. To reproduce the full parameter sweeps used in the manuscript, modify the parameter grid in `execution.py` and rerun the simulation.

Full manuscript simulations can generate a large number of raw trajectory files and may require substantial computation time and storage. For convenience, the aggregated outputs used for figure generation are included in the repository.

### Step 2: Aggregate Raw Outputs

The script `aggregate_data.py` documents how raw simulation outputs were processed into aggregated `.csv` files.

```bash
cd "Data Processing and Plotting"
python aggregate_data.py
```

This step calculates fitted diffusion curves, network metrics, and summary matrices used in the manuscript figures.

Because the full raw simulation outputs are large, the repository provides the processed aggregated files in:

```text
Data Processing and Plotting/data/
```

Readers who only want to reproduce the manuscript figures can skip Step 1 and Step 2.

### Step 3: Reproduce Manuscript Figures

Open and run:

```text
Data Processing and Plotting/Plotting.ipynb
```

The notebook reads the aggregated `.csv` files in `Data Processing and Plotting/data/` and reproduces the simulation-based figures reported in the manuscript.

---

## Notes on Figure Reproduction

- **Figure 1** in the manuscript is a conceptual flowchart of the model procedure and is not generated from simulation output.
- The simulation-based figures are reproduced from the aggregated data and plotting notebook.
- Some panels that display individual simulation trajectories are based on selected runs from stochastic simulations.
- Minor visual differences may occur because of software versions, interpolation settings, or random sampling of illustrative trajectories. These differences do not affect the aggregated results or substantive conclusions.

---

## Data Availability

The repository includes lightweight aggregated `.csv` files sufficient for reproducing the manuscript figures. The full raw simulation trajectories are substantially larger and can be regenerated using `execution.py` by running the corresponding parameter sweeps.

---

## Requirements

The code was developed in Python and uses standard scientific-computing packages, including:

```text
numpy
pandas
networkx
scipy
matplotlib
seaborn
scikit-learn
jupyter
```

Jupyter Notebook is required to run `Plotting.ipynb`.

---

## Citation

If you use this code or model, please cite the corresponding manuscript:

Lu, H.-C., & Lee, H.-W.  
**Modeling Competing Narratives in Adaptive Networks: How Social Pressure and Network Dynamics Drive Tipping and Persistence.**

---

## Contact

For questions about the code or replication materials, please contact:

Hsiu-Chi Lu  
Network Science Institute, Northeastern University  
lu.hsiu@northeastern.edu
