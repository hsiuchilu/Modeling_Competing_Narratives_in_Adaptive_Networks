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
