# Firm Dynamics ABM

## Overview

This project presents an agent-based model (ABM) that simulates firm dynamics within a market environment. It builds upon the foundational work of Robert Axtell's research on firm dynamics, exploring the intricate interactions between workers and firms, and focusing on the processes of employment dynamics, production, and distribution within a simulated economy.

The motivation behind this project stems from Robert Axtell's seminal work on firm dynamics, which provides a comprehensive exploration of how firms grow and interact within economic systems. This project seeks to extend Axtell's model by incorporating insights from Ole Peters and the Ergodicity Economics research program. Specifically, it aims to examine the implications of shifting the primary worker maximand from utility to the growth rate of wealth.

## Features

- **Simulation of Firm Dynamics**: Models the lifecycle of firms and their interactions with workers in a competitive market, with a unique focus on the growth rate of wealth as the primary worker maximand.
- **Data Analysis**: Utilizes analytical techniques to derive insights into market dynamics, firm growth patterns, and the impact of wealth growth maximization on employment and economic stability.
- **Visualization**: Employs various visualization tools to represent the outcomes of simulations, facilitating a clearer understanding of complex economic interactions and dynamics.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/firm_dynamics.git
cd firm_dynamics
pip install -r requirements.txt
```

## Usage

To run the simulation, navigate to the src directory and execute the main script:

```bash
python main.py
```

For analyzing the simulation results, scripts located in analysis directory can be used:

```bash
python analysis/analyze_results.py
```

## Documentation
Generate the documentation using Sphinx:

```bash
cd docs
make html
```

The generated HTML documentation can be found in docs/build/html/index.html.

## Contributing
Contributions to the Firm Dynamics ABM project are welcome. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to make contributions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
