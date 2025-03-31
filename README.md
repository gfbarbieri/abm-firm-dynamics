# Agent-Based Model of Firm Dynamics

This repository contains an agent-based model that explores the emergence of increasing returns to scale (IRS) at the firm level while maintaining constant returns to scale (CRS) in the aggregate. The model demonstrates how this pattern naturally arises from risk-sharing under multiplicative dynamics.

## Overview

The model simulates a labor market where:
- Workers have multiplicative wealth dynamics characterized by expected return rates ($\mu$) and volatility ($\sigma$)
- Firms form as cooperative groups that share output and pool risk
- Workers search for employment opportunities that maximize their growth potential
- Firms evaluate potential hires based on their impact on firm-level growth rates

## Key Features

- **Risk-Sharing Mechanism**: Firms reduce volatility through diversification, creating IRS at the firm level
- **Endogenous Firm Formation**: Firms emerge and stabilize based on optimal risk-sharing configurations
- **Convergence Properties**: Model naturally converges to a state where no mutually beneficial trades exist
- **Flexible Configuration**: Supports various parameter settings for activation rates, search behavior, and correlation structures

## Model Structure

The model consists of three main components:

1. **Workers** (`worker.py`):
   - Characterized by $\mu$ (expected return) and $\sigma$ (volatility)
   - Search for employment opportunities
   - Produce output based on multiplicative dynamics

2. **Firms** (`firm.py`):
   - Employ workers and facilitate production
   - Evaluate potential hires based on growth rate impact
   - Distribute output among workers

3. **Market** (`market.py`):
   - Manages interactions between workers and firms
   - Handles job search and matching process
   - Tracks convergence through mutual trade checking

## Installing Dependencies

To install with Poetry, run:

```bash
poetry install
```

To install with pip, run:

```bash
pip install -r requirements.txt
```

To create a requirements.txt file using Poetry:

```bash
poetry export --without-hashes --format=requirements.txt > requirements.txt
```

To create a virtual environment before using pip:

```bash
python -m venv venv
source venv/bin/activate
```

## Usage

```python
from src.models.market import Market

# Initialize model with parameters. Without num_steps, the model will
# run until convergence.
model = Market(
    num_agents=100,
    activation=0.1,
    mutual_acceptance=True,
    global_search_rate=1.0
)

# Run until convergence
model.run_model()
```

## Key Parameters

- `num_agents`: Number of workers in the model
- `activation`: Rate at which workers search for new jobs
- `mutual_acceptance`: Whether both worker and firm must agree to a match
- `global_search_rate`: Rate at which workers search outside their network
- `constant_mu`/`constant_sigma`: Optional fixed values for worker characteristics
- `track_wealth`: Whether to update worker wealth over time
- `correlation_matrix`: Configuration for worker correlation structure

## Data Collection

The model collects data on:
- Firm size distribution
- Worker mobility
- Growth rates
- Job creation/destruction
- Output production

## Requirements

- Python 3.x
- Mesa (Agent-Based Modeling framework)
- NumPy

## License

[Your chosen license]

## Citation

If you use this model in your research, please cite:
[Your citation information]
