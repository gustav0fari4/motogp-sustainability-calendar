# MotoGP Sustainability Calendar Optimisation (Python)

Python project that models MotoGP season logistics and applies **metaheuristic optimisation** to improve the race calendar with the goal of reducing total travel distance while respecting operational constraints (e.g., temperature limits and summer shutdown).

The project includes:
- A travel-distance simulator (Haversine-based)
- Constraint checks (temperature, shutdown period, calendar validity)
- Optimisation approaches:
  - **Simulated Annealing** (`simanneal`)
  - **Genetic Algorithm** (`deap`)
  - **Particle Swarm Optimisation** (`pyswarms`)
- **Unit tests** (`unittest`) to validate correctness

---

## Features
- Reads race weeks and track/location data from CSV files
- Calculates season travel distance assuming a “home” track
- Applies penalty-based constraints:
  - Temperature range per race week (min/max)
  - Summer shutdown weeks (no races in weeks 29–31)
  - No duplicate/missing race weeks
  - Valencia fixed on its original week
  - No triple headers (3+ consecutive race weekends)
- Compares optimisation strategies and prints the best calendar found

---

## Tech Stack
- Python 3.x
- `numpy`
- `pyswarms`
- `simanneal`
- `deap`

---

## Project Structure
- `motogp-calendar-base.py` — main script (simulation + optimisation + tests)
- `race-weekends.csv` — baseline race week numbers (input)
- `track-locations.csv` — track names + coordinates + weekly temperatures (input)

> Important: keep the CSV files in the **same folder** as the `.py` file.

---

## Setup

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

## Conda setup (recommended)
```bash
conda env create -f environment.yml
conda activate numerical-optimisation
python motogp-calendar-base.py
