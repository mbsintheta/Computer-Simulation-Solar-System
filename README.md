# Computer Simulation Project
## Overview

This project simulates the motion of planets in a solar system using numerical integration methods. It calculates orbital motion, tracks energy conservation, detects orbital periods, and analyses the effect of timestep size on simulation accuracy.

The simulation uses Newtonian gravity and supports multiple numerical integration methods including Beeman, Euler, and Euler-Cromer.

---

## File List & Descriptions

### Body.py

Defines the `Body` class.

**Purpose:**
- Stores properties of each celestial body:
  - Mass
  - Position
  - Velocity
  - Orbital parameters
- Used by `Simulation.py` to manage celestial objects.

---

### Simulation.py

Implements the main `Simulation` class.

Reads parameters from:

```
parameters_solar.json
```

#### Key Methods

**calc_acceleration**  
Calculates gravitational acceleration between bodies.

**update_positions, update_velocities**  
Updates positions and velocities using Beeman integration (default).

**track_orbits**
- Detects orbit completion
- Detects planetary alignments
- Saves plot snapshots when alignment occurs

**calc_KE, calc_PE, calc_tot_energy**
Calculates:
- Kinetic Energy
- Potential Energy
- Total Energy

**plot_total_energy**
Plots total energy vs time.

**energy_conservation**
Calculates and displays energy conservation statistics.

**run_simulation**
- Runs the full simulation
- Displays live plots
- Tracks energy and orbits
- Prints statistics

**alternative_methods**
Allows switching integration methods:
- Beeman (default)
- Forward Euler
- Euler-Cromer

---

### main.py

Main driver script.

**Functions:**
- Initializes simulation from JSON file
- Runs simulation
- Displays animation
- Saves energy data

Energy values are written to:

```
Energy_data.csv
```

every 100 simulation steps.

---

### Further_Analysis.py

Contains function:

```python
timestep_period()
```

**Purpose:**
- Varies timestep from:

```
1.0 → 0.0001 years
```

- Runs simulation at each timestep
- Measures orbital period
- Plots:

```
Orbital Period vs Timestep (log scale)
```

---

### parameters_solar.json

Contains simulation input parameters.

Includes:

- Gravitational constant (`G`)
- Default timestep (`dt`)
- Number of iterations (`num_iterations`)
- Body definitions:
  - Name
  - Mass
  - Orbital radius
  - Color

---

## How to Run

### Default Simulation

Open terminal in project folder and run:

```bash
python main.py
```

This will:

- Initialize the solar system
- Run simulation
- Display live animation
- Generate energy plots
- Save energy data to CSV

---

### Disable or Enable Plotting

In `main.py`, modify:

```python
animation = True
energy_plot = True
```

Set to `False` to disable plotting.

---

### Run Timestep vs Period Experiment

Run:

```bash
python Further_Analysis.py
```

This will generate a log plot of orbital period vs timestep.

---

## Customisation

You can modify simulation parameters in `main.py`:

```python
sim.num_iterations = 10000
sim.timestep = 0.001
```

You can also edit:

```
parameters_solar.json
```

to:

- Add planets
- Change masses
- Change orbital radii
- Change timestep

---

## Output Files

Generated automatically:

```
Energy_data.csv
```

Optional outputs:

- Energy plots
- Orbit plots
- Alignment snapshot images

---

## Numerical Methods Implemented

- Beeman Integration (default)
- Forward Euler
- Euler-Cromer

---

## Requirements

Python 3.x

Required libraries:

```
numpy
matplotlib
json
csv
```

Install using:

```bash
pip install numpy matplotlib
```

---

## Future Improvements

Possible extensions include:

- Add moons and satellites
- Add 3D simulations
- Add more integration methods
- Improve performance
- Add interactive controls

---

## Author

Mehul Bandhu  
Student ID: s2500976  

Computer Simulation Project
