README
------
Name: Mehul Bandhu
s2500976
Computer Simulation Project

------

1. File List & Descriptions

   • Body.py
     - Defines the Body class, which stores each planet's mass, position, velocity,
       and other orbital parameters.
     - Used by Simulation.py to create and manage individual celestial body objects.

   • Simulation.py
     - Implements the Simulation class.
     - Reads parameters from a JSON file (parameters_solar.json) via the read_input_data function.
     - Key methods include:
       - calc_acceleration: Calculates gravitational accelerations
       - update_positions, update_velocities: Updates positions/velocities (using Beeman integration by default)
       - track_orbits: Detects orbit completions and planetary alignments, and saves a plot whenever alignment occurs
       - calc_KE, calc_PE, calc_tot_energy: Computes kinetic, potential, and total energies
       - plot_total_energy: Plots total energy over time for individual planets/bodies
       - energy_conservation: Calculates and displays statistics on energy conservation
       - run_simulation: Runs the simulation and displays real-time plots, calculates energies, and prints orbit/energy stats
       - alternative_methods: Lets you pick different numerical methods (Euler, Euler-Cromer) for simulations, orbit tracking, and energy analysis
       - (Optional) Generates real-time plots and saves snapshots whenever an alignment occurs.

   • main.py
     - A main driver script that:
       - Initializes the Simulation from parameters_solar.json.
       - Runs the simulation (plot = True by default) for a set number of iterations.
       - Writes total energy values to "Energy_data.csv" every 100 steps.

   • Further_Analysis.py
     - Contains the function `timestep_period()`, which:
       - Varies the integration timestep from 1.0 down to 0.0001 years.
       - Runs the simulation at each timestep.
       - Collects the first detected orbital period for each planet.
       - Plots “Orbital Period vs. Timestep” on a log scale.

   • parameters_solar.json
     - Stores simulation parameters such as the gravitational constant (G), number of iterations,
       and properties (name, mass, orbital radius, color) for each celestial body.

2. How to Run the Code

   a. Default Simulation with Plots:
      - Open a terminal/command prompt in the folder containing main.py.
      - Run:  `python main.py`
      - The program reads parameters_solar.json, initializes the system,
        and runs for the specified number of iterations.
      - A live plot of the solar system is displayed and updated repeatedly.
      - Plots of individual planets’ total energies are also displayed (this can be optional if you already have "Energy_data.csv").
      - All energies are written to "Energy_data.csv" periodically.
      - When complete, a message confirms that the CSV file has been saved.

   b. Changing or Disabling Plotting:
      - In main.py, look for the flags that control animation (animation, energy_plot).
      - Switch them to True or False as needed (if you only want energy plots or if you don’t need any animation).

   c. Running the Timestep vs. Period Experiment:
      - Run:  `python Further_Analysis.py`
      - This script imports the Simulation class, loops over multiple timesteps,
        and plots the resulting orbital periods for each planet on a single figure.

3. Notes

   • The simulation relies on parameters_solar.json for:
     - The default timestep (dt), if not overridden in code.
     - Number of iterations (num_iterations).
     - Body definitions (e.g., mass, initial orbital radius, color).
   • If you wish to customize run duration or the timestep in main.py,
     you can modify sim.num_iterations or sim.timestep before calling run_simulation().

4. Future Work

   • This code is open to extension: you can add new planets, satellites, or alternative
     integration schemes (Forward Euler, Euler-Cromer). Just add entries to
     parameters_solar.json and/or call alternative_methods in Simulation.py.

Thank you!
