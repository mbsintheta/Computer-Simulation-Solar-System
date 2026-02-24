

# Name: Mehul Bandhu
# s2500976
# Computer Simulation Project


import numpy as np
import matplotlib.pyplot as plt
from Simulation import Simulation


def timestep_period():
    """
    Demonstrates how the first-detected orbital periods of planets vary
    as a function of the chosen timestep. 
    Runs multiple simulations with timesteps ranging from 1.0 to 0.0001 years,
    each covering a total duration of 200 years, and plots the resulting
    orbital periods.
    """
    # Generating a sequence of timesteps from 1 down to 0.0001 (40 points)
    timesteps = np.linspace(1, 0.0001, 40)

    # Temporary simulation to discover planet names (excluding Sun)
    temp_sim = Simulation.read_input_data('parameters_solar.json', False)
    body_names = [
        body.name for body in temp_sim.body_list if body.name.lower() != 'sun']
    n_bodies = len(body_names)

    # array to store each planet's reported orbit time at each timestep
    time_periods = np.zeros((len(timesteps), n_bodies))

    # By default, animation and energy plotting are turned off for speed
    animation, energy_plot = False, False

    # Main loop over timesteps
    for i, dt in enumerate(timesteps):
        # Number of iterations so that total simulated time is approx. 200 years
        iterations = int(200 / dt)

        # Creates a fresh Simulation object, override timestep and iteration count
        sim = Simulation.read_input_data('parameters_solar.json', False)
        sim.timestep = dt
        sim.num_iterations = iterations

        # Running the simulation with no animation or energy plots
        sim.run_simulation(animation, energy_plot)

        # Accessing 'orbit_time_reported' for each planet
        col_idx = 0
        for body in sim.body_list:
            if body.name.lower() != 'sun':
                time_periods[i, col_idx] = body.orbit_time_reported
                col_idx += 1

    # Plotting results
    plt.figure()
    for j, name in enumerate(body_names):
        # For planets that never completed an orbit (time <= 0), plot as NaN
        y = time_periods[:, j].copy()
        y[y <= 0] = np.nan
        plt.plot(timesteps, y, linewidth=1.5,
                 marker='o', markersize=2, label=name)

    plt.xlabel("Timestep (years)")
    plt.ylabel("Orbital Period (years)")
    plt.yscale('log')
    plt.title("Orbital Period vs. Timestep")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    timestep_period()
