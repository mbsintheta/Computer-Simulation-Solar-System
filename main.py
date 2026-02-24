

# Name: Mehul Bandhu
# s2500976
# Computer Simulation Project

import csv
import numpy as np
from Simulation import Simulation


def main():
    """
    main method - Allows the user to choose an integration method
    at runtime, then runs the simulation accordingly and writes energy data to a CSV.
    """
    # Prompting user to pick a method
    choice = input(
        "Choose:\n1) Beeman (default)\n2) Euler\n3) Euler-Cromer\n"
    ).strip()

    # Setting flags for animation (orbital display) and energy plotting
    animation = True
    energy_plot = False

    # Loading parameters from JSON file and creating Simulation object
    sim = Simulation.read_input_data("parameters_solar.json")

    # Running simulation based on user choice
    if choice == "2":
        sim.alternative_methods("Euler", energy_plot)
    elif choice == "3":
        sim.alternative_methods("Euler-Cromer", energy_plot)
    else:
        sim.run_simulation(animation, energy_plot)

    # Writing energy data to a CSV file at intervals of 100 steps
    with open("Energy_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        body_names = [body.name for body in sim.body_list]

        header = [
            "Time (years)"] + [f"{name} Energy (J)" for name in body_names] + ["System Total (J)"]
        writer.writerow(header)

        for step in range(sim.num_iterations):
            if step % 100 == 0:
                time_now = step * sim.timestep
                energies = [body.TE[step] for body in sim.body_list]
                total_sys_energy = sum(energies)
                row = [time_now] + energies + [total_sys_energy]
                writer.writerow(row)

    print("Energy data successfully written to energy_data.csv.")


if __name__ == "__main__":
    main()
