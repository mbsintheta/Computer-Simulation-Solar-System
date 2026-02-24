

# Name: Mehul Bandhu
# s2500976
# Computer Simulation Project

import numpy as np
import json
import matplotlib.pyplot as plt
from math import sqrt, atan2, pi
from Body import Body


class Simulation:
    """Holds all simulation data and methods related to orbital motion."""

    def __init__(self, timestep, num_iterations, grav_const, body_list, alignment):
        """Initializes key simulation parameters and keeps track of alignment state."""
        self.timestep = timestep
        self.num_iterations = num_iterations
        self.grav_const = grav_const
        self.body_list = body_list
        self.alignment = alignment

    def read_input_data(json_file):
        """Reads JSON parameters (time step, iterations, bodies, G) and returns a Simulation object."""
        # accessing the JSON file for parameters, bodies
        with open(json_file) as f:
            params = json.load(f)
        dt = params["timestep"]
        num_iterations = params["num_iterations"]
        G = params["grav_const"]
        body_list = []
        sun_mass = None

        # Identifying Sun's mass to help compute initial velocities
        for b in params["bodies"]:
            if b["name"].lower() == "sun":
                sun_mass = b["mass"]
                break

        # Create each Body object from data
        for b in params["bodies"]:
            name = b["name"]
            colour = b["colour"]
            mass = b["mass"]
            orbital_radius = b["orbital_radius"]
            position = [orbital_radius, 0.0]
            if orbital_radius > 0 and sun_mass is not None:
                # Assigning initial velocities
                v_initial = sqrt(G * sun_mass / orbital_radius)
                velocity = [0.0, v_initial]
            else:
                velocity = [0.0, 0.0]
            current_acc = [0.0, 0.0]
            previous_acc = [0.0, 0.0]

            # creating separate body objects for the bodies in JSON file, with tha parameters from JSON
            new_body = Body(name, colour, mass, position,
                            velocity, current_acc, previous_acc)
            body_list.append(new_body)
            alignment = False  # checks whether the inner planets are aligned

        # returns Simulation object
        return Simulation(dt, num_iterations, G, body_list, alignment)

    def calc_acceleration(self, update=True):
        """Computes gravitational accelerations for all bodies. """
        new_acc = []  # store the new/updated acceleration at each step

        # nested for loops to calculate the accelerations on each body due to all bodies
        for i, body_i in enumerate(self.body_list):
            a_i = np.zeros_like(body_i.position)
            for j, body_j in enumerate(self.body_list):
                if i != j:
                    r_ij = body_i.position - body_j.position
                    dist = np.linalg.norm(r_ij)
                    if dist > 0:
                        # Adding accelerations from each other body
                        a_i += -self.grav_const * \
                            body_j.mass * r_ij / (dist**3)
            new_acc.append(a_i)

        # if true, this block updates the previous and current accelerations of all bodies
        # this is only used for the first step, and after this, the accelerations are updated from 'update_velocities'
        if update:
            for i, body in enumerate(self.body_list):
                body.prev_acc = body.current_acc
                body.current_acc = new_acc[i]
        return new_acc

    def update_positions(self):
        """Updates positions using the Beeman integration position formula."""
        dt = self.timestep
        for body in self.body_list:
            body.position += (
                body.velocity * dt
                + (1/6) * (4*body.current_acc - body.prev_acc) * (dt**2)
            )

    def update_velocities(self):
        """Updates velocities using the Beeman integration velocity formula."""
        dt = self.timestep
        # returns the updated acceleration for a given step
        next_acc_list = self.calc_acceleration(update=False)
        for i, body in enumerate(self.body_list):
            body.velocity += (
                (1.0/6.0)
                * (2.0*next_acc_list[i] + 5.0*body.current_acc - body.prev_acc)
                * dt
            )
            # updating previous and current accelerations
            body.prev_acc = body.current_acc
            body.current_acc = next_acc_list[i]

    def track_orbits(self, step):
        """Detects orbit completions and alignment events among the inner planets."""
        dt = self.timestep
        time_now = step * dt
        inner_planets = self.body_list[1:6]
        deviation = 5 * (pi / 180)  # converting degrees to radians

        def normalize_angle_diff(angle_diff):
            # Re-map angle differences into [-π, π]
            if angle_diff > pi:
                return angle_diff - 2 * pi
            elif angle_diff < -pi:
                return angle_diff + 2 * pi
            return angle_diff

        # Checking orbit completions
        for body in self.body_list:
            if body.name.lower() == 'sun':
                continue
            angle = atan2(body.position[1], body.position[0])
            dtheta = normalize_angle_diff(angle - body.last_angle)
            body.total_angle += dtheta
            body.last_angle = angle

            # Detecting first orbit completion
            # After first orbit, this code doesn't print any more completions
            # For the further analysis part, 'if body.orbits_completed == 0:' must be removed
            if abs(body.total_angle) >= 2 * pi:
                if body.orbits_completed == 0:
                    print(
                        f"{body.name} completed orbit #{body.orbits_completed+1} in {time_now:.4f}")
                    body.orbit_time_reported = time_now
                    body.orbits_completed += 1
                body.total_angle -= np.sign(body.total_angle) * 2 * pi

        # Checking for alignment
        angles = [p.total_angle for p in inner_planets]
        mean_angle = atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        within_deviation = all(
            abs(normalize_angle_diff(a - mean_angle)) <= deviation for a in angles
        )
        if within_deviation and not self.alignment:
            print(
                f"All inner planets are within 5 degrees of the mean at time: {time_now}")
            self.alignment = True
        elif not within_deviation:
            self.alignment = False

    def calc_KE(self):
        """Returns a list of the kinetic energy of each body."""
        KE = []
        for body in self.body_list:
            velocity = np.linalg.norm(body.velocity)
            KE.append(0.5 * body.mass * (velocity**2))
        return KE

    def calc_PE(self):
        """Computes potential energy for each body."""
        PE = []
        for body_i in self.body_list:
            PE_i = 0.0
            for body_j in self.body_list:
                if body_i != body_j:
                    r_vec = body_i.position - body_j.position
                    distance = np.linalg.norm(r_vec)
                    if distance > 0:
                        PE_i += -self.grav_const * body_i.mass * body_j.mass / distance
            PE.append(PE_i)
        return PE

    def calc_tot_energy(self):
        """Converts energies to SI units and returns total energy per body."""
        earth_mass_kg = 5.972e24
        au_m = 1.496e11
        year_s = 3.154e7
        energy_si = (earth_mass_kg * (au_m**2)) / (year_s**2)
        KE = [e*energy_si for e in self.calc_KE()]
        PE = [e*energy_si for e in self.calc_PE()]
        return [KE[i] + PE[i] for i in range(len(KE))]

    def plot_total_energy(self, TE, method_label):
        """Plots total energy vs. time for each body, with the given method label."""
        time = [i * self.timestep for i in range(self.num_iterations)]
        plt.figure()
        for i, body in enumerate(self.body_list):
            plt.plot(time, TE[:, i], label=f"{body.name}")
            plt.xlabel('Time (years)')
            plt.ylabel('Total Energy (J)')
            plt.title(
                f"Total Energy vs. Time for {body.name} ({method_label})")
            plt.legend(fontsize=8, loc='upper right', frameon=False)
            plt.tight_layout()
            plt.show()

    def energy_conservation(self, TE):
        """Returns (body name, max % error) for each body's total energy drift."""
        stats = []
        for i, body in enumerate(self.body_list):
            energy_series = TE[:, i]
            initial_energy = energy_series[0]

            # calculates the maximum of the difference series between first energy value and subsequent energies
            # this gives an idea about the error for beeman method
            max_energy_diff = np.max(np.abs(energy_series - initial_energy))
            max_err_percent = (max_energy_diff / abs(initial_energy)) * 100
            stats.append((body.name, max_err_percent))
        return stats

    def run_simulation(self, animation, energy_plots):
        """Runs main loop with optional animation, and can produce energy plots at the end."""
        if animation:
            plt.ion()
            fig, ax = plt.subplots()
            fixed_xlim, fixed_ylim = (-10, 10), (-10, 10)

            def configure_axes():
                ax.set_xlim(fixed_xlim)
                ax.set_ylim(fixed_ylim)
                ax.set_aspect('equal', 'box')
                ax.set_xlabel("X Position (AU)")
                ax.set_ylabel("Y Position (AU)")

            configure_axes()

        # to store the first set of previous and current values
        self.calc_acceleration(update=True)
        n_bodies = len(self.body_list)
        TE = np.zeros((self.num_iterations, n_bodies))

        for step in range(self.num_iterations):
            self.update_positions()
            self.update_velocities()
            self.track_orbits(step)
            TE[step, :] = self.calc_tot_energy()

            if animation and step % 50 == 0:
                ax.clear()
                configure_axes()
                for b in self.body_list:
                    ax.scatter(b.position[0], b.position[1],
                               color=b.color, s=80, label=b.name)
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                ax.set_title(
                    f"Time {step*self.timestep:.2f} / {self.num_iterations*self.timestep:.2f} years")
                if self.alignment:
                    fig.savefig(f"alignment_{self.timestep*step:.2f}.png")
                plt.pause(0.001)

        if animation:
            plt.ioff()
            plt.show()

        for i, body in enumerate(self.body_list):
            body.TE = TE[:, i]

        # if true, then the individual total energies over the course of simulation is graphed for every body
        if energy_plots:
            self.plot_total_energy(TE, method_label='Beeman')

        # printing the percentage error in energy from intital to the max at any step during the course of simulation
        print(self.energy_conservation(TE))

    def alternative_methods(self, method_name, energy_plots):
        """Applies chosen integrator (Euler or Euler-Cromer) and collects total energies."""
        n_bodies = len(self.body_list)
        TE = np.zeros((self.num_iterations, n_bodies))

        for step in range(self.num_iterations):
            new_acc = self.calc_acceleration(update=False)
            for i, body in enumerate(self.body_list):
                if method_name == "Euler":
                    # Euler update case
                    body.position += body.velocity * self.timestep
                    body.velocity += new_acc[i] * self.timestep
                elif method_name == "Euler-Cromer":
                    # Euler-Cromer update case
                    body.velocity += new_acc[i] * self.timestep
                    body.position += body.velocity * self.timestep

            self.track_orbits(step)
            TE[step, :] = np.array(self.calc_tot_energy())

        for i, body in enumerate(self.body_list):
            body.TE = TE[:, i]

        if energy_plots:
            self.plot_total_energy(TE, method_label=method_name)

        print(self.energy_conservation(TE))
