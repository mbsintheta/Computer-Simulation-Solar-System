

# Name: Mehul Bandhu
# s2500976
# Computer Simulation Project


import numpy as np


class Body:
    def __init__(self, name, color, mass, position, velocity, current_acceleration, previous_acceleration):
        """
        Method to initialize object type: Body
        """
        self.name = name
        self.color = color
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.current_acc = np.array(current_acceleration, dtype=float)
        self.prev_acc = np.array(previous_acceleration, dtype=float)
        self.last_angle = np.arctan2(self.position[1], self.position[0])
        self.total_angle = 0.0
        self.orbits_completed = 0
        self.orbit_time_reported = 0
        self.TE = []
