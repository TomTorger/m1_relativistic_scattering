"""
Defines the ElasticScatteringExperiment class for simulating collisions.

This module provides the primary simulation engine that takes two initial 
particle states, solves for their final states after a relativistic elastic 
collision, and provides methods to display the results from both the standard
and the Momentum-First (M-First) perspectives.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .particle import Particle, ConcreteParticle

class ElasticScatteringExperiment:
    """
    Simulates a relativistic elastic scattering event between two particles.

    This class sets up an initial state with two particles, solves for their
    final momentum vectors given a scattering direction for one particle, and
    verifies the conservation laws from both standard and M-First viewpoints.
    """
    def __init__(
        self,
        initial_particle_A: Particle,
        initial_particle_B: Particle,
        escape_direction: np.ndarray
    ):
        """
        Initializes the scattering experiment.

        Args:
            initial_particle_A (Particle): The first particle in its initial state.
            initial_particle_B (Particle): The second particle in its initial state.
            escape_direction (np.ndarray): A 3D unit vector specifying the direction
                of Particle A after the collision.
        """
        self.initial_A = initial_particle_A
        self.initial_B = initial_particle_B
        
        # Ensure the escape direction is a normalized unit vector
        esc_vec = np.array(escape_direction, dtype=float)
        norm = np.linalg.norm(esc_vec)
        if norm < 1e-12:
            raise ValueError("Escape direction vector must be non-zero.")
        self.escape_direction = esc_vec / norm

        # Final states will be populated by the solve() method
        self.final_A: Particle | None = None
        self.final_B: Particle | None = None
        self.solved = False

    def solve(self) -> Tuple[Particle, Particle]:
        """
        Solves the elastic scattering problem using standard conservation laws.

        The method calculates the final momentum vectors for both particles based
        on the conservation of total energy and total vector momentum. It solves
        for the magnitude of particle A's final momentum along the specified
        escape direction.

        The core equation being solved is for the conservation of energy:
        E_A_final + E_B_final = E_total_initial
        
        Where:
        E_A_final = sqrt((m_A*c^2)^2 + (x*c)^2)
        E_B_final = sqrt((m_B*c^2)^2 + ||P_total - x*v||^2 * c^2)
        
        and 'x' is the magnitude of particle A's final momentum we solve for.

        Returns:
            Tuple[Particle, Particle]: A tuple containing the two particles
            (A, B) in their final, post-collision states.
            
        Raises:
            RuntimeError: If no valid physical solution is found.
        """
        # Get initial properties using M-First terminology
        mA, mB = self.initial_A.m0, self.initial_B.m0
        c = self.initial_A.c
        
        pA_initial_vec = self.initial_A.bosic_momentum
        pB_initial_vec = self.initial_B.bosic_momentum

        # Calculate total initial energy and momentum (standard model)
        total_initial_momentum = pA_initial_vec + pB_initial_vec
        total_initial_energy = self.initial_A.energy + self.initial_B.energy

        # Define the function f(x) = 0 that we need to solve for x,
        # where x is the magnitude of particle A's final momentum.
        def f(x: float) -> float:
            # Energy of particle A with final momentum magnitude x
            E_A_final = np.sqrt((mA * c**2)**2 + (x * c)**2)
            
            # Momentum vector of particle B by conservation
            pB_final_vec = total_initial_momentum - x * self.escape_direction
            pB_final_mag = np.linalg.norm(pB_final_vec)
            
            # Energy of particle B
            E_B_final = np.sqrt((mB * c**2)**2 + (pB_final_mag * c)**2)
            
            return E_A_final + E_B_final - total_initial_energy
        
        # Find a reasonable search interval for the root-finding algorithm.
        # x must be positive. An upper bound is the total momentum magnitude.
        x_lower_bound = 0.0
        x_upper_bound = np.linalg.norm(total_initial_momentum) + self.initial_A.fermic_momentum + self.initial_B.fermic_momentum
        
        try:
            # Use Brent's method to find the root x
            final_pA_magnitude = brentq(f, x_lower_bound, x_upper_bound, rtol=1e-7)
        except ValueError:
            raise RuntimeError("No valid physical solution found for the final "
                               "momentum of Particle A in the given direction.")

        # Construct the final particle states
        final_pA_vector = final_pA_magnitude * self.escape_direction
        final_pB_vector = total_initial_momentum - final_pA_vector

        self.final_A = ConcreteParticle(m0=mA, bosic_momentum=final_pA_vector, c=c)
        self.final_B = ConcreteParticle(m0=mB, bosic_momentum=final_pB_vector, c=c)
        
        self.solved = True
        return self.final_A, self.final_B

    def print_traditional_table(self):
        """
        Prints a table showing conservation from the standard viewpoint.
        
        The table shows the components of vector momentum (px, py, pz) and
        energy for each particle before and after the collision, verifying
        that their sums are conserved.
        """
        if not self.solved:
            self.solve()

        states = {
            "A_before": self.initial_A,
            "B_before": self.initial_B,
            "A_after": self.final_A,
            "B_after": self.final_B,
        }

        data = []
        components = ["px", "py", "pz", "Energy"]
        
        for i, comp_name in enumerate(components):
            row = {"Component": comp_name}
            # Initial states
            val_A_before = states["A_before"].energy if comp_name == "Energy" else states["A_before"].bosic_momentum[i]
            val_B_before = states["B_before"].energy if comp_name == "Energy" else states["B_before"].bosic_momentum[i]
            total_before = val_A_before + val_B_before
            # Final states
            val_A_after = states["A_after"].energy if comp_name == "Energy" else states["A_after"].bosic_momentum[i]
            val_B_after = states["B_after"].energy if comp_name == "Energy" else states["B_after"].bosic_momentum[i]
            total_after = val_A_after + val_B_after
            
            row.update({
                "A_before": val_A_before, "B_before": val_B_before, "Total (Before)": total_before,
                "A_after": val_A_after, "B_after": val_B_after, "Total (After)": total_after,
                "Balance": total_before - total_after
            })
            data.append(row)
        
        self._last_traditional_data = data  # Store the data before creating the DataFrame

        df = pd.DataFrame(data).set_index("Component")
        print("--- Standard Conservation Table ---\n")
        print(df.to_string(float_format="%.6f"))
        print("\n" + "="*80 + "\n")

    def print_momentum_first_table(self):
        """
        Prints a table showing conservation from the M-First viewpoint.

        The table shows the six directional momentum components (x+, x-, etc.)
        for each particle, verifying that each is independently conserved.
        """
        if not self.solved:
            self.solve()
            
        states = {
            "A_before": self.initial_A.directional_components,
            "B_before": self.initial_B.directional_components,
            "A_after": self.final_A.directional_components,
            "B_after": self.final_B.directional_components,
        }
        
        data = []
        # The order of components is determined by the dictionary keys.
        components = list(states["A_before"].keys())

        for comp_name in components:
            row = {"Component": comp_name}
            
            val_A_before = states["A_before"][comp_name]
            val_B_before = states["B_before"][comp_name]
            total_before = val_A_before + val_B_before

            val_A_after = states["A_after"][comp_name]
            val_B_after = states["B_after"][comp_name]
            total_after = val_A_after + val_B_after

            row.update({
                "A_before": val_A_before, "B_before": val_B_before, "Total (Before)": total_before,
                "A_after": val_A_after, "B_after": val_B_after, "Total (After)": total_after,
                "Balance": total_before - total_after
            })
            data.append(row)
            
        self._last_mfirst_data = data  # Store the data before creating the DataFrame
        df = pd.DataFrame(data).set_index("Component")
        print("--- Momentum-First Conservation Table ---\n")
        # Re-order for intuitive display
        df = df.reindex(['x+', 'x-', 'y+', 'y-', 'z+', 'z-'])
        print(df.to_string(float_format="%.6f"))
        print("\n" + "="*80 + "\n")

    def __repr__(self) -> str:
        """Provides an unambiguous string representation of the experiment setup."""
        return (
            f"ElasticScatteringExperiment(\n"
            f"  initial_A={self.initial_A},\n"
            f"  initial_B={self.initial_B},\n"
            f"  escape_direction={self.escape_direction.tolist()}\n)"
        )