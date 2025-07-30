"""
Defines the Particle class for the Momentum-First (M-First) framework.

This module provides the `ConcreteParticle` class, which encapsulates the 
kinematic properties of a particle according to the M-First postulates. It is the
foundational object for all simulations in this repository.
"""

from typing import Dict
import numpy as np

class Particle:
    """
    An abstract base class representing a particle's kinematic state.

    This class should not be instantiated directly. Use a concrete subclass
    like `ConcreteParticle`.
    """
    def __init__(self, m0: float, bosic_momentum: np.ndarray, c: float = 1.0):
        """
        Initializes a particle's state.

        Args:
            m0 (float): The particle's rest mass.
            bosic_momentum (np.ndarray): The 3D external momentum vector, corresponding
                to the 'bosic momentum' (p-vec) in the M-First framework.
            c (float): The speed of light, used to convert between mass and momentum units.
        """
        self.m0 = m0
        self.bosic_momentum = np.array(bosic_momentum, dtype=float)
        self.c = c

    @property
    def bosic_momentum_magnitude(self) -> float:
        """
        Calculates the scalar magnitude of the bosic momentum vector.

        This corresponds to 'p' in the M-First energy-momentum relations.

        Returns:
            float: The magnitude |p-vec|.
        """
        return np.linalg.norm(self.bosic_momentum)

    @property
    def fermic_momentum(self) -> float:
        """
        Calculates the particle's intrinsic fermic momentum.

        This is the invariant momentum scale of the particle, defined as p_f = m0 * c.

        Returns:
            float: The fermic momentum (p_f).
        """
        return self.m0 * self.c

    @property
    def core_momentum(self) -> float:
        """
        Calculates the particle's total scalar core momentum.

        This is the total kinematic scale of the moving particle, defined by the 
        relation M(p) = sqrt(p_f^2 + p^2).

        Returns:
            float: The core momentum M(p).
        """
        return np.sqrt(self.fermic_momentum**2 + self.bosic_momentum_magnitude**2)

    @property
    def energy(self) -> float:
        """
        Calculates the particle's total relativistic energy.

        In the M-First framework, energy is an emergent property derived from the
        core momentum: E = M(p) * c. This is equivalent to the standard
        relativistic formula E = sqrt((p*c)^2 + (m0*c^2)^2).

        Returns:
            float: The total energy (E).
        """
        return self.core_momentum * self.c

    @property
    def directional_components(self) -> Dict[str, float]:
        """
        Calculates the six fundamental directional momentum components.

        This is the core of the M-First postulate. It implements the formula:
            p_{k^±} = M(p) ∓ (1/2) * p_k
        
        where k is one of the Cartesian axes (x, y, z). These six values are
        the fundamental quantities that are independently conserved in any
        isolated interaction according to M-First.

        Returns:
            Dict[str, float]: A dictionary mapping each directional component
                              (e.g., 'x+', 'x-') to its scalar value.
        """
        M = self.core_momentum
        half_p = self.bosic_momentum / 2.0
        
        return {
            'x+': M - half_p[0],
            'x-': M + half_p[0],
            'y+': M - half_p[1],
            'y-': M + half_p[1],
            'z+': M - half_p[2],
            'z-': M + half_p[2],
        }

    @property
    def unit_vector(self) -> np.ndarray:
        """
        Calculates the 3D unit vector in the direction of the bosic momentum.

        If the momentum is zero, returns a default vector [0, 0, 1] to avoid
        division-by-zero errors. This is a convention for calculation and does
        not imply a physical direction for a particle at rest.

        Returns:
            np.ndarray: A 3D unit vector.
        """
        norm = self.bosic_momentum_magnitude
        if norm < 1e-12:
            return np.array([0, 0, 1], dtype=float)
        return self.bosic_momentum / norm

    @unit_vector.setter
    def unit_vector(self, new_dir: np.ndarray):
        """
        Sets a new direction for the particle's momentum while preserving its magnitude.

        Args:
            new_dir (np.ndarray): The new direction vector (will be normalized).
        
        Raises:
            ValueError: If the new direction vector has zero magnitude.
        """
        new_dir_vec = np.array(new_dir, dtype=float)
        norm = np.linalg.norm(new_dir_vec)
        if norm < 1e-12:
            raise ValueError("New direction vector must be non-zero.")
        
        new_unit_vector = new_dir_vec / norm
        current_magnitude = self.bosic_momentum_magnitude
        self.bosic_momentum = current_magnitude * new_unit_vector

    def __repr__(self) -> str:
        """Provides an unambiguous string representation of the particle."""
        return (f"{self.__class__.__name__}("
                f"m0={self.m0}, "
                f"bosic_momentum={self.bosic_momentum.tolist()}, "
                f"c={self.c})")


# A concrete implementation of the Particle class for instantiation.
class ConcreteParticle(Particle):
    """A concrete, instantiable version of the Particle class."""
    pass