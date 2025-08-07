"""
M-First Kinematics Source Package
=================================

This package provides the core classes for simulating and validating the
kinematics of the Momentum-First (M-First) framework.

The main classes exposed at the package level are:
  - Particle: The abstract base class for a particle.
  - ConcreteParticle: An instantiable particle class.
  - ElasticScatteringExperiment: The main simulation engine.
"""

from .particle import Particle, ConcreteParticle
from .elastic_scattering import ElasticScatteringExperiment
from .comparison import ScatteringComparison

# Define the public API of the 'src' package.
__all__ = [
    "Particle",
    "ConcreteParticle",
    "ElasticScatteringExperiment",
    "ScatteringComparison",
]