"""
A simple command-line demonstration of the M-First Kinematics validation.

This script runs a few pre-defined elastic scattering scenarios and prints the
results from both the standard and the Momentum-First (M-First) viewpoints,
demonstrating their mathematical equivalence.
"""

import numpy as np
import sys
import os

# Adjust the Python path to include the project's root directory
# This allows the script to find the 'src' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.particle import ConcreteParticle
from src.elastic_scattering import ElasticScatteringExperiment

def run_head_on_collision_scenario():
    """
    Scenario 1: A massive particle strikes a stationary target head-on
                and scatters directly backward.
    """
    print("--- SCENARIO 1: Head-On Collision ---")
    print("A particle (m=3) with momentum [0, 0, 4] hits a stationary particle (m=6).")
    print("Particle A is assumed to scatter directly backward.\n")

    # Define the initial particles
    particle_A_initial = ConcreteParticle(m0=3, bosic_momentum=np.array([0, 0, 4]), c=1.0)
    particle_B_initial = ConcreteParticle(m0=6, bosic_momentum=np.array([0, 0, 0]), c=1.0)

    # Define the escape direction for Particle A (straight back)
    escape_direction = np.array([0, 0, -1])

    # Create and solve the experiment
    try:
        experiment = ElasticScatteringExperiment(particle_A_initial, particle_B_initial, escape_direction)
        experiment.solve()

        # Print both conservation tables
        experiment.print_traditional_table()
        experiment.print_momentum_first_table()
    except RuntimeError as e:
        print(f"Error running scenario: {e}")


def run_glancing_blow_scenario():
    """
    Scenario 2: A massive particle strikes a stationary target and scatters
                at a 45-degree angle.
    """
    print("--- SCENARIO 2: Glancing Blow ---")
    print("A particle (m=3) with momentum [0, 0, 4] hits a stationary particle (m=6).")
    print("Particle A is assumed to scatter at 45 degrees in the x-z plane.\n")

    # Define the initial particles (same as before)
    particle_A_initial = ConcreteParticle(m0=3, bosic_momentum=np.array([0, 0, 4]), c=1.0)
    particle_B_initial = ConcreteParticle(m0=6, bosic_momentum=np.array([0, 0, 0]), c=1.0)

    # Define the escape direction for Particle A (45 degrees in x-z plane)
    # cos(45) = sin(45) = 1/sqrt(2) approx 0.7071
    escape_direction = np.array([0.70710678, 0, 0.70710678])

    # Create and solve the experiment
    try:
        experiment = ElasticScatteringExperiment(particle_A_initial, particle_B_initial, escape_direction)
        experiment.solve()

        # Print both conservation tables
        experiment.print_traditional_table()
        experiment.print_momentum_first_table()
    except RuntimeError as e:
        print(f"Error running scenario: {e}")


def run_massless_particle_scenario():
    """
    Scenario 3: A massless particle (e.g., a photon) scatters off a
                stationary massive particle at 90 degrees (Compton Scattering analog).
    """
    print("--- SCENARIO 3: Massless Particle Scattering ---")
    print("A massless particle (m=0) with momentum [0, 0, 5] hits a stationary particle (m=4).")
    print("The massless particle is assumed to scatter at 90 degrees into the y-axis.\n")

    # Define the initial particles
    particle_A_initial = ConcreteParticle(m0=0, bosic_momentum=np.array([0, 0, 5]), c=1.0)
    particle_B_initial = ConcreteParticle(m0=4, bosic_momentum=np.array([0, 0, 0]), c=1.0)

    # Define the escape direction for Particle A (90 degrees into the y-axis)
    escape_direction = np.array([0, 1, 0])

    # Create and solve the experiment
    try:
        experiment = ElasticScatteringExperiment(particle_A_initial, particle_B_initial, escape_direction)
        experiment.solve()

        # Print both conservation tables
        experiment.print_traditional_table()
        experiment.print_momentum_first_table()
    except RuntimeError as e:
        print(f"Error running scenario: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("  Running M-First Kinematics Validation Demonstrations")
    print("=" * 80 + "\n")
    
    # Run all defined scenarios
    run_head_on_collision_scenario()
    run_glancing_blow_scenario()
    run_massless_particle_scenario()

    print("All demonstrations complete.")