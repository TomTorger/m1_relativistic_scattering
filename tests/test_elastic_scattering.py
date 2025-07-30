"""
Tests for the ElasticScatteringExperiment class.

This module verifies that the simulation correctly conserves both the standard
physical quantities (energy, vector momentum) and the specific directional
momentum components postulated by the M-First framework.
"""

import numpy as np
import pytest
from src.particle import ConcreteParticle
from src.elastic_scattering import ElasticScatteringExperiment

# A standard set of parameters for reuse in multiple tests
P_A_INITIAL = ConcreteParticle(m0=3, bosic_momentum=np.array([0, 0, 4]), c=1.0)
P_B_INITIAL = ConcreteParticle(m0=6, bosic_momentum=np.array([0, 0, 0]), c=1.0)
ESCAPE_DIR = np.array([0.70710678, 0, -0.70710678]) # 45 degrees off -z

def test_standard_conservation():
    """
    Tests that total energy and total vector momentum are conserved.
    """
    experiment = ElasticScatteringExperiment(P_A_INITIAL, P_B_INITIAL, ESCAPE_DIR)
    A_final, B_final = experiment.solve()

    # Vector momentum conservation
    initial_total_momentum = P_A_INITIAL.bosic_momentum + P_B_INITIAL.bosic_momentum
    final_total_momentum = A_final.bosic_momentum + B_final.bosic_momentum
    np.testing.assert_allclose(initial_total_momentum, final_total_momentum,
                               err_msg="Total vector momentum is not conserved.")

    # Energy conservation
    initial_total_energy = P_A_INITIAL.energy + P_B_INITIAL.energy
    final_total_energy = A_final.energy + B_final.energy
    assert np.isclose(initial_total_energy, final_total_energy), \
        "Total energy is not conserved."

def test_mfirst_directional_conservation():
    """
    Tests that each of the six M-First directional components is conserved.
    This is the core validation of the M-First postulate for interactions.
    """
    experiment = ElasticScatteringExperiment(P_A_INITIAL, P_B_INITIAL, ESCAPE_DIR)
    A_final, B_final = experiment.solve()

    initial_comps_A = P_A_INITIAL.directional_components
    initial_comps_B = P_B_INITIAL.directional_components
    final_comps_A = A_final.directional_components
    final_comps_B = B_final.directional_components

    # Check conservation for each of the six components independently
    for key in initial_comps_A:
        initial_total = initial_comps_A[key] + initial_comps_B[key]
        final_total = final_comps_A[key] + final_comps_B[key]
        assert np.isclose(initial_total, final_total), \
            f"Directional component '{key}' is not conserved."

@pytest.mark.parametrize("mA, pA, mB, pB, escape_direction", [
    (3, [0, 0, 4], 6, [0, 0, 0], [0, 0, -1]),          # Head-on
    (4, [5, 0, 0], 8, [0, 0, 0], [1, 0, 0]),          # Forward scatter
    (2, [1, 2, 3], 5, [0, 1, 0], [1, 3, 3]),          # Both moving
    (2, [0, 0, 10], 4, [1, 1, 0], [0, 0.707, -0.707]) # High energy
])
def test_scattering_varied_inputs(mA, pA, mB, pB, escape_direction):
    """
    Tests both standard and M-First conservation for varied inputs.
    """
    pA, pB = np.array(pA), np.array(pB)
    escape_direction = np.array(escape_direction)
    
    particle_A = ConcreteParticle(m0=mA, bosic_momentum=pA, c=1.0)
    particle_B = ConcreteParticle(m0=mB, bosic_momentum=pB, c=1.0)
    
    experiment = ElasticScatteringExperiment(particle_A, particle_B, escape_direction)
    A_final, B_final = experiment.solve()

    # Standard check
    initial_total_momentum = particle_A.bosic_momentum + particle_B.bosic_momentum
    final_total_momentum = A_final.bosic_momentum + B_final.bosic_momentum
    np.testing.assert_allclose(initial_total_momentum, final_total_momentum)

    initial_total_energy = particle_A.energy + particle_B.energy
    final_total_energy = A_final.energy + B_final.energy
    assert np.isclose(initial_total_energy, final_total_energy)

    # M-First check
    initial_comps_A = particle_A.directional_components
    initial_comps_B = particle_B.directional_components
    final_comps_A = A_final.directional_components
    final_comps_B = B_final.directional_components
    for key in initial_comps_A:
        initial_total = initial_comps_A[key] + initial_comps_B[key]
        final_total = final_comps_A[key] + final_comps_B[key]
        assert np.isclose(initial_total, final_total)