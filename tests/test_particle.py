"""
Tests for the Particle class in the M-First framework.

This module verifies that the kinematic properties of the ConcreteParticle class
are calculated correctly according to both standard relativistic mechanics and
the specific postulates of the Momentum-First framework.
"""

import numpy as np
import pytest
from src.particle import ConcreteParticle

# --- Test standard relativistic properties with M-First terminology ---

@pytest.mark.parametrize("m0, p_vec", [
    (1.0, [0, 0, 0]),
    (1.0, [1, 0, 0]),
    (2.5, [3, 4, 0]),    # p_mag = 5
    (10.0, [3, 4, 12]),  # p_mag = 13
    (0.5, [-1, -1, 0]),  # p_mag = sqrt(2)
])
def test_energy_and_core_momentum(m0, p_vec):
    """
    Tests that energy and core momentum are computed correctly.
    E = M*c = sqrt((m0*c^2)^2 + (p*c)^2)
    """
    p_vec = np.array(p_vec)
    particle = ConcreteParticle(m0=m0, bosic_momentum=p_vec, c=1.0)
    
    p_mag = np.linalg.norm(p_vec)
    expected_core_momentum = np.sqrt(m0**2 + p_mag**2)
    expected_energy = expected_core_momentum  # Since c=1.0

    assert np.isclose(particle.core_momentum, expected_core_momentum)
    assert np.isclose(particle.energy, expected_energy)
    assert np.isclose(particle.fermic_momentum, m0)
    assert np.isclose(particle.bosic_momentum_magnitude, p_mag)

def test_unit_vector_update():
    """
    Tests that setting a new unit_vector updates the momentum direction
    while preserving the original momentum magnitude.
    """
    particle = ConcreteParticle(m0=2.0, bosic_momentum=np.array([3, 0, 0]), c=1.0)
    original_magnitude = particle.bosic_momentum_magnitude
    
    # Change direction to the y-axis
    particle.unit_vector = np.array([0, 1, 0])
    expected_new_momentum = np.array([0, 3.0, 0])
    
    assert np.isclose(particle.bosic_momentum_magnitude, original_magnitude)
    np.testing.assert_allclose(particle.bosic_momentum, expected_new_momentum)

# --- Test the core M-First postulate implementation ---

@pytest.mark.parametrize("m0, p_vec", [
    (3.0, [4, 0, 0]),      # Motion along +x
    (5.0, [0, -12, 0]),    # Motion along -y
    (8.0, [0, 0, 15]),     # Motion along +z
    (1.0, [3, 4, 12]),    # General motion vector
    (10.0, [0, 0, 0]),     # At rest
])
def test_directional_components_calculation(m0, p_vec):
    """
    Validates the calculation of the six directional momentum components.

    This is the most critical test for the M-First implementation, as it
    directly checks the formula: p_{k^±} = M(p) ∓ (1/2) * p_k
    """
    p_vec = np.array(p_vec)
    particle = ConcreteParticle(m0=m0, bosic_momentum=p_vec, c=1.0)

    # Manually calculate expected values
    M = particle.core_momentum
    px, py, pz = p_vec[0], p_vec[1], p_vec[2]

    expected = {
        'x+': M - px / 2.0,
        'x-': M + px / 2.0,
        'y+': M - py / 2.0,
        'y-': M + py / 2.0,
        'z+': M - pz / 2.0,
        'z-': M + pz / 2.0,
    }

    # Get the calculated components from the particle
    calculated = particle.directional_components

    # Assert that each component is calculated correctly
    for key in expected:
        assert key in calculated, f"Key '{key}' not found in directional_components."
        assert np.isclose(calculated[key], expected[key]), (
            f"Component '{key}' incorrect for p_vec={p_vec}. "
            f"Expected {expected[key]}, got {calculated[key]}."
        )