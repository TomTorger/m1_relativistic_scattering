"""
Tests for scattering events involving massless particles.
"""

import numpy as np
import pytest
from src.particle import ConcreteParticle
from src.elastic_scattering import ElasticScatteringExperiment

@pytest.mark.parametrize("pA_vec, mB, pB_vec, escape_direction", [
    # A (massless) hits B (at rest)
    ([0, 0, 5], 4, [0, 0, 0], [0, 1, 0]),
    ([2, 0, 3], 6, [0, 0, 0], [1, 0, 0]),
    # A (massless) hits B (moving)
    ([1, 2, 3], 4, [0, 1, 0], [0, -1, 0]),
    ([0, 0, 10], 3, [1, 0, 0], [-1, 0, 0]),
])
def test_massless_particle_scattering(pA_vec, mB, pB_vec, escape_direction):
    """
    Parameterized test for scattering where Particle A is massless (m0=0).
    Verifies conservation of standard energy and vector momentum.
    """
    pA_vec, pB_vec = np.array(pA_vec), np.array(pB_vec)
    escape_direction = np.array(escape_direction)

    # Particle A is massless
    particle_A = ConcreteParticle(m0=0, bosic_momentum=pA_vec, c=1.0)
    particle_B = ConcreteParticle(m0=mB, bosic_momentum=pB_vec, c=1.0)

    experiment = ElasticScatteringExperiment(particle_A, particle_B, escape_direction)
    A_final, B_final = experiment.solve()

    # Check conservation of total vector momentum
    initial_total_momentum = particle_A.bosic_momentum + particle_B.bosic_momentum
    final_total_momentum = A_final.bosic_momentum + B_final.bosic_momentum
    np.testing.assert_allclose(initial_total_momentum, final_total_momentum,
                               err_msg="Momentum not conserved in massless scattering.")

    # Check conservation of total energy
    initial_total_energy = particle_A.energy + particle_B.energy
    final_total_energy = A_final.energy + B_final.energy
    assert np.isclose(initial_total_energy, final_total_energy), \
        "Energy not conserved in massless scattering."