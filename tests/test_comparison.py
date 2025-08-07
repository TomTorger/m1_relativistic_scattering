"""Tests for the ScatteringComparison helper class."""

import numpy as np

from src import ConcreteParticle, ElasticScatteringExperiment, ScatteringComparison


P_A_INITIAL = ConcreteParticle(m0=3, bosic_momentum=np.array([0, 0, 4]), c=1.0)
P_B_INITIAL = ConcreteParticle(m0=6, bosic_momentum=np.array([0, 0, 0]), c=1.0)
ESCAPE_DIR = np.array([0.70710678, 0, -0.70710678])


def test_tables_and_verification():
    experiment = ElasticScatteringExperiment(P_A_INITIAL, P_B_INITIAL, ESCAPE_DIR)
    comparison = ScatteringComparison(experiment)
    trad_df, mfirst_df = comparison.tables()

    # Expect four rows for traditional and six for M1 tables
    assert list(trad_df.index) == ["px", "py", "pz", "Energy"]
    assert list(mfirst_df.index) == ["x+", "x-", "y+", "y-", "z+", "z-"]

    # Verification should pass with zero balances
    assert np.allclose(trad_df["Balance"].values, 0)
    assert np.allclose(mfirst_df["Balance"].values, 0)
    assert comparison.verify()
