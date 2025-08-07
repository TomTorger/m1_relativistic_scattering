"""Utility class for comparing standard and Momentum-First scattering results.

This module provides the :class:`ScatteringComparison` class, which wraps an
:class:`~src.elastic_scattering.ElasticScatteringExperiment` and generates
conservation tables for both the traditional relativistic formulation and the
Momentum-First (M1) framework.  It also offers a convenience method to verify
that the balances in both tables vanish, confirming conservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .elastic_scattering import ElasticScatteringExperiment


@dataclass
class ScatteringComparison:
    """Generate and verify conservation tables for an experiment.

    Parameters
    ----------
    experiment:
        The :class:`ElasticScatteringExperiment` to analyse.  The experiment
        will be solved lazily when tables are requested.
    """

    experiment: ElasticScatteringExperiment

    def tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return conservation tables for standard and M1 formulations.

        Returns
        -------
        Tuple[pandas.DataFrame, pandas.DataFrame]
            Two data frames corresponding to the traditional (energy and vector
            momentum) and the Momentum-First directional momentum components.
        """
        if not self.experiment.solved:
            self.experiment.solve()

        # --- Traditional table ---
        states = {
            "A_before": self.experiment.initial_A,
            "B_before": self.experiment.initial_B,
            "A_after": self.experiment.final_A,
            "B_after": self.experiment.final_B,
        }

        trad_data = []
        components = ["px", "py", "pz", "Energy"]
        for i, comp_name in enumerate(components):
            row = {"Component": comp_name}
            val_A_before = (
                states["A_before"].energy
                if comp_name == "Energy"
                else states["A_before"].bosic_momentum[i]
            )
            val_B_before = (
                states["B_before"].energy
                if comp_name == "Energy"
                else states["B_before"].bosic_momentum[i]
            )
            total_before = val_A_before + val_B_before

            val_A_after = (
                states["A_after"].energy
                if comp_name == "Energy"
                else states["A_after"].bosic_momentum[i]
            )
            val_B_after = (
                states["B_after"].energy
                if comp_name == "Energy"
                else states["B_after"].bosic_momentum[i]
            )
            total_after = val_A_after + val_B_after

            row.update(
                {
                    "A_before": val_A_before,
                    "B_before": val_B_before,
                    "Total (Before)": total_before,
                    "A_after": val_A_after,
                    "B_after": val_B_after,
                    "Total (After)": total_after,
                    "Balance": total_before - total_after,
                }
            )
            trad_data.append(row)

        trad_df = pd.DataFrame(trad_data).set_index("Component")

        # --- Momentum-First table ---
        states_m = {
            "A_before": self.experiment.initial_A.directional_components,
            "B_before": self.experiment.initial_B.directional_components,
            "A_after": self.experiment.final_A.directional_components,
            "B_after": self.experiment.final_B.directional_components,
        }

        mfirst_data = []
        components_m = list(states_m["A_before"].keys())
        for comp_name in components_m:
            row = {"Component": comp_name}
            val_A_before = states_m["A_before"][comp_name]
            val_B_before = states_m["B_before"][comp_name]
            total_before = val_A_before + val_B_before

            val_A_after = states_m["A_after"][comp_name]
            val_B_after = states_m["B_after"][comp_name]
            total_after = val_A_after + val_B_after

            row.update(
                {
                    "A_before": val_A_before,
                    "B_before": val_B_before,
                    "Total (Before)": total_before,
                    "A_after": val_A_after,
                    "B_after": val_B_after,
                    "Total (After)": total_after,
                    "Balance": total_before - total_after,
                }
            )
            mfirst_data.append(row)

        mfirst_df = pd.DataFrame(mfirst_data).set_index("Component")
        mfirst_df = mfirst_df.reindex(["x+", "x-", "y+", "y-", "z+", "z-"])

        return trad_df, mfirst_df

    def verify(self, atol: float = 1e-8) -> bool:
        """Check that both conservation tables balance to zero within tolerance."""
        trad_df, mfirst_df = self.tables()
        ok_trad = np.allclose(trad_df["Balance"].values, 0, atol=atol)
        ok_mfirst = np.allclose(mfirst_df["Balance"].values, 0, atol=atol)
        return bool(ok_trad and ok_mfirst)
