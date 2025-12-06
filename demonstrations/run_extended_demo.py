"""
Run the extended set of elastic scattering demonstrations defined in YAML.

Scenarios are declared in demonstrations/extended_scenarios.yaml. Each scenario
describes incoming particles and an escape direction; the script prints both
standard and Momentum-First conservation tables for each physical solution.
"""

import os
import sys
import numpy as np
import yaml

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.elastic_scattering import ElasticScatteringExperiment
from src.particle import ConcreteParticle


def load_scenarios(filename: str):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or []


def make_particle(cfg: dict) -> ConcreteParticle:
    return ConcreteParticle(
        m0=float(cfg["mass"]),
        bosic_momentum=np.array(cfg["momentum"], dtype=float),
        c=1.0,
    )


def run_scenario(index: int, scenario: dict) -> None:
    print(f"--- SCENARIO {index}: {scenario['name']} ---")
    description = scenario.get("description", "")
    if description:
        print(description)
    print()

    particle_A = make_particle(scenario["particle_A"])
    particle_B = make_particle(scenario["particle_B"])
    escape_direction = np.array(scenario["escape_direction"], dtype=float)

    try:
        experiment = ElasticScatteringExperiment(particle_A, particle_B, escape_direction)
        experiment.solve()
        experiment.print_traditional_table()
        experiment.print_momentum_first_table()
    except (RuntimeError, ValueError) as exc:
        print(f"Skipped: {exc}\n")


def main():
    scenarios = load_scenarios("extended_scenarios.yaml")

    print("=" * 80)
    print("  Running Extended M-First Kinematics Demonstrations")
    print("=" * 80 + "\n")

    for idx, scenario in enumerate(scenarios, start=1):
        run_scenario(idx, scenario)

    print("Extended demonstrations complete.")


if __name__ == "__main__":
    main()
