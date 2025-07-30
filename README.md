# A Computational Validation of the Momentum-First Kinematic Framework

[![CI](https://github.com/your-username/M-First-Kinematics/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/M-First-Kinematics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a computational proof that for relativistic elastic scattering, the Momentum-First (M-First) framework's postulate of **Absolute Directional Momentum Conservation** is mathematically equivalent to the standard model's conservation of total energy and vector momentum.

The theoretical basis for this work is outlined in the paper ["Momentum Is All You Need"](https://www.authorea.com/users/695998/articles/713606-momentum-is-all-you-need).

---

## The Core Claim

The M-First framework redefines the fundamental conserved quantities in an interaction. Instead of conserving total vector momentum ($\vec{P}$) and total energy ($E$) as separate entities, it postulates that the six **directional momentum components** ($p_{k^\pm}$) are independently conserved for each Cartesian half-axis ($+x, -x, +y, -y, +z, -z$).

The components for a single particle are defined as:
$$
p_{k^\pm} = M(p) \mp \frac{1}{2} p_k
$$
where $p_k$ is the standard momentum component along axis $k$, and $M(p) = \sqrt{p_f^2 + p^2}$ is the particle's **Core Momentum** ($p_f = m_0c$ being the **Fermic Momentum**).

This repository demonstrates that for any two-particle elastic collision:
$$
\sum_i p_{k^\pm, \text{initial}}^{(i)} = \sum_j p_{k^\pm, \text{final}}^{(j)} \quad \iff \quad
\begin{cases}
\sum_i E_{\text{initial}}^{(i)} = \sum_j E_{\text{final}}^{(j)} \\
\sum_i \vec{P}_{\text{initial}}^{(i)} = \sum_j \vec{P}_{\text{final}}^{(j)}
\end{cases}
$$

## How It Works

We simulate a two-particle collision using a numerical solver that enforces the standard conservation of energy and vector momentum. We then display the "before" and "after" states of the system in two tables:

1. **The Standard Table:** Shows conservation of $E, p_x, p_y, p_z$.
2. **The M-First Table:** Shows conservation of $p_{x^+}, p_{x^-}, p_{y^+}, p_{y^-}, p_{z^+}, p_{z^-}$.

The demonstration shows that the "Balance" (Total Before - Total After) column in both tables is zero (within numerical precision), proving the equivalence of the conservation laws.

## Installation

To get started, clone the repository and install the required packages.

```bash
git clone https://github.com/your-username/M-First-Kinematics.git
cd M-First-Kinematics
pip install -r requirements.txt
```

## How to Run the Demonstration

You have two options to see the validation in action.

### 1. The Quick Way (Command Line)

For a quick demonstration of a few pre-defined scenarios, run the following command from the root directory:

```bash
python demonstrations/run_basic_demo.py
```

This will print the conservation tables directly to your terminal.

### 2. The In-Depth Way (Jupyter Notebook)

For a detailed, step-by-step explanation that weaves together the theory, code, and results, we highly recommend using the Jupyter Notebook.

```bash
# Launch Jupyter Notebook from the root directory
jupyter notebook
```

Then, navigate to and open `demonstrations/M-First_Demonstration.ipynb` in your browser.

## Running the Tests

To verify the correctness of the simulation code and the implementation of the M-First postulates, you can run the full test suite using `pytest`.

This will discover and run all tests in the tests/ directory. The CI workflow automatically runs these tests on every push.
