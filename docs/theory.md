# Momentum-First Theory Notes

## Core claim
The M-First framework replaces the usual pair of conserved quantities (total vector momentum $\vec{P}$ and total energy $E$) with six independently conserved **directional momentum components** $p_{k^\pm}$, one for each half-axis $+x, -x, +y, -y, +z, -z$. For any elastic two-particle interaction, conserving those six components is equivalent to conserving $E$ and $\vec{P}$.

## Directional momentum definition
For a single particle:
$$
p_{k^\pm} = M(p) \mp \tfrac{1}{2} p_k
$$
where $p_k$ is the standard momentum along axis $k$ and $M(p) = \sqrt{p_f^2 + p^2}$ is the **Core Momentum**. Here $p_f = m_0 c$ is the **Fermic Momentum**, with $m_0$ the rest mass.

## Equivalence statement
For any elastic collision:
$$
\sum_i p_{k^\pm,\text{initial}}^{(i)} = \sum_j p_{k^\pm,\text{final}}^{(j)} \quad \iff \quad
\begin{cases}
\sum_i E_{\text{initial}}^{(i)} = \sum_j E_{\text{final}}^{(j)} \\
\sum_i \vec{P}_{\text{initial}}^{(i)} = \sum_j \vec{P}_{\text{final}}^{(j)}
\end{cases}
$$
The demo enumerates three scattering configurations and shows that both the standard and M-First tables close their balances to zero (numerical precision), illustrating the equivalence.

## Simulation approach
- Solve for final-state momenta by enforcing standard relativistic conservation of $E$ and $\vec{P}$ for two particles.
- Compute the M-First directional components for the same before/after states.
- Present both formulations side by side with a "Balance" column (Total Before - Total After); zero balances indicate the conservation laws agree.

## Further reading
The conceptual motivation and derivations are described in the paper ["Momentum Is All You Need"](https://www.authorea.com/users/933348/articles/1304846-momentum-is-all-you-need).
