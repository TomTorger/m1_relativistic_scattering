# Momentum-First Theory Notes

## Core claim
The M-First framework replaces the usual pair of conserved quantities (total vector momentum $\vec{P}$ and total energy $E$) with six independently conserved **directional momentum components** $p_{k^\pm}$, one for each half-axis $+x, -x, +y, -y, +z, -z$. For any isolated process, conserving those six components is exactly equivalent to conserving $E$ and $\vec{P}$ (i.e., four-momentum conservation).

## Directional momentum definition
For a single particle:
$$
p_{k^\pm} = M(p) \pm \tfrac{1}{2} p_k
$$
where the $+$/$-$ half-axis take the matching sign of $p_k$. Here $p_k$ is the standard momentum along axis $k$ and $M(p) = \sqrt{p_f^2 + p^2}$ is the **Core Momentum**. The fermic momentum is $p_f = m_0 c$ with $m_0$ the rest mass. These $p_{k^\pm}$ are additive over constituents and exchange under parity $p_k \mapsto -p_k$.

The linear change of variables is invertible:
$$
\begin{pmatrix} p_{k^+} \\ p_{k^-} \end{pmatrix}
= \begin{pmatrix} 1 & \tfrac{1}{2} \\ 1 & -\tfrac{1}{2} \end{pmatrix}
\begin{pmatrix} M \\ p_k \end{pmatrix}, \qquad
\begin{pmatrix} M \\ p_k \end{pmatrix}
= \begin{pmatrix} \tfrac{1}{2} & \tfrac{1}{2} \\ 1 & -1 \end{pmatrix}
\begin{pmatrix} p_{k^+} \\ p_{k^-} \end{pmatrix}.
$$
Summing over particles, $\sum (p_{k^+}+p_{k^-}) = 2 \sum M$ and $\sum (p_{k^+}-p_{k^-}) = \sum p_k$, so conserving both $p_{k^+}$ and $p_{k^-}$ on each axis is equivalent to conserving $M$ and the usual $p_k$ on that axis.

## Equivalence statement
For any isolated process:
$$
\sum_i p_{k^\pm,\text{initial}}^{(i)} = \sum_j p_{k^\pm,\text{final}}^{(j)} \quad \iff \quad
\begin{cases}
\sum_i E_{\text{initial}}^{(i)} = \sum_j E_{\text{final}}^{(j)} \\
\sum_i \vec{P}_{\text{initial}}^{(i)} = \sum_j \vec{P}_{\text{final}}^{(j)}
\end{cases}
$$
Assuming the standard dispersion relation $M = \sqrt{p_f^2 + \vec{p}^2}$, this is exactly the usual four-momentum conservation in a different basis. A uniqueness argument (additivity, isotropy, parity, local invertibility) fixes the affine form $p_{k^\pm} = \alpha M \pm \beta p_k$ up to a common scale; choosing the odd part to match $p_k$ and the even part to equal $2M$ recovers the canonical $\alpha=1, \beta=\tfrac{1}{2}$ used above.

## Immediate corollaries
- Speed–momentum relation: $\vec{v} = c\,\vec{p}/M$, so $v = p\,c/M$.
- Lorentz factor: $\gamma = M/p_f$ with $p_f = m_0 c$.
- Massless limit: $p_f \to 0$ gives $M \to \|\vec{p}\|$, $E \to c\|\vec{p}\|$, $v \to c$.
- Newtonian expansion: $E = m_0 c^2 + \tfrac{\vec{p}^{\,2}}{2 m_0} - \tfrac{\vec{p}^{\,4}}{8 m_0^3 c^2} + \cdots$.

## Simulation approach
- Solve for final-state momenta by enforcing standard relativistic conservation of $E$ and $\vec{P}$ for two particles.
- Compute the M-First directional components for the same before/after states.
- Present both formulations side by side with a Balance column; zero balances show the conserved quantities agree. The M-First table also reports Net_Before/Net_After (derived $p_k$ from $p_{k^\pm}$) on the $+$ rows for quick comparison to the standard table.
