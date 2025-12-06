# A Computational Validation of the Momentum-First Kinematic Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project numerically validates that the Momentum-First (M-First) postulate of absolute directional momentum conservation matches the standard relativistic conservation of energy and vector momentum for two-particle elastic scattering. The demo prints both formulations so you can see every component balance to zero.

## Run it

```bash
pip install -r requirements.txt
python demonstrations/run_basic_demo.py
```

## Current demo scenarios
- Head-on collision: particle A rebounds straight back from a stationary partner.
- Glancing blow: particle A scatters 45 degrees in the x-z plane.
- Massless particle scattering: photon-like particle deflects 90 degrees into the y-axis.
- More scenarios coming soon.

## Demo output
Below is the output from a fresh run of `python demonstrations/run_basic_demo.py`:

```
================================================================================
  Running M-First Kinematics Validation Demonstrations
================================================================================

--- SCENARIO 1: Head-On Collision ---
A particle (m=3) with momentum [0, 0, 4] hits a stationary particle (m=6).
Particle A is assumed to scatter directly backward.

--- Standard Conservation Table ---

           A_before  B_before  Total (Before)   A_after  B_after  Total (After)  Balance
Component
px         0.000000  0.000000        0.000000  0.000000 0.000000       0.000000 0.000000
py         0.000000  0.000000        0.000000  0.000000 0.000000       0.000000 0.000000
pz         4.000000  0.000000        4.000000 -1.028571 5.028571       4.000000 0.000000
Energy     5.000000  6.000000       11.000000  3.171429 7.828571      11.000000 0.000000

================================================================================

--- Momentum-First Conservation Table ---

           A_before  B_before  Total (Before)  A_after   B_after  Total (After)  Balance
Component
x+         5.000000  6.000000       11.000000 3.171429  7.828571      11.000000 0.000000
x-         5.000000  6.000000       11.000000 3.171429  7.828571      11.000000 0.000000
y+         5.000000  6.000000       11.000000 3.171429  7.828571      11.000000 0.000000
y-         5.000000  6.000000       11.000000 3.171429  7.828571      11.000000 0.000000
z+         3.000000  6.000000        9.000000 3.685714  5.314286       9.000000 0.000000
z-         7.000000  6.000000       13.000000 2.657143 10.342857      13.000000 0.000000

================================================================================

--- SCENARIO 2: Glancing Blow ---
A particle (m=3) with momentum [0, 0, 4] hits a stationary particle (m=6).
Particle A is assumed to scatter at 45 degrees in the x-z plane.

--- Standard Conservation Table ---

           A_before  B_before  Total (Before)  A_after   B_after  Total (After)   Balance
Component
px         0.000000  0.000000        0.000000 2.235572 -2.235572       0.000000  0.000000
py         0.000000  0.000000        0.000000 0.000000  0.000000       0.000000  0.000000
pz         4.000000  0.000000        4.000000 2.235572  1.764428       4.000000  0.000000
Energy     5.000000  6.000000       11.000000 4.358390  6.641610      11.000000 -0.000000

================================================================================

--- Momentum-First Conservation Table ---

           A_before  B_before  Total (Before)  A_after  B_after  Total (After)   Balance
Component
x+         5.000000  6.000000       11.000000 3.240604 7.759396      11.000000 -0.000000
x-         5.000000  6.000000       11.000000 5.476175 5.523825      11.000000 -0.000000
y+         5.000000  6.000000       11.000000 4.358390 6.641610      11.000000 -0.000000
y-         5.000000  6.000000       11.000000 4.358390 6.641610      11.000000 -0.000000
z+         3.000000  6.000000        9.000000 3.240604 5.759396       9.000000 -0.000000
z-         7.000000  6.000000       13.000000 5.476175 7.523825      13.000000 -0.000000

================================================================================

--- SCENARIO 3: Massless Particle Scattering ---
A massless particle (m=0) with momentum [0, 0, 5] hits a stationary particle (m=4).
The massless particle is assumed to scatter at 90 degrees into the y-axis.

--- Standard Conservation Table ---

           A_before  B_before  Total (Before)  A_after   B_after  Total (After)   Balance
Component
px         0.000000  0.000000        0.000000 0.000000  0.000000       0.000000  0.000000
py         0.000000  0.000000        0.000000 2.222222 -2.222222       0.000000  0.000000
pz         5.000000  0.000000        5.000000 0.000000  5.000000       5.000000  0.000000
Energy     5.000000  4.000000        9.000000 2.222222  6.777778       9.000000 -0.000000

================================================================================

--- Momentum-First Conservation Table ---

           A_before  B_before  Total (Before)  A_after  B_after  Total (After)   Balance
Component
x+         5.000000  4.000000        9.000000 2.222222 6.777778       9.000000 -0.000000
x-         5.000000  4.000000        9.000000 2.222222 6.777778       9.000000 -0.000000
y+         5.000000  4.000000        9.000000 1.111111 7.888889       9.000000 -0.000000
y-         5.000000  4.000000        9.000000 3.333333 5.666667       9.000000 -0.000000
z+         2.500000  4.000000        6.500000 2.222222 4.277778       6.500000 -0.000000
z-         7.500000  4.000000       11.500000 2.222222 9.277778      11.500000 -0.000000

================================================================================

All demonstrations complete.
```

## Theory background
- Full explanation of Absolute Directional Momentum and the equivalence proof lives in `docs/theory.md`.
- The theoretical basis is outlined in the paper ["Momentum Is All You Need"](https://www.authorea.com/users/695998/articles/713606-momentum-is-all-you-need).

## Running the tests

```bash
pytest
```
