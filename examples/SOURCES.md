# Example sources

Each example below has one authoritative source. The cited source is the one
used for the mathematical formulation implemented in its solver script.
Examples developed within SMS-OCP are identified explicitly instead of being
assigned an unrelated external reference.

## Space Shuttle Reentry Trajectory

J. T. Betts, *Practical Methods for Optimal Control and Estimation Using
Nonlinear Programming*, 2nd ed., SIAM, 2010, Section 6.1.

Implementation note: the dynamics and path constraint are expressed in scaled
coordinates, and finite state and terminal-time bounds are added for numerical
robustness.

## Constrained Double Integrator

A. E. Bryson and Y.-C. Ho, *Applied Optimal Control: Optimization,
Estimation, and Control*, Hemisphere Publishing, 1975.

## Constrained Cart-Pole Swing-Up

Experimental model developed in this project.

## Jacobson–Lele State-Constraint Example

D. H. Jacobson and M. M. Lele, “A Transformation Technique for Optimal
Control Problems with a State Variable Inequality Constraint,” *IEEE
Transactions on Automatic Control*, 1969, Example 1.

## High-Frequency Narrow-Corridor Problem

Experimental model developed in this project.

## Penicillin Fed-Batch Fermentation

J. Fu, J. M. M. Faust, B. Chachuat, and A. Mitsos, “Local Optimization of
Dynamic Programs with Guaranteed Satisfaction of Path Constraints,”
*Automatica*, 2015, Appendix A.1.

Implementation note: positive lower bounds are additionally imposed on the
states to preserve their physical domain during numerical integration.

## Rayleigh Mixed-Constraint Problem

M. Gerdts, “Global Convergence of a Nonsmooth Newton Method for
Control-State Constrained Optimal Control Problems,” *SIAM Journal on
Optimization*, 2008, Section 5.1.

## Robot Path Planning

The `Robot Path Planning` example distributed in the
[QuITO v.2 software repository](https://github.com/chatterjee-d/QuITOv2).

## State-Constrained Van der Pol Oscillator

T. W. C. Chen and V. S. Vassiliadis, “Inequality Path Constraints in
Optimal Control: A Finite Iteration ε-Convergent Scheme Based on Pointwise
Discretization,” *Journal of Process Control*, 2005, Section 6.1.
