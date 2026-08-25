# SMSOCP

SMSOCP is a Python package built on CasADi for solving nonlinear optimal
control problems with inequality path constraints. It combines
integral-transcription path-constraint upper bounds with multiple shooting to
guarantee satisfaction of path constraints over continuous time.

The SMS algorithm solves a sequence of nonlinear programming subproblems. It
adaptively refines checking intervals associated with active path constraints
until the KKT conditions satisfy a user-specified tolerance.

## Capabilities

- Define reusable continuous-time dynamic models and optimal-control problems.
- Solve nonlinear OCPs through multiple shooting and SMS-IA integral
  transcription.
- Adaptively refine checking intervals and assess KKT conditions.

## Installation

SMSOCP requires Python 3.10 or later. After cloning the repository, install the
package and the dependencies required by the runnable examples:

```powershell
cd SMSOCP
pip install -e ".[examples]"
```

Verify that Python imports the local package:

```powershell
python -c "import sms_ocp; print(sms_ocp.__file__)"
```

## Quick start

Run the constrained double-integrator example from the project root:

```powershell
python examples/solve_constrained_double_integrator.py
```

The example solves

```text
minimize    1/2 ∫ u(t)^2 dt

subject to  x1_dot = x2
            x2_dot = u
            x(0) = (0, 1)
            x(1) = (0, -1)
            x1(t) <= 1/9,  for all t in [0, 1].
```

It demonstrates the complete SMSOCP workflow: dynamic-model definition,
optimal-control-problem construction, SMS-IA path-constraint transcription,
Phase I feasibility restoration, Phase II optimization, and KKT checking.
The script writes a numerical solution and a path-constraint figure to
`examples/outputs/`.

## Public API

The primary user-facing interface is available directly from `sms_ocp`:

```python
from sms_ocp import (
    BaseDynamicModel,
    OptimalControlProblem,
    SMSAlgorithmOptions,
    SMSIAOptions,
    SMSKKTOptions,
    pack_initial_guess,
    solve_sms_ocp,
)
```

The usual workflow is:

1. Subclass `BaseDynamicModel` to define `xdot = f(t, x, u, p)`.
2. Create an `OptimalControlProblem` and add objectives, variable bounds, and
   path, initial, and terminal constraints.
3. Choose a normalized shooting grid and construct an initial guess with
   `pack_initial_guess`.
4. Call `solve_sms_ocp` with optional SMS-IA, algorithm, and KKT settings.

`SMSIAOptions` configures the initial SMS-IA approximation, including
smoothing and checking-interval overrides. `SMSAlgorithmOptions` configures
the refinement procedure. `SMSKKTOptions` configures sampled KKT checking.

## Control parameterization

Controls are currently parameterized as piecewise constant over the shooting
intervals.

## Examples

The repository includes examples from multiple application domains. See
[examples/README.md](examples/README.md) for the runnable scripts and
reference results, and [examples/SOURCES.md](examples/SOURCES.md) for the
authoritative source of each formulation.

## Testing

Install the test dependency and run the test suite from the project root:

```powershell
pip install -e ".[test]"
pytest
```

## License

SMSOCP is released under the [GPL-3.0-only license](LICENSE).
