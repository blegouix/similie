# ONELAB `.silpro` Interface

`similie_onelab` now uses a SimiLie-specific problem description file with extension `.silpro`.
The syntax is intentionally close to GetDP `.pro` files:

- named sections
- brace-delimited blocks
- `Key Value;` assignments
- quoted strings for paths and ONELAB parameter names
- `#` and `//` comments

## Supported structure

```text
Problem {
  Name "Problem name";
  Physics Magnetostatics;
  Solver MinimizeStrongFormulationResidual;
}

Solver {
  MaxIterations 10000;
  RelativeTolerance 1e-10;
  JacobiMaxBlockSize 1;
  UseMatrixFree 1;
}
```

For magnetostatics, the `.silpro` file currently only selects the physics and solver settings.
Problem-specific input quantities must come from the driving ONELAB model itself.

For the structured inductor example, the `.geo` file publishes:

- `Input/90SimiLie/0Coil current density magnitude z [A/m^2]`
- `Input/90SimiLie/1Core magnetic permeability [H/m]`

Semantics:

- `UseMatrixFree 1` keeps the matrix-free operator as the default backend.
- the ONELAB interface always checks hexahedricity and rectilinearity before solving.
- the ONELAB interface always exports its internal `.pos` view to the mesh output directory.

## Scalar Field

```text
ScalarFieldWithPowerCoupling {
  Mass 1.0;
  CouplingConstant 0.0;
  CouplingPower 4.0;
}
```

This physics is parsed and assembled by the interface, but the current ONELAB runtime path is still focused on the structured magnetostatics workflow.

## Example

The inductor example uses:

- [similie_linear_magnetostatics.silpro](/home/cart3sianbear/similie/examples/onelab_inductor/similie_linear_magnetostatics.silpro)
- [run_similie_onelab.sh](/home/cart3sianbear/similie/examples/onelab_inductor/run_similie_onelab.sh)

The shell script passes the `.silpro` file through:

`0Modules/SimiLie/0Control/Problem file`
