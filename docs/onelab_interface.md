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

The physics-specific section must match the `Problem/Physics` value.

## Magnetostatics

```text
Magnetostatics {
  CurrentRmsParameter "Input/4Coil Parameters/0Current (rms) [A]";
  NumberOfTurnsParameter "Input/4Coil Parameters/1Number of turns";
  CoreRelativePermeabilityParameter "Input/42Core relative permeability";
  CoilWidthParameter "Input/10Geometric dimensions/03Coil width [m]";
  CoilHeightParameter "Input/10Geometric dimensions/04Coil height [m]";

  CurrentRms 10.0;
  NumberOfTurns 288.0;
  CoreRelativePermeability 2000.0;
  CoilWidth 0.03;
  CoilHeight 0.09;

}
```

Semantics:

- `*Parameter` entries point to ONELAB numeric inputs to read first.
- scalar values are fallback defaults when the ONELAB model does not provide them.
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
