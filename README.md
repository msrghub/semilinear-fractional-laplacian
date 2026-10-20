# Semilinear Fractional Laplacian

This repository contains a Julia implementation for a semilinear fractional Laplacian problem with exterior Dirichlet boundary conditions on the interval \((-1,1)\).

## Numerical methods

The code compares six finite difference weight families:

- Grünwald–Letnikov (GL)
- Piecewise Constant (PC)
- Spectral (SP)
- Periodic (PER)
- T-type (T)
- Q-type (Q)

The nonlinear systems are solved using Newton's method and a damped Picard iteration.

## Requirements

Julia and the following packages are required:

```julia
using Pkg
Pkg.add(["SpecialFunctions", "HypergeometricFunctions", "Plots"])
```

## Run the code

Run the Julia script from the terminal:

```bash
julia semilinear_fractional_laplacian.jl
```

## Reproducibility

The numerical results depend on the fractional order, grid resolution, truncation parameter, and solver tolerances specified in the source code.

## Authors

Mohammad Sadegh Rezaei and Khadije Nedaiasl.
