# 2D-and-3D-Gravity-data-inversion
We present a numerical method to estimate gravity anomalies in the Fourier domain for complex topography. It combines Nagy’s formula, Gauss-FFT, and Parker’s approach. Using differential evolution, it reconstructs basement geometry accurately, validated on synthetic and real 3D basin models.

Objectives

Develop an efficient forward modeling approach for 3D gravity data

Handle complex basin geometry and variable density

Reconstruct basement depth using global optimization

Improve computational efficiency using FFT-based methods

Input Data

The model requires:

Gravity anomaly data – Observed surface gravity values

Grid coordinates (x, y, z) – Spatial discretization

Density model (Δρ) – Constant or depth-varying

Initial basin geometry – Defined via control points

Methodology
1. Forward Modeling

Two approaches are used:

a. Space Domain (Nagy’s Method)

Models gravity response of rectangular prisms

High accuracy but computationally expensive

b. Frequency Domain (Gauss-FFT)

Uses Fourier transform for fast computation

Vertical integration via Gaussian quadrature

Efficient for large-scale 3D problems

2. Model Parameterization

Basin geometry represented using cubic B-spline surfaces

Defined by a grid of control points (e.g., 5×5, 8×8)

Ensures smooth and realistic subsurface representation

3. Objective Function

The inversion minimizes the misfit between observed and modeled gravity:

Root Mean Square Error (RMSE)

Optional regularization for stability

4. Optimization (Differential Evolution)

Population-based global optimization method

Steps involved:

Initialization of population

Mutation

Crossover

Selection

Robust against local minima
