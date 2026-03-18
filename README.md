# 2D-and-3D-Gravity-data-inversion
We present a numerical method to estimate gravity anomalies in the Fourier domain for complex topography. It combines Nagy’s formula, Gauss-FFT, and Parker’s approach. Using differential evolution, it reconstructs basement geometry accurately, validated on synthetic and real 3D basin models.


For forward modeling, two approaches are used. In the space domain, Nagy’s analytical solution is applied to compute gravity effects of rectangular prisms, providing high accuracy but at a higher computational cost. In the frequency domain, a Gauss-FFT-based method is implemented, which combines Gaussian quadrature for vertical integration and Fast Fourier Transform for horizontal convolution. This significantly improves computational efficiency, especially for large-scale 3D problems.

The basin geometry is parameterized using cubic B-spline surfaces defined by a set of control points. This ensures a smooth, continuous, and geologically realistic representation of the subsurface while reducing the number of parameters required for inversion.

The inversion problem is formulated as an optimization task, where the objective is to minimize the difference between observed and modeled gravity anomalies using an RMSE-based cost function. Differential Evolution (DE), a population-based global optimization algorithm, is used to explore the model space efficiently and avoid local minima. In some cases, local optimization methods can be applied after DE for further refinement.

The workflow begins with input data, including observed gravity anomalies and an assumed density model. An initial basin model is defined using a B-spline control grid with specified parameter bounds. Forward modeling is then performed to compute the predicted gravity response. The misfit between observed and modeled data is calculated, and DE iteratively updates the model parameters through mutation, crossover, and selection. This process continues until convergence is achieved or a maximum number of iterations is reached. The final output includes the recovered basement geometry, modeled gravity response, and residual errors.

The method has been validated on both synthetic and real datasets, demonstrating its ability to accurately reconstruct simple and complex basin structures, including cases with variable density. The Gauss-FFT approach provides a significant reduction in computation time while maintaining high accuracy.

Overall, this framework offers a fast, flexible, and reliable solution for 3D gravity inversion. It is particularly useful for sedimentary basin analysis, basement depth estimation, and gravity data interpretation in geophysical studies, with potential for further improvements such as integration with seismic data and parallel computation.
