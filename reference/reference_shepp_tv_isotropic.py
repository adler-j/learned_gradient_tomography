"""Reference TV reconstruction for ellipse data."""

import numpy as np
import odl


# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

# Creat parallel beam geometry
geometry = odl.tomo.parallel_beam_geometry(space, angles=30)

# Create ray transform operator
operator = odl.tomo.RayTransform(space, geometry)

# Create pseudoinverse
pseudoinverse = odl.tomo.fbp_op(operator)


# --- Generate artificial data --- #


# Create phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create sinogram of forward projected phantom with noise
data = operator(phantom)
np.random.seed(0)
data += odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05


# --- Set up the inverse problem --- #


# Initialize gradient operator
gradient = odl.Gradient(space)

# Column vector of two operators
op = odl.BroadcastOperator(operator, gradient)

# Do not use the g functional, set it to zero.
g = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(operator.range).translated(data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.3 * odl.solvers.L1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 1000  # Number of iterations
tau = 0.1  # Step size for the primal variable
sigma = 1.0 / (op_norm ** 2 * tau)  # Step size for the dual variable
gamma = 0.1

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrint(lambda x: odl.util.psnr(phantom, x)) &
            odl.solvers.CallbackShow(clim=[0.1, 0.4]))

# Choose a starting point
x = pseudoinverse(data)

# Run the algorithm
odl.solvers.pdhg(
    x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma,
    callback=None)

print('psnr = {}'.format(odl.util.psnr(phantom, x)))

# Display images
x.show('Shepp-Logan TV windowed', clim=[0.1, 0.4])
