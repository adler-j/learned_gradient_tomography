"""Reference TV reconstruction for head data."""

import numpy as np
import odl
from odl.contrib import fom

mu_water = 0.02
photons_per_pixel = 10000.0
epsilon = 1.0 / photons_per_pixel
use_artificial_data = True

# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting=1.0)

# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 1000, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
# Detector: uniformly sampled, n = 1000, min = -360, max = 360
detector_partition = odl.uniform_partition(-360, 360, 1000)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)

# Create ray transform operator
operator = odl.tomo.RayTransform(space, geometry)

# Create pseudoinverse
pseudoinverse = odl.tomo.fbp_op(operator, filter_type='Hann')

# Create nonlinear forward operator using composition
nonlinear_operator = odl.ufunc_ops.exp(operator.range) * (- mu_water * operator)

# --- Generate artificial data --- #
if use_artificial_data:
    # Example with generated data
    phantom = odl.phantom.shepp_logan(space)
else:
    # Use true data
    file_path = ''  # INSERT FILE PATH
    phantom = space.element(np.rot90(np.load(file_path), -1))
    phantom /= 1000.0  # convert go g/cm^3

# Create artificial data and add noise
data = nonlinear_operator(phantom)
noisy_data = odl.phantom.poisson_noise(data * photons_per_pixel) / photons_per_pixel


# --- Set up the inverse problem --- #


# Initialize gradient operator
gradient = odl.Gradient(space)

# Column vector of two operators
# scaling the operator acts as a pre-conditioner, improving convergence.
op = odl.BroadcastOperator(nonlinear_operator, gradient)

# Do not use the g functional, set it to zero.
f = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
data_discr = odl.solvers.KullbackLeibler(operator.range, noisy_data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.0002 * odl.solvers.L1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
g = odl.solvers.SeparableSum(data_discr, l1_norm)


# --- Select solver parameters and solve using Chambolle-Pock --- #

# Choose a starting point
x = pseudoinverse(-np.log(epsilon + noisy_data) / mu_water)

# Estimated operator norm to ensure ||K||_2^2 * sigma * tau < 1
op_norm = odl.power_method_opnorm(op.derivative(x))

niter = 1000  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
gamma = 0.01

# Pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrint(lambda x: fom.psnr(x, phantom)) &
            odl.solvers.CallbackShow(clim=[0.8, 1.2]))

odl.solvers.pdhg(
    x, f, g, op, tau=tau, sigma=sigma, niter=niter, gamma=gamma,
    callback=callback)

print('psnr = {}'.format(fom.psnr(phantom, x)))

# Display images
x.show('head TV', clim=[0.8, 1.2])
