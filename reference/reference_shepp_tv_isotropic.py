"""Reference TV reconstruction for ellipse data."""

import numpy as np
import odl
from odl.contrib import fom

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

# Creat parallel beam geometry
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)

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
f = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(operator.range).translated(data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.26 * odl.solvers.GroupL1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
g = odl.solvers.SeparableSum(l2_norm, l1_norm)


# --- Select solver parameters and solve using Chambolle-Pock --- #


# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrint(lambda x: fom.psnr(x, phantom)) &
            odl.solvers.CallbackShow(clim=[0.1, 0.4]))

# Choose a starting point
x = pseudoinverse(data)

# Run the algorithm
odl.solvers.pdhg(
    x, f, g, op, niter=1000, gamma=0.3,
    callback=None)

print('psnr = {}'.format(fom.psnr(x, phantom)))

# Display images
x.show('Shepp-Logan TV windowed', clim=[0.1, 0.4])
