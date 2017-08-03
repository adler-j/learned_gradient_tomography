"""Reference Huber reconstruction for ellipse data."""

import numpy as np
import odl


# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

# Creat parallel beam geometry
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)

# Create ray transform operator
operator = odl.tomo.RayTransform(space, geometry)

# Create pseudoinverse
pseudoinverse = odl.tomo.fbp_op(operator, filter_type='Hann')


# --- Generate artificial data --- #


# Create phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create sinogram of forward projected phantom with noise
data = operator(phantom)
data += odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05


# --- Set up optimization problem and solve --- #

# Create data term ||Ax - b||_2^2 as composition of the squared L2 norm and the
# ray trafo translated by the data.
l2_norm = odl.solvers.L2NormSquared(operator.range)
data_discrepancy = l2_norm * (operator - data)

# Create initial estimate of the inverse Hessian by a diagonal estimate
opnorm = odl.power_method_opnorm(operator)
hessinv_estimate = odl.ScalingOperator(space, 0.5 / opnorm ** 2)

sigma = 0.03
lam = 0.3

# Create regularizing functional || |grad(x)| ||_1 and smooth the functional
# using the Moreau envelope.
# The parameter sigma controls the strength of the regularization.
gradient = odl.Gradient(space)
l1_norm = odl.solvers.GroupL1Norm(gradient.range)
smoothed_l1 = odl.solvers.MoreauEnvelope(l1_norm, sigma=sigma)
regularizer = smoothed_l1 * gradient

# Create full objective functional
obj_fun = data_discrepancy + lam * regularizer

# Pick parameters
maxiter = 200
num_store = 10  # only save some vectors (Limited memory)


# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrint(lambda x: odl.util.psnr(phantom, x)) &
            odl.solvers.CallbackShow(clim=[0.1, 0.4]))

# Choose a starting point
x = pseudoinverse(data)

# Run the algorithm
odl.solvers.bfgs_method(
    obj_fun, x, maxiter=maxiter, num_store=num_store,
    hessinv_estimate=hessinv_estimate,
    callback=callback)

print('psnr = {}'.format(odl.util.psnr(phantom, x)))

# Display images
x.show('Shepp-Logan TV windowed', clim=[0.1, 0.4])


