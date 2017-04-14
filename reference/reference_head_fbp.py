"""Reference FBP for head data."""

import numpy as np
import odl

mu_water = 0.02
photons_per_pixel = 10000.0
epsilon = 1.0 / photons_per_pixel
use_artificial_data = True

# Create ODL data structures
size = 512
space = odl.uniform_discr([-128, -128], [128, 128], [size, size],
                          dtype='float32', weighting='const')

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
pseudoinverse = odl.tomo.fbp_op(operator, filter_type='Hann',
                                frequency_scaling=0.5)

# Create nonlinear forward operator using composition and ufuncs
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
data = odl.phantom.poisson_noise(data * photons_per_pixel) / photons_per_pixel

# Reconstruct using FBP
recon = pseudoinverse(-np.log(epsilon + data) / mu_water)

print('psnr = {}'.format(odl.util.psnr(phantom, recon)))

# Display results
phantom.show('phantom', clim=[0.8, 1.2])
data.show('data')
recon.show('recon', clim=[0.8, 1.2])
