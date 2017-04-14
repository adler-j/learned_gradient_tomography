"""Reference FBP reconstruction for ellipse data."""

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
pseudoinverse = odl.tomo.fbp_op(operator, filter_type='Hann')


# --- Generate artificial data --- #


# Create phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create sinogram of forward projected phantom with noise
data = operator(phantom)
data += odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05

recon = pseudoinverse(data)

print('psnr = {}'.format(odl.util.psnr(phantom, recon)))

# Display images
data.show('Data')
recon.show('Shepp-Logan FBP', clim=[0.1, 0.4])
