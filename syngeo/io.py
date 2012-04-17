# stardard library
import sys, os

# external libraries
import numpy as np
from ray import imio, evaluate, morpho

def add_anything(a, b):
    return a + b

def write_synapse_to_vtk(neurons, coords, fn, im=None, t=(2,0,1), s=(1,-1,1),
        margin=None):
    """Output neuron shapes around pre- and post-synapse coordinates.
    
    The coordinate array is a (n+1) x m array, where n is the number of 
    post-synaptic sites (fly neurons are polyadic) and m = neurons.ndim, the
    number of dimensions of the image.
    """
    neuron_ids = neurons[zip(*(coords[:,t]*s))]
    synapse_volume = reduce(add_anything, 
        [(i+1)*(neurons==j) for i, j in enumerate(neuron_ids)])
    imio.write_vtk(synapse_volume, fn)
    if im is not None:
        imio.write_vtk(im, 
            os.path.join(os.path.dirname(fn), 'image.' + os.path.basename(fn)))

def get_box(a, coords, margin):
    """Obtain a box of size 2*margin+1 around coords in array a.

    Boxes close to the boundary are trimmed accordingly.
    """
    coords = np.array(coords)[np.newaxis, :]
    origin = np.zeros(coords.shape, dtype=int)
    shape = np.array(a.shape)[np.newaxis, :]
    topleft = np.concatenate((coords-margin, origin), axis=0).max(axis=0)
    bottomright = np.concatenate((coords+margin+1, shape), axis=0).min(axis=0)
    box = [slice(top, bottom) for top, bottom in zip(topleft, bottomright)]
    return a[box].copy()
