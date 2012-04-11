# stardard library
import sys, os

# external libraries
import numpy as np
from ray import imio, evaluate

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
