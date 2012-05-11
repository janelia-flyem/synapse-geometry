# stardard library
import sys, os
import json

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
    coords = coords[:, t]*s
    neuron_ids = neurons[zip(*coords)]
    mean_coords = coords.mean(axis=0).astype(np.uint)
    neurons = get_box(neurons, mean_coords, margin)
    synapse_volume = reduce(add_anything, 
        [(i+1)*(neurons==j) for i, j in enumerate(neuron_ids)])
    imio.write_vtk(synapse_volume, fn)
    if im is not None:
        im = get_box(im, mean_coords, margin)
        imio.write_vtk(im, 
            os.path.join(os.path.dirname(fn), 'image.' + os.path.basename(fn)))

def all_postsynaptic_sites(synapses):
    tbars, posts = zip(*synapses)
    return list(it.chain(*posts))

def get_box(a, coords, margin):
    """Obtain a box of size 2*margin+1 around coords in array a.

    Boxes close to the boundary are trimmed accordingly.
    """
    if margin is None:
        return a
    coords = np.array(coords)[np.newaxis, :]
    origin = np.zeros(coords.shape, dtype=int)
    shape = np.array(a.shape)[np.newaxis, :]
    topleft = np.concatenate((coords-margin, origin), axis=0).max(axis=0)
    bottomright = np.concatenate((coords+margin+1, shape), axis=0).min(axis=0)
    box = [slice(top, bottom) for top, bottom in zip(topleft, bottomright)]
    return a[box].copy()

def synapses_from_raveler_session_data(fn):
    with open(fn) as f:
        d = pck.load(f)
    annots = d['annotations']['point']
    tbars = [a for a in annots if annots[a]['kind'] == 'T-bar']
    posts = [annots[a] for a in tbars]
    posts = [eval(p['value'].replace('false', 'False').replace('true', 'True'))
             for p in posts]
    posts = [p['partners'] for p in posts]
    posts = [map(lambda x: x[0], p) for p in posts]
    return zip(tbars, posts)


def raveler_synapse_annotations_to_coords(fn, output_format='pairs'):
    """Obtain pre- and post-synaptic coordinates from Raveler annotations."""
    with open(fn, 'r') as f:
        syn = json.load(f)['data']
    tbars = [np.array(s['T-bar']['location']) for s in syn]
    posts = [np.array([p['location'] for p in s['partners']]) for s in syn]
    if output_format == 'pairs':
        return zip(tbars, posts)
    elif output_format == 'arrays':
        return [np.concatenate((t[np.newaxis, :], p), axis=0)
                                        for t, p in zip(tbars, posts)]

def write_all_synapses_to_vtk(neurons, list_of_coords, fn, im, t=(2,0,1), 
        s=(1,-1,1), margin=None, single_pairs=True):
    for i, coords in enumerate(list_of_coords):
        if single_pairs:
            pre = coords[0]
            for j, post in enumerate(coords[1]):
                pair_coords = np.concatenate(
                    (pre[np.newaxis, :], post[np.newaxis, :]), axis=0)
                fn = fn%(i, j)
                write_synapse_to_vtk(neurons, pair_coords, fn, im, t, s, margin)
        else:
            fn = fn%i
            write_synapse_to_vtk(neurons, coords, fn, im, t, s, margin)
