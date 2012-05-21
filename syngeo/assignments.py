# built-ins
import os
import argparse
from math import floor, ceil
from random import shuffle
import json
import cPickle as pck
import itertools as it
from operator import __or__, __not__
from functools import partial
import fnmatch

# external libraries
import numpy as np
from ray import imio
from scipy import spatial

# local modules
from syngeo import io

# "constants"
pat_session_data = '/groups/flyem/data/temp/temp-session/noise/rivlinp.20120508.noise1_0_ground_truth/sessiondata'
proofreaders = ['roxanne', 'leiann', 'shirley', 'ashley', 'omotara', 'mat', 'chris', 'satoko']
ress = ['5nm', '7.5nm', '10nm', '15nm']
noise_levels = [0, 1, 2, 4, 8, 16]
d = '/groups/flyem/data/image_analysis_experiment/assignments'

def compose(f1, f2):
    def f(*args, **kwargs):
        return f1(f2(*args, **kwargs))
    return f

def on_boundary(coords, volshape, margin=20):
    on = [(coords[i] < margin or coords[i] > volshape[i] - margin) 
        for i in range(len(coords))]
    return reduce(__or__, on)

def stratified_slices(total, nslices):
    if total % nslices == 0:
        size = total/nslices
        starts, ends = range(0, total, size), range(size, total+size, size)
    else:
        size_l = int(ceil(float(total)/nslices))
        size_s = int(floor(total/nslices))
        num_l = total % nslices
        num_s = nslices - num_l
        switch = num_l * size_l
        starts = range(0, switch, size_l) + range(switch, total, size_s)
        ends = range(size_l, switch, size_l) + range(switch, total+1, size_s)
    return [slice(s, e) for s, e in zip(starts, ends)]

def tracing_accuracy_matrix(base_dir, gt_vol, proofreaders=proofreaders, 
                nposts=161, assign_dir='assignments', exports_dir='exports'):
    """Return synapse-tracing accuracy statistics.

    Note: assume all proofreaders have same set of assignments.
    """
    assign_dir = os.path.join(base_dir, assign_dir)
    num_conds = len(os.listdir(os.path.join(assign_dir, proofreaders[0])))
    conds = [
        proofreader_stack_to_resolution(assign_dir, proofreaders[0], '%02i'%c)
        for c in range(num_conds)]
    ress = np.unique(zip(*conds)[0])
    noises = np.unique(zip(*conds)[1])
    acc = np.zeros((nposts, len(proofreaders), len(ress), len(noises), 2))
    all_coords = []
    unc = np.zeros((nposts, len(proofreaders), len(ress), len(noises)))
    for c in range(num_conds):
        for p, prf in enumerate(proofreaders):
            r, n = proofreader_stack_to_resolution(assign_dir, prf, '%02i'%c)
            ri, ni = np.flatnonzero(r==ress), np.flatnonzero(n==noises)
            fn = os.path.join(base_dir, exports_dir, prf, '%02i'%c, 
                'annotations-bookmarks.json')
            coords, uncertain = bookmark_coords(fn, r/10.0)
            all_coords.extend(coords)
            ids, starts, ends = map(compose(np.floor, np.array), zip(*coords))
            ids = ids.astype(int)
            for i in range(starts.ndim):
                if starts[0, i] < 0:
                    starts[:, i] += gt_vol.shape[i]
                    ends[:, i] += gt_vol.shape[i]
            startsr = np.ravel_multi_index(starts.T.astype(int), gt_vol.shape)
            endsr = np.ravel_multi_index(ends.T.astype(int), gt_vol.shape)
            acc[ids, p, ri, ni, 1] += 1
            start_ids = gt_vol.ravel()[startsr]
            end_ids = gt_vol.ravel()[endsr]
            if (ids == 157).sum() > 0:
                start_ids[np.flatnonzero(ids==157)] = 277078
            correct = np.flatnonzero(start_ids == end_ids)
            acc[ids[correct], p, ri, ni, 0] += 1
            unc[uncertain, p, ri, ni] += 1
    return acc, all_coords, unc


def proofreader_stack_to_resolution(assign_dir, proofreader, stack_num_str, 
                                                        prefix='postsyn'):
    d = os.path.join(assign_dir, proofreader)
    fn = fnmatch.filter(os.listdir(d), prefix+'-'+stack_num_str+'-*.json')[0]
    fnsplit = fn.rsplit('.', 1)[0].split('-')
    return float(fnsplit[3].rstrip('unm')), int(fnsplit[5])

def get_bookmark_distances(bookmarks):
    return [(nid, spatial.distance.euclidean(loc1, loc2)) 
        for nid, loc1, loc2 in bookmarks]

def bookmark_coords(fn, scale=1.0, transform=True):
    """Return triples of (bookmark-id, start-coord, end-coord)."""
    bookmarks = {}
    uncertain = []
    with open(fn, 'r') as f: data = json.load(f)['data']
    for textid, location in sorted([(d['text'], 
            scale * io.raveler_to_numpy_coord_transform(d['location']))
            for d in data]):
        if textid.endswith('-start'):
            nid = int(textid.split('-')[0])
            if bookmarks.has_key(nid):
                bookmarks[nid].insert(0, location)
            else:
                bookmarks[nid] = [location]
        else:
            nid = int(textid.rstrip('?'))
            if bookmarks.has_key(nid):
                bookmarks[nid].append(location)
            else:
                bookmarks[nid] = [location]
            if textid.endswith('?'):
                uncertain.append(nid)
    return [(nid, loc1, loc2) for nid, (loc1, loc2) in bookmarks.items()], \
        uncertain

parser = argparse.ArgumentParser(
    description='Create synapse tracing assignments for proofreaders.')
parser.add_argument('session/json', help='The session or synapse json '+
    'containing the synapses to be traced.')
parser.add_argument('output-dir', help='Where to write the data.')
parser.add_argument('-P', '--proofreaders', nargs='+',
    help='The names of all the proofreaders.')
parser.add_argument('-r', '--resolutions', type=str, nargs='+', default=ress,
    help='The resolutions of the base stacks.')
parser.add_argument('-R', '--annotation-resolution', type=float,
    help='The resolution at which the annotations were produced.')
parser.add_argument('-N', '--noise-levels', type=int, nargs='+',
    default=noise_levels, help='The noise levels of the base stacks.')
parser.add_argument('-s', '--shape', type=partial(eval, globals={}),
    help='The shape of the volume.', default=(1500, 1100, 280))
parser.add_argument('-m', '--margin', type=int, default=20,
    help='The margin in which to remove postsynaptics.')

if __name__ == '__main__':
    args = parser.parse_args()

    posts = io.all_postsynaptic_sites(io.synapses_from_raveler_session_data(
        getattr(args, 'session/json')))
    posts2 = filter(compose(__not__,
        partial(on_boundary, volshape=args.shape, margin=args.margin)), posts)
    posts2 = [p for p in posts2 if not on_boundary(p, args.shape, 20)]
    float_ress = np.array([float(r.rstrip('unm')) for r in args.resolutions])
    relative_float_ress = float_ress / args.annotation_resolution

    npo = len(posts2)
    ids = range(npo)
    aps = np.array(posts2)
    apss = [(aps/r).round() for r in relative_float_ress]
    apss = [np.concatenate((np.array(ids)[:, np.newaxis], ap), axis=1)
        for ap in apss]
    apsd = {n: ap for n, ap in zip(args.resolutions, apss)}
    conds = list(it.product(args.resolutions, args.noise_levels))
    which = stratified_slices(npo, len(conds))
    for pr in args.proofreaders:
        shuffle(conds)
        shuffle(ids)
        odir = os.path.join(getattr(args, 'output-dir'), pr)
        os.makedirs(odir)
        for i, (r, n) in enumerate(conds):
            locations = map(list, list(apsd[r]))
            locations = [locations[j] for j in ids[which[i]]]
            bkmarks = [{'location': loc[1:], 'body ID': -1, 
                'text': str(loc[0])+'-start'} for loc in locations]
            bmdict = {'metadata': 
                {'description': 'bookmarks', 'file version': 1},
                'data': bkmarks}
            fn = os.path.join(odir, 
                            'postsyn-%02i-res-%s-noise-%02i.json' % (i, r, n))
            with open(fn, 'w') as f:
                json.dump(bmdict, f, indent=4)
