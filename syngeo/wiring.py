# stardard library
import sys, os
import json
import cPickle as pck
import itertools as it

# external libraries
import numpy as np
from gala import imio, evaluate, morpho
import networkx as nx

def synapses_to_network(vol, syns):
    """Compute a wiring diagram from a volume and synapse locations."""
    network = nx.MultiDiGraph()
    for pre, posts in syns:
        for post in posts:
            network.add_edge(vol[tuple(pre)], vol[tuple(post)])
    return network

