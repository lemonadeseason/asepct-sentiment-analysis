"""
Basic operations on trees.
"""

import numpy as np
from collections import defaultdict

import copy

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_):    #return the root of this tree
    """
    Convert a sequence of head indexes into a tree object.
    """
    if isinstance(head, list) == False:     #convert from tensor to list
        tokens = tokens[:len_].tolist()
        head = head[:len_].tolist()
    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i    #下标？可以随意添加？
        nodes[i].dist = -1 # just a filler
        if h == 0:
            root = nodes[i]     #如果在head中，某一处的值为0，就代表在dependency parsing中，此处为root
        else:
            nodes[h-1].add_child(nodes[i])

    assert root is not None
    return root

def tree_to_adj(sent_len, tree, directed=False, self_loop=True):  #direct是有向，un是无
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]   #队列中一开始只有root
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]    #dequeue

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1     #head行、dependent列为1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret

