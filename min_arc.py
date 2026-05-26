import random
from functools import partial
import itertools
from itertools import product
from collections import defaultdict
from typing import OrderedDict, Tuple, List, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from einops import rearrange


class VirtualArray:
    """A wrapper around numpy array that supports custom origin for indexing.
    
    Allows negative and positive indices relative to a custom origin point,
    useful for transformations centered at different points in the grid.
    
    Example:
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        vgrid = VirtualArray(grid, origin=(1, 1))
        vgrid[-1, -1]  # accesses grid[0, 0] = 1
        vgrid[0, 0]    # accesses grid[1, 1] = 5
        vgrid[1, 1]    # accesses grid[2, 2] = 9
    """
    
    def __init__(self, array, origin=(0, 0)):
        self.array = array
        self.origin = origin
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            # Translate only the first len(origin) dimensions (if they're integers, not slices)
            # Leave remaining dimensions as-is (for multi-dimensional arrays like granges.grid)
            translated = []
            for i, k in enumerate(key):
                if i < len(self.origin) and isinstance(k, int):
                    translated.append(k + self.origin[i])
                else:
                    translated.append(k)
            return self.array[tuple(translated)]
        else:
            # Single index (1D case)
            return self.array[key + self.origin[0]]
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            # Translate only the first len(origin) dimensions (if they're integers, not slices)
            translated = []
            for i, k in enumerate(key):
                if i < len(self.origin) and isinstance(k, int):
                    translated.append(k + self.origin[i])
                else:
                    translated.append(k)
            self.array[tuple(translated)] = value
        else:
            self.array[key + self.origin[0]] = value
    
    def __eq__(self, other):
        """Support comparison operations like vgrid == 5"""
        return self.array == other
    
    @property
    def shape(self): return self.array.shape
    
    @property
    def dtype(self): return self.array.dtype
    
    def __repr__(self): return f"VirtualArray(origin={self.origin}, shape={self.array.shape})\n{self.array}"


color_names = [
    "K",  # "black",
    "B",  # "blue",
    "R",  # "red",
    "G",  # "green",
    "Y",  # "yellow",
    "E",  # "grey",
    "P",  # "pink",
    "O",  # "orange",
    "T",  # "teal",
    "M",  # "maroon",
]

# neighbor positional relations
def at(x, y, x0, y0):
    # Return True if (x, y) is at (x0, y0).
    # e.g. at(*(2, 6), *(2, 6)) == True. (2, 6) is at (2, 6)
    return x == x0 and y == y0

def left_of(x, y, x0, y0):
    # Return True if (x, y) is the left neighbor of (x0, y0).
    # e.g. left_of(*(1, 6), *(2, 6)) == True. (1, 6) is the left neighbor of (2, 6)
    return x == x0 - 1 and y == y0

def right_of(x, y, x0, y0):
    # Return True if (x, y) is the right neighbor of (x0, y0).
    # e.g. right_of(*(3, 6), *(2, 6)) == True. (3, 6) is the right neighbor of (2, 6)
    return x == x0 + 1 and y == y0

def above(x, y, x0, y0):
    # Return True if (x, y) is the upper neighbor of (x0, y0).
    # e.g. above(*(2, 5), *(2, 6)) == True. (2, 5) is the upper neighbor of (2, 6)
    return x == x0 and y == y0 - 1

def below(x, y, x0, y0):
    # Return True if (x, y) is the lower neighbor of (x0, y0).
    # e.g. below(*(2, 7), *(2, 6)) == True. (2, 7) is the lower neighbor of (2, 6)
    return x == x0 and y == y0 + 1

def upper_left_of(x, y, x0, y0):
    # Return True if (x, y) is the upper left neighbor of (x0, y0).
    # e.g. upper_left_of(*(1, 5), *(2, 6)) == True. (1, 5) is the upper left neighbor of (2, 6)
    return x == x0 - 1 and y == y0 - 1

def lower_left_of(x, y, x0, y0):
    # Return True if (x, y) is the lower left neighbor of (x0, y0).
    # e.g. lower_left_of(*(1, 7), *(2, 6)) == True. (1, 7) is the lower left neighbor of (2, 6)
    return x == x0 - 1 and y == y0 + 1

def upper_right_of(x, y, x0, y0):
    # Return True if (x, y) is the upper right neighbor of (x0, y0).
    # e.g. upper_right_of(*(3, 5), *(2, 6)) == True. (3, 5) is the upper right neighbor of (2, 6)
    return x == x0 + 1 and y == y0 - 1

def lower_right_of(x, y, x0, y0):
    # Return True if (x, y) is the lower right neighbor of (x0, y0).
    # e.g. lower_right_of(*(3, 7), *(2, 6)) == True. (3, 7) is the lower right neighbor of (2, 6)
    return x == x0 + 1 and y == y0 + 1


# mirror and rotate positional relations
def mirror_x_of(x, y, x0, y0):
    # Return True if (x, y) is the mirror reflection of (x0, y0) across the y-axis.
    # e.g. mirror_x_of(*(-2, 6), *(2, 6)) == True. (-2, 6) is the mirror reflection of (2, 6) across the y-axis
    return x == -x0 and y == y0

def mirror_y_of(x, y, x0, y0):
    # Return True if (x, y) is the mirror reflection of (x0, y0) across the x-axis.
    # e.g. mirror_y_of(*(2, -6), *(2, 6)) == True. (2, -6) is the mirror reflection of (2, 6) across the x-axis
    return x == x0 and y == -y0

def mirror_origin_of(x, y, x0, y0):
    # Return True if (x, y) is the mirror reflection of (x0, y0) across the origin.
    # e.g. mirror_origin_of(*(-2, -6), *(2, 6)) == True. (-2, -6) is the mirror reflection of (2, 6) across the origin
    return x == -x0 and y == -y0

def mirror_diag1_of(x, y, x0, y0):
    # Return True if (x, y) is the mirror reflection of (x0, y0) across the line y = x.
    # e.g. mirror_diag1_of(*(6, 2), *(2, 6)) == True. (6, 2) is the mirror reflection of (2, 6) across y = x
    return x == y0 and y == x0

def mirror_diag2_of(x, y, x0, y0):
    # Return True if (x, y) is the mirror reflection of (x0, y0) across the line y = -x.
    # e.g. mirror_diag2_of(*(-6, -2), *(2, 6)) == True. (-6, -2) is the mirror reflection of (2, 6) across y = -x
    return x == -y0 and y == -x0

def rotate_cw_of(x, y, x0, y0):
    # Return True if (x, y) is the 90-degree clockwise rotation of (x0, y0) around the origin.
    # e.g. rotate_cw_of(*(6, -2), *(2, 6)) == True. (6, -2) is the 90° clockwise rotation of (2, 6)
    return x == y0 and y == -x0

def rotate_ccw_of(x, y, x0, y0):
    # Return True if (x, y) is the 90-degree counterclockwise rotation of (x0, y0) around the origin.
    # e.g. rotate_ccw_of(*(-6, 2), *(2, 6)) == True. (-6, 2) is the 90° counterclockwise rotation of (2, 6)
    return x == -y0 and y == x0


rel_pos2fn = {(0, 0): at, (-1, 0): left_of, (1, 0): right_of, (0, -1): above, (0, 1): below,
              (-1, -1): upper_left_of, (-1, 1): lower_left_of, (1, -1): upper_right_of, (1, 1): lower_right_of}

dxs = dys = [-1, 0, 1]
dys = [-1, 0, 1]
rel_positions = np.array([[dx, dy] for dx in dxs for dy in dys])
rel_positions = rel_positions[(rel_positions[:, 0] == 0) | (rel_positions[:, 1] == 0)]
# rel_positions = rel_positions[rel_positions[:, 0] != 0 & rel_positions[:, 1] != 0]
side_neighbor_relations_old = [rel_pos2fn[tuple(rel_pos)] for rel_pos in rel_positions]

# Convenience lists of relation functions
neighbor_relations = [at, left_of, right_of, above, below, upper_left_of, lower_left_of, upper_right_of, lower_right_of]
mirror_rotate_relations = [mirror_x_of, mirror_y_of, mirror_origin_of, mirror_diag1_of, mirror_diag2_of, rotate_cw_of, rotate_ccw_of]
all_relations = neighbor_relations + mirror_rotate_relations
side_neighbor_relations = [left_of, right_of, above, below]  # orthogonal
corner_neighbor_relations = [upper_left_of, lower_left_of, upper_right_of, lower_right_of]  # diagonal

# Reverse mapping: function name to function (auto-generated from all_relations)
rel_name2fn = {fn.__name__: fn for fn in all_relations}

def relation_type(rel_fn):
    if rel_fn.__name__ in [r.__name__ for r in neighbor_relations]:
        return 'neighbor'
    elif rel_fn.__name__ in [r.__name__ for r in mirror_rotate_relations]:
        return 'mirror_rotate'
    else:
        assert False, f'Unknown relation function: {rel_fn}'

# Direct computation mapping: given (x0, y0), compute the related position for each relation
# This is much more efficient than searching the entire grid
# Using function names (strings) as keys to avoid issues with function object identity in notebooks
_rel_fn_compute_map = {
    'at': lambda x0, y0: (x0, y0),
    'left_of': lambda x0, y0: (x0 - 1, y0),
    'right_of': lambda x0, y0: (x0 + 1, y0),
    'above': lambda x0, y0: (x0, y0 - 1),
    'below': lambda x0, y0: (x0, y0 + 1),
    'upper_left_of': lambda x0, y0: (x0 - 1, y0 - 1),
    'lower_left_of': lambda x0, y0: (x0 - 1, y0 + 1),
    'upper_right_of': lambda x0, y0: (x0 + 1, y0 - 1),
    'lower_right_of': lambda x0, y0: (x0 + 1, y0 + 1),
    'mirror_x_of': lambda x0, y0: (-x0, y0),
    'mirror_y_of': lambda x0, y0: (x0, -y0),
    'mirror_origin_of': lambda x0, y0: (-x0, -y0),
    'mirror_diag1_of': lambda x0, y0: (y0, x0),
    'mirror_diag2_of': lambda x0, y0: (-y0, -x0),
    'rotate_cw_of': lambda x0, y0: (y0, -x0),
    'rotate_ccw_of': lambda x0, y0: (-y0, x0),
}

# Wrapper to accept function objects (looks up by name)
def rel_fn_compute(rel_func, x0, y0):
    """Compute the related position for a given relation function and position."""
    return _rel_fn_compute_map[rel_func.__name__](x0, y0)

def rel_fn2name(rel_fn):
    return 'the ' + rel_fn.__name__.replace("above", "upper_of").replace("below", "lower_of") \
        .replace("_", " ").replace(" of", " neighbor of") \
        if rel_fn.__name__ != 'at' else 'at'

side_neighbor_fn_defs = """
def at(x,y, x0,y0):
    # Return True if (x,y) is at (x0,y0).
    # e.g. at(*(2,6), *(2,6)) == True. (2,6) is at (2,6)
    return x == x0 and y == y0
    
def left_of(x,y, x0,y0):
    # Return True if (x,y) is the left neighbor of (x0,y0).
    # e.g. left_of(*(1,6), *(2,6)) == True. (1,6) is the left neighbor of (2,6)
    return x == x0 - 1 and y == y0

def right_of(x,y, x0,y0):
    # Return True if (x,y) is the right neighbor of (x0,y0).
    # e.g. right_of(*(3,6), *(2,6)) == True. (3,6) is the right neighbor of (2,6)
    return x == x0 + 1 and y == y0

def above(x,y, x0,y0):
    # Return True if (x,y) is the upper neighbor of (x0,y0).
    # e.g. above(*(2,5), *(2,6)) == True. (2,5) is the upper neighbor of (2,6)
    return x == x0 and y == y0 - 1

def below(x,y, x0,y0):
    # Return True if (x,y) is the lower neighbor of (x0,y0).
    # e.g. below(*(2,7), *(2,6)) == True. (2,7) is the lower neighbor of (2,6)
    return x == x0 and y == y0 + 1
"""

corner_neighbor_fn_defs = """
def at(x,y, x0,y0):
    # Return True if (x,y) is at (x0,y0).
    # e.g. at(*(2,6), *(2,6)) == True. (2,6) is at (2,6)
    return x == x0 and y == y0

def upper_left_of(x,y, x0,y0):
    # Return True if (x,y) is the upper left neighbor of (x0,y0).
    # e.g. upper_left_of(*(1,5), *(2,6)) == True. (1,5) is the upper left neighbor of (2,6)
    return x == x0 - 1 and y == y0 - 1

def lower_left_of(x,y, x0,y0):
    # Return True if (x,y) is the lower left neighbor of (x0,y0).
    # e.g. lower_left_of(*(1,7), *(2,6)) == True. (1,7) is the lower left neighbor of (2,6)
    return x == x0 - 1 and y == y0 + 1

def upper_right_of(x,y, x0,y0):
    # Return True if (x,y) is the upper right neighbor of (x0,y0).
    # e.g. upper_right_of(*(3,5), *(2,6)) == True. (3,5) is the upper right neighbor of (2,6)
    return x == x0 + 1 and y == y0 - 1

def lower_right_of(x,y, x0,y0):
    # Return True if (x,y) is the lower right neighbor of (x0,y0).
    # e.g. lower_right_of(*(3,7), *(2,6)) == True. (3,7) is the lower right neighbor of (2,6)
    return x == x0 + 1 and y == y0 + 1
"""

mirror_rotate_fn_defs = """
def at(x,y, x0,y0):
    # Return True if (x,y) is at (x0,y0).
    # e.g. at(*(2,6), *(2,6)) == True. (2,6) is at (2,6)
    return x == x0 and y == y0
    
def mirror_x_of(x,y, x0,y0):
    # Return True if (x,y) is the mirror reflection of (x0,y0) across the y-axis.
    # e.g. mirror_x_of(*(-2,6), *(2,6)) == True. (-2,6) is the mirror reflection of (2,6) across the y-axis
    return x == -x0 and y == y0

def mirror_y_of(x,y, x0,y0):
    # Return True if (x,y) is the mirror reflection of (x0,y0) across the x-axis.
    # e.g. mirror_y_of(*(2,-6), *(2,6)) == True. (2,-6) is the mirror reflection of (2,6) across the x-axis
    return x == x0 and y == -y0

def mirror_origin_of(x,y, x0,y0):
    # Return True if (x,y) is the mirror reflection of (x0,y0) across the origin.
    # e.g. mirror_origin_of(*(-2,-6), *(2,6)) == True. (-2,-6) is the mirror reflection of (2,6) across the origin
    return x == -x0 and y == -y0

def mirror_diag1_of(x,y, x0,y0):
    # Return True if (x,y) is the mirror reflection of (x0,y0) across the line y = x.
    # e.g. mirror_diag1_of(*(6,2), *(2,6)) == True. (6,2) is the mirror reflection of (2,6) across y = x
    return x == y0 and y == x0

def mirror_diag2_of(x,y, x0,y0):
    # Return True if (x,y) is the mirror reflection of (x0,y0) across the line y = -x.
    # e.g. mirror_diag2_of(*(-6,-2), *(2,6)) == True. (-6,-2) is the mirror reflection of (2,6) across y = -x
    return x == -y0 and y == -x0

def rotate_cw_of(x,y, x0,y0):
    # Return True if (x,y) is the 90-degree clockwise rotation of (x0,y0) around the origin.
    # e.g. rotate_cw_of(*(6,-2), *(2,6)) == True. (6,-2) is the 90° clockwise rotation of (2,6)
    return x == y0 and y == -x0

def rotate_ccw_of(x,y, x0,y0):
    # Return True if (x,y) is the 90-degree counterclockwise rotation of (x0,y0) around the origin.
    # e.g. rotate_ccw_of(*(-6,2), *(2,6)) == True. (-6,2) is the 90° counterclockwise rotation of (2,6)
    return x == -y0 and y == x0
"""

grid_desc = {}
grid_desc['dict'] = """# A grid is represented as a dict of cells, 
# where the keys are the (x,y) coordinates of cells and the values are their colors. e.g.:"""

grid_desc['list'] = """# A grid is represented as a list of (x,y,color) tuples corresponding to its cells, 
# where x and y are the coordinates of a cell, and color is the color abbreviation of a cell. e.g.:"""

grid_desc['markdown table'] = """# A grid is represented as a markdown table. 
# Each of its cells has x and y coordinates and a color (e.g. k=Black, g=Green, b=Blue, y=Yellow, r=Red)."""

example_grid = np.array([[1, 0, 1], [2, 3, 4], [0, 1, 0]])
# verbalize_grid(example_grid, color_names, style='dict', color_quote='"')

instructions = """# Below are functions defining the positional relations between two cells (x,y) and (x0,y0) in a grid,
# where (x,y) is the candidate neighbor cell and (x0,y0) is the queried center cell.
{fn_defs}

# pos_relation_fn() is one of the above positional relation functions used to answer the example and test questions below.
# Infer this positional relation function from the given examples and answer the test question.
"""

system_prompt0 = "You are a helpful assistant."
system_prompt = "Give the final one-word answer directly without any intermediate thinking process."
system_prompt2 = 'Answer with the color directly without any intermediate thinking process.'
system_prompt3 = 'Respond with "Answer: <Color>" directly without any intermediate thinking process.'
# system_prompt = "By inferring from the given examples, what is the positional relation function, left_of, right_of, above or below?"
# system_prompt = "Give the final answer of function_name directly without any intermediate thinking process."


user_turn = '<|im_end|>\n<|im_start|>user'
assistant_turn = '<|im_end|>\n<|im_start|>assistant'


@dataclass
class GridRanges:
    prefix: dict[str, tuple[int, int]] = None
    grid: np.ndarray | VirtualArray = None
    suffix: dict[str, tuple[int, int]] = None

@dataclass
class Item:
    key: tuple[int, int]
    value: int
    key_ranges: dict[str, tuple[int, int]] = None # only used by query
    value_ranges: dict[str, tuple[int, int]] = None

@dataclass
class QA:
    query: Item
    answer: Item

@dataclass
class Example:
    input: np.ndarray | VirtualArray
    output: list[QA]
    input_ranges: GridRanges = None


class TokenizedStrBuffer:
    def __init__(self, tokenizer=None, base=0, space_token='Ġ', newline_token='Ċ', print_out=True):
        self.tokenizer = tokenizer
        self.base0 = self.base = base
        self.str = ''
        self.space_token = space_token
        self.newline_token = newline_token
        self.print_out = print_out

    def add(self, s, n_tokens=None, end='\n', return_ranges=False):
        s += end
        self.str += s
        ranges = None
        if self.tokenizer is not None:
            if n_tokens is None:
                tokens = self.tokenizer.tokenize(s)
                if return_ranges:
                    ranges = {t.replace(self.space_token, ' ').replace(self.newline_token, '\n'): (i, i + 1) 
                              for i, t in enumerate(tokens, start=self.base)}
                self.base += len(tokens)
            else:
                self.base += n_tokens + len(self.tokenizer.tokenize(end))
        if self.print_out: print(s, end='')
        return ranges
    
    def assert_and_count_tokens(self):
        if self.tokenizer is not None:
            assert len(self.tokenizer.tokenize(self.str)) == self.base - self.base0, \
                f'{len(self.tokenizer.tokenize(self.str))} != {self.base - self.base0}'
            return self.base - self.base0
        return None


def verbalize_grid(grid, color_map, grid_name='grid', transpose=True, style='dict', color_quote='', 
                   tokenizer=None, base=0, print_out=True):
    cq = color_quote
    tsb = TokenizedStrBuffer(tokenizer=tokenizer, base=base, print_out=print_out) # unuseful when tokenizer is None
    print = tsb.add
    granges = GridRanges()
    
    # Detect if grid is a VirtualArray and compute coordinate ranges
    is_virtual = isinstance(grid, VirtualArray)
    if is_virtual:
        ox, oy = grid.origin
        n, m = grid.shape
        x_range = range(-ox, n - ox)
        y_range = range(-oy, m - oy)
    else:
        ox, oy = 0, 0
        n, m = grid.shape
        x_range = range(n)
        y_range = range(m)
    
    if style == 'dict':  # default
        prefix = f'{grid_name} = {{'
        item_tokens = ['(', 'x', ',', 'y', '):', 'c', ',']
    elif style == 'list':
        prefix = f'{grid_name} = ['   
    elif style == 'markdown table':
        print(f'{grid_name}:')
        print(f'|   |' + '|'.join(f'x={x}' for x in x_range) + '|')
        print(f'|---|' + '|'.join('---'    for x in x_range) + '|')
    granges.prefix = print(prefix, return_ranges=True)
    
    # Wrap granges.grid with VirtualArray for consistent coordinate indexing
    raw_granges_grid = np.empty((n, m, len(item_tokens), 2), dtype=int)
    granges.grid = VirtualArray(raw_granges_grid, origin=(ox, oy))
    
    if transpose:
        for y in y_range:
            if style == 'dict':  # default
                base = tsb.base
                print(' ' + ', '.join(f'({x},{y}):{cq}{color_map[grid[x, y]]}{cq}' for x in x_range) + ',')
                for x in x_range:
                    granges.grid[x, y, :, 0] = base + np.arange(len(item_tokens))  # start
                    granges.grid[x, y, :, 1] = granges.grid[x, y, :, 0] + 1  # end
                    base += len(item_tokens)
            elif style == 'list':
                print(' ' + ', '.join(f'({x},{y},{cq}{color_map[grid[x, y]]}{cq})' for x in x_range) + ',')
            elif style == 'markdown table':
                print(f'|y={y}|' + '|'.join(f'{color_map[grid[x, y]]}' for x in x_range) + '|')
    else:
        for x in x_range:
            # print(' ' + ','.join(f'({x},{y},"{color_map[grid[x, y]]}")' for y in y_range) + ',')
            print(' ' + ','.join(f'{{x={x},y={y},color="{color_map[grid[x, y]]}"}}' for y in y_range) + ',')
    if style == 'dict': suffix = '}'
    elif style == 'list': suffix = ']'
    elif style == 'markdown table': suffix = ''
    # let '\n\n' merge with suffix to avoid tokenization discrepancy issue
    granges.suffix = print(suffix, end='\n\n', return_ranges=True)
    n_tokens = tsb.assert_and_count_tokens()
    return tsb.str, n_tokens, item_tokens, granges


def gen_puzzle(rel_functions, n=5, m=5, origin_position='corner', n_color=5, n_train=3, n_test=1, Q_train=4, Q_test=1, seed=None):
    """Generate puzzle with positional relations. All rel_functions should be callable functions."""
    if seed is not None: np.random.seed(seed); random.seed(seed)
    is_seed_reset = False
    rel_fn = random.choice(rel_functions)
    rel_type = relation_type(rel_fn)
    k = 0
    puzzle = {'rel_fn': rel_fn.__name__, 'train': [], 'test': []}
    
    def get_related_pos(pos, rel_func):
        x, y = pos
        rx, ry = rel_fn_compute(rel_func, x, y)
        return (rx, ry)
    
    if origin_position == 'center':  # for mirror/rotate relations with positive and negative coordinates
        assert n % 2 == 1 and m % 2 == 1, f'n={n}, m={m}'
        x_start, x_stop = -(n - 1) // 2, (n + 1) // 2
        y_start, y_stop = -(m - 1) // 2, (m + 1) // 2
        grid_origin = ((n - 1) // 2, (m - 1) // 2)
    else:  # origin_position == 'corner' for neighbor relations with all positive coordinates
        x_start, x_stop = 0, n
        y_start, y_stop = 0, m
        grid_origin = (0, 0)

    for i in range((n_train + n_test) * 100):
        section, q_per_grid = ('train', Q_train) if k < n_train else ('test', Q_test)
        if k == n_train and seed is not None and not is_seed_reset:  # reset seed for test set
            np.random.seed(seed + 100); random.seed(seed + 100)
            is_seed_reset = True
        
        # Create raw grid and wrap it with VirtualArray for elegant negative indexing
        grid = np.random.randint(0, n_color, (n, m))
        if origin_position == 'center': grid = VirtualArray(grid, origin=grid_origin)
        
        candidates = []
        for pos in list(product(range(x_start, x_stop), range(y_start, y_stop))):
            x, y = pos
            if rel_type == 'neighbor':  # exclude borders
                if x in [x_start, x_stop - 1] or y in [y_start, y_stop - 1]: continue
            elif rel_type == 'mirror_rotate':  # exclude axes and diagonals
                if x == 0 or y == 0 or x == y or x == -y: continue

            rel_pos = get_related_pos(pos, rel_fn)
            # Check if related position is within bounds
            if not (x_start <= rel_pos[0] < x_stop and y_start <= rel_pos[1] < y_stop):
                continue
            # Count how many other relations map to the same color
            count = sum(1 for rp in rel_functions
                       if (other_pos := get_related_pos(pos, rp)) is not None 
                       and x_start <= other_pos[0] < x_stop and y_start <= other_pos[1] < y_stop
                       and grid[other_pos] == grid[rel_pos])
            if count <= 1:
                candidates.append((pos, (grid == grid[rel_pos]).sum()))
        
        if len(candidates) >= q_per_grid:
            # puzzle[section].append({
            #     'input': raw_grid,  # Store the original numpy array
            #     'origin': grid_origin,
            #     'output': OrderedDict({tuple(pos): (rel_pos := get_related_pos(pos, rel_fn), grid[rel_pos])
            #         for pos, _ in random.sample(candidates, q_per_grid)})
            # })
            puzzle[section].append(Example(
                input=grid,
                output=[QA(query=Item(key=tuple(pos), value=grid[tuple(pos)]),
                           answer=Item(key=(rel_pos := get_related_pos(pos, rel_fn)), value=grid[rel_pos]))
                    for pos, _ in random.sample(candidates, q_per_grid)]
            ))
            k += 1
        if k == n_train + n_test: return puzzle
    assert False, f'seed={seed}, n_train={n_train}, len_train={len(puzzle["train"])}, len_test={len(puzzle["test"])}'


def verbalize_puzzle(puzzle, style='dict', color_quote='', tokenizer=None, base=0, print_out=True,):
    n_train, n_test = len(puzzle['train']), len(puzzle['test'])
    cq = color_quote

    tsb = TokenizedStrBuffer(tokenizer=tokenizer, base=base, print_out=print_out)  # unuseful when tokenizer is None
    print = tsb.add
    if tokenizer is not None: assert cq == ''
    # puzzle_ranges = defaultdict(list)
    for k in range(n_train + n_test):
        section = 'train' if k < n_train else 'test'
        example = puzzle[section][k % n_train]
        grid, qas = example.input, example.output

        print('-' * 30)
        print(f'# Example {k+1}' if section == 'train' else '# Test')
        # grid_name = f'example_grid_{k+1}' if section == 'train' else 'test_grid'
        grid_str, n_tokens, item_tokens, granges = verbalize_grid(grid, color_names, #grid_name=grid_name,
            style=style, color_quote=color_quote, tokenizer=tokenizer, base=tsb.base, print_out=False)
        print(grid_str, end='', n_tokens=n_tokens)
        # ex_ranges = defaultdict(list)
        # ex_ranges['input'] = granges
        example.input_ranges = granges
        for i, qa in enumerate(qas):
            pos, c0 = qa.query.key, qa.query.value
            rel_pos, c = qa.answer.key, qa.answer.value
            color0 = f'{cq}{color_names[c0]}{cq}'
            color = f'{cq}{color_names[c]}{cq}'
            # print(f'\n# What is the color of the cell that is {rel_name} ({pos[0]},{pos[1]})?')
            # print(f'[color for x, y, color in grid if {rel_fn.__name__}(*(x, y), *({pos[0]},{pos[1]}))][0] = ?')
            print('# What is the color of the cell in positional relation pos_relation_fn to cell', end='')
            # ex_ranges['queries'].append({t: (i, i + 1) for i, t in enumerate(item_tokens, start=tsb.base)})
            # item_tokens = ['(', 'x', ',', 'y', '):', 'c', ',/?']
            ranges = [(t, (i, i + 1)) for i, t in enumerate(item_tokens, start=tsb.base)]
            qa.query.key_ranges = OrderedDict(ranges[:5])  # ['(', 'x', ',', 'y', '):']
            qa.query.value_ranges = OrderedDict(ranges[5:])  # ['c', ',/?']
            print(f' ({pos[0]},{pos[1]}):{color0}?')
            # print(f'\n# Given the positional relation defined by pos_relation_fn(), '
            #       f'what is the color of the cell in this relation to cell ({pos[0]},{pos[1]})?')

            if style == 'dict':  # default
                print(f'[color for (x,y), color in grid.items() if pos_relation_fn(*(x,y),', end='')
                # ex_ranges['query_in_code'].append({t: (i, i + 1) for i, t in 
                #           enumerate(item_tokens[:5], start=tsb.base)}) # [' *(', 'x', ',', 'y', '))']
                qa.query.key_ranges_in_code = OrderedDict([
                    (t, (i, i + 1)) for i, t in enumerate(item_tokens[:5], start=tsb.base)]) # [' *(', 'x', ',', 'y', '))']
                print(f' *({pos[0]},{pos[1]}))][0] = ?') #, end='\n' if k < n_train else '\n\n')
            elif style == 'list': 
                print(f'[color for x, y, color in grid if pos_relation_fn(*(x, y), *({pos[0]},{pos[1]}))][0] = ?')
            if section == 'test': print(assistant_turn)
            print('Answer:', end='')
            # ex_ranges['answers'].append({'c': (tsb.base, tsb.base + 1)})
            qa.answer.value_ranges = OrderedDict([('c', (tsb.base, tsb.base + 1))])
            # let '\n\n' merge with color to avoid tokenization discrepancy issue
            print(f' {color}', end='\n\n' if section == 'train' else '')
            if section == 'test' and i < len(qas) - 1: print(user_turn)

        # puzzle_ranges[section].append(ex_ranges)
    n_tokens = tsb.assert_and_count_tokens()
    return tsb.str, n_tokens, puzzle


def gen_prompt(puzzle, fn_defs=None, tokenizer=None, print_out=True, remove_instructions=False, **kwargs): # style='dict', color_quote=''
    apply_chat_template = partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True)
    dummy_content = 'dummy_content'
    messages = [ # official: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        {"role": "system", "content": system_prompt0},
        {"role": "user", "content": dummy_content}
    ]
    s = apply_chat_template(messages)
    prefix = s[:s.index(dummy_content)]
    tsb = TokenizedStrBuffer(tokenizer=tokenizer, print_out=print_out); print = tsb.add
    print(prefix, end='')
    print(grid_desc['dict'])
    grid_str, n_tokens, _, _ = verbalize_grid(example_grid, color_names,
        tokenizer=tokenizer, base=tsb.base, print_out=False, **kwargs)
    print(grid_str, end='', n_tokens=n_tokens)
    if not remove_instructions: 
        print(instructions.format(fn_defs=fn_defs))
    print(f'# **{system_prompt3}**\n')
    
    puzzle_str, n_tokens, puzzle = verbalize_puzzle(puzzle, 
        tokenizer=tokenizer, base=tsb.base, print_out=False, **kwargs)
    print(puzzle_str, end='', n_tokens=n_tokens)
    # print(f'**{system_prompt}**')
    n_tokens = tsb.assert_and_count_tokens()

    return tsb.str, n_tokens, puzzle


@dataclass
class Result:
    index: int
    prompt: str
    model: str
    answers: list[str]
    answer_indices: list[int] = None
    candidate_ids: list[int] = None
    queries: list[tuple[int, int]] = None
    keys: list[tuple[int, int]] = None
    labels: list[int] = None
    responses: list[str] = field(default_factory=list)
    is_corrects: list[bool] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    rel_fn: str = None
    n_train: int = None
    Q_train: int = None
    n_test: int = None
    Q_test: int = None
    n_tokens: int = None
    puzzle: dict = None
    latency_sec: float = 0.0
    retries: int = 0
    index_map: list = None

def abs_sum(t): return sum(abs(x) for x in t)


def mean_elementwise(series):
    """Returns elementwise mean as a list"""
    return np.array(series.tolist()).mean(axis=0).round(3).tolist()

def mean_of_mean(series):
    """Returns the overall mean (mean of elementwise means)"""
    return round(np.array(series.tolist()).mean(), 3)

