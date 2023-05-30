import numpy as np

tree = np.array(
    [
        [-1, 1 + 1j, 2 + 2j],
        [0,  3 + 1j,  4 + 2j],
        [0,  5 + 1j,  6 + 2j],
        [1,  0,  0], 
        [1,  3 + 1j,  4 + 2j],
        [2, 0, 0],
        [2,  9 + 1j, 10 + 2j],
        [4, 0, 0], 
        [4, 0, 0],
        [6, 0, 0],
        [6, 0, 0]
    ]
)

def get_leafs(tree):
    shift = tree - 1j
    return np.nonzero(shift[:, 1].imag)

def get_postorder(tree):
    pass
# np.argwhere()
# postorder = 
print(tree, get_leafs(tree))