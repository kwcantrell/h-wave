import numpy as np
import cmath
# tree = np.array(
#     [
#         [-1, 1 + 1j, 2 + 2j],
#         [0,  3 + 1j,  4 + 2j],
#         [0,  5 + 1j,  6 + 2j],
#         [1,  0,  0], 
#         [1,  3 + 1j,  4 + 2j],
#         [2, 0, 0],
#         [2,  9 + 1j, 10 + 2j],
#         [4, 0, 0], 
#         [4, 0, 0],
#         [6, 0, 0],
#         [6, 0, 0]
#     ]
# )
# basis [-l, -1] are ancestors where -1 is direct parent, -2 is grand parent, ...
#       [0] is brach length
#       [1, n] are children where n is max # children
basis = np.exp(2*np.pi*1j*np.arange(-3, 3, 1))[np.newaxis, :]
print(basis.shape)
tree = np.array(
    [
        [-1, -1, -1, -1, 1, 2],
        [-1, -1, 0, 5, 3, 4],
        [-1, -1, 0, 1, 5, 6],
        [-1, 1, 0, 3, -1, -1],
        [-1, 1, 0, 1, 7, 8],
        [-1, 2, 0, 2, -1, -1],
        [-1, 2, 0, 3, 9, 10],
        [4, 1, 0, 4, -1, -1],
        [4, 1, 0, 1, -1, -1],
        [6, 2, 0, 2, -1, -1],
        [6, 2, 0, 8, -1, -1]
    ], dtype=np.complex128
)
tree += 1
tree *= np.multiply(tree, basis)
tree = np.add.reduce(tree, axis=1)
print(cmath.phase(tree[0]))
def get_leafs(tree):
    # shift = tree*np.exp(2*np.pi*1j)
    print(tree.shape, np.apply_along_axis(lambda z: 1j*cmath.phase(z), 0, tree))
    # print((np.real(tree)*1j*np.angle(tree)*tree - np.real(tree)*1j*np.angle(tree)).imag)
    # print(shift)
    return np.nonzero(tree.real)
# get_leafs(tree)
# def get_postorder(tree):
#     pass
# # np.argwhere()
# # postorder = 