import numpy as np
from scipy.integrate import quad_vec

# basis [-l, -1] are ancestors where -1 is direct parent, -2 is grand parent, ...
#       [0] is brach length
#       [1, n] are children where n is max # children
# basis = lambda t: np.exp(2*np.pi*1j*np.arange(-3, 3, 1)*t)
# def f(x, c):
#     return (np.add.reduce(c*basis(x), axis=1)*np.exp(-2*np.pi*1j*x)).real

basis = np.exp(2*np.pi*1j*np.arange(-3, 3, 1), dtype=np.complex256)
print(basis)
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
    ], dtype=np.complex256
)

tree = np.add.reduce(np.nan_to_num((tree*basis - tree) / (2*np.pi*1j*np.arange(-3, 3, 1)), nan=0.), axis=1)

# tree = np.add.reduce(tree*basis/(np.sum(2*np.pi*1j*np.arange(-3, 3, 1))), axis=1) - np.add.reduce(tree/((np.sum(2*np.pi*1j*np.arange(-3, 3, 1)))), axis=1)
 
print('??????', tree)
# print('??????', quad_vec(f, 0, 1, args=(tree,)))
# tree += 1
# tree = np.multiply(tree, np.exp(2*np.pi*1j*np.arange(-3, 3, 1)))
# tree = np.add.reduce(tree, axis=1)
# print('????', tree)
# def get_leafs(tree):
#     # shift = tree*np.exp(2*np.pi*1j)
#     print(0.5*(tree.real**2 +tree.imag**2))
#     # print(shift)
#     return np.nonzero(tree.real)
# get_leafs(tree)
# # def get_postorder(tree):
# #     pass
# # # np.argwhere()
# # # postorder = 