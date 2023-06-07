import math
import numpy as np
# from scipy.sparse import csc_array, coo_array, csr_array
import torch
from skbio import TreeNode, read
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# row =         np.array([0,   1,  1,  2,  2])
# col =         np.array([0,   0,  1,  1,  3])
# data_approx = np.array([3.5, 3,  2,  1,  7.5])
# data_detail = np.array([1.5, 1,  -1, 0, -1.5])
# approx_space = csc_array((data_approx, (row, col)))
# detail_space = csc_array((data_detail, (row, col)))
# indices =    [[0,   1,  1,  2,  2],
#               [0,   0,  1,  1,  3]]
# data_approx = [3.5, 3,  2,  1,  7.5]
# data_detail = [1.5, 1,  -1, 0, -1.5]
# approx_space = torch.sparse_coo_tensor(indices, data_approx, size=(3, 4), device=device).coalesce()
# detail_space = torch.sparse_coo_tensor(indices, data_detail, size=(3, 4), device=device).coalesce()
# max_children = 2
# indices =       [[0,  1, 1,   1,   2,   2,   2, 2,   2, 2,   2,   2,   2, 2],
#                  [0,  0, 1,   2,   0,   1,   2, 3,   4, 5,   6,   7,   8, 9]]
# data_approx = [14.5, 11, 5, 9.5, 7.5, 2.5, 4.5, 1, 1.5, 0, 2.5, 1.5, 5.5, 3]
# data_detail = [ 0.5,  1, 6, 0.5, 0.5, 3.5, 0.5, 2, 0.5, 1,-0.5, 2.5,-0.5, 4]
# approx_space = torch.sparse_coo_tensor(indices, data_approx, size=(3, 10), device=device).coalesce()
# detail_space = torch.sparse_coo_tensor(indices, data_detail, size=(3, 10), device=device).coalesce()
# max_children = 3
row_ind = torch.tensor([ 1,  1,  2,  2,  2,  2, 3, 3, 3, 3])
col_ind = torch.tensor([ 0,  1,  0,  1,  2,  3, 2, 3, 6, 7])
lengths = torch.tensor([ 5,  2,  4,  2,  1,  3, 1, 1, 6, 9])
max_children=2

row_ind =  torch.tensor([ 1,  1,  2,  2,  2,  2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  3,  3,  3,  3,  3], device=device)
col_ind =  torch.tensor([ 0,  1,  0,  1,  2,  3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], device=device)
lengths =  torch.tensor([15, 14, 12, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2,  3,  4,  5,  6,  7], device=device)
#                       [ 0,  1,  2,  3,  4,  5, 6, 7, 8, 9,10,11,12,13,14,15,16, 17, 18, 19, 20]
max_children=3
class HWave():
    def __init__(self, row_ind, col_ind, lengths, max_children=2):
        self.row_ind, self.col_ind, self.lengths = row_ind, col_ind, lengths
        self.nr, self.nc = torch.max(self.row_ind)+1, torch.max(self.col_ind)+1
        self.max_children=max_children
        self.gaps = torch.tensor([1])
        for _ in range(self.nr): # levels in tree
            self.gaps = torch.tile(self.gaps, (self.max_children,))
            self.gaps[-1] += 1
        self.gaps = torch.concatenate((torch.tensor([0]), self.gaps))
        self.gaps = torch.cumsum(self.gaps, 0).to(device)

    # torch.compile()
    def postorder(self):
        """Returns dense array of postorder branch lengths
        """
        gap_shift = ((self.max_children**(self.nr-self.row_ind)-self.max_children)/(self.max_children-1))*(self.col_ind+1)
        blocks = self.gaps[self.col_ind] + gap_shift
        return self.lengths[torch.argsort(blocks)]

    def _rank(col_vals):
        x = col_vals
        uq_vals,inverse_ixs = torch.unique(x, sorted=True, return_inverse=True)
        return inverse_ixs
    
    torch.compile
    def _match(left, right):
        """
        return.values is bool mask for right and return.indices of matching index positions in right
        """
        l_s, r_s = left.shape[0], right.shape[0]
        return torch.max(left[:, None].expand((l_s, r_s)) == right[None, :].expand((l_s, r_s)), dim=0)

    torch.compile
    def total_branch_lengths(self):
        return torch.sum(self.lengths)

    torch.compile(mode='max-autotune')
    def tips_to_root_length(self, tip_rows, tip_cols):
        sums  = torch.zeros_like(tip_rows)
        
        for l in range(self.nr-1, 0, -1):
            tree_indices = (self.row_ind == l)
            tip_indices = (tip_rows ==l)

            # need to match column values
            tip_col_values = tip_cols[tip_indices]
            match_info = HWave._match(tip_col_values, self.col_ind).values

            # create tree length indices
            tree_match_indices = tree_indices & match_info
            tree_lengths = self.lengths[tree_match_indices]

            tip_col_match = HWave._match(self.col_ind[tree_match_indices], tip_col_values).indices

            tip_rows[tip_indices] -= 1
            tip_cols[tip_indices] //= self.max_children

            sums[tip_indices] += tree_lengths[tip_col_match]
        
        return sums
    
    # def sum_unique_path(self, rows, cols):



#     def from_treenode(self, treenode):
#         pass

tree = HWave(row_ind, col_ind, lengths, max_children)
# print(tree.tips_to_root_length(torch.tensor([2, 3, 3, 2, 3, 3]),
#                                torch.tensor([0, 2, 3, 2, 6, 7])))
rows = torch.tensor([3, 3, 3, 3, 3, 3,  3,  3,  3,  3,  3], device=device)
cols = torch.tensor([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14], device=device)
# print(tree.tips_to_root_length(rows,cols))

start = time.time()
print('!!!!', tree.tips_to_root_length(torch.clone(rows),torch.clone(cols)))
for _ in range(10000):
    tree.tips_to_root_length(torch.clone(rows),torch.clone(cols))
print(f'hwave time elapsed: {time.time() - start}')
# tree.postorder()

# treenode = read('tree.nwk', format='newick', into=TreeNode)
# start = time.time()
# for _ in range(100000):
#     sum = 0
#     for node in treenode.levelorder(include_self=True):
#         sum += node.length
# print(f'treenode time elapsed: {time.time() - start}')


# class PSimHWaveTree():
#     def __init__(self, tree):
#         self.tree = tree

#     def _get_packets_level(self, level, num_wavelets):
#         packets = []
#         for q in range(num_wavelets):
#             packets.append([level, q, 0])
#             packets.append([level, q, 1])        
#         return packets
        
#     def get_level_order_packets(self):
#         packets = []
#         for level_wavelets in self.tree:
#             level = len(level_wavelets) // 2
#             for packet in self._get_packets_level(level, len(level_wavelets)):
#                 packets.append(packet)
#         return packets

#     def get_leaf_packets(self):
#         num_levels = len(self.tree) - 1
#         num_wavelets = math.pow(2, num_levels)
#         if len(self.tree[num_levels]) == num_wavelets:
#             return self._get_packets_level(num_levels, int(num_wavelets))    

#     def get_branch_length(self, packet):
#         level, q, r = packet
#         a, d = self.tree[level][q]
#         return a + math.pow(-1, r)*d

#     def sum_branch_to_root(self, packet):
#         end = packet
#         while not self.is_root_packet(end):
#             end = self.get_mother_packet(end)
#         return self.sum_ancestor_path(packet, end, True)
        

#     def sum_ancestor_path(self, start, end, include_end=False):
#         sum = 0
#         while not self.is_same_packet(start, end):
#             sum += self.get_branch_length(start)
#             start = self.get_mother_packet(start)
#         if include_end:
#             sum += self.get_branch_length(start)
#         return sum

#     def get_leaf_packet(self, leaf):
#         return self.get_mother_packet([len(self.tree), leaf, 0])

#     def get_mother_packet(self, packet):
#         level, q, _ = packet
#         if level == 0:
#             return []
#         return [level - 1, q // 2, q % 2]
    
#     def is_same_packet(self, a, b, same_child=True):
#         if same_child:
#             return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

#         return a[0] == b[0] and a[1] == b[1]
    
#     def is_root_packet(self, packet):
#         return packet[0] == 0
    
#     def is_ancestor(self, a, b):
#         # checks if b is ancestor of a
#         if self.get_packet_level(a) <= self.get_packet_level(b):
#             return False

#         while self.get_packet_level(a) > self.get_packet_level(b):
#             a = self.get_mother_packet(a)
            
#         return self.is_same_packet(a, b)

#     def is_valid_packet(self, packet):
#         try:
#             level, q, _ = packet
#             check = self.tree[level, q]
#             return True
#         except:
#             return False

#     def is_leaf(self, packet):
#         if self.get_packet_level(packet) == len(self.tree) -1:
#             return True
        
#         packet = self.get_mother_packet(packet)
#         return not self.is_valid_packet(packet)
    
#     def get_packet_level(self, packet):
#         return packet[0]

#     def lca(self, a, b):
#         while self.get_packet_level(a) > self.get_packet_level(b):
#             a = self.get_mother_packet(a)

#         while self.get_packet_level(b) > self.get_packet_level(a):
#             b = self.get_mother_packet(b)

#         while not self.is_same_packet(a, b, same_child=False):
#             a, b = self.get_mother_packet(a), self.get_mother_packet(b)
#         return a, b

#     def pairwise_distance(self, a, b):
#         if self.is_same_packet(a, b):
#             return 0
#         if self.is_same_packet(a, b, same_child=False):
#             return self.get_branch_length(a) + self.get_branch_length(b)
#         if self.is_ancestor(a, b):
#             return self.sum_ancestor_path(a, b)
#         if self.is_ancestor(b, a):
#             return self.sum_ancestor_path(b, a)
        
#         lca_a, lca_b = self.lca(a,b)
#         sum = self.sum_ancestor_path(a, lca_a, True)
#         sum += self.sum_ancestor_path(b, lca_b, True)
#         return sum
