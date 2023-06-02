import math
import numpy as np
from scipy.sparse import csc_array, coo_array, csr_array
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
indices =    [[0,   1,  1,  2,  2],
              [0,   0,  1,  1,  3]]
data_approx = [3.5, 3,  2,  1,  7.5]
data_detail = [1.5, 1,  -1, 0, -1.5]
approx_space = torch.sparse_coo_tensor(indices, data_approx, size=(3, 4)).to(device).coalesce()
detail_space = torch.sparse_coo_tensor(indices, data_detail, size=(3, 4)).to(device).coalesce()
print(approx_space.values())
class HWave():
    def __init__(self, approx_space, detail_space):
        self.approx_space = approx_space
        self.detail_space = detail_space
        (self.nr, self.nc) = self.approx_space.shape
    
    def postorder(self):
        """Returns dense array of postorder branch lengths
        """
        gaps = np.array([1])
        for _ in range(3): # levels in tree
            gaps = np.tile(gaps, 2)
            gaps[-1] += 1
        nodes = self.approx_space.nonzero()

        def postorder_info(r, c, gaps):
            gap_shift = 2**(self.nr-r) - 2
            cur_block = c * 2
            left_block = np.sum(gaps[:cur_block] + gap_shift) + gap_shift
            right_block = left_block+gaps[cur_block] + gap_shift
            approx, detail = self.approx_space[r, c], self.detail_space[r, c]
            left_value = approx + detail
            right_value = approx - detail
            return left_block, right_block, left_value, right_value
        func = lambda r, c: postorder_info(r, c, gaps)
        vfunc = np.vectorize(func, [np.int32, np.int32, np.float32, np.float32], cache=True)
        
        info = vfunc(*nodes)
        col_indices = np.concatenate(info[:2])
        post_order_data = np.concatenate(info[2:])
        post_order_array = csr_array((post_order_data, col_indices, np.array([0, col_indices.size])))
        post_order_array.sort_indices()
        return post_order_array[:, post_order_array.nonzero()[1]].A
    
    def total_branch_lengths(self):
        return torch.sum(self.approx_space.values() * 2)

    def from_treenode(self, treenode):
        pass

treenode = read('tree.nwk', format='newick', into=TreeNode)
tree = HWave(approx_space, detail_space)
start = time.time()
print(tree.total_branch_lengths())
for _ in range(10000):
    tree.total_branch_lengths()
print(f'hwave time elapsed: {time.time() - start}')

start = time.time()
for _ in range(10000):
    sum = 0
    for node in treenode.levelorder(include_self=True):
        sum += node.length
print(f'treenode time elapsed: {time.time() - start}')


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
