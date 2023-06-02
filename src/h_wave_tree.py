import math
import numpy as np
from scipy.sparse import csc_array, coo_array, csr_array


row =         np.array([0,   1,  1,  2,  2])
col =         np.array([0,   0,  1,  1,  3])
data_approx = np.array([3.5, 3,  2,  1,  7.5])
data_detail = np.array([1.5, 1,  -1, 0, -1.5])
approx_space = csc_array((data_approx, (row, col)))
detail_space = csc_array((data_detail, (row, col)))
# print(approx_space.A)
# for i in range(3):
#     for j in range(4):
#         print(approx_space[i,j], detail_space[i,j])
# print(approx_space.getcol(0).indptr, approx_space.getcol(0).indices[-1], approx_space.getcol(0).data)

def postorder(a, d):
    (nr, nc) = a.shape
    # start = a.getcol(0).indices[-1]
    gaps = np.array([1])
    for g in range(3): # levels in tree
        gaps = np.tile(gaps, 2)
        gaps[-1] += 1
    nodes = a.nonzero()

    post_indices = []
    post_order_data = []
    def get_start_blocks(r, c, nr, post_indices, post_order_data, a, d, gaps):
        gap_shift = 2**(nr-r) - 2 
        left_block = 

        cur_block = c // 2
        left_pos = start_block
        if cur_block > 0:
            left_pos += gaps[cur_block-1] - 1

        if r == nr - 1:
            right_pos = left_pos + 1
        else:
            right_pos = 

        approx, detail = a[r, c], d[r, c]
        haar_packet = (approx, detail)
        left = approx + detail
        right = approx - detail
        # left_col = 
        # print(r, c, haar_packet, left, right, start_block, left_pos)
        print(r, c, haar_packet, start_block, left_pos)
        return r, c, 5., 5.

    func = lambda r, c: get_start_blocks(r, c, nr, post_indices, post_order_data, a, d, gaps)
    vfunc = np.vectorize(func, [np.int32, np.int32, np.float32, np.float32])
    
    print(vfunc(*nodes))
    

postorder(approx_space, detail_space)
class PSimHWaveTree():
    def __init__(self, tree):
        self.tree = tree

    def _get_packets_level(self, level, num_wavelets):
        packets = []
        for q in range(num_wavelets):
            packets.append([level, q, 0])
            packets.append([level, q, 1])        
        return packets
        
    def get_level_order_packets(self):
        packets = []
        for level_wavelets in self.tree:
            level = len(level_wavelets) // 2
            for packet in self._get_packets_level(level, len(level_wavelets)):
                packets.append(packet)
        return packets

    def get_leaf_packets(self):
        num_levels = len(self.tree) - 1
        num_wavelets = math.pow(2, num_levels)
        if len(self.tree[num_levels]) == num_wavelets:
            return self._get_packets_level(num_levels, int(num_wavelets))    

    def get_branch_length(self, packet):
        level, q, r = packet
        a, d = self.tree[level][q]
        return a + math.pow(-1, r)*d

    def sum_branch_to_root(self, packet):
        end = packet
        while not self.is_root_packet(end):
            end = self.get_mother_packet(end)
        return self.sum_ancestor_path(packet, end, True)
        

    def sum_ancestor_path(self, start, end, include_end=False):
        sum = 0
        while not self.is_same_packet(start, end):
            sum += self.get_branch_length(start)
            start = self.get_mother_packet(start)
        if include_end:
            sum += self.get_branch_length(start)
        return sum

    def get_leaf_packet(self, leaf):
        return self.get_mother_packet([len(self.tree), leaf, 0])

    def get_mother_packet(self, packet):
        level, q, _ = packet
        if level == 0:
            return []
        return [level - 1, q // 2, q % 2]
    
    def is_same_packet(self, a, b, same_child=True):
        if same_child:
            return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

        return a[0] == b[0] and a[1] == b[1]
    
    def is_root_packet(self, packet):
        return packet[0] == 0
    
    def is_ancestor(self, a, b):
        # checks if b is ancestor of a
        if self.get_packet_level(a) <= self.get_packet_level(b):
            return False

        while self.get_packet_level(a) > self.get_packet_level(b):
            a = self.get_mother_packet(a)
            
        return self.is_same_packet(a, b)

    def is_valid_packet(self, packet):
        try:
            level, q, _ = packet
            check = self.tree[level, q]
            return True
        except:
            return False

    def is_leaf(self, packet):
        if self.get_packet_level(packet) == len(self.tree) -1:
            return True
        
        packet = self.get_mother_packet(packet)
        return not self.is_valid_packet(packet)
    
    def get_packet_level(self, packet):
        return packet[0]

    def lca(self, a, b):
        while self.get_packet_level(a) > self.get_packet_level(b):
            a = self.get_mother_packet(a)

        while self.get_packet_level(b) > self.get_packet_level(a):
            b = self.get_mother_packet(b)

        while not self.is_same_packet(a, b, same_child=False):
            a, b = self.get_mother_packet(a), self.get_mother_packet(b)
        return a, b

    def pairwise_distance(self, a, b):
        if self.is_same_packet(a, b):
            return 0
        if self.is_same_packet(a, b, same_child=False):
            return self.get_branch_length(a) + self.get_branch_length(b)
        if self.is_ancestor(a, b):
            return self.sum_ancestor_path(a, b)
        if self.is_ancestor(b, a):
            return self.sum_ancestor_path(b, a)
        
        lca_a, lca_b = self.lca(a,b)
        sum = self.sum_ancestor_path(a, lca_a, True)
        sum += self.sum_ancestor_path(b, lca_b, True)
        return sum
