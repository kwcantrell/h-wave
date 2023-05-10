import sys
sys.path.append("src")
from h_wave_tree import PSimHWaveTree

class TestFullPSimHWaveTree():
    def __init__(self):
        tree = [
            [
                [1.5,-0.5]
            ],
            [
                [2, 1], [3, -1]
            ],
            [
                [1.5, -0.5], [2, 0], [4, -1], [1, 0]
            ]
        ]
        self.h_wave = PSimHWaveTree(tree)
    def print_test(self, test, passed, msg=None):
        if msg:
            print(f'Test {test}: Passed {passed}, {msg}')
        else:
            print(f'Test {test}: Passed {passed}')

    def test_get_level_order_packets(self):
        test='get_level_order_packets'
        level_order_packets = [
            [0,0,0], [0,0,1],
            [1,0,0], [1,0,1], [1,1,0], [1,1,1],
            [2,0,0], [2,0,1], [2,1,0], [2,1,1], [2,2,0], [2,2,1], [2,3,0], [2,3,1],
        ]
        result = self.h_wave.get_level_order_packets()
        if len(level_order_packets) != len(result):
            return self.print_test(test, False, f'different sizes {len(level_order_packets), len(result)}')

        for true_packet, res_packet in zip(level_order_packets, result):
            if not self.h_wave.is_same_packet(true_packet, res_packet):
                return self.print_test(test, False, f'wrong packet {true_packet, res_packet}')
        return self.print_test(test, True)
    
    def test_get_leaf_packets(self):
        test = 'get_leaf_packets'
        leaf_packets = [
            [2,0,0], [2,0,1], [2,1,0], [2,1,1], [2,2,0], [2,2,1], [2,3,0], [2,3,1],
        ]
        result = self.h_wave.get_leaf_packets()
        if len(leaf_packets) != len(result):
            return self.print_test(test, False, f'different sizes {len(leaf_packets), len(result)}')
        for true_leaf, res_leaf in zip(leaf_packets, result):
            if not self.h_wave.is_same_packet(true_leaf, res_leaf):
                return self.print_test(test, False, f'wrong packet {true_leaf, res_leaf}')
        return self.print_test(test, True)

    def test_get_branch_length(self):
        test = 'get_branch_length'
        # iterate over the tree in level order traversal
        b_lengths = [1,2,3,1,2,4,1,2,2,2,3,5,1,1]

        result = []
        level_packets = self.h_wave.get_level_order_packets()
        for packet in level_packets:
            result.append(self.h_wave.get_branch_length(packet))

        if len(b_lengths) != len(result):
            return self.print_test(test, False, f'different sizes {len(b_lengths), len(result)}')

        for true_val, res_val in zip(b_lengths, result):
            if true_val != res_val:
                return self.print_test(test, False, f'wrong length {true_val, res_val}')
                
        return self.print_test(test, True)

    def test_sum_branch_to_root(self):
        test = 'get_sum_leaf_to_root'
        path_sums = [5,6,4,4,7,9,7,7]

        result = []
        for leaf in self.h_wave. get_leaf_packets():
            result.append(self.h_wave.sum_branch_to_root(leaf))
        
        if len(path_sums) != len(result):
            return self.print_test(test, False, f'different sizes {len(path_sums), len(result)}')

        for true_val, res_val in zip(path_sums, result):
            if true_val != res_val:
                return self.print_test(test, False, f'wrong length {true_val, res_val}')
                
        return self.print_test(test, True)

    def test_pairwise_distance(self):
        test = 'pairwise_distance'

        #iterate in level order
        pair_distances = [
            0,3,3,1,5,7,4,5,3,3,8,10,8,8,0,3,3,1,5,7,4,5,3,3,8,10,8,8
        ]
        result = []
        packets = self.h_wave.get_level_order_packets()
        for packet in packets:
            result.append(self.h_wave.pairwise_distance(packets[0], packet))
        for packet in packets:
            result.append(self.h_wave.pairwise_distance(packet, packets[0]))

        if len(pair_distances) != len(result):
            return self.print_test(test, False, f'different sizes {len(pair_distances), len(result)}')

        for i, (true_val, res_val) in enumerate(zip(pair_distances, result)):
            if true_val != res_val:
                return self.print_test(test, False, f'wrong length {i, true_val, res_val}')
                
        return self.print_test(test, True)

test = TestFullPSimHWaveTree()
test.test_get_level_order_packets()
test.test_get_leaf_packets()
test.test_get_branch_length()
test.test_sum_branch_to_root()
test.test_pairwise_distance()
