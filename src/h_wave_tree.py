import torch
from skbio import TreeNode, read
import time
from bp import parse_newick

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HWave():
    def __init__(self, row_ind: torch.Tensor, level_ind:torch.Tensor, parent_ind:torch.Tensor, lengths: torch.Tensor, max_children: int):
        self.row_ind, self.level_ind, self.parent_ind = row_ind, level_ind, parent_ind
        self.lengths = lengths
        self.nr = self.row_ind.max().add_(1)
        self.max_children=max_children

        # self.gaps = torch.tensor([1])
        # for _ in torch.arange(self.nr): # levels in tree
        #     self.gaps = self.gaps.tile((self.max_children,))
        #     self.gaps[-1] += 1
        # self.gaps = torch.concatenate((torch.tensor([0]), self.gaps)).cumsum_(0)

    # @torch.jit.export
    # def postorder(self):
    #     """Returns dense array of postorder branch lengths
    #     """
    #     gap_shift = torch.tensor([self.max_children]).pow(self.row_ind.sub(self.nr).mul_(-1)).sub_(self.max_children).div_(self.max_children-1, rounding_mode='trunc').mul_(self.col_ind.add(1))
    #     blocks = self.gaps.gather(0, self.col_ind).add_(gap_shift).argsort()
    #     return self.lengths.take(blocks)

    @torch.jit.export
    def _match(self, left: torch.Tensor, right: torch.Tensor):
        """returns values w.r.t. right
        """
        l_s  = left.size(dim=0)# if left.dim() > 0 else 1
        r_s = right.size(dim=0)# if right.dim() > 0 else 1
        return left.unsqueeze(1).expand((l_s, r_s)).eq(right.unsqueeze(0).expand((l_s, r_s))).max(dim=0)
    
    @torch.jit.export
    def _match_values(self, left: torch.Tensor, right: torch.Tensor):
        """returns values w.r.t. right
        """
        l_s  = left.size(dim=0)# if left.dim() > 0 else 1
        r_s = right.size(dim=0)# if right.dim() > 0 else 1
        return left.unsqueeze(1).expand((l_s, r_s)).eq(right.unsqueeze(0).expand((l_s, r_s))).amax(dim=0)
    
    @torch.jit.export
    def _match_indices(self, left: torch.Tensor, right: torch.Tensor):
        l_s  = left.size(dim=0)# if left.dim() > 0 else 1
        r_s = right.size(dim=0)# if right.dim() > 0 else 1
        return left.unsqueeze(1).expand((l_s, r_s)).eq(right.unsqueeze(0).expand((l_s, r_s))).argmax(dim=0)

    @torch.jit.export
    def total_branch_lengths(self):
        return self.lengths.sum()
    
    @torch.jit.export
    def tips_to_root_length(self, tip_rows: torch.Tensor, tip_levels: torch.Tensor):
        sums  = torch.zeros(tip_rows.size(0), dtype=torch.float64)
        tip_levels = torch.clone(tip_levels)
        for l in torch.arange(self.nr-1, 0, -1):
            tip_indices = tip_rows.ge(l).argwhere().flatten()
            tip_level_values = tip_levels.take(tip_indices)

            tree_mask = self.row_ind.ge(l).bitwise_and_(self._match_values(tip_level_values, self.level_ind))
            tree_level_values = self.level_ind.masked_select(tree_mask)
            tree_match_tip_indices = self._match(tree_level_values, tip_level_values).indices
          
            tree_parent_values = self.parent_ind.masked_select(tree_mask).take(tree_match_tip_indices)
            tree_length_values = self.lengths.masked_select(tree_mask).take(tree_match_tip_indices)

            sums.scatter_add_(0, tip_indices, tree_length_values)
            tip_levels.scatter_(0, tip_indices, tree_parent_values)
        return sums
    
    @torch.jit.export
    def tips(self):
        level_indices = self.row_ind.eq(self.nr-1).argwhere().flatten()
        tip_levels = [self.level_ind.take(level_indices)]
        tip_parents = [self.parent_ind.take(level_indices)]
        tip_rows = [torch.full(tip_levels[0].size(), fill_value=self.nr-1)]
        for l in torch.arange(self.nr-2, 1, -1):
            level_indices = self.row_ind.eq(l).argwhere().flatten()
            level_nodes = self.level_ind.take(level_indices)
            parent_nodes = self.parent_ind.take(level_indices)
            tip_mask = self._match_values(tip_parents[-1], level_nodes).bitwise_not_()
            tips = level_nodes.masked_select(tip_mask)
            parents = parent_nodes.masked_select(tip_mask)
            if tips.size(0) != 0:
                tip_rows.append(torch.full(tips.size(), fill_value=l))
                tip_levels.append(tips)
                tip_parents.append(parents)

        tip_rows = torch.concat(tip_rows)
        tip_levels = torch.concat(tip_levels)
        tip_parents = torch.concat(tip_parents)
        return tip_rows, tip_levels, tip_parents

def from_treenode(treenode):
    max_children = 0
    for level_node in treenode.levelorder(include_self=False):
        if max_children < len(level_node.children):
            max_children = len(level_node.children)
    
    cur_level_nodes = [(0, treenode.children)]
    cur_level = 1
    next_level_nodes = []
    row_ind = []
    level_ind = []
    parent_ind = []
    lengths = []
    while len(cur_level_nodes) > 0:
        cur_ind = 0
        for (pi, level_nodes) in cur_level_nodes:
            for i, node in enumerate(level_nodes):
                row_ind.append(cur_level)
                level_ind.append(cur_ind)
                parent_ind.append(pi)
                
                if node.length and node.length > 0:
                    lengths.append(node.length)
                else:
                    lengths.append(0)

                if len(node.children) > 0:
                    next_level_nodes.append((cur_ind, node.children))
                cur_ind += 1


        cur_level_nodes = next_level_nodes
        next_level_nodes = []
        cur_level += 1

    return (torch.tensor(row_ind), torch.tensor(level_ind), torch.tensor(parent_ind), torch.tensor(lengths, dtype=torch.float64), max_children)

def compare_trees(tree_type):
    if tree_type == 'ternary':
        # row_ind    = torch.tensor([ 1,   1,   2,  2,  2,  2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  3,  3,  3,  3,  3], device=device)
        # level_ind  = torch.tensor([ 0,   1,   0,  1,  2,  3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], device=device)
        # parent_ind = torch.tensor([ 0,   0,   0,  0,  0,  1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3,  3,  3,  4,  4,  4], device=device)
        # lengths =  torch.tensor([  15,  14,  12, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2,  3,  4,  5,  6,  7], device=device, dtype=torch.float64)
        # max_children=3
        tree_path = 'tree-ternary.nwk'

    elif tree_type == 'binary':
        # row_ind    = torch.tensor([ 1,  1,  2,  2,  2,  2, 3, 3, 3, 3])
        # level_ind  = torch.tensor([ 0,  1,  0,  1,  2,  3, 0, 1, 2, 3])
        # parent_ind = torch.tensor([ 0,  0,  0,  0,  1,  1, 1, 1, 3, 3])
        # lengths   = torch.tensor([ 5,  2,  4,  2,  1,  3, 1, 1, 6, 9], dtype=torch.float64)
        # max_children=2
        tree_path = 'tree-binary.nwk'

    else:
        # row_ind =  torch.tensor([ 1,  1,  2,  2,  2,  2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  3,  3,  3,  3,  3], device=device)
        # col_ind =  torch.tensor([ 0,   1,   0,  1,  2,  3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], device=device)
        # lengths =  torch.tensor([15, 14, 12, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2,  3,  4,  5,  6,  7], device=device, dtype=torch.float64)
        # max_children=3
        tree_path = 'test.nwk'
    treenode = read(tree_path, format='newick', into=TreeNode)
    row_ind, level_ind, parent_ind, lengths, max_children = from_treenode(treenode)
    tree_hwave = torch.jit.script(HWave(row_ind, level_ind, parent_ind, lengths, max_children))
    tip_rows, tip_levels, tip_parents = tree_hwave.tips()
    print('YEAH!!!!!!!!!!!!')
    # print(tree_hwave.tips_to_root_length(tip_rows, tip_levels))
    wave_lengths = lambda: tree_hwave.tips_to_root_length(tip_rows, tip_levels)
    wave_postorder = lambda: tree_hwave.postorder()
    wave_tips = lambda: tree_hwave.tips()
    def tips_lengths():
        sums = []
        for node in treenode.postorder():
            if node.is_tip():
                sums.append(node.accumulate_to_ancestor(treenode))
        return sums
    def postorder():
        p = []
        for node in treenode.postorder():
            p.append(node.length)
        return p
    def tips():
        t = []
        for n in treenode.postorder():
            if n.is_tip():
                t.append(t)
        return t
    def time_tree(tree_func):   
        print(tree_func())
        # start = time.time()
        # for _ in range(100):
        #     tree_func()
        # print(f'hwave time elapsed: {time.time() - start}')

    # time_tree(wave_lengths)
    time_tree(tips_lengths)

compare_trees('test')
