import math

# done - Make it work with linked nodes.
# done - Make it work with Array implementation.
# todo_someday - add 'sum'

# 2e744d696cc14587a241f53c233ca826   Link to Notability
class StNode:
    def __init__(self, i, j, parent=None):
        self.value = None
        self.i = i
        self.j = j
        self.left = None
        self.right = None
        self.parent = parent

    def __repr__(self):
        return "({}, {}) {}".format(self.i, self.j, self.value)


class SegmentTree():
    def __init__(self, arr, func):
        # E.g data:
        # [2, 5, 1, 4, 9, 3]   len(L) = 6.  pivot=ceil(6/2)=3
        # [0, 1, 2, 3, 4, 5, 6, 8]
        self.root = StNode(0, len(arr)-1)
        self.func = func

        # Generate Tree Structure
        to_proccess = [self.root]
        leaf_nodes = []
        while to_proccess:
            node = to_proccess.pop()
            i, j = node.i, node.j
            assert i <= j

            if j == i:  # Leaf.
                node.value = arr[i]
                leaf_nodes += [node]
                continue

            pivot_index = int(math.ceil((float(j) - float(i) + 1.0) / 2.0) + float(i))  # 4,6,7 pi->7,   4,6  pi->6  #pi leans right for odds.
            li, lj = i, pivot_index -1  # l = left.
            ri, rj = pivot_index, j     # r = right

            node.left = StNode(li, lj, node)
            node.right = StNode(ri, rj, node)
            to_proccess += [node.left, node.right]

        # Compute function for all internal nodes.
        to_proccess = leaf_nodes
        while to_proccess:
            stnode = to_proccess.pop()
            parent = stnode.parent

            if parent is None:  # root.
                continue

            # Special note: Internal nodes will always have 2 children. Otherwise they are leafs.
            # This makes it easier to process when buddling up.
            assert parent.left is not None
            assert parent.right is not None

            if parent.value is None:
                parent.value = stnode.value
            else:
                parent.value = func(parent.value, stnode.value)
                to_proccess += [parent]
        pass

    def query(self, i, j):
        # Since we're dealing with complete tree's, Tree depth is less than 200 even for large data sets (100k+)
        # Thus recursion (which has a limit of 1000) is acceptable for this implementation.
        def _query(node, i, j):
            assert node is not None  # Since we have a complete graph
            if node.i >= i and node.j <= j:  # If node is within i, j, return node val.
                return node.value
            elif node.i > j or node.j < i:  # if node outside i, j, return inf.
                return node_out_of_range
            else:
                return self.func(_query(node.left, i, j), _query(node.right, i, j))

        if self.func is min:
            node_out_of_range = float("inf")
            return _query(self.root, i, j)

        if self.func is max:
            node_out_of_range = -float("inf")
            return _query(self.root, i, j)



# Below is an
from math import floor, ceil, log
class SegmentTreeArr():
    # SegmentTree implemented with a numpy array for "efficiency". However tests show that
    def __init__(self, arr, func):
        self.func = func
        st_arr_size = (2 ** ceil(log(len(arr), 2) + 1.0)) - 1
        st = [None] * int(st_arr_size)   # todo_someday - use a numpy array instead of list & converting to numpy.

        st[0] = (0, len(arr) - 1, None)  # (i, j, val)

        to_proccess_indexes = [0]
        leaf_node_indexes = []

        while to_proccess_indexes:
            node_index = to_proccess_indexes.pop()
            i, j, _ = st[node_index]
            assert i <= j

            if i == j:  # Leaf
                st[node_index] = (i, j, arr[i])
                leaf_node_indexes.append(node_index)
                continue

            pivot_index = int(math.ceil((float(j) - float(i) + 1.0) / 2.0) + float(i))
            li, lj = i, pivot_index - 1  # l = left.
            ri, rj = pivot_index, j  # r = right
            left_index = node_index * 2 + 1
            right_index = node_index * 2 + 2
            st[left_index] = (li, lj, None)
            st[right_index] = (ri, rj, None)
            to_proccess_indexes += [left_index, right_index]

        to_proccess_indexes = leaf_node_indexes
        while to_proccess_indexes:
            node_index = to_proccess_indexes.pop()
            i, j, value = st[node_index]

            if node_index == 0:
                continue # root.
            parent_index = int(floor((node_index - 1.0) / 2.0))
            # if st[parent_index] is None:  # root
            #     continue

            # Special note: Internal nodes will always have 2 children. Otherwise they are leafs.
            # This makes it easier to process when bubbling up.
            p_left = parent_index*2 + 1
            p_right = parent_index*2 + 2
            assert st[p_left] is not None
            assert st[p_right] is not None

            pi, pj, pval = st[parent_index]
            if pval is None:
                pval = value
            else:
                pval = func(pval, value)
                to_proccess_indexes.append(parent_index)
            st[parent_index] = (pi, pj, pval)

        # self.st = numpy.asarray(st)  # Using numpy doesn't seem to make any performance difference in this case.
        self.st = st

    def query(self, i, j):
        st = self.st

        def _query_min(node_index, i, j):
            # Since we have a complete graph. (nodes have 2 children). No null check needed.
            assert st[node_index] is not None
            node_i, node_j, node_value = st[node_index]

            if node_i >= i and node_j <= j:  # If node is within i, j, return node val.
                return node_value
            elif node_i > j or node_j < i:  # if node outside i, j, return inf.
                return float("inf")
            else:
                node_left_index = node_index*2 + 1
                node_right_index = node_index*2 + 2
                return min(_query_min(node_left_index, i, j), _query_min(node_right_index, i, j))

        if self.func is min:
            return _query_min(0, i, j)




