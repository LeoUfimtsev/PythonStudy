
# TODO - BST print in level traversal.
    # TODO - use level traversal to visually represent tree.
# TODO - inorder
# TODO - preorder
# TODO - postorder.

# Example:
# bst = BST([4,2,1,3,6,5,7])
# bst.display()
#   _4_
#  /   \
#  2   6
# / \ / \
# 1 3 5 7


from collections import deque
class Node:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.val)


class BST:
    def __init__(self, initList=None):
        self.root = None
        if initList:
            if isinstance(initList, list):
                # O( n log n)
                for val in initList:
                    self.insert(val)
            else:
                raise ValueError("Invalid argument type.").with_traceback(type(initList))

    # O(log n)
    def insert(self, val):
        if not self.root:
            self.root = Node(val)
        else:
            curr = self.root
            while True:
                if val < curr.val:
                    if curr.left:
                        curr = curr.left
                    else:
                        curr.left = Node(val)
                        break
                else:  # Duplicate keys are put on the right side. Not tested.
                    if curr.right:
                        curr = curr.right
                    else:
                        curr.right = Node(val)
                        break

    # @ O(log n)
    def contains(self, val):
        queue = deque()
        queue.append(self.root)
        while queue:
            node = queue.popleft()
            if not node:
                continue
            elif node.val == val:
                return True
            else:
                for child_node in [node.left, node.right]:
                    queue.append(child_node)  # append = BFS.   appendLeft = DFS
        return False

    # O( n )
    def get_level_order(self):
        return_list = []
        queue = deque()
        queue.append(self.root)
        while queue:
            node = queue.popleft()
            if node:
                return_list += [node.val]
                for node in [node.left, node.right]:
                    queue.append(node)
        return return_list

    def get_in_order(self):
        # in-order : left, root, right
        # pre-order: root, left, right   # pre-refers to root.
        # post-order: left, right, root, # post-refers to root.

        # Ex problem: https://www.hackerrank.com/challenges/tree-inorder-traversal/problem
        out = []
        curr = self.root
        to_process = []
        while curr or to_process:
            if curr:
                to_process.append(curr)
                curr = curr.left
            elif to_process:
                curr = to_process.pop()
                out += [curr.val]
                curr = curr.right
        return out

    def get_pre_order(self):
        # in-order: left, root, right
        # pre-order: root, left, right   # pre-refers to root.
        # post-order: left, right, root  # post-refers to root.
        out = []
        to_visit = [self.root]
        while to_visit:
            node = to_visit.pop()
            out.append(node.val)
            # (!) Careful with order that you add things to stack. Can be counter-intuitive.
            # First item to be added will be evaluated last. So make sure to add the item that you want to visit last.
            # (i.e, kinda think in reverse order.
            for child in [node.right, node.left]:
                if child:
                    to_visit.append(child)
        return out


    def compute_sums(self):
        """Augment Node to contain:  node.sum, which is the sum of itself and sum of all it's children.
           Insight: Here we use 2 stacks.
            - First explore all nodes via pre-order and do some preliminary meta work. Append exlored nodes to 2nd stack.
            - Then pop all items off the 2nd stack and compute upwards.
            This emulates Breadth first recursion.
        """
        self.root.parent = None  # Augment nodes to have a parent.
        to_explore = [self.root]
        to_compute = []
        while to_explore:
            node = to_explore.pop()
            node.sum = node.val
            to_compute.append(node)
            for child in [node.left, node.right]:
                if child:
                    child.parent = node
                    to_explore.append(child)
        while to_compute:
            node = to_compute.pop()
            if node.parent:
                node.parent.sum += node.sum


    def get_post_order(self):
        # Very elegant 2 stack solution. 1 stack is possible but tricky.
        # Src: https://www.geeksforgeeks.org/iterative-postorder-traversal/
        # Basically pop from stack 1 onto stack 2. Append left/right child to stack 1.
        # reverse stack2 & print.
        stack1 = [self.root]
        stack2 = []
        while stack1:
            node = stack1.pop()
            stack2.append(node)
            for child in [node.left, node.right]:
                if child:
                    stack1.append(child)
        stack2.reverse()
        return [node.val for node in stack2]


    def get_height_recursive(self):
        # Recursion in Python is limited to ~1000 stack frames.
        # This approach can easily generate a StackOverflow exception. See: TreeTests.py test_get_height_big.
        # This method is a demonstration of why you shouldn't use recursion in python.Iterative approaches scale better.
        # (I've failed a google interview once because I used recursion for a DFS/BFS search). DON'T DO IT.
        def helper(node, curr_depth):
            if node is None:
                return curr_depth - 1
            else:
                return max(helper(node.left, curr_depth + 1),
                           helper(node.right, curr_depth + 1))
        return max(helper(self.root.left, 1), helper(self.root.right, 1))

    def get_height_iterative(self):
        """Pack Node along with it's depth into tuple and put the tuple into a queue"""
        max_so_far = 0
        nodes_queue = deque()
        nodes_queue.append((self.root, 0))
        while nodes_queue:
            node, depth = nodes_queue.popleft()
            max_so_far = max(max_so_far, depth)
            if node.left:
                nodes_queue.append((node.left, depth + 1))
            if node.right:
                nodes_queue.append((node.right, depth + 1))
        return max_so_far

    def get_height_iterative_WithNestedClass(self):
        """Solve the problem by using a custom class to hold Node meta data. Use if metadata is very complex"""
        class NodeMeta:
            def __init__(self, node, depth):
                self.node = node
                self.depth = depth
            def __repr__(self):
                return "Node: {}  Depth: {}".format(self.node, self.depth)

        max_so_far = 0
        nodes_queue = deque()
        nodes_queue.append(NodeMeta(self.root, 0))
        while nodes_queue:
            curr = nodes_queue.popleft()
            if not curr.node:
                continue
            max_so_far = max(max_so_far, curr.depth)
            for node in [curr.node.left, curr.node.right]:
                nodes_queue.append(NodeMeta(node, curr.depth + 1))
        return max_so_far

    def get_height_iterative_augmentNode(self):
        """In Python, we can dynamically augment classes and add fields to a class. Useful for quick hacks."""
        max_so_far = 0
        nodes_queue = deque()
        self.root.depth = 0
        nodes_queue.append(self.root)
        while nodes_queue:
            curr = nodes_queue.popleft()
            max_so_far = max(max_so_far, curr.depth)
            for node in [curr.left, curr.right]:
                if node is None:
                    continue
                node.depth = curr.depth + 1
                nodes_queue.append(node)
        return max_so_far


    # Leo Note: I didn't write the code for this myself, but rather adapted it from here:
    # https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python/34014370
    # See TestBST.test_display_tree for example.
    def display(self):
        lines, _, _, _ = self._display_aux(self.root)
        for line in lines:
            print(line)

    def _display_aux(self, node):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if node.right is None and node.left is None:
            line = '%s' % node.val
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if node.right is None:
            lines, n, p, x = self._display_aux(node.left)
            s = '%s' % node.val
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if node.left is None:
            lines, n, p, x = self._display_aux(node.right)
            s = '%s' % node.val
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self._display_aux(node.left)
        right, m, q, y = self._display_aux(node.right)
        s = '%s' % node.val
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = list(zip(left, right))
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

# Problems that I came across:
# Q: How to handle duplicate keys?
#   > Put them left|right, or augment tree.  See: https://www.geeksforgeeks.org/how-to-handle-duplicates-in-binary-search-tree/