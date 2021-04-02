from unittest import TestCase
from unittest import skip, skipIf
# from unittest import main   # Might be required if running from CMD.
from Tree import *


class TestBST(TestCase):
    SKIP_LONG_TESTS = True

    T0 = [3, 5, 2, 1, 4, 6, 7]  # height = 3
    #   3_
    #  /  \
    #  2  5
    # /  / \
    # 1  4 6
    #       \
    #       7
    # Src: https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem
    T0_in_order = [1, 2, 3, 4, 5, 6, 7]  # left, root, right.  sorted(T0)
    T0_pre_order = [3, 2, 1, 5, 4, 6, 7]   # root, left, right
    T0_post_order = [1, 2, 4, 7, 6, 5, 3]  # left, right, root

    T2 = [3, 1, 7, 5, 4]        # height = 3
    #  3__
    # /   \
    # 1   7
    #    /
    #    5
    #   /
    #   4
    T2_in_order = [1, 3, 4, 5, 7]   # left, root, right. sorted(T2)
    T2_pre_order = [3, 1, 7, 5, 4]  # root, left, right  # pre-referring to root.
    T2_post_order = [1, 4, 5, 7, 3]

    T1 = [4, 2, 1, 3, 6, 5, 7, 8]
    #   _4_
    #  /   \
    #  2   6
    # / \ / \
    # 1 3 5 7
    #        \
    #        8

    def test_compute_sum(self):
        bst = BST(self.T1)
        bst.compute_sums()
        self.assertEqual(bst.root.sum, sum(self.T1))
        self.assertEqual(bst.root.left.sum, 6)
        self.assertEqual(bst.root.right.sum, 26)

    def test_display_tree(self):
        trees = [(TestBST.T0, "T0"), (TestBST.T2, "T2")]
        for tree, name in trees:
            print(name)
            bst = BST(tree).display()
            print("")

    def test_insert(self):
        bst = BST()
        for i in TestBST.T0:
            bst.insert(i)
        self.assertEqual(bst.root.val, 3)
        self.assertEqual(bst.root.left.val, 2)
        self.assertEqual(bst.root.left.left.val, 1)

        self.assertEqual(bst.root.right.val, 5)
        self.assertEqual(bst.root.right.left.val, 4)
        self.assertEqual(bst.root.right.right.val, 6)
        self.assertEqual(bst.root.right.right.right.val, 7)

    def test_contains(self):
        bst = BST(self.T0)
        for i in self.T0:
            self.assertTrue(bst.contains(i))
        pass

    def test_contains_not(self):
        bst = BST(self.T0)
        NOT_IN_T0 = [10, 8, 0, -1, 99, 55]
        for i in NOT_IN_T0:
            self.assertFalse(bst.contains(i), "bst shouldn't contain: {}".format(i))

    # Tree heights.
    # https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem
    def test_get_height_0(self):
        bst = BST(TestBST.T0)
        self.assertEqual(bst.get_height_recursive(), 3)
        self.assertEqual(bst.get_height_iterative(), 3)
        self.assertEqual(bst.get_height_iterative_WithNestedClass(), 3)
        self.assertEqual(bst.get_height_iterative_augmentNode(), 3)

    def test_get_height_1(self):
        bst = BST([15])
        self.assertEqual(bst.get_height_recursive(), 0)
        self.assertEqual(bst.get_height_iterative(), 0)
        self.assertEqual(bst.get_height_iterative_WithNestedClass(), 0)
        self.assertEqual(bst.get_height_iterative_augmentNode(), 0)

    def test_get_height_2(self):
        bst = BST(TestBST.T2)
        self.assertEqual(bst.get_height_recursive(), 3)
        self.assertEqual(bst.get_height_iterative(), 3)
        self.assertEqual(bst.get_height_iterative_WithNestedClass(), 3)
        self.assertEqual(bst.get_height_iterative_augmentNode(), 3)

    def test_get_level_order(self):
        bst = BST(self.T0)
        level_order = bst.get_level_order()
        self.assertEqual(level_order, [3, 2, 5, 1, 4, 6, 7])

    def test_get_in_order_t0(self):
        self.assertEqual(BST(self.T0).get_in_order(), self.T0_in_order)

    def test_get_in_order_t2(self):
        self.assertEqual(BST(self.T2).get_in_order(), self.T2_in_order)

    def test_get_pre_order_t0(self):
        self.assertEqual(BST(self.T0).get_pre_order(), self.T0_pre_order)

    def test_get_pre_order_t2(self):
        self.assertEqual(BST(self.T2).get_pre_order(), self.T2_pre_order)

    def test_get_post_order_t0(self):
        self.assertEqual(BST(self.T0).get_post_order(), self.T0_post_order)

    def test_get_post_order_t2(self):
        self.assertEqual(BST(self.T2).get_post_order(), self.T2_post_order)


    @skipIf(SKIP_LONG_TESTS, "SKIP_LONG_TESTS is True")
    def test_long_get_height_big_recursive(self):
        try:
            self.assertEqual( BST(list(range(2001))).get_height_recursive(), 2000)  # An example where recursive approach due to max depth reached.
        except RuntimeError:
            print("INFO: Recursive approach  (as expected) ran into a 'RuntimeError: maximum recursion depth exceeded'")

    @skipIf(SKIP_LONG_TESTS, "SKIP_LONG_TESTS is True")
    def test_long_get_height_big_iterative_0(self):
        self.assertEqual(BST(list(range(2001))).get_height_iterative(), 2000)

    @skipIf(SKIP_LONG_TESTS, "SKIP_LONG_TESTS is True")
    def test_long_height_big_iterative_augmentedNode(self):
        self.assertEqual(BST(list(range(2001))).get_height_iterative_augmentNode(), 2000)




############### Parked items. Maybe useful.
# TODO_SOMEDAY - Create a note file for unittest and put in there.

# v1)  Mechanism to attach pdb if a test raises an unexpected exception.
# # Problem: Haven't figured out how to tie it into PyCharm's debugger. It opens a CMD prompt.
# # https://stackoverflow.com/questions/4398967/python-unit-testing-automatically-running-the-debugger-when-a-test-fails
# import unittest
# import sys
# import pdb
# import functools
# import traceback
# def debug_on(*exceptions):
#     if not exceptions:
#         exceptions = (AssertionError, AttributeError)  ## << Add your own exceptions.
#     def decorator(f):
#         @functools.wraps(f)
#         def wrapper(*args, **kwargs):
#             try:
#                 return f(*args, **kwargs)
#             except exceptions:
#                 info = sys.exc_info()
#                 traceback.print_exception(*info)
#                 pdb.post_mortem(info[2])
#         return wrapper
#     return decorator
#
# #     @debug_on()
# #     def test_hello(): ...
#
# V2)
# For Pycharm support, I should consider this: https://stackoverflow.com/questions/14081343/is-there-a-way-to-catch-unittest-exceptions-with-pycharm
# Meh, lazy. if I run into the issue, will evaluate.