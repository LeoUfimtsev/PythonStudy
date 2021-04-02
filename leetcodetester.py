
def leet_test(funcs, args, expected, test_name=""): # Leet code test runner.
    """Class testing code"""
    obj, overall = eval(funcs[0] + "(" + str(args[0])[1:-1] + ")"), True
    print(test_name + " ", obj.__class__.__name__)
    for i in range(1, len(funcs)):
        arg, exp = str(args[i])[1:-1], expected[i]
        cmd = "{}.{}({})".format("obj", funcs[i], str(arg))
        res = eval(cmd)
        if res != exp: overall = False
        print("{:<5} {:<15} arg:{:<5} res:{:<5} exp:{:<5}".format(str(res == exp), str(cmd), str(arg), str(res), str(exp)))
    print("::", overall, "\n")

null = None # Leetcode tends to use 'null'
if __name__ == '__main__':
    leet_test(  # l1: funcs to call. l2: args. l3: expc out

    )


# EXAMPLE: (Ported From python2)
# # Copy and paste the leet_test runner into your file. (It uses reflection, eval(..) fails).
# # URL: https://leetcode.com/problems/kth-largest-element-in-a-stream/submissions/
# from heapq import heapreplace, heappush, heapify
# class KthLargest(object):
#     def __init__(self, k, nums):
#         self.heap = []
#         self.k = k
#         for i in nums:
#             self.add(i)
#     def add(self, val):
#         if len(self.heap) < self.k:
#             heappush(self.heap, val)
#         elif val > self.heap[0]:
#             heapreplace(self.heap, val)
#         return self.heap[0]
#
# def leet_test(funcs, args, expected, test_name=""): # Leet code test runner.
#     obj = eval(funcs[0] + "(" + str(args[0])[1:-1] + ")")
#     print test_name + " ", obj.__class__.__name__
#     overall = True
#     for i in range(1, len(funcs)):
#         arg = str(args[i])[1:-1]
#         cmd = "{}.{}({})".format("obj", funcs[i], str(arg))
#         res = eval(cmd)
#         exp = expected[i]
#         if res != exp:
#             overall = False
#         # print res == exp, "a:"+arg, "r:"+str(res), "e:"+str(exp)
#         print "{} arg:{:<5} res:{:<5} exp:{:<5}".format(res == exp, arg, res, exp)
#     print "::", overall
#     print
#
# null = None
#
# if __name__ == '__main__':
#     leet_test(
#         ["KthLargest","add","add","add","add","add"],
#         [[3,[4,5,8,2]],[3],[5],[10],[9],[4]],
#         [None,4,5,5,8,8],
#         "Provided test"
#     )
#
#     leet_test(
#         ["KthLargest","add","add","add","add","add"],
#         [[1,[]],[-3],[-2],[-4],[0],[4]],
#         [null,-3,-2,-2,0,4],
#         "Initial Empty Heap"
#     )