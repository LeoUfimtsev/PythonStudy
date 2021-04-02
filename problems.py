##################
# CATEGORIZATION:
##################
# I Categorize my past problems into sets to make it easier to find them via simple query later.
# E.g: to find all problems of difficultly 'easy' which use a 'tree' data structure:
# find(diff.easy, ds.tree)
# E.g to find all problems I've completed this month:
# find_by_date(months_ago=0)
# It also allows me to print some statistics about problems that I've been working on for motivation.
# E.g:
# 19/08/16 Fri:
# Total Problems solved:  65
# Problems solved in recent days: 1 0 0 0 0 0 1 0 0 0 0 4 2 0 0 0 0 3 0 0 1 2 1 1 0 3 1 0 0 1
# Problems solved in recent weeks:  1 5 5 8 2 10 11 4 1 3
# Problems solved in recent Months:  8 34 8 10
# Difficulty of problems: Easy:29  Medium: 27 Hard: 9
# Data Structures stats
#    arrays 16    hashtable 14    tree 14    stacks 10    graph 8    heaps 6    linked_list 4    queue 2
# Time stats
#    min_30 21    hour_2 13    min_15 11    hour_5 8    hour 6    days 2
# Sources stats
#    hackerrank 45    leetcode 15    misc_interview_questions 2    reddit 1
# Process finished with exit code 0

class SetStats:
    """Designed to be a parent class, so that you can get nice stats from child classes."""
    @classmethod
    def _get_all_count(cls):
        """
        :rtype dict

        Print Statistics about problems solved via:
        for key, value in CLASS._get_all_count().iteritems():
             print "  ", key, value
        """
        members = filter(lambda x : x[0:1] != "_", dir(cls))
        d = dict()
        for member in members:
            d[member] = getattr(cls, member)
        for member_type in d.keys():
            d[member_type] = len(d[member_type])
        return d

class ds(SetStats):  # DataStructures.
    tree = set()
    graph = set()
    arrays = set()
    stacks = set()
    queue = set()
    linked_list = set()
    hashtable = set()
    heaps = set()

class algo:  # Algorithms
    dfs_bfs = set()  # Problems that could be solved either way.
    bfs = set()      # BFS specific problems, e.g nearest path problems.
    bisect = set()
    search = set()
    mst = set()
    kruskal = set()

    dynamic = set()

class diff:  # Difficulty.
    easy = set()
    med = set()
    hard = set()

class time(SetStats):  # rounded to nearest. Note, this is the time it took *me* to solve them, rather than an estimate for you :-).
    min_15 = set()
    min_30 = set()
    hour = set()
    hour_2 = set()
    hour_5 = set()
    days = set()

class tag(SetStats):
    optimization = set()  # Problems where it may be common to implement O(n^2) but better solutions are available.

    interview_material = set()  # Good problem to use for interviewing people or teaching.
    non_interview = set() # Problems not likley to be asked on interviews due to their length/complexity.

    insight = set()  # Problem that required some out-of-the-box thinking.
    random_test = set()  # Problem for which I wrote random tests
    recursion = set()  # A problem that has a recursive solution.

    failed = set()   # Problems I've tried and failed at. Maybe do some other day.
    amazon = set()   # Problem (or kinda-like it) was asked on an Aamzon interview.
    strings = set()  # Lots of work with String data.

    math = set()  # Problems involving match. E.g series.

    interesting = set()  # Problems that just interest me. (as of 19/07/30). Cool/funky problems.

class source(SetStats):
    hackerrank = set()
    leetcode = set()
    reddit = set()
    misc_interview_questions = set() # Random interview questions I've bumped into. Often don't have tests or detailed description.
    instacart = set()

problem_categories = [ds, algo, diff, time, tag]
all_problems = set()  # dynamically populated set with all problems I've worked on. (or those added via addto(..))

from datetime import date, timedelta
prob_dates = []
from collections import namedtuple
Date_and_func = namedtuple("Date_and_func", ["date", "func"])

from itertools import product
def addto(problem, *args):
    for arg in args:
        if isinstance(arg, date):
            prob_dates.append(Date_and_func(arg, problem.__name__))
        elif isinstance(arg, set):
            arg.add(problem)
    all_problems.add(problem)


################
################ Problems
################

# Leet code uses typing in it's Solution signatures:
from typing import List, Set, Tuple, Dict

# ######## Templates:
def __():
    pass

def merge_kSortedLists():
    # URL: https://leetcode.com/problems/merge-k-sorted-lists/submissions/
    # see also: https://www.geeksforgeeks.org/merge-k-sorted-arrays/
    # Intuition:
    # Convert into regular lists
    # Use heapq.merge which uses a min-heap internally to merge lists. -> O(n  k Log k)
    # Convert back to Linked list.
    # Alternative solution is to merge them in order. (1st & 2nd, then merge result with 3rd, then with 4th etc...)

    # Runtime: 128 ms, faster than 52.95% of Python3 online submissions for Merge k Sorted Lists.
    # Memory Usage: 20.7 MB, less than 6.06% of Python3 online submissions for Merge k Sorted Lists.
    # Definition for singly-linked list.
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None
    import heapq
    class Solution:
        def mergeKLists(self, lists: List[ListNode]) -> ListNode:
            normal_lists = []
            for ll in lists:
                lst = []
                curr = ll
                while curr:
                    lst.append(curr.val)
                    curr = curr.next
                normal_lists.append(lst)
            sorted_list = heapq.merge(*normal_lists)
            head = None
            curr = None
            for i in sorted_list:
                if head is None:
                    head = ListNode(i)
                    curr = head
                else:
                    curr.next = ListNode(i)
                    curr = curr.next
            return head
addto(merge_kSortedLists, ds.heaps, diff.med, date(2019,9,12), time.min_15, source.leetcode,)

def heap__kthLargest():
    # Inuition: Use a min-heap. Replace top element if new element *is bigger*. Return what's left on the heap.
    # 5a44c60be8684dd7906618899bb10931
    # Runtime: 76 ms, faster than 80.34% of Python3 online submissions for Kth Largest Element in an Array.
    # Memory Usage: 15 MB, less than 10.00% of Python3 online submissions for Kth Largest Element in an Array.
    from typing import List
    from heapq import heappush, heapreplace
    class Solution:
        def findKthLargest(self, nums, k) -> int:
            mheap = []
            for i in nums:
                if len(mheap) < k:
                    heappush(mheap, i)
                else:
                    if i > mheap[0]:
                        heapreplace(mheap, i)
            return mheap[0]

    # 1 line solution:
    import heapq
    def findKthLargest(self, nums, k) -> int:
        return heapq.nlargest(k, nums)[-1]

addto(heap__kthLargest, ds.heaps, diff.med, time.min_15, date(2019,9,12), source.leetcode, tag.interview_material)

def Array__BestTimeToBuyAndSellStock():
    # URL:https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    # Find the biggest increase.
    # Intuitivley, keep track of current min and find the biggest difference.
    class Solution:
        def maxProfit(self, prices: List[int]) -> int:
            if len(prices) <= 1:
                return 0
            curr_min = prices[0]
            max_diff = 0
            for i in prices:
                if i < curr_min:
                    curr_min = i
                    continue
                else:
                    diff = i - curr_min
                    max_diff = max(max_diff, diff)
            return max_diff
addto(Array__BestTimeToBuyAndSellStock, ds.arrays, diff.easy, time.min_15, date(2019,8,16), source.leetcode, tag.interview_material)

def Hashtable__HashDS_with_timestamps():
    # URL: https://www.glassdoor.ca/Interview/Problem-broken-into-three-phases-1-create-a-hash-like-data-structure-class-that-handles-get-and-set-2-Allow-key-to-ho-QTN_3147006.htm
    # Software Engineer Interview
    # Problem broken into three phases
    # 1. create a hash like data  structure class that handles get and set.
    # 2. Allow key to hold many values with each value having a timestamp. if get receives just they key, return latest value. If get receives both key and time, return the appropriate value.
    # 3. modify get to return closest value with timestamp lesser than argument timestamp.

    from collections import defaultdict, OrderedDict
    from time import time
    from bisect import bisect_left, bisect

    class DictTimeStamps:
        def __init__(self):
            self.dic_vals = defaultdict(list)
            self.dic_times = defaultdict(list)

        def set(self, key, val, timestamp=None):
            if timestamp is None:
                timestamp = time()
            if timestamp in self.dic_times[key]:  # Overwrite an existing timestamp.
                ts_index = self.dic_times[key].index(timestamp)
                self.dic_vals[key][ts_index] = val
            else:
                self.dic_vals[key].append(val)
                self.dic_times[key].append(timestamp)
            return timestamp

        def get(self, key, timestamp=None):
            if key not in self.dic_vals:
                raise KeyError
            if timestamp is None:
                return self.dic_vals[key][-1]
            else:
                if timestamp in self.dic_times[key]:
                    indx = bisect_left(self.dic_times[key], timestamp)
                else:
                    indx = bisect(self.dic_times[key], timestamp) - 1
                return self.dic_vals[key][indx]

    dts = DictTimeStamps()
    ts0 = dts.set(1, "a")
    ts1 = dts.set(1, "b")

    dts.set(1, "c", ts1)
    dts.set(1, "z")

    print(dts.get(1))       # exp: z
    print(dts.get(1, ts0))  # exp: a

    dts.set(2, "val1", 5)
    dts.set(2, "val2", 10)
    print(dts.get(2, 7))    # exp: "val1"
addto(Hashtable__HashDS_with_timestamps, ds.hashtable, algo.bisect, diff.med, time.hour, date(2019,8,10), source.instacart, tag.interview_material, tag.interesting)

def LinkedLists__MergeTwoLinkedLists():
    # URL: https://leetcode.com/problems/merge-two-sorted-lists/
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None

    # Solution: Create new list. (Alternative, merge l2 into l1)
    # Runtime: 44 ms, faster than 63.71% of Python3 online submissions for Merge Two Sorted Lists.
    # Memory Usage: 13.9 MB, less than 5.08% of Python3 online submissions for Merge Two Sorted Lists.
    class Solution:
        def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
            head = None
            curr = None
            while l1 or l2:
                if l1 and l2:
                    if l1.val < l2.val:
                        val = l1.val
                        l1 = l1.next
                    else:
                        val = l2.val
                        l2 = l2.next
                elif l1:
                    val = l1.val
                    l1 = l1.next
                else:
                    val = l2.val
                    l2 = l2.next

                if head is None:
                    head = ListNode(val)
                    curr = head
                else:
                    curr.next = ListNode(val)
                    curr = curr.next
            return head
addto(LinkedLists__MergeTwoLinkedLists, ds.linked_list, time.min_30, diff.easy, date(2019,8,5), source.leetcode, tag.interview_material, tag.amazon)

def Hashmap__ThreeSumAdv():
    # O(n^2) with optimizations.
    # Intuition:
    # (I use i,j,k instead of a,b,c)
    # - nested i,j loop to produce every possible (i,j) in O(n^2) time.
    # - for each i,j, check if nums contains a k such that k is after j. Using a dict, this is O(1)
    #   - This is accomplished by keeping a dictionary that keeps the last index of any given value.
    # - to keep output tuples unique
    #   - use a set to store output (automatically deduplicates) -> O(1)
    #   - sort tuple's i,j,k and use the tuple as key ot the set.
    # - Last test case is very large [0,0,0,0,0,0....0]. It times out.
    #   We observe that tuple consists of 3 elements, so remove any number that occurs more than 3 times.
    # Runtime: 724 ms, faster than 87.02% of Python3 online submissions for 3Sum.
    # Memory Usage: 18.1 MB, less than 5.46% of Python3 online submissions for 3Sum.
    from collections import defaultdict
    class Solution:
        def threeSum(self, nums_in: List[int]) -> List[List[int]]:
            # Optimization: Don't permit more than 3 duplicates
            nums_count = defaultdict(int)
            nums = []
            for n in nums_in:
                nums_count[n] += 1
                if nums_count[n] <= 3:
                    nums.append(n)

            # Keep track of last index of any given number. (i + j + want = 0)
            num_last_i = dict()
            for i, n in enumerate(nums):
                num_last_i[n] = i

            out = set()
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    want = - (nums[i] + nums[j])  # i+j+w=0 -> w = -(i+j)
                    if want in num_last_i:
                        last_w_i = num_last_i[want]
                        if last_w_i > j:
                            out_tup = tuple(sorted([nums[i], nums[j], nums[last_w_i]]))
                            out.add(out_tup)
            return [list(o) for o in out]

    res = Solution().threeSum([-1, 0, 1, 2, -1, -4])
    print("[")
    for l in res:
        print("  ", l)
    print("]")

    # Brute force:
    # O(n^3) -> Time Limit Exceeded. Can you do better?
    class Solution:
        def threeSum(self, nums: List[int]) -> List[List[int]]:
            out = set()
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    for k in range(j + 1, len(nums)):
                        ival, jval, kval = nums[i], nums[j], nums[k]
                        if (ival + jval + kval) == 0:
                            out.add(tuple(sorted([ival, jval, kval])))
            return [list(o) for o in out]
addto(Hashmap__ThreeSumAdv, ds.hashtable, diff.med, time.hour_2, date(2019,8,5), source.leetcode, tag.amazon, tag.interesting, tag.interview_material)

def Graph__MultiDFS_NumberOfIslands():
    # URL:https://leetcode.com/problems/number-of-islands/submissions/
    very_similar_to = Graphs__MultiBFS_maxRegion() # except here only up/down/left/right. Easier to code up.
    class Solution:
        def numIslands(self, grid: List[List[str]]) -> int:
            if len(grid) == 0 or len(grid[0]) == 0:
                return 0
            rows = len(grid)
            cols = len(grid[0])

            island_count = 0
            for row in range(rows):
                for col in range(cols):
                    cell = grid[row][col]
                    if cell == "1":
                        island_count += 1
                        grid[row][col] = "0"  # mark as visited when adding to queue.
                        to_explore = [(row, col)]
                        while to_explore:
                            r, c = to_explore.pop()
                            for near_r, near_c in [(r, c - 1), (r - 1, c), (r, c + 1), (r + 1, c)]:
                                if 0 <= near_r < rows and 0 <= near_c < cols:
                                    if grid[near_r][near_c] == "1":
                                        grid[near_r][near_c] = "0"
                                        to_explore.append((near_r, near_c))
            return island_count
addto(Graph__MultiDFS_NumberOfIslands, ds.graph, algo.bfs, algo.dynamic, diff.med, time.min_30, date(2019,8,5), source.leetcode, tag.interesting, tag.interview_material)


def Search__LongestPalidromicSubstring():
    # URL: https://leetcode.com/problems/longest-palindromic-substring
    # Dynamic Programming (supposedly?)
    # Approach: Expand Around Center
    # Time: O(n^2)
    # Space: O(1)
    # Better: Manacher's Algo. O(n). But too complex for 45 mins.
    #           https://www.hackerrank.com/topics/manachers-algorithm
    class Pali:
        def __init__(self, i, j):
            self.i = i
            self.j = j
            self.diff = j - i


    def expand_center(s, lower, upper):
        while (lower - 1) >= 0 and (upper + 1) < len(s) and \
                s[lower - 1] == s[upper + 1]:
            lower, upper = lower - 1, upper + 1
        return Pali(lower, upper)


    class Solution:
        def longestPalindrome(self, s: str) -> str:
            if s == "":
                return ""
            mx_pali = Pali(0, 0)
            for i in range(len(s)):
                odd_pali = expand_center(s, i, i)
                if (i + 1) < len(s) and s[i] == s[i + 1]:
                    even_pali = expand_center(s, i, i + 1)
                else:
                    even_pali = Pali(0, 0)
                mx_pali = max(mx_pali, odd_pali, even_pali, key=lambda p: p.diff)
            return s[mx_pali.i: mx_pali.j + 1]


    s = Solution()
    tests = [["babad", "bab"],
             ["cbbd", "bb"],
             ["zgeegf", "geeg"],
             ["zgefegfr", "gefeg"],
             ["", ""]]
    results = []
    for t, e in tests:
        res = s.longestPalindrome(t)
        results.append(res)
        print(res == e, t, e, res)
    print("::", all(results))


    # Alternative (2x slower but easier? to read) solution would be to insert "#" between everything. aba -> #a#b#a#
    #  so no ambiguity about palindrome position.
    # Runtime: 2504 ms, faster than 45.47% of Python3 online submissions for Longest Palindromic Substring.
    # Memory Usage: 13.9 MB, less than 22.89% of Python3 online submissions for Longest Palindromic Substring.
    # ...
    class Solution:
        def longestPalindrome(self, s: str) -> str:
            if s == "":
                return ""
            s = "".join(["#", "#".join(s), "#"])
            mx = Pali(0, 0)
            for i in range(len(s)):
                pali = expand_center(s, i, i)
                mx = max(mx, pali, key=lambda p: p.diff)
            pali_str = s[mx.i: mx.j + 1]
            pali_str = "".join([c if c != "#" else "" for c in pali_str])
            return pali_str
    # ...
addto(Search__LongestPalidromicSubstring, ds.arrays, algo.dynamic, diff.med, time.hour, date(2019,8,5), source.leetcode, tag.interesting, tag.amazon, tag.optimization)

def OrderedDict__LRU_Cache():
    # URL: https://leetcode.com/problems/lru-cache/
    # 54e9b14946354840b9292408008bf7af

    # Solution that extends OrderedDict() itself:
    # Runtime: 208 ms, faster than 77.37% of Python3 online submissions for LRU Cache.
    # Memory Usage: 23 MB, less than 5.45% of Python3 online submissions for LRU Cache.
    from collections import OrderedDict
    class LRUCache(OrderedDict):  # Extend OrederdDict instead of initiating local copy.
        def __init__(self, capacity: int):
            self.capacity = capacity

        def get(self, key: int) -> int:
            if key in self:
                self.move_to_end(key)
                return self[key]
            else:
                return -1

        def put(self, key: int, value: int) -> None:
            self[key] = value
            self.move_to_end(key)
            if len(self) > self.capacity:
                del (self[next(iter(self.keys()))])

    # Solution that adds self.d=OrderedDict()
    # Runtime: 204 ms, faster than 88.32% of Python3 online submissions for LRU Cache.
    # Memory Usage: 22.9 MB, less than 5.45% of Python3 online submissions for LRU Cache.
    from collections import OrderedDict
    class LRUCache:
        def __init__(self, capacity: int):
            self.d = OrderedDict()
            self.capacity = capacity

        def get(self, key: int) -> int:
            if key in self.d:
                self.d.move_to_end(key)
                return self.d[key]
            else:
                return -1

        def put(self, key: int, value: int) -> None:
            # Consider case where we overwrite a key. Key needs to move to front.
            self.d[key] = value
            self.d.move_to_end(key)
            if len(self.d) > self.capacity:
                del (self.d[next(iter(self.d))])

    #-------- Testing code:
    def leet_test(funcs, args, expected, test_name=""):  # Leet code test runner.
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

    null = None  # Leetcode tends to use 'null'
    if __name__ == '__main__':
        leet_test(  # l1: funcs to call. l2: args. l3: expc out
            ["LRUCache", "get", "put", "get", "put", "put", "get", "get"],
            [[2], [2], [2, 6], [1], [1, 5], [1, 2], [1], [2]],
            [null, -1, null, -1, null, null, 2, 6]
            #
        )
        leet_test(
            ["LRUCache", "put", "put", "put", "put", "get", "get"],
            [[2], [2, 1], [1, 1], [2, 3], [4, 1], [1], [2]],
            [null, null, null, null, null, -1, 3]
            # out[null,       null,   null,   null,   null,   1,     -1]
            # state?         2=1   2=1,1=1   1=1,2=3 2=3,4=1
        )

    '''
    Input
    ["LRUCache","put","put","put","put","get","get"]
    [[2],[2,1],[1,1],[2,3],[4,1],[1],[2]]
    Output
    [null,null,null,null,null,1,-1]
    Expected
    [null,null,null,null,null,-1,3]

    '''

    '''
    Failing
    Input
    ["LRUCache","get","put","get","put","put","get","get"]
    [[2],[2],[2,6],[1],[1,5],[1,2],[1],[2]]
    Output
    [null,-1,null,-1,null,null,2,-1]
    Expected
    [null,-1,null,-1,null,null,2,6]

    '''

    ''' Working test:
    Your input
    ["LRUCache","put","put","get","put","get","put","get","get","get"]
    [[2],[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]
    Output
    [null,null,null,1,null,-1,null,-1,3,4]
    Expected
    [null,null,null,1,null,-1,null,-1,3,4]
    '''
addto(OrderedDict__LRU_Cache, ds.hashtable, diff.med, time.hour_2, date(2019,8,4), source.leetcode, tag.amazon, tag.interesting, tag.insight)
def LinkedList__AddTwoNumbers():
    # URL: https://leetcode.com/problems/add-two-numbers/submissions/
    # Given two linked lists, where a node hold a single digit, reversed,
    # add both numbers and return a reversed list.

    # Testing:
    class ListNode:
        def __init__(self, x):
            self.val = x
            self.next = None


    # Solution 1: Convert list to integers.
    # This sol is a bit more intuitive.
    # Runtime: 84 ms, faster than 31.32% of Python3 online submissions for Add Two Numbers.
    # Memory Usage: 14 MB, less than 5.20% of Python3 online submissions for Add Two Numbers
    class Solution:
        def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
            def ll_to_num(lst):
                num = []
                curr = lst
                while curr:
                    num.append(curr.val)
                    curr = curr.next
                num.reverse()
                return int("".join(map(str, num)))

            def num_to_ll(num):
                reversed_digits = list(map(int, str(num)[::-1]))
                head = ListNode(reversed_digits[0])
                curr = head
                for i in reversed_digits[1:]:
                    curr.next = ListNode(i)
                    curr = curr.next
                return head

            n1 = ll_to_num(l1)
            n2 = ll_to_num(l2)
            return num_to_ll(n1 + n2)


    # Solution 2: Traverse both lists in parallel.
    # This solution uses less memory and arguably does less work.
    # Runtime: 68 ms, faster than 97.55% of Python3 online submissions for Add Two Numbers.
    # Memory Usage: 13.9 MB, less than 5.20% of Python3 online submissions for Add Two Numbers.
    class Solution:
        def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
            head = None
            carry = 0
            while l1 or l2 or carry != 0:
                l1_val = l1.val if l1 else 0
                l2_val = l2.val if l2 else 0
                nsum = l1_val + l2_val + carry
                val = nsum % 10
                carry = nsum // 10
                if head is None:
                    head = ListNode(val)
                    curr = head
                else:
                    curr.next = ListNode(val)
                    curr = curr.next
                l1 = l1.next if l1 else None
                l2 = l2.next if l2 else None
            return head


    """ Testing:
    [2,4,3]
    [5,6,4]
    -> [7,0,8]

    [1,8]
    [0]
    -> [1,8]

    [5]
    [5]
    -> [0, 1]
    """
addto(LinkedList__AddTwoNumbers, ds.linked_list, diff.easy, time.min_30, date(2019,8,4), source.leetcode, tag.amazon)

def Hashtable__Find_Majority():
    # URL: https://leetcode.com/problems/majority-element/submissions/

    # Attempt 4: (Review of other people's code:
    # Runtime: 192 ms, faster than 79.52% of Python3 online submissions for Majority Element.
    # Memory Usage: 14.9 MB, less than 5.11% of Python3 online submissions for Majority Element.
    from collections import Counter
    class Solution:
        def majorityElement(self, nums):
            counts = Counter(nums)
            return max(counts, key=counts.get)

    # Simmilar to counter, except with default dict.
    from collections import defaultdict
    class Solution:
        def majorityElement(self, nums):
            d = defaultdict(int)
            for i in nums:
                d[i] += 1
            return max(d, key=d.get)

    # Attempt 3:
    # O(n) solution.
    # Count instances via hash table O(n). Find max value in the hashtable O(k) (where k = # of unique keys)
    # Runtime: 196 ms, faster than 65.03% of Python3 online submissions for Majority Element.
    # Memory Usage: 15.2 MB, less than 5.43% of Python3 online submissions for Majority Element.
    from collections import defaultdict
    from typing import List
    class Solution:
        def majorityElement(self, nums: List[int]) -> int:
            d = defaultdict(int)
            for i in nums:
                d[i] += 1
            max_val = -1
            max_key = None
            for i in d.keys():
                if d[i] > max_val:
                    max_val = d[i]
                    max_key = i
            return max_key

    # Attempt 2
    # O (n log n)
    # Sort. Traverse. Track element counts. Return max counts.
    # Runtime: 204 ms, faster than 38.62% of Python3 online submissions for Majority Element.
    # Memory Usage: 15 MB, less than 5.43% of Python3 online submissions for Majority Element.
    from collections import defaultdict
    class Solution:
        def majorityElement(self, nums: List[int]) -> int:
            nums.sort()
            curr_i = None
            max_element = None
            max_count = 0
            for i in nums:
                if curr_i is None or i != curr_i:
                    curr_i = i
                    curr_count = 1
                else:
                    curr_count += 1
                if curr_count > max_count:
                    max_count = curr_count
                    max_element = i
            return max_element

    # Attemp 1
    # O(n log n)
    # Count items via hashtable. Sort Hash table according to key count. O (k log k), where k=unique keys.
    # Runtime: 192 ms, faster than 79.88% of Python3 online submissions for Majority Element.
    # Memory Usage: 15.3 MB, less than 5.43% of Python3 online submissions for Majority Element.
    from typing import List
    from collections import defaultdict
    class Solution:
        def majorityElement(self, nums: List[int]) -> int:
            counts = defaultdict(int)
            for i in nums:
                counts[i] += 1
            count_list = list(counts.items())
            count_list.sort(key=lambda x: x[1])
            return count_list[-1][0]
addto(Hashtable__Find_Majority, ds.hashtable, time.min_30, diff.easy, source.leetcode, tag.interview_material, tag.interesting, tag.optimization)

def Array__TwoSum__IceCreamVersion():
    # URL: https://www.hackerrank.com/challenges/ctci-ice-cream-parlor/problem
    # O(n)
    def whatFlavors(cost, money):
        seen = dict()
        for i in range(len(cost)):
            flavor_cost = cost[i]
            remaining = money - flavor_cost
            if remaining in seen:
                print(seen[remaining], i + 1)
            else:
                seen[flavor_cost] = i + 1
addto(Array__TwoSum__IceCreamVersion, ds.hashtable, diff.easy, time.min_15, date(2019, 7, 30), source.hackerrank, tag.interview_material)

def Search__Triple_tuple():
    # URL: https://www.hackerrank.com/challenges/triple-sum/problem
    # Given arrays A,B,C find count of (p,q,r), such that p<= q >= r.  (duplicates, unsorted).
    # !/bin/python3
    # f1b36e7abb66483cba6fefb551a676bb
    # O(a log a +  b log b + c log c)
    def triplets(a, b, c):
        a, b, c = [list(set(i)) for i in [a, b, c]]  # Deduplicate.
        a.sort()
        b.sort()
        c.sort()

        tuple_count = 0
        ai, ci = 0, 0  # a_index, c_index
        for b_val in b:
            while a[ai] < b_val:
                if ai < len(a) - 1 and a[ai + 1] <= b_val:  # Consider [1,4,10..] not increments of 1.
                    ai += 1
                else:
                    break
            if a[ai] > b_val:
                continue

            while c[ci] < b_val:
                if ci < len(c) - 1 and c[ci + 1] <= b_val:
                    ci += 1
                else:
                    break
            if c[ci] > b_val:
                continue

            tuple_count += (ai + 1) * (ci + 1)
        return tuple_count

    ### Same solution, but more concise:
    def triplets(A, B, C):
        A, B, C = [sorted(set(i)) for i in [A, B, C]]  # Deduplicate & sort.
        tuple_count = 0

        def move_up(L, li, b):
            while L[li] < b and li < len(L) - 1 and L[li + 1] <= b:
                li += 1
            return li

        ai, ci = 0, 0  # a_index, c_index
        for b in B:
            ai = move_up(A, ai, b)
            ci = move_up(C, ci, b)
            if A[ai] > b or C[ci] > b:
                continue
            tuple_count += (ai + 1) * (ci + 1)
        return tuple_count

    ####
    # O(a log a +  b log b +  c log c + b log a + b logc)
    # Slower solution but easier to implement/write. Still passes all test cases.
    #...
    from bisect import bisect
    def triplets(A, B, C):
        A, B, C = [list(set(i)) for i in [A, B, C]]  # Deduplicate.
        A.sort()
        B.sort()
        C.sort()
        tuple_count = 0
        for b in B:
            ai = bisect(A, b)  # A=[2,3,4], b[5] -> ai = 3.  b[1]-> 0
            bi = bisect(C, b)
            if ai == 0 or bi == 0:
                continue
            tuple_count += ai * bi
        return tuple_count
    # ..
addto(Search__Triple_tuple, ds.arrays, algo.search, diff.med, time.hour_2, date(2019,7,30), source.misc_interview_questions, tag.interview_material, tag.insight, tag.interesting)

def Search__Permutated_Substrings():
    # URL: --- Found brief mention on youtube.

    # Given s and b, find all permutations of s in b.
    s = "xacxzaa"
    b = "fxaazxacaaxzoecazxaxaz"

    # More simple:
    # s = aabc
    # b = aacbacab
    #

    # Attempt 1: (Works but slow).
    # O(bs). For every substring, compare dictionary.
    from collections import Counter
    def find_permutated_substrings(needle, haystack):
        s_counted = Counter(needle)
        out = []
        for i in range(len(haystack) - len(needle)):
            substr = haystack[i:i + len(needle)]
            substr_counted = Counter(substr)
            if s_counted == substr_counted:
                out.append(substr)
        return out

    print(find_permutated_substrings(s, b))

    # Attempt 2: (Fast but buggy).
    # O(b). Generate prime numbers & assign to characters. Maintain running prime_sum.
    # If running prime sum is same as s prime sum, it's a permutation.
    # -> Works-ish, but unreliable.
    # (seems to have minor bug that rarely shows up, probably summing primes generates ambiguity/conflict. e.g 2+5=7).
    from collections import deque
    def find_permutated_substrings_prime(needle, haystack):
        def prime_generator():  # pg = prime_generator(); next(pg)
            first_100_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
                                113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
                                241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
                                383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521,
                                523, 541]
            for p in first_100_primes:
                yield p
            call_count = 1
            i = first_100_primes[-1] + 1
            while True:
                i += 1
                for j in range(2, i):
                    if i % j == 0:
                        break
                else:
                    print(call_count)
                    call_count += 1
                    yield i

        char_to_prime = dict()
        pg = prime_generator()  # Optimization: Pre-compute primes.

        # Assign primes to unique characters in input.
        for chr in needle + haystack:
            if chr not in char_to_prime:
                char_to_prime[chr] = next(pg)

        s_prime_sum = sum([char_to_prime[c] for c in needle])

        out = []
        stream = deque()
        running_prime_sum = 0

        for i in range(len(haystack)):
            stream.append(haystack[i])
            running_prime_sum += char_to_prime[haystack[i]]

            if len(stream) > len(needle):
                left_char = stream.popleft()
                running_prime_sum -= char_to_prime[left_char]

            if running_prime_sum == s_prime_sum:
                out.append("".join(stream))

        return out

    print(find_permutated_substrings_prime(s, b))

    # Attempt 3: (Fast and reliable)
    # O(n)
    # Have a streaming dictionary, starting with negative counts of s. (e.g aabc -> a=-2, b=-1, c=-1).
    # Add/remove characters from stream. If len(dict) == 0, we have a match.
    from collections import Counter, deque, defaultdict
    def find_permutated_substrings_streamingdict(s, b):
        # s = "abc"
        # b = "acbaacb"
        # s = aabc
        # b = aacbacab -> 4 matches.
        dictstream = Counter(s)
        for k in dictstream.keys():
            dictstream[k] = -dictstream[k]
        charstream = deque()
        out = []
        for c in b:
            charstream.append(c)
            dictstream[c] += 1
            if dictstream[c] == 0:
                del (dictstream[c])

            if len(charstream) > len(s):
                popped = charstream.popleft()
                dictstream[popped] -= 1
                if dictstream[popped] == 0:
                    del (dictstream[popped])

            if len(dictstream) == 0:
                out.append("".join(charstream))

        return out

    print(find_permutated_substrings_streamingdict(s, b))

    # Tests and comparisons:
    import random
    import string
    s = "".join([random.choice(string.ascii_lowercase) for _ in range(1000)])
    b = "".join([random.choice(string.ascii_lowercase) for _ in range(100000)])
    print("p1")
    r1 = find_permutated_substrings(s, b)
    print("p2")
    r2 = find_permutated_substrings_prime(s, b)
    print("p3")
    r3 = find_permutated_substrings_streamingdict(s, b)

    # Most of the time r1==r2==r3, although there seems to be a subtle situation where they sometimes off by 1?. Meh.
    if r1 == r2 == r3:
        print("Equal", len(r1))
    else:
        print("Not equal", len(r1), len(r2), len(r3))

    import timeit
    print(find_permutated_substrings.__name__, timeit.timeit(lambda: find_permutated_substrings(s, b), number=5))
    print(find_permutated_substrings_prime.__name__, timeit.timeit(lambda: find_permutated_substrings_prime(s, b), number=5))
    print(find_permutated_substrings_streamingdict.__name__, timeit.timeit(lambda: find_permutated_substrings_streamingdict(s, b), number=5))

    # Performance Analysis.
    # Very big performance difference if s is large. (1000+)
    # find_permutated_substrings 24.026676302  O(sb)
    # find_permutated_substrings_prime 0.3006969350000013 O(b)
    # find_permutated_substrings_streamingdict 0.6048754650000028 O(b)
addto(Search__Permutated_Substrings, ds.hashtable, diff.med, time.hour_2, date(2019,7,30), source.misc_interview_questions, tag.insight, tag.interview_material, tag.interesting)


def Graphs__TopologicalSort_CourseScheduling():
    # Time:  O(N + E)
    # Space: O(N) #DFS stack.
    # Alternative: Node Indegree
    # 01651c3d70d74c76812d8ee54e25dcfa Notability/notes.

    # Solution with class + pointers. Most readable.
    # Runtime: 112 ms, faster than 90.15%-27% of Python3 online submissions for Course Schedule II.
    # Memory Usage: 15 MB, less than 51.44% of Python3 online submissions for Course Schedule II.
    from enum import Enum, auto
    class Col(Enum):  # Py>=3.4. Slower than regular int comparison, but makes debuging a lot easier. -> Worth using.
        White = auto()  # New
        Gray = auto()  # Visiting (DFS in progress)
        Black = auto()  # Visited


    class Node:
        def __init__(self, id):
            self.id = id
            self.edges = list()
            self.col = Col.White


    class Solution:
        def findOrder(self, numCourses, prerequisites):
            g = [Node(i) for i in range(numCourses)]
            for edg in prerequisites:
                from_n, to_n = g[edg[1]], g[edg[0]]
                from_n.edges.append(to_n)

            stack = []
            for node in g:
                if node.col == Col.Black:
                    continue

                dfs_stack = [node]
                while dfs_stack:
                    node = dfs_stack[-1]
                    if node.col == Col.White:
                        node.col = Col.Gray

                    while node.edges:
                        to_node = node.edges.pop()
                        if to_node.col == Col.Black:
                            continue
                        if to_node.col == Col.Gray:  # Cycle found.
                            return []
                        dfs_stack.append(to_node)
                        break
                    else:  # Note, else only executed if no 'break' was invoked.
                        node.col = Col.Black
                        dfs_stack.pop()
                        stack.append(node.id)

            stack.reverse()
            return stack


    ## TOPOLOGICAL SORT
    # Solution with no Enums & pointers.
    # Runtime: 120 ms, faster than 50.00% of Python3 online submissions for Course Schedule II.
    # Memory Usage: 15.3 MB, less than 51.08% of Python3 online submissions for Course Schedule II.
    from typing import Set
    WHITE, GRAY, BLACK = 0, 1, 2  # new, visiting, visited

    class Node: # Leet code runs on Py3.6, no suppoort for Dataclasses yet.
        def __init__(self, id):
            self.id = id
            self.edges = set()
            self.state = WHITE

        def __repr__(self):
            return "Node(id={}, state={})".format(self.id, self.state)

        def __eq__(self, other):  # So that we can add node to set or use as dict key.
            return self.id == other.id

        def __hash__(self):  # Needs to be implemented if we override __hash___
            return hash(self.id)


    class Solution:
        def findOrder(self, n, edges):
            g = [Node(i) for i in range(n)]

            for e in edges:
                g[e[1]].edges.add(g[e[0]])

            stack = []
            for node in g:
                if node.state == BLACK:
                    continue

                dfs_stack = [node]
                while dfs_stack:
                    node = dfs_stack[-1]
                    node.state = GRAY
                    if node.edges:
                        while node.edges:
                            adj_node = node.edges.pop()
                            if adj_node.state == GRAY:  # Cycle found.
                                return []
                            if adj_node.state == BLACK:
                                continue
                            dfs_stack.append(adj_node)
                            break
                    else:
                        node.state = BLACK
                        stack.append(node.id)
                        dfs_stack.pop()
            stack.reverse()
            return stack


    # Solution that uses adjacency list  (node -> [edges]) and dict for mapping.
    # Very simple, no fancy data structures.
    # Runtime: 120 ms, faster than 50.00% of Python3 online submissions for Course Schedule II.
    # Memory Usage: 15.2 MB, less than 51.86% of Python3 online submissions for Course Schedule II.
    from collections import defaultdict
    New = 0  # aka White
    Visiting = 1  # aka Gray
    Visited = 2  # aka Black

    class Solution:
        def findOrder(self, numCourses, prerequisites):
            g = defaultdict(list)
            g_state = defaultdict(lambda: 0)  # < dict for state mapping.
            for edg in prerequisites:
                g[edg[1]].append(edg[0])

            stack = []
            for nid in range(numCourses):
                if g_state[nid] == Visited:
                    continue

                dfs_stack = [nid]
                while dfs_stack:
                    nid = dfs_stack[-1]
                    if g_state[nid] == New:
                        g_state[nid] = Visiting
                    while len(g[nid]) > 0:
                        toid = g[nid].pop()
                        if g_state[toid] == Visited:
                            continue
                        if g_state[toid] == Visiting:  # Cycle found.
                            return []
                        dfs_stack.append(toid)
                        break
                    else:
                        g_state[nid] = Visited
                        dfs_stack.pop()
                        stack.append(nid)

            stack.reverse()
            return stack


    # As simple as possible, no classes etc..
    # Good runtime, but hard to read and very hard to debug (due to mental mapping). Would not recommend.
    # Runtime: 116 ms, faster than 76.72% of Python3 online submissions for Course Schedule II.
    # Memory Usage: 14.8 MB, less than 60.07% of Python3 online submissions for Course Schedule II.
    WHITE, GRAY, BLACK = 0, 1, 2
    edge_ids, color = 0, 1

    class Solution:
        def findOrder(self, n, edges):
            g = [[[], WHITE] for i in range(n)]
            for e in edges:
                g[e[1]][edge_ids].append(e[0])

            stack = []
            for i in range(n):
                if g[i][color] == BLACK:
                    continue

                dfs_stack = [i]
                while dfs_stack:
                    i = dfs_stack[-1]
                    g[i][color] = GRAY

                    while g[i][edge_ids]:
                        adj_nid = g[i][edge_ids].pop()
                        if g[adj_nid][color] == GRAY: return []  # Cycle
                        if g[adj_nid][color] == BLACK: continue  # Already visited.
                        dfs_stack.append(adj_nid)
                        break
                    else:  # only executed if no 'break' was called. I.e, if dfs_node.edges were empty or visited.
                        dfs_stack.pop()
                        stack.append(i)
                        g[i][color] = BLACK
            stack.reverse()
            return stack

addto(Graphs__TopologicalSort_CourseScheduling, ds.graph, date(2019,7,27), diff.med, time.hour_2, source.leetcode, tag.interview_material)


def Hashtables__Triplets():
    # URL: https://www.hackerrank.com/challenges/count-triplets-1/problem
    ## Passes all tests
    # O(n) ish.
    # dae9ccff5aea4a8ca6e087a7c16bd70d Notability notes
    """
    This problem was rather difficult (5 hours to solve).
    - Insight: Have to think of 'what tripplets can be constructed' rather than count all possibility.
        w/o this, tests 6,11,10 fail.
    - Gotta watch out for integer devision. (17/3 = 5). Can produce missleading 'previous'
    """
    from collections import defaultdict
    from dataclasses import dataclass

    @dataclass
    class I:
        idx: int
        cnt: int


    def countTriplets(arr, r):
        d = defaultdict(list)
        prev_count = defaultdict(int)  #
        triple_count = 0
        for i, v in enumerate(arr):
            prev = v / r  # (!) Integer division can be wrong.  17 // 3 -> 5. This builds incorrect previous (5, 17)
            prev_prev = (prev / r, prev)

            if prev_prev in d:
                # cnt = sum([i.cnt for i in d[prev_prev]])  # Counting the whole chain can be O(n) ish. Tests 6,11 fail.
                cnt = prev_count[(prev / r, prev, "sum")]  # Optimization, keep rolling sum. -> O(1)
                triple_count += cnt
            if prev in d:
                prev_c = len(d[prev])  # O(1)
                d[(prev, v)].append(I(i, prev_c))
                prev_count[(prev, v, "sum")] += prev_c  # Keep rolling su.
            d[v].append(i)

        return triple_count

    _, r = [int(i) for i in input().split()]
    arr = [float(i) for i in input().split()]
    print(countTriplets(arr, r))

    #### wip entries
    # T (Submission 6) ->  (integer devision issue.
    # 100000 3
    # 1 17 80 68 5 5 58 17 38 81 26 44 38 6 12 ...
    # expr:  2325652489
    # Act :   667065187 << wrong, under count.
    # ac2 : 19107507001 << wrong, over count. (integer devision issue.
    # ac3:   2325652489
addto(Hashtables__Triplets, ds.hashtable, diff.hard, date(2019,7,26), time.hour_5, source.hackerrank, tag.insight, tag.optimization)
def Hashtables__series__AnnagramSubstrings():
    # URL: https://www.hackerrank.com/challenges/sherlock-and-anagrams/problem
    from collections import defaultdict
    # 80f5e0dd45bd4b37b97cefc79c2e8541   # Notability scrap work kinda explains better.
    # O(n^2)

    # Consider generating all substrings, such that they are character sorted.
    # We are only interested if multiple matches are found.  (discard n=1)
    # kkkk ->                        n
    #  len 1 = k k k k     -> k    * 4
    #  len 2 = kk, kk , kk -> kk   * 3
    #  len 3 = kkk, kkk    -> kkk  * 2
    #  len 4 = kkkk

    # abba ->
    #  len 1: a b b a      -> a  * 2
    #                         b  * 2
    #  len 2: ab bb ab     -> ab * 2
    #  len 3: abb, bba
    #  len 4: abba

    # For every substring, how many matching substrings remain?
    # i=0 k -> k, k k,  i=1 k -> k k   i=2, k -> k   = 3 + 2  + 1 = 6
    # i=0 a -> a   i=1, b -> b     = 1 + 1 = 2
    # i=0 ab -> ab    i=1 bb->  i=2 ab->ab     = 1+1 = 2

    # We observe series-like behaviour
    # k = 4 ->       4-1 = 3,    3-1 = 2,    2-1=1,   1-1 = 0 ->   3+2+1 = 6
    # 1 + 2 + 3 + n = (n * (n+1)) / 2
    # We have 1 offset, so:
    # m = n-1 ->   (m * (m+1) / 2)

    # so for every value, perform (m * (m+1)) //2


    def every_substring(s):  # e.g abba
        ss = []
        for l in range(1, len(s) + 1):  # 1..4
            for i in range(0, len(s) - l + 1):  # l=1, i=0..3
                yield s[i:i + l]

    def sherlockAndAnagrams(s):
        wrdc = defaultdict(lambda: 0)  # word count dict

        # Generate every possible substring
        # Use a dict to count number of duplicate sorted sub strings.
        for subs in every_substring(s):
            sorted_subs = "".join(sorted(subs))
            wrdc[sorted_subs] += 1

        anna_sum = 0
        for n in wrdc.values():
            k = n - 1
            anna_sum += (k * (k + 1)) // 2  # Series: 1 + 2 + 3 + n = (n * (n+1)) / 2)
        return anna_sum

    qc = int(input())
    for _ in range(qc):
        s = input()
        print(sherlockAndAnagrams(s))
addto(Hashtables__series__AnnagramSubstrings, ds.hashtable, diff.med, date(2019,7,26), time.hour_2, source.hackerrank, tag.math, tag.optimization)

def Hashtables__twoChars():
    # URL: https://www.hackerrank.com/challenges/two-strings/problem?
    # Complete the twoStrings function below.
    def twoStrings(s1, s2):
        d = set(s1)
        for c in s2:
            if c in s1:
                return "YES"
        return "NO"
addto(Hashtables__twoChars, ds.hashtable, diff.easy, time.min_15, date(2019, 7, 25), source.hackerrank)

def Hashtables__WordsInMag():
    # Given words and words in magazine, can you form sentence?
    # https://www.hackerrank.com/challenges/ctci-ransom-note/problem
    import math
    import os
    import random
    import re
    import sys

    from collections import defaultdict
    def checkMagazine(mag, note):
        m = defaultdict(lambda: 0)
        for w in mag:
            m[w] += 1
        for w in note:
            if w in m and m[w] > 0:
                m[w] -= 1
            else:
                return "No"
        return "Yes"

    if __name__ == '__main__':
        mn = input().split()
        m = int(mn[0])
        n = int(mn[1])
        magazine = input().rstrip().split()
        note = input().rstrip().split()
        res = checkMagazine(magazine, note)
        print(res)
addto(Hashtables__WordsInMag, ds.hashtable, diff.easy, date(2019,7,24), time.min_15, source.hackerrank, tag.interview_material)

def Graphs__MultiBFS_maxRegion():
    # URL: https://www.hackerrank.com/challenges/ctci-connected-cell-in-a-grid/problem
    # This is excellent Interview material.
    # 4
    # 4
    # 1 1 0 0
    # 0 1 1 0
    # 0 0 1 0 << has max region of 5  1's adjacent to each other.  (adjacent = above/below/left/right/diagonal).
    # 1 0 0 0
    very_similar = Graph__MultiDFS_NumberOfIslands()
    import math, os, random, re, sys
    from itertools import product
    def maxRegion(grid):
        to_explore = []
        row_count = len(grid)
        col_count = len(grid[0])
        max_region = 0
        for r, c in product(range(row_count), range(col_count)):
            if grid[r][c] == 1:
                to_explore.append((r, c))

        while to_explore:
            region_sum = 0
            r, c = to_explore.pop()
            if grid[r][c] == 1:
                grid[r][c] = -1
                local_nodes = [(r, c)]

            while local_nodes:
                r, c = local_nodes.pop()
                region_sum += 1
                max_region = max(max_region, region_sum)
                adj_rows = [i + r for i in [-1, 0, 1] if 0 <= i + r < row_count]
                adj_cols = [i + c for i in [-1, 0, 1] if 0 <= i + c < col_count]
                for ar, ac in product(adj_rows, adj_cols):
                    if grid[ar][ac] == 1:
                        grid[ar][ac] = -1  # Need to be careful about neighbouring nodes adding same neighbour.
                        local_nodes.append((ar, ac))
        return max_region

    if __name__ == '__main__':
        fptr = open(os.environ['OUTPUT_PATH'], 'w')
        n = int(input())
        m = int(input())
        grid = []
        for _ in range(n):
            grid.append(list(map(int, input().rstrip().split())))
        res = maxRegion(grid)
        fptr.write(str(res) + '\n')
        fptr.close()
addto(Graphs__MultiBFS_maxRegion, ds.graph, algo.dynamic, diff.hard, time.hour_2, date(2019,7,22), algo.bfs, source.hackerrank, tag.interview_material, tag.insight)
def Graphs__ShortestReachGraph():
    # URL: https://www.hackerrank.com/challenges/ctci-bfs-shortest-reach/problem
    # Passes all test cases.
    # I believe this to be an efficent algorithm.
    # Complexity: O(N) traversal of nodes, O(E)
    #            -> O(N + E)
    from collections import defaultdict
    from collections import deque

    # wisdom: I make mistakes if I use  arra[index] for nodes/edges.
    #         Sol: -> append i to index values. E.g  nodei edgei
    class Node:
        def __init__(self):
            self.edges = set()
            self.dist = -1


    def get_shortest_reach(g, si):
        g[si].dist = 0
        to_explore = deque([si])
        while to_explore:
            node = g[to_explore.popleft()]  # BFS. But could use DFS.
            for edge_ni in node.edges:
                edge_n = g[edge_ni]
                if edge_n.dist == -1:
                    edge_n.dist = node.dist + 6
                    to_explore.append(edge_ni)
        return [node.dist for node in g if node.dist != 0]  # Filter dummy and start.

    queries = int(input())
    for _ in range(queries):
        n_len, e_len = [int(value) for value in input().split()]
        g = [Node() for _ in range(n_len + 1)]
        g[0].dist = 0  # Dummy to aling array index with node id. Filtered later.
        for _ in range(e_len):
            xi, yi = [int(i) for i in input().split()]
            g[xi].edges.add(yi)
            g[yi].edges.add(xi)
        si = int(input())
        edge_dists = get_shortest_reach(g, si)
        print(" ".join(map(str, edge_dists)))
addto(Graphs__ShortestReachGraph, ds.graph, diff.hard, time.hour, date(2019,7,22), algo.dfs_bfs, source.hackerrank)
def Graphs__MultiBFS_NearestClone():
    # URL:https://www.hackerrank.com/challenges/find-the-nearest-clone/problem

    # !/bin/python3
    import math
    import os
    import random
    import re
    import sys

    from collections import deque
    from dataclasses import dataclass, field
    from collections import defaultdict

    @dataclass()
    class Node:
        edges: set = field(default_factory=set)  # list/set/dict need to be field_generated.
        desired: bool = False
        min_path: int = -1

    def findShortest(graph_nodes, graph_from, graph_to, ids, val):
        g = defaultdict(Node)
        to_explore = deque()
        for i in range(1, graph_nodes + 1):
            desired = ids[i - 1] == val
            g[i].desired = desired
            if desired:
                to_explore.append(i)
                g[i].min_path = 0
        for i in range(len(graph_from)):
            frm, to = graph_from[i], graph_to[i]
            g[frm].edges.add(to)
            g[to].edges.add(frm)

        while to_explore:
            node_id = to_explore.popleft()
            node = g[node_id]
            for n in node.edges:

                if g[n].desired:
                    return node.min_path + 1

                if g[n].min_path != -1:
                    return node.min_path + 1 + g[n].min_path

                g[n].min_path = node.min_path + 1
                g[n].edges.remove(node_id)
                to_explore.append(n)
        return -1

    if __name__ == '__main__':
        fptr = open(os.environ['OUTPUT_PATH'], 'w')
        graph_nodes, graph_edges = map(int, input().split())
        graph_from = [0] * graph_edges
        graph_to = [0] * graph_edges
        for i in range(graph_edges):
            graph_from[i], graph_to[i] = map(int, input().split())
        ids = list(map(int, input().rstrip().split()))
        val = int(input())
        ans = findShortest(graph_nodes, graph_from, graph_to, ids, val)
        fptr.write(str(ans) + '\n')
        fptr.close()
addto(Graphs__MultiBFS_NearestClone, ds.graph, diff.med, date(2019,7,22), algo.bfs, time.hour_2, source.hackerrank, tag.optimization)

def Graphs__KruskalsMST():
    # !/bin/python3
    # URL: https://www.hackerrank.com/challenges/kruskalmstrsub/problem#!
    # efd709addc104599b65e9fd3204f3a42  Notability / Cheat sheet.
    # Passes all tests.
    # Sample 5 has 1000 nodes and 10,000 edges
    # Refs:
    # https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
    # https://www.geeksforgeeks.org/union-find/
    # https://www.geeksforgeeks.org/union-find-algorithm-set-2-union-by-rank/

    from dataclasses import dataclass
    import sys

    path_compression_optimization = True  # for 1000 nodes, 10,000 edges,  improves 350ms -> 72ms. # Also keeps set-tree depth very low.
    union_by_rank_optimization = True
    if path_compression_optimization or union_by_rank_optimization:
        # Note, thanks to either optimization, set-tree's depth is kept very low.
        # As a result, algo works with 1000 nodes & 10,000 edges with recursion limit of only 14!
        # >>> Thus, a recursive solution is acceptable.
        sys.setrecursionlimit(14)


    @dataclass()
    class Edge:
        src: int  # node id
        dst: int  # node id
        weight: int


    @dataclass()
    class Node:
        id: int  # node id also used as 'set id' that nodes belong to.
        parent: int = -1
        set_sum: int = 1  # Used by Union by Rank optimization.


    def find_set_of(node_id: int, nodes):
        node = nodes[node_id]
        if node.parent == -1:
            return node_id
        parent_id = find_set_of(node.parent, nodes)
        if path_compression_optimization:
            node.parent = parent_id  # Path compression optimization.
        return parent_id


    def kruskals(node_count, edges):
        edges.sort(key=lambda e: (e.weight, e.src + e.dst + e.weight))  # Compare Tuples: if same weights, use src+dst+weight to break tie.
        nodes = [Node(i) for i in range(0, node_count + 1)]
        nodes[0] = None  # node 0 used as dummy so that node_id's match array index.
        mst_edge_weights = []
        for edge in edges:
            src_set_id = find_set_of(edge.src, nodes)
            dst_set_id = find_set_of(edge.dst, nodes)
            if src_set_id == dst_set_id:  # Cycle found.
                continue
            else:
                if not union_by_rank_optimization:
                    # Union:
                    nodes[src_set_id].parent = dst_set_id
                else:
                    # Union by Rank Optimization: (Verified to work).
                    # Does fairly little to improve performance in practice. (01%). Get's dominated by path compression.
                    # However, if used alone, keeps tree depth very low.
                    if nodes[dst_set_id].set_sum > nodes[src_set_id].set_sum:
                        bigger_set_id = dst_set_id
                        smaller_set_id = src_set_id
                    else:
                        bigger_set_id = src_set_id
                        smaller_set_id = dst_set_id

                    nodes[smaller_set_id].parent = bigger_set_id  # Union.
                    nodes[bigger_set_id].set_sum += nodes[smaller_set_id].set_sum

                mst_edge_weights.append(edge.weight)
            if len(mst_edge_weights) == (node_count - 1):
                break
        return sum(mst_edge_weights)

    if __name__ == '__main__':
        # Note, I re-wrote parsing logic.
        node_count, edge_count = map(int, input().rstrip().split())  # Number of Node, Number of Edges
        edges = []
        for _ in range(edge_count):
            edges.append(Edge(*map(int, input().rstrip().split())))
        print(kruskals(node_count, edges))
addto(Graphs__KruskalsMST, ds.graph, diff.med, date(2019,7,21), algo.mst, algo.kruskal, time.hour_5, source.hackerrank)


def Graphs__Roads_And_libraries__BFS_graph():
    # URL: https://www.hackerrank.com/challenges/torque-and-development/problem
    # ...
    from collections import defaultdict
    class Node:
        def __init__(self):
            self.edges = set()
            self.visited = False

    # Complete the roadsAndLibraries function below.
    def roadsAndLibraries(n, cost_lib, cost_road, city_pairs):
        if cost_lib < cost_road:  # Cheaper to build libraries in every city
            return n * cost_lib
        # We shall refer to city=node, road=edge

        # Generate Graph.
        g = defaultdict(Node)
        for node_id1, node_id2 in city_pairs:
            g[node_id1].edges.add(node_id2)
            g[node_id2].edges.add(node_id1)

        # Count number of graphs & edges by BFS-ing.
        graph_count, edge_count = 0, 0
        node_ids_to_explore = set(range(1, n + 1))  # Caveat, some nodes are disconnected.
        while node_ids_to_explore:
            node = g[node_ids_to_explore.pop()]
            graph_count += 1
            node.visited = True
            explore_local = node.edges  # node ids
            while explore_local:
                node_id = explore_local.pop()
                node = g[node_id]
                if not node.visited:
                    node.visited = True
                    edge_count += 1
                    node_ids_to_explore.remove(node_id)
                    explore_local |= node.edges

        cost = (edge_count * cost_road) + (graph_count * cost_lib)
        return cost
    # ...
addto(Graphs__Roads_And_libraries__BFS_graph, ds.graph, diff.med, date(2019,7,18), algo.dfs_bfs, time.hour_2, source.hackerrank)

#######
####### Python 2 to Python3 migration:   (19/07/20)
####### Problems below were Python2 ported to Python3 via 2to3-3.7
#######

def Heaps__Running_median():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/find-the-running-median/problem
    # Initially implemented meh solution. Read the comments/editorial. Implemented better solution. Proud of self.

    #####
    ##### Solution 2
    #####
    ##### O(.5n log  .5n)  via two heaps.
    #####
    from heapq import heappush, heappop
    # Intuition:
    #          m   < medium is inbetween smaller/bigger.
    # [smaller] [bigger]
    #        ^    ^
    #     want    want
    # (max heap)   (min heap)
    # - We need to balance smaller/Bigger to be +- 1 of same size.

    # We don't keep median separate. Instead either one of smaller/bigger is larger:
    # [ss][b]  median: s[-1]
    # [s][bb]  median: b[0]
    # [ss][bb]  medina:  (s[1] + b[0]) / 2

    class maxh:
        def __init__(self):
            self.h = []
            self.push = lambda x: heappush(self.h, - x)
            self.pop = lambda: -heappop(self.h)
            self.peak = lambda: -self.h[0]
            self.len = lambda: len(self.h)


    class minh:
        def __init__(self):
            self.h = []
            self.push = lambda x: heappush(self.h, x)
            self.pop = lambda: heappop(self.h)
            self.peak = lambda: self.h[0]
            self.len = lambda: len(self.h)


    smaller = maxh()
    bigger = minh()

    last_median = None
    i_count = int(input()) - 1

    last_median = int(input())
    smaller.push(last_median)
    out = ["{:.1f}".format(last_median)]

    for _ in range(i_count):
        i = int(input())
        if i < last_median:
            smaller.push(i)
        else:
            bigger.push(i)

        # Balance our heaps.
        if smaller.len() - bigger.len() > 1:  # smaller has too many elements
            bigger.push(smaller.pop())
        if bigger.len() - smaller.len() > 1:  # bigger has too many elements
            smaller.push(bigger.pop())

        if smaller.len() > bigger.len():
            val = smaller.peak()
        elif smaller.len() < bigger.len():
            val = bigger.peak()
        else:
            v1, v2 = bigger.peak(), smaller.peak()
            val = (v1 + v2) / 2.0
        out.append("{:.1f}".format(val))
        last_median = val
    for i in out:
        print(i)

    #####
    ##### Solution 1
    #####
    ##### O(.5n log  .5n)  via two heaps.
    ##### Life lesson: I made a lot of mistakes with using methods for min/max heaps. Better implement classes for min/max heap.
    from heapq import heappush, heappop

    smaller_maxh = []
    slen = lambda: len(smaller_maxh)
    bigger_minh = []
    blen = lambda: len(bigger_minh)
    last_median = None

    i_count = int(input())

    # Base case.
    last_median = int(input())
    heappush(smaller_maxh, -last_median)
    out = []
    out.append("{:.1f}".format(last_median))
    i_count -= 1

    def heappush_max(h, val):
        heappush(h, -val)

    def heappop_max(h):
        return -heappop(h)

    def heap_peak_max(h):
        return -h[0]

    def heap_peak(h):
        return h[0]

    for _ in range(i_count):
        i = int(input())
        if i < last_median:
            heappush_max(smaller_maxh, i)
        else:
            heappush(bigger_minh, i)

        # Balance our heaps.
        if slen() - blen() > 1:  # smaller has too many elements
            val = heappop_max(smaller_maxh)
            heappush(bigger_minh, val)
        if blen() - slen() > 1:  # bigger has too many elements
            val = heappop(bigger_minh)
            heappush_max(smaller_maxh, val)

        if slen() > blen():
            val = heap_peak_max(smaller_maxh)
        elif slen() < blen():
            val = heap_peak(bigger_minh)
        else:
            v1, v2 = heap_peak(bigger_minh), heap_peak_max(smaller_maxh)
            val = (v1 + v2) / 2.0
        out.append("{:.1f}".format(val))
        last_median = val
    for i in out:
        print(i)

    ####
    #### Solution 0
    ####
    #### Brute force solution. (insert, sort, get median)
    #### O (n * nlogn)  # Bad runtime, but passes all the test cases.
    import bisect
    from math import ceil, floor
    def median(L):
        if len(L) % 2 == 1:
            med_i = ((len(L) + 1) / 2) - 1
            return L[int(med_i)]
        else:
            med_i = ((len(L) + 1.0) / 2.0) - 1
            left = int(floor(med_i))
            right = int(ceil(med_i))
            return (L[left] + L[right]) / 2.0

    a_count = int(input())
    a = []
    for _ in range(a_count):
        a_item = int(input())
        bisect.insort(a, a_item)
        print("{:.1f}".format(median(a)))
addto(Heaps__Running_median, ds.heaps, diff.hard, date(2019,7,14), time.hour_5, tag.optimization, tag.insight, source.hackerrank, tag.failed)

def Heaps__deleteHeapElement_Plus_heap_ops():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/qheap1/problem

    # Tricky part is deletion.
    # Current solution is O(n log n)
    # Could be imporved to O(log n) if I were to use a hashtable to keep track of indexes. todo_someday.

    # Manual implementation of heap functions.
    # Hacky implemenatin of deleting element from heap. (find index, bubble up and pop).
    # Passes all test cases.
    q = int(input())
    heap = []

    def leo_heappop(h):
        lastval = h.pop()
        if len(h) == 0:  # < Keep in mind last element in heap.
            return lastval
        retval = h[0]
        h[0] = lastval
        leo_heapifydown(h, 0)
        return retval

    def leo_heapifydown(h, i):  # Verified working.
        # swap with min child below.
        while True:
            li, ri = i * 2 + 1, i * 2 + 2
            # Complete Tree property: If it has right, must have a left.
            if ri > len(h) - 1:
                if li > len(h) - 1:
                    break  # 0: leaf.
                if h[li] < h[i]:  # 1: Only Left
                    h[li], h[i] = h[i], h[li]
                    i = li
                    continue
                else:
                    break  # In the right spot.
            # 2: Both children
            small_i = li if h[li] < h[ri] else ri
            if h[small_i] < h[i]:
                h[small_i], h[i] = h[i], h[small_i]
                i = small_i
            else:
                break  # In the right spot.
            # Case Zero: leaf
            # Case one: Only left child
            # Case Two: Both children

    # Leo: Implemented myself. Works, yay. There is no build-in heapdelete.
    # Alternative is swap with the last element, remove last and move replaced element into the right spot.
    def leo_heapdelete(h, val):
        ival = h.index(val)  # index of val
        h[ival] = - float("inf")
        while ival != 0:
            iparent = (ival - 1) >> 1
            h[iparent], h[ival] = h[ival], h[iparent]
            ival = iparent
        leo_heappop(h)

    def leo_heapifyup(h, i):
        while i > 0:  # if i=0,  ip= -1.
            ip = (i - 1) >> 1  # ip = index of parent
            if h[ip] > h[i]:
                h[ip], h[i] = h[i], h[ip]
                i = ip
            else:  # note, this also handles the case where i == 0.
                break

    def leo_heapqpush(h, val):
        h.append(val)
        leo_heapifyup(h, len(h) - 1)

    for _ in range(q):
        cmd = list(map(int, input().split()))
        if cmd[0] == 1:
            leo_heapqpush(heap, cmd[1])
        if cmd[0] == 2:
            leo_heapdelete(heap, cmd[1])
        if cmd[0] == 3:
            print(heap[0])
addto(Heaps__deleteHeapElement_Plus_heap_ops, ds.heaps, date(2019,7,14), diff.med, time.hour_2, tag.interview_material, tag.optimization, source.hackerrank)
def Heaps__Jessie_and_cookies__MinHeap():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/jesse-and-cookies/problem
    #from __future__ import print_function
    import os, sys

    # stdin:
    # 6 7
    # 1 2 3 9 10 12
    # exp: 2

    # O(n log n)
    # Min heap implementation.
    # Passes all test cases.
    from heapq import heapify, heappush, heappop
    def cookies(k, A):
        heapify(A)
        ops = 0
        while True:
            if A[0] >= k:
                return ops
            if len(A) < 2:
                return -1
            lesssweet = heappop(A)
            moresweet = heappop(A)
            new = lesssweet + 2 * moresweet
            heappush(A, new)
            ops += 1

    # For comparison, bad time complexity implementation:
    # # O(n * nlogn)
    # #  ^ because after insert [log n], might have to shift many elements [n].
    # # Works on Lower test cases.
    # # Times out on large test cases 20,21,22,23.  E.g for test 20, N=105823341
    # from bisect import insort
    # def cookies(k, A):
    #     A.sort(reverse=True)
    #     ops = 0
    #     while True:
    #         if A[-1] >= k:
    #             return ops
    #         if len(A)< 2:
    #             return -1
    #         lessweet = A.pop()
    #         moresweet = A.pop()
    #         new = lessweet + 2*moresweet
    #         reverse_insort(A, new)
    #         ops += 1

    # #O (n log n)
    # def reverse_insort(a, x, lo=0, hi=None):
    #     """Insert item x in list a, and keep it reverse-sorted assuming a
    #     is reverse-sorted.
    #     If x is already in a, insert it to the right of the rightmost x.
    #     Optional args lo (default 0) and hi (default len(a)) bound the
    #     slice of a to be searched.
    #     """
    #     if lo < 0:
    #         raise ValueError('lo must be non-negative')
    #     if hi is None:
    #         hi = len(a)
    #     while lo < hi:
    #         mid = (lo+hi)//2
    #         if x > a[mid]: hi = mid
    #         else: lo = mid+1
    #     a.insert(lo, x)

    if __name__ == '__main__':
        fptr = open(os.environ['OUTPUT_PATH'], 'w')
        nk = input().split()
        n = int(nk[0])
        k = int(nk[1])
        A = list(map(int, input().rstrip().split()))
        result = cookies(k, A)
        fptr.write(str(result) + '\n')
        fptr.close()
addto(Heaps__Jessie_and_cookies__MinHeap, ds.heaps, date(2019,7,14), diff.easy, time.min_30, tag.interview_material, tag.optimization, source.hackerrank)

def Hashmap__SortCharsByFrequency():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://leetcode.com/problems/sort-characters-by-frequency/
    # O(n)
    #   O(n) to count all items.
    #   the sort is constant because there are only 26 letters in the alphabet.
    # 15 min with some procrastination in between
    # Runtime: 32 ms, faster than 93.29% of Python online submissions for Sort Characters By Frequency.
    # Memory Usage: 14.5 MB, less than 42.59% of Python online submissions for Sort Characters By Frequency.
    from collections import defaultdict
    class Solution(object):
        def frequencySort(self, s):
            seen = defaultdict(int)
            for c in s: seen[c] += 1
            ltrs = list(seen.items())
            ltrs.sort(key=lambda x: x[1], reverse=True)
            return "".join([l[0] * l[1] for l in ltrs])
addto(Hashmap__SortCharsByFrequency, ds.hashtable, date(2019,7,13), diff.med, time.min_30, tag.interview_material, tag.strings, source.leetcode)

def Array__TwoSum():
    # Python2 ported to Python3 via 2to3-3.7
    # O(n) beacuse we use one-pass hashtable & look back to see if there is a matching value.
    # Runtime: 36 ms, faster than 84.65% of Python online submissions for Two Sum.
    # Memory Usage: 13.3 MB, less than 16.91% of Python online submissions for Two Sum.
    class Solution(object):
        def twoSum(self, arr, target):
            seen_vals = dict()  # seen_vals[val] = index
            for i in range(len(arr)):
                j_val = target - arr[i]
                if j_val in seen_vals:
                    return [seen_vals[j_val], i]
                else:
                    seen_vals[arr[i]] = i
addto(Array__TwoSum, ds.arrays, ds.hashtable, diff.easy, date(2019,7,13), time.min_15, tag.interview_material, tag.optimization)

def Heaps__KclosestPoints():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://leetcode.com/problems/k-closest-points-to-origin/submissions/
    # O (n log k)
    # Implementation that uses a tuple inside a heap
    # Runtime: 588 ms, faster than 74.10% of Python online submissions for K Closest Points to Origin.
    # Memory Usage: 18.6 MB, less than 11.82% of Python online submissions for K Closest Points to Origin.
    from heapq import heappush, heapreplace
    class Solution(object):
        def kClosest(self, coords, k):
            h = []
            for c in coords:
                x, y = c
                d = -(
                            x ** 2 + y ** 2)  # sqrt(...) not needed since all points will be scaled. # Negation is to implement max-heap with heapq (heapq is minheap)
                if len(h) < k:
                    heappush(h, (d, x, y))
                elif h[0][0] < d:
                    heapreplace(h, (d, x, y))
            return [[p[1], p[2]] for p in h]

    # O( n log k)
    ### Implementation that uses a custom class.
    # Runtime: 756 ms, faster than 10.17% of Python online submissions for K Closest Points to Origin.
    # Memory Usage: 19.4 MB, less than 5.10% of Python online submissions for K Closest Points to Origin.
    from math import sqrt
    from heapq import heappush, heapreplace

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.dist = sqrt(x ** 2 + y ** 2)

        def __lt__(self, o):
            return self.dist > o.dist  # for max heap implementation of heapq, we invert less than op.

    class Solution(object):
        def kClosest(self, coords, k):
            h = []
            for c in coords:
                p = Point(c[0], c[1])
                if len(h) < k:
                    heappush(h, p)
                elif h[0].dist > p.dist:
                    heapreplace(h, p)
            return [[p.x, p.y] for p in h]


    # O (n log n).  Worse Time complexity but interestingly enough runs faster than heap approach.
    # Runtime: 540 ms, faster than 99.07% of Python online submissions for K Closest Points to Origin.
    # Memory Usage: 17.4 MB, less than 79.62% of Python online submissions for K Closest Points to Origin.
    from heapq import heappush, heapreplace
    class Solution(object):
        def kClosest(self, coords, k):
            return sorted(coords, key=lambda x: x[0] ** 2 + x[1] ** 2)[:k]

    # Testing code:
    print(Solution().kClosest([[1, 3], [-2, 2]], 1), "expecting:,", [[-2, 2]])
    print(Solution().kClosest([[1, 3], [-2, 2], [1, 1], [0, 0]], 2), "expecting:,", [[0, 0], [1, 1]])
addto(Heaps__KclosestPoints, ds.heaps, diff.med, date(2019,7,13), time.hour_2, source.leetcode, tag.interview_material, tag.insight, tag.optimization)


def Arrays__MergeTwoSortedArrays():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.reddit.com/r/Python/comments/9sqvjn/a_python_interview_question/
    # Guy failed an intreview.
    # I'm thinking "Dumbass", but to be fair I struggled with this in University during my algorithm class as well.
    # Merge Two sorted arrays(/lists) I and J
    # O(I + J) via manual merge.
    I = [1, 3, 6, 10]
    J = [2, 3, 5, 9, 11]
    i, j = 0, 0
    M = []
    while True:
        if I[i] < J[j]:
            M.append(I[i])
            i += 1
        else:
            M.append(J[j])
            j += 1
        if i == len(I):
            while j != len(J):
                M.append(J[j])
                j += 1
            break
        if j == len(J):
            while i != len(I):
                M.append(I[i])
                i += 1
            break
    print(M)
    I.extend(J)
    I.sort()
    print(I)
    print(M == I)
addto(Arrays__MergeTwoSortedArrays, ds.arrays, diff.easy, date(2019,7,13), source.reddit, tag.interview_material)

def Heaps__LastStoneWeight():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://leetcode.com/problems/last-stone-weight/submissions/
    # Barley looked at cheat sheet.
    # Runtime: 16 ms, faster than 84.37% of Python online submissions for Last Stone Weight.
    # Memory Usage: 11.8 MB, less than 100.00% of Python online submissions for Last Stone Weight.
    from heapq import heappush, heappop
    class Solution(object):
        def lastStoneWeight(self, stones):
            """
            :type stones: List[int]
            :rtype: int
            """
            if len(stones) == 0:
                return 0
            h = []
            for i in stones:
                heappush(h, -i)
            while len(h) > 1:
                y = -heappop(h)
                x = -heappop(h)
                if x != y:
                    z = y - x
                    heappush(h, -z)
            if len(h) == 0:
                return 0
            else:
                return -h[0]
addto(Heaps__LastStoneWeight, ds.heaps, diff.easy, date(2019,7,10), source.leetcode)



def Heaps__KthLargest_in_Stream():
    # Python2 ported to Python3 via 2to3-3.7
    # Notability: 65fdd808fcad43b2b2726062aeaa108d
    # Runtime: 100 ms, faster than 90.53% of Python online submissions for Kth Largest Element in a Stream.
    # Memory Usage: 15.9 MB, less than 42.32% of Python online submissions for Kth Largest Element in a Stream.
    # URL: https://leetcode.com/problems/kth-largest-element-in-a-stream/submissions/
    from heapq import heapreplace, heappush, heapify
    class KthLargest(object):
        def __init__(self, k, nums):
            self.heap = []
            self.k = k
            for i in nums:
                self.add(i)

        def add(self, val):
            if len(self.heap) < self.k:
                heappush(self.heap, val)
            elif val > self.heap[0]:
                heapreplace(self.heap, val)
            return self.heap[0]


    def leet_test(funcs, args, expected, test_name=""):  # Leet code test runner.
        obj, overall = eval(funcs[0] + "(" + str(args[0])[1:-1] + ")"), False
        print(test_name + " ", obj.__class__.__name__)
        for i in range(1, len(funcs)):
            arg, exp = str(args[i])[1:-1], expected[i]
            cmd = "{}.{}({})".format("obj", funcs[i], str(arg))
            res = eval(cmd)
            if res != exp: overall = False
            print("{} arg:{:<5} res:{:<5} exp:{:<5}".format(res == exp, arg, res, exp))
        print("::", overall, "\n")

    null = None

    if __name__ == '__main__':
        leet_test(
            ["KthLargest", "add", "add", "add", "add", "add"],
            [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]],
            [None, 4, 5, 5, 8, 8],
            "Provided test"
        )

        leet_test(
            ["KthLargest", "add", "add", "add", "add", "add"],
            [[1, []], [-3], [-2], [-4], [0], [4]],
            [null, -3, -2, -2, 0, 4],
            "Initial Empty Heap"
        )
addto(Heaps__KthLargest_in_Stream, ds.heaps, date(2019,7,13), diff.med, time.hour_2, source.leetcode, tag.amazon, tag.insight, tag.interview_material)

def Trees__BST_Max_path_sum():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://leetcode.com/problems/binary-tree-maximum-path-sum/submissions/

    # Key Lessons learned:
    # - ** DFS can be implemented quite easily via iterative approach and closely mimics recusive behaviour
    # - Keep it simple. (2 if's can be better than fancy for-looping over pointers).
    #   Short/simple code is often easier to read/maintain.
    # - Recursion can be faster than iterative approach w/ good memory utilization in some cases

    # Iterative Solution.
    # Leo: Chosen best solution for very low memory footprint. Reasonably fast & very low memory consumption.
    # Runtime: 96 ms, faster than 26.77% of Python online submissions for Binary Tree Maximum Path Sum.
    # Memory Usage: 23.8 MB, less than 99.10% of Python online submissions for Binary Tree Maximum Path Sum.
    class Solution(object):
        def maxPathSum(self, root):
            to_visit = [root]
            mx = -float("inf")
            while to_visit:
                node = to_visit[-1]
                lmax, rmax = 0, 0
                if node.left is not None:
                    if not hasattr(node.left, "max_sum"):
                        to_visit.append(node.left)
                        continue
                    else:
                        lmax = node.left.max_sum
                if node.right is not None:
                    if not hasattr(node.right, "max_sum"):
                        to_visit.append(node.right)
                        continue
                    else:
                        rmax = node.right.max_sum
                to_visit.pop()
                bridged_sum = lmax + rmax + node.val
                node.max_sum = max(lmax, rmax, 0) + node.val
                mx = max(mx, bridged_sum, node.max_sum)
            return mx


    # Alternative:

    # Recursive Solution.
    # Leo: Interestingly enough, runs a little bit faster than the iterative one.
    # Runtime: 80 ms, faster than 84.94% of Python online submissions for Binary Tree Maximum Path Sum.
    # Memory Usage: 24.8 MB, less than 41.64% of Python online submissions for Binary Tree Maximum Path Sum.
    class Solution_recursive(object):
        def maxPathSum(self, root):
            varz = dict()
            varz["max"] = -float("inf")  # Nested functions can read but cannot rebind variables.

            def max_path(node):
                varz["max"] = max(varz["max"], node.val)
                lmax = max_path(node.left) if node.left else 0
                rmax = max_path(node.right) if node.right else 0
                bridged_max = lmax + rmax + node.val
                node_max = max(lmax, rmax, 0) + node.val
                varz["max"] = max(varz["max"], bridged_max, node_max)
                return node_max

            max_path(root)
            return varz["max"]

    ####
    # Testing Code
    ###
    class TreeNode(object):
        def __init__(self, x):
            self.val = x
            self.left = None
            self.right = None

        def __repr__(self):
            return str(self.val)

    tn = TreeNode
    # T_a # Answ 4
    root = TreeNode(1)
    root.left = TreeNode(-2)
    root.right = TreeNode(3)
    print(4, Solution().maxPathSum(root))

    # T_b? -> expt 42
    r = tn(-10)
    r.left, r.right = tn(9), tn(20)
    r.right.left, r.right.right = tn(15), tn(7)

    print(42, Solution().maxPathSum(r))

    r = tn(2)
    r.left = tn(-1)
    print(2, Solution().maxPathSum(r))


    ###
    # Alternative ineffective implementations:
    ###
    # Iterative solution with nested function.
    # Poor speed but Oddly enough uses the least memory.
    # # Runtime: 132 ms, faster than 5.31% of Python online submissions for Binary Tree Maximum Path Sum.
    # # Memory Usage: 23.6 MB, less than 100.00% of Python online submissions for Binary Tree Maximum Path Sum.
    class Solution_DFS_nested_Func(object):
        # Notes:
        # - Problem: Have nested loop (while->for) & nested loop needs to break out of outer loop.
        #   Workaround: Define nested function and inside nested function return.
        # - Problem: Nested functions can only read but not rebind variables.
        #   Workaround: Use a dictionary key/value 'varz' (btw, vars conflicts with build-in, so use varz)
        # - I implemented a nested for-loop to practice with trees that have N-Number of children.
        #   For basic BST trees with only left/right nodes, it's easier to have 2 if statements.
        def maxPathSum(self, root):
            to_visit = [root]
            varz = dict()
            varz["mx"] = -float("inf")

            def visit(node):
                child_max, child_sum = None, 0
                for child in [x for x in [node.left, node.right] if x is not None]:
                    if not hasattr(child, "max_sum"):  # max_sum attribute also means node vas visited.
                        to_visit.append(child)
                        return
                    else:
                        child_max = max(child_max, child.max_sum) if child_max is not None else child.max_sum
                        child_sum += child.max_sum
                to_visit.pop()
                bridged_sum = child_sum + node.val
                node.max_sum = max(node.val, node.val + child_max) if child_max is not None else node.val
                varz["mx"] = max(varz["mx"], bridged_sum, node.max_sum)

            while to_visit:
                visit(to_visit[-1])
            return varz["mx"]


    # TODO deduplicate and put into my notes.
    # TODO Add note about nested for loops somewhere.

    # Continuing an outer loop via try/catch custom exception.
    # Runtime: 140 ms, faster than 5.31% of Python online submissions for Binary Tree Maximum Path Sum.
    # Memory Usage: 23.5 MB, less than 100.00% of Python online submissions for Binary Tree Maximum Path Sum.
    class OuterContinue(Exception):
        pass


    class Solution_DFS(object):
        def maxPathSum(self, root):
            to_visit = [root]
            mx = -float("inf")
            while to_visit:
                node = to_visit[-1]
                child_max, child_sum = None, 0
                try:
                    for child in [x for x in [node.left, node.right] if x is not None]:
                        if not hasattr(child, "max_sum"):  # max_sum attribute also means node vas visited.
                            to_visit.append(child)
                            raise OuterContinue
                        else:
                            child_max = max(child_max, child.max_sum) if child_max is not None else child.max_sum
                            child_sum += child.max_sum
                except OuterContinue:
                    continue
                to_visit.pop()
                bridged_sum = child_sum + node.val
                node.max_sum = max(node.val, node.val + child_max) if child_max is not None else node.val
                mx = max(mx, bridged_sum, node.max_sum)
            return mx


    # Use.
    # Has two inner loops. Use a variable to break out of outer loop.
    # Runtime: 136 ms, faster than 5.31% of Python online submissions for Binary Tree Maximum Path Sum.
    # Memory Usage: 23.5 MB, less than 100.00% of Python online submissions for Binary Tree Maximum Path Sum.
    class Solution_DFS(object):
        def maxPathSum(self, root):
            to_visit = [root]
            mx = -float("inf")
            while to_visit:
                node = to_visit[-1]
                child_max, child_sum = None, 0
                outer_break = False
                for child in [x for x in [node.left, node.right] if x is not None]:
                    if not hasattr(child, "max_sum"):  # max_sum attribute also means node vas visited.
                        to_visit.append(child)
                        outer_break = True
                        break
                    else:
                        child_max = max(child_max, child.max_sum) if child_max is not None else child.max_sum
                        child_sum += child.max_sum
                if outer_break:
                    continue
                to_visit.pop()
                bridged_sum = child_sum + node.val
                node.max_sum = max(node.val, node.val + child_max) if child_max is not None else node.val
                mx = max(mx, bridged_sum, node.max_sum)
            return mx


    # My first try :'-).
    # BFS from leafs up.
    # Works and passes all test cases.
    # Poor performance. Too much code, too much overhead.
    # Runtime: 156 ms, faster than 5.31% of Python online submissions for Binary Tree Maximum Path Sum.
    # Memory Usage: 49 MB, less than 5.07% of Python online submissions for Binary Tree Maximum Path Sum.
    class Solution_BFS(object):
        def maxPathSum(self, root):
            max_so_far = - float("inf")

            def ccount(node):
                count = 0
                if node.left is not None:
                    count += 1
                if node.right is not None:
                    count += 1
                return count

            leaf_nodes = []
            to_process = [root]
            root.parent = None
            root.cpath = []
            while to_process:
                node = to_process.pop()
                node.cpath = []
                node.sum = node.val
                max_so_far = max(node.sum,
                                 max_so_far)  # Maybe individual node has longest path if all other nodes are negative.

                if node.left is None and node.right is None:
                    leaf_nodes.append(node)
                    continue
                else:
                    for child in [node.left, node.right]:
                        if child:
                            child.parent = node
                            to_process.append(child)

            to_process = leaf_nodes
            while to_process:
                node = to_process.pop()
                parent = node.parent
                if parent is not None:  # for root.
                    parent.cpath.append(node.sum)
                    if ccount(parent) == len(parent.cpath):  # All children have appended their best paths.
                        bridge_path_sum = parent.sum + sum(parent.cpath)
                        max_so_far = max(bridge_path_sum, max_so_far)

                        # Find biggest sum-path for either parent + (left or right)
                        parent.sum = parent.sum + max(max(parent.cpath), 0)  # Children can be negative.
                        max_so_far = max(max_so_far, parent.sum)
                        to_process.append(parent)
            return max_so_far
addto(Trees__BST_Max_path_sum, ds.tree, date(2019,7,9), diff.hard, time.hour_5, source.leetcode, tag.amazon, tag.recursion, tag.insight, tag.interview_material, algo.dfs_bfs)


def Array__shuffle():
    # Python2 ported to Python3 via 2to3-3.7
    # Bit of a meh question, but apeared on an Amazon interview.
    # URL: https://leetcode.com/problems/shuffle-an-array/submissions/
    from random import shuffle
    class Solution(object):
        def __init__(self, nums):
            """
            :type nums: List[int]
            """
            self.nums = nums

        def reset(self):
            """
            Resets the array to its original configuration and return it.
            :rtype: List[int]
            """
            return self.nums

        def shuffle(self):
            """
            Returns a random shuffling of the array.
            :rtype: List[int]
            """
            random_list = list(self.nums)
            shuffle(random_list)
            # Alternative would be to loop over every item and randomly shuffle it:
            #         for i in xrange(len(self.now) - 1):
            #             idx = random.randint(i,len(self.now) - 1)
            #             self.now[i],self.now[idx] = self.now[idx],self.now[i]
            return random_list
addto(Array__shuffle, ds.arrays, date(2019,7,7), diff.easy, time.min_15, source.leetcode, tag.amazon)


def Trees__CheckBST2():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/is-binary-search-tree/problem
    # Duplicate of a problem I've already solved: Trees__CheckBST  Good pactice thou.

    # Interesting aspect is that brute force solution would be O(n^2) (check all children for every node).
    # Optimal solution (I believe) is O(n) thou.
    # Not intiutive.
    # 6b25dc8214dc4abfb9eac0a835d77e54 << Notability Diagram/rough work.
    def check_binary_search_tree_(root):
        to_proccess = [(root, None, None)]
        while to_proccess:
            node, last_up_left, last_up_right = to_proccess.pop()
            # (!) >= checks for duplicates. '>' alone wouldn't.
            if (last_up_right is not None and last_up_right >= node.data):
                return False
            if (last_up_left is not None and last_up_left <= node.data):
                return False
            if node.left:
                to_proccess.append((node.left, node.data, last_up_right))
            if node.right:
                to_proccess.append((node.right, last_up_left, node.data))
        return True
addto(Trees__CheckBST2, ds.tree, ds.stacks, date(2019,7,7), diff.med, time.min_30, tag.interview_material, tag.optimization, tag.insight, source.hackerrank)

def Trees__bst_insert():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/binary-search-tree-insertion/problem
    class Node:
        def __init__(self, value):
            self.value = value
            self.left = None
            self.right = None

    def insert(self, val):
        if not self.root:
            self.root = Node(val)
            return self.root
        curr = self.root
        while True:
            if curr.info < val:
                if curr.right:
                    curr = curr.right
                else:
                    curr.right = Node(val)
                    break
            else:
                if curr.left:
                    curr = curr.left
                else:
                    curr.left = Node(val)
                    break
addto(Trees__bst_insert, ds.tree, date(2019,7,7), diff.easy, time.min_15, tag.interview_material, source.hackerrank)


def Trees__Level_order_traversal():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/tree-level-order-traversal/problem
    from collections import deque
    def levelOrder(root):
        curr_lvl, next_lvl, out = deque([root]), deque(), []
        while curr_lvl or next_lvl:
            if curr_lvl:
                node = curr_lvl.popleft()
                out.append(node.info)
                for child in [node.left, node.right]:
                    if child:
                        next_lvl.append(child)
            else:
                curr_lvl = next_lvl
                next_lvl = deque()
        print(" ".join(map(str, out)))
addto(Trees__Level_order_traversal, ds.tree, ds.queue, date(2019,7,7), diff.easy, time.min_30, tag.interview_material, source.hackerrank)

def Trees__top_view():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/tree-top-view/problem
    # Passes all test cases.
    # O(n) Traversal & conversions.
    # Logic: Level order traversal with 2 stacks.
    #        Use a hashtable to track items seen in positions,
    #            where positions is 0 for root, -1 for left child, +1 for right child.
    #        Afterwarrds convert hashtable's keys to list, sort and print it's values.
    from collections import deque
    def topView(root):
        curr_lvl = deque([(root, 0)])  # (node, position)
        next_lvl = deque()
        seen = dict()  # pos/value
        while curr_lvl or next_lvl:
            if curr_lvl:
                node, pos = curr_lvl.popleft()   # Careful with traversal order (order in which we pop/append items).
                if pos not in seen:
                    seen[pos] = node.info
                if node.left:
                    next_lvl.append((node.left, pos - 1))
                if node.right:
                    next_lvl.append((node.right, pos + 1))
            else:
                curr_lvl = next_lvl
                next_lvl = deque()
        positions = list(seen.keys())
        positions.sort()
        top_values = [seen[i] for i in positions]
        print(" ".join(map(str, top_values)))
addto(Trees__top_view, ds.tree, ds.hashtable, date(2019,7,7), diff.easy, time.min_30, tag.interview_material, tag.insight, source.hackerrank)

def Warmup__ArraySum():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/simple-array-sum/problem
    # For personal entertainment :-D...
    input()
    print(sum(map(int, input().split())))
addto(Warmup__ArraySum, ds.arrays, diff.easy, date(2019,7,7), time.min_15, source.hackerrank, tag.interview_material)


def Search__PairDiff():
    # Python2 ported to Python3 via 2to3-3.7
    # Find pair (sum/diff)  O(n)
    # - Place arr into hashmap.
    # - Look for counter part.
    # - Brute force: O(n chose k)
    # - Efficient: O(n)  (ex)
    # URL: https://www.hackerrank.com/challenges/pairs/problem
    _, k = list(map(int, input().split()))
    arr = list(map(int, input().split()))
    def pairs(k, arr):  # O(n)
        items, match = set(arr), 0
        for i in arr:
            # k = i - j    # Fancy algebra.
            # j = -k + i
            j = i - k
            if j in items:
                match += 1
        return match
    print(pairs(k, arr))
addto(Search__PairDiff, algo.search, ds.hashtable, date(2019,7,6), diff.med, time.min_15, source.hackerrank, tag.interview_material)

def SegmentTRees_rmq():
    # Python2 ported to Python3 via 2to3-3.7
    # Spent a lot of time implementing segment trees.  See:
    import TreesSegment
    import TreesSegmentTests
    # Used them for a hard problem: https://www.hackerrank.com/challenges/array-pairs/problem
    # But I couldn't figure out how to optimize the combinatorial part.
addto(SegmentTRees_rmq, ds.tree, diff.hard, time.days, source.hackerrank, tag.failed)

def Trees_postOrder_traversal():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/tree-postorder-traversal/problem
    # Very elegant 2 stack solution. 1 stack is possible but tricky.
    # Src: https://www.geeksforgeeks.org/iterative-postorder-traversal/
    # Basically pop from stack 1 onto stack 2. Append left/right child to stack 1.
    # reverse stack2 & print.
    def postOrder(root):
        stack1 = [root]
        stack2 = []
        while stack1:
            node = stack1.pop()
            stack2.append(node)
            for child in [node.left, node.right]:
                if child:
                    stack1.append(child)
        stack2.reverse()
        print(" ".join([str(node.info) for node in stack2]))

    # Marking a node as visisted/not visited.
    # Not ideal because we have to modify or keep state on node.
    def postOrder_use_node_property(root):
        # inorder: left, root, right
        # preorder: root, left, right
        # postorder: left, right, root.   1, 3, 2, 5, 7 ,6 ,4  (think leafs up.)
        #      4
        #    2    6
        # 1    3  5  7
        out = []
        curr = root
        later = []
        hasVisited = set()
        while True:
            if curr.left and curr.left not in hasVisited:
                later.append(curr)
                curr = curr.left
                continue
            if curr.right and curr.right not in hasVisited:
                later.append(curr)
                curr = curr.right
                continue
            out.append(curr.info)
            hasVisited.add(curr)
            if later:
                curr = later.pop()
            else:
                break
        print(" ".join(map(str, out)))

    def postOrder_recursive(root):
        def _postOrder(node):
            out = []
            for child in [node.left, node.right]:
                if child:
                    out.extend(_postOrder(child))
            out.append(node.info)
            return out

        print(" ".join(map(str, _postOrder(root))))
addto(Trees_postOrder_traversal, ds.tree, ds.stacks, date(2019,7,6), diff.easy, time.min_30, source.hackerrank, tag.recursion, tag.interview_material)


def Trees_preOrder_traversal():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/tree-preorder-traversal/problem
    """
    Node is defined as
    self.left (the left child of the node)
    self.right (the right child of the node)
    self.info (the value of the node)
    """
    def preOrder(root):
        # inorder: left root right
        # preorder: root, left, right  5,3,1,4,7,6,8
        # postorder: left,right, root
        #    5
        #  3    7
        # 1  4  6  8
        out = []
        to_proccess = [root]
        while to_proccess:
            node = to_proccess.pop()
            out.append(node.info)
            for child in [node.right, node.left]:
                if child:
                    to_proccess.append(child)
        print(" ".join(map(str, out)))

    def preOrder_recursive(root):
        def _preOrder(node):
            out = []
            out.append(node.info)
            for child in [node.left, node.right]:
                if child:
                    out.extend(_preOrder(child))
            return out
        print(" ".join(map(str, _preOrder(root))))
addto(Trees_preOrder_traversal, ds.tree, ds.stacks, date(2019,7,5), diff.easy, time.min_30, source.hackerrank, tag.recursion, tag.interview_material)

def Trees_inOrder_traversal():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/tree-inorder-traversal/problem
    # Best Aproach: Iterative "Go to target & push things to followup-on onto a stack."
    def inOrder(root):
        # left, root, right.
        out = []
        process_later = []
        curr = root
        while curr or process_later:
            if curr:
                process_later.append(curr)
                curr = curr.left
            elif process_later:
                curr = process_later.pop()
                out.append(curr.info)
                curr = curr.right
        print(" ".join(map(str, out)))

    # I experimented "Peak before you go aparoach".
    # However, this requires annotating nodes that we visit in some way.
    # This is not particularly very clean.
    def inOrder_iterative_peek(root):
        out = []
        queue = [root]
        while queue:
            node = queue[-1]  # peak.
            if node.left and not hasattr(node.left, "visited"):
                queue.append(node.left)
                continue
            if not hasattr(node, "visited"):
                out.append(node.info)
                node.visited = True
            if node.right and not hasattr(node.right, "visited"):
                queue.append(node.right)
                continue
            queue.pop()
        print(" ".join(map(str, out)))

    def inOrder_recursive(root):
        # Write your code here
        def _inOrder(root):
            out = []
            if root.left:
                out.extend(_inOrder(root.left))
            out.append(root.info)
            if root.right:
                out.extend(_inOrder(root.right))
            return out

        print(" ".join(map(str, _inOrder(root))))
addto(Trees_inOrder_traversal, ds.tree, ds.stacks, date(2019,7,4), diff.easy, time.min_30, source.hackerrank, tag.recursion, tag.interview_material)


def Trees__Graphs__BalancedForest():
    # Python2 ported to Python3 via 2to3-3.7
    # URL: https://www.hackerrank.com/challenges/balanced-forest/problem
    # *Very hard* problem. Took me whole day and still didn't quite work.

    # Well, my solution works for 3 test cases. 2 produce wrong answer and 3 time out. I tired my best, but need to move on.
    # Take a graph, convert it to a Tree.

    # The complex part here is that you're actually given a Graph and you need to do some clever logic to break it into
    # a tree and do some pre-computation to improve operations later.

    # Potential Improvements:
    # - Figure out why some test cases are failing.
    # - Implement some improvements:
    #   - LCA can be computed in constant time via RMQ mod:
    #       https://www.topcoder.com/community/competitive-programming/tutorials/range-minimum-query-and-lowest-common-ancestor/
    #       See also comment section.
    # - My Logic on finding subtrees is very poor. Just tries every combo. Look through disscussions to find smoe tricks!
    # - The iterative approach makes it hard to read the code. Consider instead increasing recursion limit
    #   via sys.setrecursionlimit(6000).
    from itertools import combinations
    class Node:
        def __init__(self, value, nodeid):
            self.value = value
            self.nodeid = nodeid
            self.children = dict()  # ptr to dec[node] = tree_sum  # todo why not use set() here? Too late to code well.
            self.edges = set()  # Used during Graph -> Tree conversion.
            self.parent = None
            self.sum = None  # Accumulated sum from all children + self.value. To be computed.
            self.level = None

        def __repr__(self):
            if not self.parent:
                type = "root"
            elif self.parent and len(self.children) > 0:
                type = "node"
            elif self.parent and len(self.children) == 0:
                type = "leaf"
            else:
                type = "????"
            return "({} i{} v{} s{} l{})".format(type, self.nodeid, self.value, self.sum, self.level)


    def balancedForest(nodes, edges):  # nodes = [1,2,3] ...   edges_raw = [[1,2], [2,3] .. ]
        # - Generate list of actual nodes classes rather than just values.
        # - and decrement edge indexes to make indexes start at 0 (to make it easier to reference nodes in python array).
        nodes = [Node(node_val, i) for i, node_val in enumerate(nodes)]  # [Node(1), Node(2) ...]
        edges = [[edge[0] - 1, edge[1] - 1] for edge in edges]  # [[0,1], [1,2] ...

        root = make_tree(nodes, edges)  # O(n)
        compute_node_sums(nodes)  # O(n)
        compute_levels(nodes[0])  # O(n)
        return compute_balanced_tree(nodes, edges)  # O((n chose 2) * log n)

    def make_tree(nodes, edges):
        # Generate a graph
        for edge in edges:
            n1 = nodes[edge[0]]
            n2 = nodes[edge[1]]
            n1.edges.add(n2)
            n2.edges.add(n1)

        # Pick a node to be the root & convert Graph into a Tree.
        # TODO_someday - good root candidate is somewhere in the middle of the graph.
        root = nodes[0]
        queue = [root]
        while queue:
            node = queue.pop()
            for child in node.edges:
                child.parent = node
                child.edges.remove(node)
                node.children[child.nodeid] = child
                queue.append(child)

        # todo - document why we add dummy node.
        dummy = Node(0, len(nodes))
        dummy.sum = 0
        dummy.level = 1
        dummy.parent = root
        nodes.append(dummy)
        root.children[dummy.nodeid] = dummy
        return root

        # for i, edge in enumerate(edges):
        #
        #     def is_bare_node(node):
        #         return node.parent is None and len(node.children) == 0
        #     # If two nodes are completely unconnected, we just link them per given order. (E.g root)
        #     n1 = nodes[edge[0]]
        #     n2 = nodes[edge[1]]
        #     if i == 0 or (is_bare_node(n1) and is_bare_node(n2)):
        #         parent = nodes[edge[0]]  # type: Node
        #         child = nodes[edge[1]]   # type: Node
        #     else:
        #         # n2 Probably parent if itself has a parent or children.
        #         if n2.parent or len(n2.children) > 0:
        #             parent, child = n2, n1
        #         else:
        #             parent, child = n1, n2
        #
        #     assert isinstance(parent, Node)
        #     assert isinstance(child,  Node)
        #     child.parent = parent
        #     if not parent.children:
        #         parent.children = dict()
        #     parent.children[child.nodeid] = child
        # return nodes[0]

    def compute_node_sums(nodes):
        """For every node, compute the sum of it + it's children. Iteratively"""
        for node in nodes:
            node.children_summed = 0  # Dynamically add a meta field to Node to improve runtime when computing sums.

        leaf_nodes = []
        for node in nodes:
            if len(node.children) == 0:
                leaf_nodes.append(node)
        to_process = leaf_nodes
        while to_process:
            node = to_process.pop()
            # if leaf_node or all child notes computed their sum.
            if len(node.children) == 0 or len(node.children) == node.children_summed:
                node.sum = node.value
                if len(node.children) > 0:
                    node.sum = node.sum + sum([child.sum for child in list(node.children.values())])
                if node.parent:
                    node.parent.children_summed += 1
                    if len(
                            node.parent.children) == node.parent.children_summed:  # all children have computed their sums
                        to_process.append(node.parent)

        for node in nodes:
            del node.children_summed

    def compute_levels(root):
        root.level = 0
        queue = [root]
        while queue:
            node = queue.pop()
            for child in list(node.children.values()):
                child.level = node.level + 1
                queue.append(child)

    def compute_balanced_tree(nodes, edges):
        # Python2 ported to Python3 via 2to3-3.7
        def get_path_to_root(node):
            nodes_on_path = set()  # We don't need a linear path. Just knowing which nodes are in the path. -> O(n) instead of (n^2)
            curr = node  # type: Node
            while curr:
                nodes_on_path.add(curr)
                curr = curr.parent
            return nodes_on_path

        solution = -1
        for e1, e2 in combinations(edges, 2):
            # Chop 2 edges. Get 3 trees:
            c1_upper, c1_root = nodes[e1[0]], nodes[e1[1]]  # T1
            c2_upper, c2_root = nodes[e2[0]], nodes[e2[1]]  # T2
            root = nodes[0]  # remaining root.              # T3

            # There can be 2 cases.
            # 1) c1_upper and c2_upper have a LCA  (i.e, are in separate sub-branches)
            # 2) c(1|2)_upper in in the subtree of c(1|2)_root
            c_higher_root, c_lower_root = (c1_root, c2_root) if c1_root.level > c2_root.level else (c2_root, c1_root)

            if c_higher_root.level == c_lower_root.level:  # Minor optimization to avoid O(lg n)
                case = 1  # LCA
            else:
                longer_path_to_root = get_path_to_root(c_lower_root)
                if c_higher_root in longer_path_to_root:
                    case = 2
                else:
                    case = 1  # LCA
            if case == 1:  # LCA
                sum_t1 = c1_root.sum
                sum_t2 = c2_root.sum
                sum_t3 = root.sum - c1_root.sum - c2_root.sum
            elif case == 2:
                sum_t1 = c_lower_root.sum
                sum_t2 = c_higher_root.sum - c_lower_root.sum
                sum_t3 = root.sum - c_higher_root.sum

            # Need two sums to be equals to each other, and the two same sums must be bigger than the other one.
            # (Because insert non-zero node into remaining tree.
            if ((sum_t1 == sum_t2) and sum_t1 > sum_t3) \
                    or ((sum_t1 == sum_t3) and sum_t1 > sum_t2) \
                    or ((sum_t2 == sum_t3) and sum_t2 > sum_t1):
                smaller = min(sum_t1, sum_t2, sum_t3)
                bigger = max(sum_t1, sum_t2, sum_t3)
                min_val = bigger - smaller
                if solution == -1:
                    solution = min_val
                else:
                    solution = min(solution, min_val)
        return solution

    FileBasedTest = True
    if FileBasedTest:
        import math, os, random, re, sys
        if __name__ == '__main__':
            fptr = open(os.environ['OUTPUT_PATH'], 'w')
            q = int(input())
            for q_itr in range(q):
                n = int(input())
                c = list(map(int, input().rstrip().split()))
                edges = []
                for _ in range(n - 1):
                    edges.append(list(map(int, input().rstrip().split())))
                result = balancedForest(c, edges)
                fptr.write(str(result) + '\n')
            fptr.close()
    else:
        # # test_0   # Given in descripiton.
        # # n id   1  2  3  4  5
        nodes0 = [1, 2, 2, 1, 1]
        edges0 = [[1, 2], [1, 3], [3, 5], [1, 4]]
        print(balancedForest(nodes0, edges0))
        # #
        # #          i,val(sum)
        # #          0,1(7)
        # # 1,2(2)   2,2(3)    3,1(1)
        # #          4,1(1)
        # Solution: 2

        # # my own test  # I made this to work out how to construct tree and comute sums. Don't know the optimal sol :-).
        # # givenid 1  2  3  4  5  6  7  8
        # # nodeid  0  1  2  3  4  5  6  7
        # nodes1 = [5, 3, 2, 1, 4, 8, 6, 9]
        # edges1 = [[1, 2], [2, 3], [3, 4], [2, 5], [1, 6], [6, 7], [6, 8]]
        # balancedForest(nodes1, edges1)

        # HackerRank Test 1
        # Problem case
        # n = 8
        # nodes: [1, 1, 1, 18, 10, 11, 5, 6]
        # edges  [[0, 1], [0, 3], [1, 2], [0, 7], [7, 6], [6, 5], [4, 6]]
addto(Trees__Graphs__BalancedForest, ds.tree, ds.graph, date(2019,7,3), diff.hard, time.days, source.hackerrank, tag.non_interview)

def Trees__decodeHuffman_Traversal():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/tree-huffman-decoding/problem
    """A bit confusing if you're not familiar with huffman decoding"""
    def decodeHuff(root, s):
        word = []
        curr = root
        for bit in s:
            if bit == "0":
                curr = curr.left
            else:
                curr = curr.right
            if curr.data != "\0":
                word.append(curr.data)
                curr = root
        print("".join(word))

        # Dev note. Initially wasn't sure what "Phie" node value was. verified it with this:
        # print 1, [root.data], "it's \ 0" if root.data == "\0" else "something else"
        # print 2, [root.left.data]
        # print 3, [root.right.data]
        # print 4, [root.left.left.data]
addto(Trees__decodeHuffman_Traversal, ds.tree, diff.med, time.min_30, date(2019,6,30),source.hackerrank)

def Trees__LCA_LowestCommonDenominator():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:# URL:https://www.hackerrank.com/challenges/binary-search-tree-lowest-common-ancestor/problem
    """LCA without links to parent"""
    '''
    class Node:
          def __init__(self,info): 
              self.info = info  
              self.left = None  
              self.right = None 
           // this is a node of the tree , which contains info as data, left , right
    '''
    def lca(root, v1, v2):
        # Find a and b. Link child nodes to parent to be able to backtrack.
        # (1) Note, we add 'parent' attribute to node dynamically via node.parent = ...
        root.parent = None
        node_stack = []
        node_stack.append(root)
        v1_node, v2_node = None, None
        while node_stack:
            node = node_stack.pop()
            if not v1_node and node.info == v1:
                v1_node = node
            if not v2_node and node.info == v2:
                v2_node = node
            for child_node in [node.left, node.right]:
                if child_node:
                    child_node.parent = node  # (1)
                    node_stack.append(child_node)

        # Generate path from A to root.
        curr = v1_node
        a_to_root = set()
        while curr:
            a_to_root.add(curr.info)
            curr = curr.parent

        # traverse up b until you come across an element in a's path to parent.
        curr = v2_node
        while curr:
            if curr.info in a_to_root:
                return curr
            else:
                curr = curr.parent

        print("Shouldn't be here, Something went wrong")

    # # Recursive. (Iterative is better, but did recursive for practice.) ~15 min.
    # # Main idea is that we count the number of v1/v2's found of the subnodes.
    # # If a node has sum of 2, we know it's the lca.
    # def lca(root, v1, v2):
    #     def lca_helper(node):
    #         ret_node = None
    #         if not node:
    #             return 0, None
    #         v_match_counter = 0
    #         if node.info in [v1, v2]:
    #             v_match_counter += 1
    #         left_count, left_node_ret = lca_helper(node.left)
    #         right_count, right_node_ret = lca_helper(node.right)
    #         v_match_counter += left_count + right_count
    #         if v_match_counter == 2:
    #             ret_node = node
    #         if left_node_ret:
    #             ret_node = left_node_ret
    #         if right_node_ret:
    #             ret_node = right_node_ret
    #         return v_match_counter, ret_node

    #     _, node = lca_helper(root)
    #     return node
addto(Trees__LCA_LowestCommonDenominator, ds.tree, algo.dfs_bfs, diff.med, time.hour, date(2019,6,29), source.hackerrank, tag.recursion)

def Trees__CheckBST():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/ctci-is-binary-search-tree/problem
    """Initial gut feeling is to just go down the tree and check that there are no duplicate and children are bigger.
        However, there is a caveat.
    """
    # O(n) solution. Passes all test cases.
    # Tricky part with leaf in left side of root being bigger than root. (or right/smaller)
    # E.g:
    #       3
    #    2     6
    #  1  4   5   7
    # Note, the 4 is bigger than the parent 3. But in a proper BST parent
    #  must be bigger than all items on parent's left side.
    # Keep track of last biggest element as we descend to children (& last smallest element)
    # val < last biggest (last_left)    #for cases like 4
    # val > last smallest  (last_right) # for mirror side.
    # Convieniently, this also ensures uniqueness.
    def checkBST(root):
        queue = []
        queue.append((root, None, None))  # node, last_left, last_right.
        while queue:
            node, last_left, last_right = queue.pop()
            if not node:
                continue
            if last_left and not node.data < last_left \
                    or last_right and not node.data > last_right:
                return False
            queue.append((node.left, node.data, last_right))
            queue.append((node.right, last_left, node.data))
        return True
addto(Trees__CheckBST, ds.tree, algo.dfs_bfs, diff.med, time.hour_2, source.hackerrank, tag.insight, tag.interview_material, date(2019, 6, 28), source.hackerrank)

def Trees__depth():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/tree-height-of-a-binary-tree/problem"
    # Avoid the recursive implementation here. See  Tree.py -> depth.
    def height(root):
        nodes_queue, max_so_far = [(root, 0)], 0
        while nodes_queue:
            node, depth = nodes_queue.pop()
            max_so_far = max(max_so_far, depth)
            for childnode in [node.left, node.right]:
                if childnode:
                    nodes_queue.append((childnode, depth + 1))
        return max_so_far
addto(Trees__depth, ds.tree, diff.easy, time.min_30, date(2019,6,24), source.hackerrank)

def Queues__TalesOfTwoStacks():
    # Python2 ported to Python3 via 2to3-3.7
    #  URL:https://www.hackerrank.com/challenges/ctci-queue-using-two-stacks
    """This is one of these problems where the simple solution is not efficient."""
    """The solution below passes all test cases."""

    class MyQueue(object):
        def __init__(self):
            self.pop_index = 0
            self.first = []
            self.second = []

        def peek(self):
            if self.pop_index == 0:
                return self.first[0]
            else:
                return self.first[self.pop_index]

        def pop(self):
            ret_val = self.first[self.pop_index]
            self.pop_index += 1
            if self.pop_index == len(self.first):  # index is out of array.
                self.first = self.second
                self.second = []
                self.pop_index = 0
            return ret_val

        def put(self, value):
            if self.pop_index == 0:
                self.first.append(value)
            else:
                self.second.append(value)

    queue = MyQueue()
    t = int(input())
    for line in range(t):
        values = list(map(int, input().split()))

        if values[0] == 1:
            queue.put(values[1])
        elif values[0] == 2:
            queue.pop()
        else:
            print(queue.peek())

    ## Simple solution that simply moves stacks along. Not efficient. Fails some test cases.
    # class MyQueue(object):
    #     def __init__(self):
    #         self.first = []
    #         self.second = []
    #
    #     def peek(self):
    #         return self.first[0]
    #     # O(n). Not efficient.
    #     def pop(self):
    #         for _ in xrange(len(self.first) -1):
    #             self.second.append(self.first.pop())
    #         ret_val = self.first.pop()
    #         for _ in xrange(len(self.second)):
    #             self.first.append(self.second.pop())
    #         return ret_val
    #
    #     def put(self, value):
    #         self.first.append(value)
    # ...
addto(Queues__TalesOfTwoStacks, ds.stacks, time.min_30, diff.med, date(2019,6,20), tag.interview_material, tag.insight, source.hackerrank)

def Stacks__LargestRect():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/largest-rectangle/problem
    # Initially implemented dumb sol, that failed on higher test cases. Then implemented proper linear-ish sol.
    # Good example of easy & efficient solutions.

    def largestRectangle(h):  # h - height arrray

        # Starting from every building, try to find the biggest rectangle stretching left and right.
        # Only expand rectangle if it makes keeps rectangle at same height.
        # Only start rectangle if you can start with a height that is bigger than a previous overlapping rectange.

        max_rect_heights = [0] * len(h)
        max_area = 0

        for i in range(len(h)):
            if h[i] > max_rect_heights[i]:
                height = h[i]
                max_rect_heights[i] = height

                # Grow as far right as possible so long as we're not reducing height of rect.
                for j in range(i, len(h)):  # start at i to avoid IndexError
                    if h[j] < height:
                        j -= 1
                        break
                    max_rect_heights[j] = max(max_rect_heights[j], height)

                # Grow as far left as possible so long as we're not reducing height of rect.
                for k in range(i, -1, -1):
                    if h[k] < height:
                        k += 1
                        break
                    max_rect_heights[k] = max(max_rect_heights[k], height)

                width = j - k + 1  # Python conveniently doesn't discard values after loop.
                area = height * width
                max_area = max(max_area, area)

        return max_area

    # # O(n^2) Solution. Try every Square.
    # # Works on Test1,2,3,45,14,15, not on 6,7,8,9,10,11,12
    def slow_largestRectangle(h):
        import itertools
        max = 0
        for r in itertools.chain(itertools.combinations(list(range(len(h) + 1)), 2)):
            area = min(itertools.islice(h, r[0], r[1])) * (r[1] - r[0])
            if area > max:
                max = area
        return max

    T1 = [1, 2, 3, 4, 5]  # 9
    T2 = [1, 2, 3, 3, 2]  # 8  # hill
    T3 = [5, 4, 3, 2, 1]  # 9  # Reverse T1.
    T4 = [2, 1, 2]  # 3 Needs to expand left and right from 1 to find.

    import random
    T5 = [random.randint(0, 1000000) for _ in range(1000)]
    # Large test case. Slow algo takes 5 seconds. Fast one is 3ms. #(Use PyCharm's profiler).

    for i, T, in enumerate([T1, T2, T3, T4, T5], 1):
        print("T", i, slow_largestRectangle(T) == largestRectangle(T))
addto(Stacks__LargestRect, ds.stacks, time.hour_5, diff.med, date(2019,6,15), tag.insight, tag.random_test, tag.optimization, source.hackerrank)

def Stacks__Game_of_Two_Stacks ():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/game-of-two-stacks/problem
    # I initially tried a dynamic aproach, but it wasn't very efficient. It might scale better with more stacks thou.

    # Linear solution: O(A + B)
    def twoStacks(x, A, B):
        from collections import deque
        a = deque(A)
        b = deque(B)
        a_taken = deque()
        b_taken = deque()  # todo_someday, don't need b_taken pile, can just keep a counter.

        curr_sum = 0
        max_cards_picked = 0

        # Go as deep as you can in pile A
        while a and curr_sum + a[0] <= x:
            card_num = a.popleft()
            curr_sum += card_num
            a_taken.append(card_num)

        max_cards_picked = len(a_taken)

        # Go as deep as you can in pile B. Put a_taken cards back if you can't take from B. Keep track of max.
        while b:
            if curr_sum + b[0] <= x:
                card_num = b.popleft()
                curr_sum += card_num
                b_taken.append(card_num)
                max_cards_picked = max(len(a_taken) + len(b_taken), max_cards_picked)
                continue

            if len(a_taken) > 0:
                card_num = a_taken.pop()
                curr_sum -= card_num
                continue
            else:
                break

        return max_cards_picked

    games = int(input())
    for _ in range(games):
        _, _, x = list(map(int, input().split()))
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
        print(twoStacks(x, A, B))

    # Dynamic solution: O( (A+B) log n)? ish.
    # Problem: This solution doesn't pass test cases 8-12 due to time-out.
    # Probably the seen dict(or set) is not O(1) in practice.
    def twoStacks(x, A, B):
        from collections import deque
        seen = dict()

        def get_max(curr_case):
            return curr_case[0] + curr_case[1]

        # (index of A, index of B, current sum), # of cards picked can be calculated.
        base_case = (-1, -1, 0)
        curr_max = get_max(base_case)
        to_try = deque()
        to_try.append(base_case)

        while to_try:  # Empty deque/list implicitly False. Non-empty is True.
            curr = to_try.popleft()
            if get_max(curr) > curr_max:
                curr_max = get_max(curr)

            next_a_i = curr[0] + 1
            if next_a_i < len(A):
                new_case = (next_a_i, curr[1], curr[2] + A[next_a_i])
                if new_case not in seen and new_case[2] <= x:
                    seen[new_case] = True
                    to_try.append(new_case)

            next_b_i = curr[1] + 1
            if next_b_i < len(B):
                new_case = (curr[0], next_b_i, curr[2] + B[next_b_i])
                if new_case not in seen and new_case[2] <= x:
                    seen[new_case] = True
                    to_try.append(new_case)

        return curr_max + 2  # +2 to account for index offset of list A & B.
addto(Stacks__Game_of_Two_Stacks, ds.stacks, diff.med, date(2019,6,13), tag.interview_material, tag.insight, source.hackerrank)

def Stacks__EqualStacks():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/equal-stacks/problem"
    # Original problem involved 3 stacks, but an extension could be "x number of stacks".
    # O(n)

    from collections import deque
    def equalStacks(h1, h2, h3):
        many_stacks = [h1, h2, h3]
        stacks = []
        stacks_sums = []
        for s in many_stacks:
            stacks.append(deque(s))
            stacks_sums.append(sum(s))

        while True:
            if stacks_sums[0] == 0 or all([i == stacks_sums[0] for i in stacks_sums]):
                break
            biggest_index = stacks_sums.index(max(stacks_sums))  # Biggest from left.
            stacks_sums[biggest_index] -= stacks[biggest_index].popleft()

        return stacks_sums[0]
addto(Stacks__EqualStacks, ds.stacks, ds.queue, diff.easy, time.min_30, date(2019,6,12), source.hackerrank)

def Stacks__BalancedBrackets():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/balanced-brackets/problem
    def isBalanced(s):
        stack = []
        for c in s:
            if c in "{[(":
                stack.append(c)
            else:
                if len(stack) == 0:  # E.g }{  #start with wrong bracket.
                    return "NO"
                if (c == "}" and stack[-1] == "{") or (c == "]" and stack[-1] == "[") or (
                        c == ")" and stack[-1] == "("):
                    stack.pop()
                else:
                    return "NO"
        if len(stack) == 0:
            return "YES"
        else:
            return "NO"
addto(Stacks__BalancedBrackets, ds.stacks, time.min_15, diff.easy, date(2019, 5, 28), tag.interview_material, source.hackerrank)

def Stacks__MaxElement():
    # Python2 ported to Python3 via 2to3-3.7
    # ABOUT: Multiple solutions can be found here. O(n^N) & O(n)
    # URL:https://www.hackerrank.com/challenges/maximum-element/problem

    # Solution: We don't care about non-max items. So only track max items seen & other items.
    class item:
        def __init__(self, value):
            self.value = value
            self.count = 1
            self.othercount = 0

    maxStack = []

    def max_push(value):
        if len(maxStack) == 0:
            maxStack.append(item(value))
        else:
            currMax = maxStack[-1]
            if currMax.value == value:
                currMax.count += 1
            elif value > currMax.value:
                maxStack.append(item(value))
            else:
                currMax.othercount += 1

    def max_pop():
        currMax = maxStack[-1]
        if currMax.othercount > 0:
            currMax.othercount -= 1
        elif currMax.count == 1:
            maxStack.pop()
        else:
            currMax.count -= 1

    for _ in range(int(input())):
        cmd = input().split()
        if cmd[0] == "1":  # push
            max_push(int(cmd[1]))
        elif cmd[0] == "2":  # pop
            max_pop()
        elif cmd[0] == "3":  # max
            print(maxStack[-1].value)


    # Solution 2: Use 2 stacks. Works, but not ideal. Why?
    # (No need to keep 2nd stack of values.
    # -------
    # class item:
    #     def __init__(self, value):
    #         self.value = value
    #         self.count = 1
    #
    # stack = []
    # maxStack = []
    #
    # for _ in range(int(raw_input())):
    #     cmd = raw_input().split()
    #     if cmd[0] == "1":   # push
    #         value = int(cmd[1])
    #         stack.append(value)
    #         if len(maxStack) == 0:
    #             maxStack.append(item(value))
    #         else:
    #             currMax = maxStack[-1]
    #             if currMax.value == value:
    #                 currMax.count += 1
    #             elif value > currMax.value:
    #                 maxStack.append(item(value))
    #     elif cmd[0] == "2": # pop
    #         value = stack.pop()
    #         currMax = maxStack[-1]
    #         if value == currMax.value:
    #             if currMax.count == 1:
    #                 maxStack.pop()
    #             else:
    #                 currMax.count -= 1
    #     elif cmd [0] == "3": # max
    #         print maxStack[-1].value

    # The following works for the first 9 test cases, but times out on the larger once.
    # Reason is max(S) is O(n), and get's very expensive for large data sets.
    # -----
    # stack = []
    # for _ in range(int(raw_input())):
    #     cmd = raw_input().split()
    #     if cmd[0] == "1":
    #         stack.append(int(cmd[1]))
    #     elif cmd[0] == "2":
    #         stack.pop()
    #     elif cmd [0] == "3":
    #         # print max(S)  # O(n) per op. Terrible run time.
addto(Stacks__MaxElement, ds.stacks, time.hour_5, diff.med, tag.optimization, source.hackerrank)

def LinkedList_reverse():
    # Python2 ported to Python3 via 2to3-3.7
    def reverse(head):
        # https://www.hackerrank.com/challenges/reverse-a-linked-list/problem 19/05/23
        # Case 0: Empty list  or  Case 1: List with only 1 element.
        if head is None or head.__next__ is None:
            return head

        # Case 2: 2+ elements. Start at 2nd element.
        prev, curr = head, head.__next__
        prev.next = None

        while True:
            nxt = curr.__next__
            curr.next = prev
            if nxt is None:
                return curr
            prev, curr = curr, nxt
addto(LinkedList_reverse, ds.linked_list, diff.easy, date(2019,5,28), source.hackerrank)

def LinkedList_hasCycle():
    # Python2 ported to Python3 via 2to3-3.7
    # Option 1: Hare and Tortoise
    # Option 2: Add nodes to hashmap and check as you go along.
    def has_cycle(head):  # ret: True=hasCycle,,  False=Doesn't.
        # https://www.hackerrank.com/challenges/detect-whether-a-linked-list-contains-a-cycle/problem 19/05/23
        if head is None:
            return False
        hare = head
        tortoise = head
        while True:
            for _ in range(2):
                hare = hare.__next__
                if hare is None:
                    return False
                if hare is tortoise:
                    return True
            tortoise = tortoise.__next__
addto(LinkedList_hasCycle, ds.linked_list, diff.med, time.hour, date(2019,5,28), source.hackerrank, tag.interview_material)


def Arrays__difference():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/crush/problem
    ## Brute Force: O(nm)  -> Times out.
    # n, queries = map(int, raw_input().split())
    # arr = [0] * n
    # for _ in range(queries):
    #     a, b, k = map(int, raw_input().split())
    #     for i in range(a-1, b):
    #         arr[i] += k
    # print max(arr)

    # Smart: Array Difference O(n):
    # https://www.youtube.com/watch?v=hDhf04AJIRs
    n, queries = list(map(int, input().split()))
    arr = [0] * (n + 1)  # n+1 to account for A[b+1]
    for _ in range(queries):
        a, b, k = list(map(int, input().split()))
        arr[a - 1] += k
        arr[b + 1 - 1] -= k

    diff_arr = [0] * n
    diff_arr[0] = arr[0]
    for i in range(1, n):
        diff_arr[i] = diff_arr[i - 1] + arr[i]
    print(max(diff_arr))
addto(Arrays__difference, ds.arrays, diff.hard, time.hour_5, date(2019,5,25), tag.insight, source.hackerrank)


def Arrays__NewYearChaos():
    # Python2 ported to Python3 via 2to3-3.7
    # URL:https://www.hackerrank.com/challenges/new-year-chaos/problem
    def efficient_sol(rt="O(n)"):
        """
        Bubble sort starting from back, with look ahead only 2 items. Thus O(2n) = O(n)
        Input is special, it's almost sorted.

        line: (front)> 2 1 5 3 4 <(Back)
        Explanation:
        - Person A_i would have moved no more than 2 places forward. So he must be in A_i-1 or A__i-2.
        - We loop in revrese, find the person and *shift* the other (one|two) person forward. We note how far the person has moved.
        """
        def bribe_count(L):  # L = line
            bribes = 0
            # from back to front
            for i in range(len(L) - 1, -1, -1):

                expected_index = i + 1
                # Case 1: Expected Person already in the right spot.
                if L[i] == expected_index:
                    continue

                # Case 2: Expected Person is 1 step away. Swap.
                elif L[i - 1] == expected_index:
                    L[i], L[i - 1] = L[i - 1], L[i]
                    bribes += 1

                # Case 3: Expected person is 2 steps away. Put into proper place and other 2 up.
                elif L[i - 2] == expected_index:
                    L[i - 2], L[i - 1], L[i] = L[i - 1], L[i], L[i - 2]
                    bribes += 2
                else:
                    return "Too chaotic"  # Expected person is more than 2 steps away.
            return bribes

        test_count = int(input())
        for _ in range(test_count):
            input()
            line = list(map(int, input().split()))
            print(bribe_count(line))

    def brute_force(rt="O(n^2"): # First thing that came to mind.

        # Brute Force: O(n^2) approach.
        # Notes:
        # Based on Bubble sort implementation in:
        # from Leo_Python_Notes import Algorithms_Sort.BubbleSort()
        # Everyone starts with 2. If count is 0 and a swap is desired: "Too chaotic"
        def bribe_count(line):
            ppl = [[x, 2] for x in line]  # #[[label, bribe_count], [label, bc2] ...]
            bribes = 0
            for i in range(len(line) - 1):
                for j in range(len(line) - i - 1):
                    if ppl[j][0] > ppl[j + 1][0]:
                        ppl[j], ppl[j + 1] = ppl[j + 1], ppl[j]
                        ppl[j + 1][1] -= 1
                        bribes += 1
                        if ppl[j + 1][1] == -1:
                            return "Too chaotic"
            return bribes

        test_count = int(input())
        for _ in range(test_count):
            input()
            line = list(map(int, input().split()))
            print(bribe_count(line))
addto(Arrays__NewYearChaos, ds.arrays, diff.hard, time.hour_5, date(2019,5,22), source.hackerrank, tag.optimization)

def Arrays__Minimum_swaps(doc="""Meta:"This took some effort" Time:40m Difficulty:Medium Date: TAG__Array TAG__Insert URL:https://www.hackerrank.com/challenges/minimum-swaps-2/problem"""):
    # Python2 ported to Python3 via 2to3-3.7
    # Interesting fact: We know where item belongs since there are no duplicate and arr[i] <= n

    # [7, 1, 3, 2, 4, 5, 6]

    # Let's try different sort approaches:

    # # O(n)
    # # Insertion Sort. Move elemen to where it belongs. -> OK?
    # [6, 1, 3, 2, 4, 5, 7] 1
    # [1, 6, 3, 2, 4, 5, 7] 2
    # [1, 5, 3, 2, 4, 6, 7] 3
    # [1, 4, 3, 2, 5, 6, 7] 4
    # [1, 2, 3, 4, 5, 6, 7] 5  -> OK.  Not like example given, but seems to work.
    #
    # # O(n^2)
    # # Selection sort like. Pick the element that 'belongs' -> TOO SLOW.
    # [7, 1, 3, 2, 4, 5, 6] 0
    # [1, 7, 3, 2, 4, 5, 6] 1
    # [1, 2, 3, 7, 4, 5, 6] 2
    # [1, 2, 3, 4, 7, 5, 6] 3
    # [1, 2, 3, 4, 5, 7, 6] 4
    # [1, 2, 3, 4, 5, 6, 7] 4
    #
    # # Let's try Insertion-Sort like behaviour for Examples 0,1,3
    # # E.g 0
    # 4 3 1 2  | 0
    # 2 3 1 4  | 1
    # 1 2 3 4  | 3  -> OK.
    #
    # # E.g 1
    # 2 3 4 1 5 | 0
    # 3 2 4 1 5 | 1
    # 4 2 3 1 5 | 2
    # 1 2 3 4 5 | 3  -> OK.
    #
    # # E.g 2
    # 1 3 5 2 4 6 7 | 0
    # 1 5 3 2 4 6 7 | 1
    # 1 4 3 2 5 6 7 | 2
    # 1 2 3 4 5 6 7 | 3 -> OK.

    # All work. Let's implement:
    _ = input()
    arr = list(map(int, input().split()))

    curr_pos = 1  # curr_pos shall mean array index starting from 1 (not 0)
    swaps = 0
    while curr_pos != len(arr):

        element_at_curr_pos = arr[curr_pos - 1]

        if element_at_curr_pos != curr_pos:
            # Swap curr element with where it belongs.
            arr[curr_pos - 1] = arr[element_at_curr_pos - 1]
            arr[element_at_curr_pos - 1] = element_at_curr_pos
            swaps += 1
        else:
            curr_pos += 1
    print(swaps)
addto(Arrays__Minimum_swaps, ds.arrays, diff.med, time.hour, date(2019,5,20), source.hackerrank)

def Arrays__LeftRotation(doc="""Meta:"" Time:15m Difficulty:Easy Date:19/05/12 TAG__Array TAG__Rotate URL:https://www.hackerrank.com/challenges/ctci-array-left-rotation/problem"""):
    # Python2 ported to Python3 via 2to3-3.7
    _, d = list(map(int, input().split()))
    arr = input().split()
    print(" ".join(arr[d:] + arr[:d]))
addto(Arrays__LeftRotation, ds.arrays, time.min_30, diff.easy, source.hackerrank)

def Arrays__2D_ArraysDS(d="""Meta: TAG__Arrays TAG__Arrays_2D Date:19/05/12 Difficulty:Easy Time:20m URL:https://www.hackerrank.com/challenges/2d-array/problem?h_l=interview&playlist_slugs%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D=arrays"""):
    # Python2 ported to Python3 via 2to3-3.7
    from itertools import chain
    L = []
    for _ in range(6):
        L.append(list(map(int, input().split())))

    max_sum = None
    for i in range(4):
        for j in range(4):
            hr_sum = sum(chain(L[i][j:j + 3], [L[i + 1][j + 1]], L[i + 2][j:j + 3]))

            if max_sum == None or hr_sum > max_sum:
                max_sum = hr_sum
    print(max_sum)
addto(Arrays__2D_ArraysDS, ds.arrays, diff.easy, time.min_30, source.hackerrank)

def Arrays__repeated_string(doc="""Meta: TAG__Array TAG__Indexing TAG__mod TAG__counting Date:19/05/12 Difficulty:Easy Time:20m URL:https://www.hackerrank.com/challenges/repeated-string/problem?h_l=interview&playlist_slugs%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D=warmup"""):
    # Python2 ported to Python3 via 2to3-3.7
    # Count ref: https://www.geeksforgeeks.org/python-count-occurrences-of-a-character-in-string/
    def repeatedString(s, n):
        a_in_s_count = sum([1 if x == "a" else 0 for x in s])
        s_in_n_count = n / len(s)
        full_a_count = s_in_n_count * a_in_s_count

        s_in_n_count_remainder = n % len(s)
        remainder_a_count = sum([1 if x == "a" else 0 for x in s[:s_in_n_count_remainder]])

        return full_a_count + remainder_a_count

    s = input()
    n = int(input())
    print(repeatedString(s, n))
addto(Arrays__repeated_string, ds.arrays, date(2019, 5, 12), diff.easy, time.min_30, source.hackerrank)

def Arrays__Jumping_clouds(doc="""Meta: TAG__Array Date:19/05/12 Difficulty:Easy Time:20m URL:https://www.hackerrank.com/challenges/jumping-on-the-clouds/problem?h_l=interview&playlist_slugs%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D=warmup"""):
    # Python2 ported to Python3 via 2to3-3.7
    print(doc)
    _, clouds = input(), list(map(int, input().split()))
    jumps, curr_pos = 0, 0
    end_pos = len(clouds) - 1
    for _ in range(len(clouds)):
        if curr_pos == end_pos:
            break
        if curr_pos == end_pos - 1:
            jumps += 1
            break
        curr_pos += 2 if clouds[curr_pos + 2] == 0 else 1
        jumps += 1
    print(jumps)
addto(Arrays__Jumping_clouds, ds.arrays, diff.easy, time.min_30, date(2019,5,12), source.hackerrank)

def Array__counting_valleys(doc="""Meta: TAG__Array  TAG__T Date: Difficulty:Easy Time:9m URL:https://www.hackerrank.com/challenges/counting-valleys/problem?h_l=interview&playlist_slugs%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D=warmup"""):
    # Python2 ported to Python3 via 2to3-3.7
    # Keep track of current position. (height wise)
    # Count # of occurences where we go down from sea level [0, -1]
    _, path, valley_count, curr_pos = input(), input(), 0, 0
    for step in path:
        if curr_pos == 0 and step == "D":
            valley_count += 1
        curr_pos += 1 if step == "U" else -1
    print(valley_count)
addto(Array__counting_valleys, ds.arrays, diff.easy, time.min_30, date(2019,5,10), source.hackerrank)

def Arrays__Matching_socks(doc="""Meta: TAG__set TAG__deduplicate Date:19/05/11 Difficulty:Easy Time:4m URL:https://www.hackerrank.com/challenges/sock-merchant/problem?h_l=interview&playlist_slugs%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D=warmup"""):
    # Python2 ported to Python3 via 2to3-3.7
    # Solved in 4 minutes.
    _ = input()
    ar = list(map(int, input().split()))
    seen_socks = set()
    pair_count = 0
    for s in ar:
        if s in seen_socks:
            pair_count += 1
            seen_socks.remove(s)
        else:
            seen_socks.add(s)
    print(pair_count)
addto(Arrays__Matching_socks, ds.arrays, diff.easy, date(2019,5,1), time.min_15, source.hackerrank)

def hackerrank_Python_Regex__Matrix_script(doc="""Meta: TAG__Regex TAG__Groups TAG__Lookahead TAG__substitution Date: Difficulty:Hard Time:65m URL:https://www.hackerrank.com/challenges/matrix-script/problem"""):
    # Python2 ported to Python3 via 2to3-3.7
    # LNOTE: This was hard :-O. Phew.

    import re
    Rows, Cols = list(map(int, input().split()))
    wrapped = "".join([input() for _ in range(Rows)])
    decoded = []
    for i in range(Cols):
        # Generate Regex:
        # with i = 0,1,2, construct (.).. .(.).  ..(.)
        rx = ["."] * Cols
        rx[i] = "(.)"
        rx_str = "".join(rx)
        for matched in re.finditer(rx_str, wrapped):
            decoded.append(matched.group(1))

    decoded_string = "".join(decoded)
    # Note, reverse lookahead cannot contain wildcards. Workaround:
    # \g<1> is reference to first group. I.e replace (group1)+[!@..] with (group1)+" "
    # (?=...) only if ahead is...
    print(re.sub("([a-zA-Z0-9]+)[!@#$%& ]+(?=.*[a-zA-Z0-9]+)", "\g<1> ", decoded_string))


def hackerrank_Python_Regex__Validating_postcode(doc="""Meta: TAG__Regex TAG__LookAhead Date:19/05/10 Difficulty:Medium Time:20m URL:https://www.hackerrank.com/challenges/validating-postalcode/problem"""):
    # Python2 ported to Python3 via 2to3-3.7
    # LNote: Tricky par is the alternating consequitive repeition without consuming.

    regex_integer_in_range = r"^[1-9]\d{5}$"  # Do not delete 'r'.
    regex_alternating_repetitive_digit_pair = r"(.)(?=.\1)"  # Do not delete 'r'.

    import re
    P = input()

    print((bool(re.match(regex_integer_in_range, P))
           and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2))


def hackerrank_Python_Regex__validate_credit_card(doc="""Meta: TAG__Regex TAG__Repeating Date:19/05/10 Difficulty:Medium Time:21m URL:https://www.hackerrank.com/challenges/validating-credit-card-number/problem"""):
    # Python2 ported to Python3 via 2to3-3.7
    # LNOTE: I tend to make the mistake of forgetting to make exact match:  ^....$

    import re
    for _ in range(int(input())):
        line = input()
        # - Validate it's either 16 digits or separeted into groups of 4.
        # - Validate [456] as first group.
        if re.match(r'^[456](\d{3}-\d{4}-\d{4}-\d{4}|\d{15})$', line):
            # Remove dashes if there are any.
            line2 = "".join(line.split("-"))
            # Must not have 4 or more repeating digits.
            if re.search(r"^(?!.*([0-9])\1{3,}).+$", line2):  # If no 4 consequive.
                print("Valid")
                continue
        print("Invalid")


def hackerrank_Python_Regex__valid_UID(doc="""Meta: TAG__Regex TAG__NoRepetition Date:19/05/10 Difficulty:Medium Time:30m URL:https://www.hackerrank.com/challenges/validating-uid/problem"""):
    # Python2 ported to Python3 via 2to3-3.7
    import re

    def is_valid_UID(UID):
        two_chars = bool(re.search(r"[A-Z].*[A-Z]", UID))
            # Better: (.*[A-Z]){2}

        three_digits = bool(re.search(r"[0-9].*[0-9].*[0-9]", UID))
        alphaNum = bool(re.search(r"^[a-zA-Z0-9]{10}$", UID))
        norepeat = bool(re.search(r"^(?!.*(\w).*\1{1,}).+$", UID))  # No Repetion. Interesting.  -> .*(.).*\1+.*
        # Explanation: https://stackoverflow.com/questions/31897806/regular-expression-to-match-string-without-repeated-characters
        return two_chars and three_digits and alphaNum and norepeat

    import os
    if "USER" not in os.environ:
        for _ in range(int(input())):
            if is_valid_UID(input()):
                print("Valid")
            else:
                print("Invalid")
        raise SystemExit

    ts = [
        ("B1CD102354", False)  # 1 repeats
        , ("B1CDEF2354", True)
        , ("ABCD123HII", False)  # II repeats
    ]
    for t in ts:
        ret = is_valid_UID(t[0])
        print(ret == t[1], t, ret)

def hackerrank_Python_Regex__Validate_hex(doc="""Meta: TAG__regex TAG__hex Date: Difficulty:Easy Time:20m URL:https://www.hackerrank.com/challenges/hex-color-code/problem"""):
    # Python2 ported to Python3 via 2to3-3.7input()
    import sys
    full_stdin = "".join([line for line in sys.stdin])
    import re
    for m in re.findall(r"(#[0-9a-f]{6}|#[0-9a-f]{3})(?=.*;)(?i)", full_stdin):
        print(m)

def hackerrank_Python_Regex__Validate_email_address(doc="""Meta: TAG__Regex TAG__emailAddress Date:19/05/10 Difficulty:Medium Time:40m URL:https://www.hackerrank.com/challenges/validating-named-email-addresses/problem"""):
    # Python2 ported to Python3 via 2to3-3.7

    import re
    def emailIsValid(emailaddr):
        rx = r"^[a-zA-Z]+[a-zA-Z0-9\-_.]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$"
        return bool(re.match(rx, emailaddr))

    import os, email.utils
    if "USER" not in os.environ:
        count = int(input())
        for _ in range((count)):
            line = input()
            name, emailaddr = email.utils.parseaddr(line)
            if email == "":
                continue

            if emailIsValid(emailaddr):
                print(line)
        raise SystemExit

    F = False
    T = True
    tests = [
        ("a@b.c", T)
        , ("bob@abc.com", T)

        # Formatting
        , ("abc.com", F)  # No user.
        , ("a@bc", F)  # No extension
        , ("@abc.com", F)  # No user, but has extension
        , ("bob-young@abc.com", T)  # '-' in username
        , ("bob.young@abc.com", T)  # '.' in username
        , ("bob_young@abc.com", T)  # '_' in username

        # Start with english char. + AlphaNumeric & -._
        , ("99bob@abc.com", F)  # not english char
        , ("bob%%@abc.com", F)  # not alpha or -._

        # Domain only contains English Alphabetic chars
        , ("bob@123.com", F)

        # Extension should be 1,2,3 length
        , ("bob@abc.", F)
        , ("bob@abc.c", T)
        , ("bob@abc.co", T)
        , ("bob@abc.com", T)
        , ("bob@abc.comz", F)
    ]
    overall = True
    for T in tests:
        retval = emailIsValid(T[0])
        compare = retval == T[1]
        if not compare:
            overall = False
        print(compare, T, retval)
    print("Overall: ", overall)

def hackerrank_Python_Regex__validate_phone_number():
    # Python2 ported to Python3 via 2to3-3.7
    """Meta: Hackerrank problem to validate a 10 digit phone number. TAG__Regex TAG__exactMatch Date:19/05/09 Difficulty:Easy Time:20m
    https://www.hackerrank.com/challenges/validating-the-phone-number/problem
    """
    import re
    count = int(input())
    for _ in range(count):
        num = input()
        if bool(re.match(r"^[789]\d{9}$", num)):
            print("YES")
        else:
            print("NO")

def hackerrank_Python_Regex__roman_numerals():
    # Python2 ported to Python3 via 2to3-3.7
    """Meta: Hackerank problem to validate roman numerals. TAG__Regex TAG__groups Date:19/05/09 Difficulty:hard
    https://www.hackerrank.com/challenges/validate-a-roman-number/problem
    LNote: This took me 2-3 hours to research & understand.
    """

    import re
    regex_pattern = r"^M{0,3}(CM|CD|D?)(CL|XC|C{0,3})(X?L|L?X{0,3})(I[XV]|V|V?I{0,3})$"
    def isRoman(string):
        return bool(re.match(regex_pattern, string))

    tests = [
        ("I", True)
        , ("II", True)
        , ("III", True)
        , ("IIII", False)
        , ("IV", True)
        , ("V", True)
        , ("VI", True)
        , ("VII", True)
        , ("VIII", True)
        , ("VIIII", False)
        , ("IX", True)
        , ("IIX", False)
        , ("X", True)
        , ("XI", True)
        , ("XII", True)
        , ("XIII", True)
        , ("XIIII", False)
        , ("XIV", True)
        , ("XV", True)  # 15
        , ("XXX", True)
        , ("XXXX", False)
        , ("XL", True)
        , ("LX", True)
        , ("LXX", True)
        , ("LXXX", True)
        , ("LXXXX", False)
        , ("XC", True)
        , ("C", True)
        , ("CL", True)
        , ("CD", True) # #400
        , ("D", True) # 500
        , ("DC", True) # 600
        , ("DCC", True)
        , ("DCCC", True)
        , ("DCCCC", False)
        , ("M", True)  # 1000
        , ("MMM", True)
        , ("MMMM", False)
        , ("MMMCMXCIX", True) #3999


        # I: 1
        # V: 5
        # X: 10
        # L: 50
        # C: 100
        # D: 500
        # M: 1000
        # 9 -> IX
        # 99 -> XCIX
        # 999 -> CM XCIX (900 + 999)
        # 3 999 -> MMM CM XCIX


    ]
    overall = True
    for t in tests:
        ret = isRoman(t[0])
        iscorrect = ret == t[1]
        if not iscorrect:
            overall = False
        print(iscorrect, t, ret)
    print("Overall: ", overall)


def hackerrank_Python_Regex_lookahead_behind():
    # Python2 ported to Python3 via 2to3-3.7
    """Meta: Hackerrank problem with regex lookahead/behind and substitution. TAG__Regex TAG__substitution Date: 19/05/09
        https://www.hackerrank.com/challenges/re-sub-regex-substitution/problem
    """

    import sys, re
    _ = input()  # Ignore initial line count.
    for line in sys.stdin:
        s1 = re.sub(r"(?<= )&&(?= )", "and", line)  # Look behind/ahead.
        s2 = re.sub(r"(?<= )\|\|(?= )", "or", s1)
        sys.stdout.write(s2)

def hackerrank_Python_Regex_search_with_start_pos():
    # Python2 ported to Python3 via 2to3-3.7
    """ Meta: Hacker Rank problem that uses regex to search with start/end pos, starting at particular pos with regexObject. TAG___Regex, TAG__regexObject TAG__lookahead Date: 19/05/09
    https://www.hackerrank.com/challenges/re-start-re-end/problem
    """
    # Version 1: Iterative search, setting starting positi
    import re
    stack = input()
    needle = input()
    rxobj = re.compile(needle)
    m = rxobj.search(stack)
    if not m:
        print("(-1, -1)")
    while m:
        print("({}, {})".format(m.start(), m.end()-1))
        m = rxobj.search(stack, m.start()+1)
    raise SystemExit

    # Version 2: Via lookahead.
    import re
    stack = input()
    needle = input()
    needle_end = len(needle) -1
    matches, m = re.finditer("(?=({}))".format(needle), stack), None
    for m in matches:
        t = (m.start(), m.start() + needle_end)
        print(t)
    if not m:
        print("(-1, -1)")



def hackerrank_Python_Regex_findall_finditer():
    # Python2 ported to Python3 via 2to3-3.7
    """ Meta: HackerRank Problem that uses regular expressions re.finditer TAG__regex TAG__findall TAG__finditer  Date: 19/05/08
        https://www.hackerrank.com/challenges/re-findall-re-finditer/problem
    """
    import re
    def sol_func(input_string):
        returnL = []
        consonants = "[qwrtypsdfghjklzxcvbnm]"
        vowels = "[aeiou]{2,}"
        rx = r"(?<={c})({v})(?={c})(?i)".format(c=consonants, v=vowels)
        match_iter = re.finditer(rx, input_string)
        for m in match_iter:
            returnL += [m.group(1)]
        return "\n".join(returnL or ["-1"])

    import os
    if 'USER' not in list(os.environ.keys()):
        sol = sol_func(input())
        print(sol)
        raise SystemExit

    tests = [
        ("zzzaazzz", "aa")

        , ("""\
    rabcdeefgyYhFjkIoomnpOeorteeeeet""",
           """\
           ee
           Ioo
           Oeo
           eeeee""")

        , ("aa", "-1")
        , ("%%aa%%", "-1")
        , ("ZAEZ", "AE")
        , ("+ZEE-S", "-1")
        , ("ZZZZeeeeeeeeeeZZZZ", 'eeeeeeeeee')
        , ("zzeezzeezz", "ee\nee")
        , ("abaabaabaabaae", "aa\naa\naa")

    ]

    for t in tests:
        out = sol_func(t[0])
        print(out == t[1], t, [out])


def hackerrank_Python_Regex_find_float__TAG_REGEX():
    # Python2 ported to Python3 via 2to3-3.7
    # Meta: Problem that uses regex to determine floating point.    TAG_REGEX_GROUP TAG_REGEX_DIGITS TAG_REGEX_SETS
    # https://www.hackerrank.com/challenges/introduction-to-regex/problem
    # Date: 19/05/08

    import re
    def isFloat(string):
        pattern = r"[+-]?\d*\.\d+"
        try:
            float(string)
        except Exception:
            return False
        return bool(re.match(pattern, string))

    # Hacker Rank
    # for _ in range(int(raw_input())):
    #     s = raw_input()
    #     print isFloat(s)

    # Useful Testing function:
    # 19/05/08 Deprecated testing mechanism. Use simpler once in newer problem
    if True:  # Test code.

        def tester(test_func, test_data):  # e.g: test_data = [(1==1, True), (1==2, False)]
            pass_count = 0
            fail_count = 0
            print("Testing...")
            for t in test_data:
                try:
                    assert test_func(t[0]) == t[1], t
                    print(str(t), "PASSED")
                    pass_count += 1
                except AssertionError as e:
                    print(str(e), "FAILED")
                    fail_count += 1
            print("Result: Passed: " + str(pass_count) + "   Failed: " + str(fail_count))

        test_func = isFloat
        test_data = [
            # (Test, Expected)
            # Number can start with +, - or . symbol.
            ("+4.50", True),
            ("-1.0", True),
            (".5", True),
            ("-.7", True),
            ("+.4", True),
            ("-+4.5", False),

            #  Number must contain at least  decimal value.
            ("12.", False),
            ("12.0", True),

            # Number must have exactly one . symbol.
            ("12.10", True),
            ("12..10", False),
            ("12.10.15", False),

            ("4.0O0", False),
            ("-1.00", True),
            ("+4.54", True),
            ("SomeRandomStuff", False)
            ]

        tester(test_func, test_data)

# problem_find_float__TAG_REGEX()
def problem_sub_string():
    # Python2 ported to Python3 via 2to3-3.7
    # Count substrings in a string.
    # Naive solution:   (KMP is better).
    main = "ABCDCDC"
    subs = "CDC"

    count = 0
    for s in range(len(main)):
        if main[s:s+len(subs)] == subs:
            count = count + 1

    print("Found: " + str(count))


def hackerrank_Python_String_print_formatted_decimal_octal_hex_binary():
    """
    # Python2 ported to Python3 via 2to3-3.7
    https://www.hackerrank.com/challenges/python-string-formatting/problem
    """
    def print_formatted(number):
        # your code goes here

        padw = len(bin(number).lstrip("0b"))
        for i in range(1, number+1):
            print(str(i).rjust(padw) + " " \
                  + str(oct(i).lstrip("0")).rjust(padw) + " " \
                  + str(hex(i).lstrip("0x").upper()).rjust(padw) + " " \
                  + str(bin(i).lstrip("0b").rjust(padw)))

    print_formatted(20)
    # 1     1     1     1
    # 2     2     2    10
    # 3     3     3    11
    # 4     4     4   100 ...



def hackerrank_Python_Python_String__alphabet_rangoli(d="""Meta: Difficulty:Easy"""):
    """https://www.hackerrank.com/challenges/alphabet-rangoli/problem"""
    # Python2 ported to Python3 via 2to3-3.7
    N = 3

    import string
    alph_section = string.ascii_lowercase[:N][::-1]  # N = 3 -> cba

    # count > 1
    def gen_line (count):
        left = "-".join(alph_section[:count])
        right = left[::-1][1:]
        return str(left + right).center(4*N -3, "-")

    # Print upper + middle
    for i in range(1, N+1):
        print(gen_line(i))

    # Print lower
    for i in range(N-1, 0, -1):
        print(gen_line(i))



# Bannana Problem
def hackerrank_Python_minion_game_inefficent (string): # Ex string=BANANA
    # Python2 ported to Python3 via 2to3-3.7
    # https://www.hackerrank.com/challenges/the-minion-game/problem

    # Generate every possible substring
    str_len = len(string)  # 6

    Kevin_score = 0  # Vowels
    Stuart_score = 0

    # for every length
    for word_len in range(1, str_len+1): # 1,2,3,4,5,6
        last_index = str_len - word_len  # 5,4,3,2,1,0

        # get every possible word of that length
        for word_index in range(0, last_index+1):
            substr =  string[word_index:word_index+word_len]
            if substr[0] in ["A", "E", "I", "O", "U"]:
                Kevin_score += 1
            else:
                Stuart_score += 1
    if Kevin_score > Stuart_score:
        print("Kevin " + str(Kevin_score))
    elif Stuart_score > Kevin_score:
        print("Stuart " + str(Stuart_score))
    else:
        print("Draw")

def hackerrank_Python_minion_game_efficent(string):
    # Python2 ported to Python3 via 2to3-3.7
    # tag_substring
    vowels = 'AEIOU'
    s_len = len(string)
    kevsc = 0
    stusc = 0
    for i in range(len(string)):  # 0,1,2,3,4,5
        if string[i] in vowels:
            kevsc += (s_len - i)
        else:
            stusc += (s_len - i)
    if kevsc > stusc:
        print("Kevin", kevsc)
    elif kevsc < stusc:
        print("Stuart", stusc)
    else:
        print("Draw")


def hackerrank_Python_merge_the_tools(string, k):
    # Python2 ported to Python3 via 2to3-3.7
    # https://www.hackerrank.com/challenges/merge-the-tools/problem

    import textwrap
    t_subsegments = textwrap.wrap(string, k)  # k is factor of len(string)

    # Break into substrings
    for segment in t_subsegments:
        # Keep track of seen chars via set. O(n) instead of re-looping whole list O(n^2)
        known_chars = set()
        u_unique_subseq = []
        for c in segment:
            if c not in known_chars:
                known_chars.add(c)
                u_unique_subseq.append(c)

        print("".join(u_unique_subseq))

################
################ Utility Functions For finding problems and printing statistics
################

def _______UTILITY__FUNCTIONS_______():
    pass

from functools import reduce
def find(*query_items, **flags):   #find(s1, s2, s3, [rt=True])
    if "rt" in flags:  # return as list instead of printing.
        return list(map(lambda x: x.__name__, list(reduce(lambda a, b: a.intersection(b), query_items))))
    else:
        for problem in reduce(lambda a, b: a.intersection(b), query_items):
            print(problem.__name__)


def find_count(*sets):
    return len(find(*sets, rt=True))

def find_by_date(days_ago=False, weeks_ago=False, months_ago=False, years_ago=False):
    assert len([True for arg in [days_ago, weeks_ago, months_ago, years_ago] if arg is not False]) != 0, "Expecting at least 1 arg"
    td = date.today()
    if days_ago is not False:
        return [df.func for df in prob_dates if df.date == td - timedelta(days=days_ago)]
    elif weeks_ago is not False:
        curr_yyyy_ww = td.isocalendar()[0:2]  # (yyyy, ww)
        return [df.func for df in prob_dates if df.date.isocalendar()[0:2] == (curr_yyyy_ww[0], curr_yyyy_ww[1] - weeks_ago)]
    elif months_ago is not False:
        return [df.func for df in prob_dates if (df.date.year, df.date.month) == (td.year, td.month - months_ago)]
    elif years_ago is not False:
        return [df.func for df in prob_dates if df.date.year == td.year - years_ago]
    print("Warning Arg not recognized")
    return []

###### Print some stats!
def f0(L): # filter 0's
    while L and L[-1] == 0: L.pop()
    return L

print("--- Some problem Statistics:", "(Accurate as of 19/07/07) ish")
print("Total Problems solved: ", find_count(all_problems))
print("Problems solved in recent days:", " ".join(map(str, f0([len(find_by_date(days_ago=i)) for i in range(30)]))))
print("Problems solved in recent weeks: ", " ".join(map(str, f0([len(find_by_date(weeks_ago=i)) for i in range(10)]))))
print( "Problems solved in recent Months: ", " ".join(map(str, f0([len(find_by_date(months_ago=i)) for i in range(10)]))))
print("Diffiulty of problems: Easy:{}  Medium: {} Hard: {}".format(find_count(diff.easy), find_count(diff.med), find_count(diff.hard)))
print("Data Structures stats")
for key, value in sorted(ds._get_all_count().items(), key=lambda x: x[1], reverse=True):
    print("  ", key, value, end=" ")

print("\nTime stats")
for key, value in sorted(time._get_all_count().items(), key=lambda x:x[1], reverse=True):
    print("  ", key, value, end=" ")

# print("Tag stats")
# for key, value in tag._get_all_count().items():
#     print("  ", key, value)

print("\nSources stats")
for key, value in source._get_all_count().items():
    print("  ", key, value, end=" ")

# "(Hint. You can set a breakpoint on this line and explore problems via problem_categories variable in debugger)"
pass