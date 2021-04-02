# LTAG_PY2_PORT - This was ported from python2. Tests pass, but there is a chance of bugs.

# DONE - Init  (null)
# DONE - Append
# DONE - AppendLeft
# DONE - ToString
# DONE - __repr__
# DONE - pop
# DONE - popleft
# DONE - isEmpty() ? (maybe override True/False comparison of sorts?)  To know how to loop over.
# DONE - merge two lists  mergeWith(otherList)
# DONE - random tests with functions?

# Notes:
# Kinda missing the functionality to iterate over the list and insert/delete something. Should implement iterable maybe.
# TODO_SOMEDAY - Insert node at index. (should override indexing of sorts).
# TODO_SOMEDAY - Delete node

# Someday:
# TODO_SOMEDAY - Read from index. https://stackoverflow.com/questions/1957780/how-to-override-the-operator-in-python
# TODO_SOMEDAY - Node.delete self? (Use case, delete a node from middle of a dll for filtering). (might need to reference dll in Node thou..).
# TODO_SOMEDAY - insert at index.
# TODO_SOMEDAY - Node.insertRight
# TODO_SOMEDAY - Node.insertLeft


# Simple Doubly Linked List implementation.
# To keep things simple, users are expected to work with the internals. (e.g manually traverse list).
class DLL:
    def __init__(self, init_value=None):
        assert init_value is None or isinstance(init_value, Node)
        self.head = init_value
        self.tail = init_value

    def append(self, node):
        assert isinstance(node, Node)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            self.tail.nxt = node
            node.prv = self.tail
            self.tail = node

    def appendLeft(self, node):
        assert isinstance(node, Node)
        if not self.head:
            self.append(node)
        else:
            node.nxt = self.head
            self.head.prv = node
            self.head = node

    def peak(self):
        if self.tail:
            return self.tail
        else:
            raise IndexError

    def peakleft(self):
        if self.head:
            return self.head
        else:
            return IndexError

    def pop(self):
        if not self.tail:
            return None
        last_node = self.tail
        #last_node = Node(1)
        if self.head == self.tail:  # Last node.
            self.head, self.tail = None, None
        else:
            self.tail.prv.nxt = None
            self.tail = self.tail.prv
        #last_node.nxt, last_node.prv = None, None  # Sanitize.
        return last_node

    def popleft(self):
        if not self.head:
            return None
        last_node = self.head
        if self.head == self.tail:  # Last node.
            self.head, self.tail = None, None
        else:
            self.head.nxt.prv = None
            self.head = self.head.nxt
        #last_node.nxt, last_node.prv = None, None  # Sanitize.
        return last_node

    def remove(self, node):
        assert isinstance(node, Node)
        if node == self.head:
            self.popleft()
        elif node == self.tail:
            self.pop()
        else:  # middle node.
            if node.prv:
                node.prv.nxt = node.nxt
            if node.nxt:
                node.nxt.prv = node.prv


    def mergeWith(self, other_dll):
        assert isinstance(other_dll, DLL)
        assert other_dll is not None
        assert other_dll.head is not None
        assert other_dll.tail is not None
        assert self.tail is not None, "Current list is empty."
        self.tail.nxt = other_dll.head
        other_dll.head.prv = self.tail
        self.tail = other_dll.tail

    def __bool__(self):         # Python2 only. Python3 uses __bool__. Ref: https://stackoverflow.com/questions/2233786/overriding-bool-for-custom-class
        return False if self.head is None else True

    def tolist(self):
        L = []
        cur = self.head
        while cur:
            L.append(cur.data)
            cur = cur.nxt
        return L

    def __repr__(self):
        return "DLL: " + str(self.tolist())

class Node:
    def __init__(self, data):
        self.data = data
        self.nxt = None  # Don't use 'next' as it conflicts with build in.
        self.prv = None

    def __repr__(self):
        return str(self.data)


##################
### Tests   (also examples of usage).
# todo_someday Move to separate module and use unittest.
##################
## Config:
BASIC_TESTS = False
RANDOM_TESTS = False

dbm = []  # Debug messages.
tests_to_run = []

def test_init():
    """Test init with and without default value."""
    dll2 = DLL(Node(1))
    dll1 = DLL()
    if dll1.head is not None or dll1.tail is not None:
        dbm.append("dll1 init head and should be None")
        return False
    if dll2.head is None or dll2.tail is None:
        dbm.append("dll2 init head and tail should not be None")
        return False
    return True
tests_to_run += [test_init]

def test_iterater_over():
    dll = DLL()
    cmplist = []
    for i in range(5):
        dll.append(Node(i))
        cmplist.append(i)
    cur = dll.head;
    result_list = []
    while cur:
        result_list.append(cur.data)
        cur = cur.nxt
    return True if cmplist == result_list else False
tests_to_run += [test_iterater_over]



# todo, test remove head node.
# todo, test remove tail node.


def test_delete_node_head():
    dll = DLL()
    cmplist = []
    for i in range(5):
        dll.append(Node(i))
        cmplist.append(i)
    del(cmplist[0])
    dll.remove(dll.head)
    return True if dll.tolist() == cmplist else False
tests_to_run += [test_delete_node_head]

def test_delete_node_tail():
    dll = DLL()
    cmplist = []
    for i in range(5):
        dll.append(Node(i))
        cmplist.append(i)
    del(cmplist[-1])
    dll.remove(dll.tail)
    return True if dll.tolist() == cmplist else False
tests_to_run += [test_delete_node_tail]


def test_delete_node_middle():
    dll = DLL()
    cmplist = []
    for i in range(5):
        dll.append(Node(i))
        cmplist.append(i)

    for i, val in enumerate(cmplist):
        if val == 3:
            del(cmplist[i])
    cur = dll.head
    while cur:
        nxt = cur.nxt
        if cur.data == 3:
            dll.remove(cur)
        cur = nxt
    return True if dll.tolist() == cmplist else False
tests_to_run += [test_delete_node_middle]


def test_peak():
    cmplist = []
    dll = DLL()
    for i in range(5):
        cmplist.append(i)
        dll.append(Node(i))
    if cmplist[-1] != dll.peak().data:
        dbm.append("peak values don't match.")
        return False
    return True
tests_to_run += [test_peak]


def test_peakleft():
    cmplist = []
    dll = DLL()
    for i in [2,3,1]:
        cmplist.append(i)
        dll.append(Node(i))
    if cmplist[0] != dll.peakleft().data:
        dbm.append("peakleft values don't match.")
        return False
    return True
tests_to_run += [test_peakleft]

def test_append():
    cmplist = []
    dll = DLL()
    for i in range(5):
        cmplist.append(i)
        dll.append(Node(i))
    return True if dll.tolist() == cmplist else False
tests_to_run += [test_append]

def test_appendLeft():
    cmplist = []
    dll = DLL()
    for i in [5,4,3,2,1]:
        dll.appendLeft(Node(i))
        cmplist.insert(0, i)
    return True if dll.tolist() == cmplist else False
tests_to_run += [test_appendLeft]

def test_pop():
    cmplist = []
    dll = DLL()
    for i in range(5):
        cmplist.append(i)
        dll.append(Node(i))
    for _ in range(5):
        cmplist.pop()
        dll.pop()
    cmplist.append(99)
    dll.append(Node(99))
    dbm.append("test_pop:" + str(cmplist) + str(dll))
    return True if dll.tolist() == cmplist else False
tests_to_run += [test_pop]

def test_popleft_single():
    cmplist = []
    dll = DLL()
    for i in range(5):
        cmplist.append(i)
        dll.append(Node(i))
    del(cmplist[0])
    dll.popleft()
    dbm.append(":-O")
    return True if dll.tolist() == cmplist else False
tests_to_run += [test_popleft_single]

def test_popleft_all():
    cmplist = []
    dll = DLL()
    for i in range(5):
        cmplist.append(i)
        dll.append(Node(i))
    for i in range(5):
        del(cmplist[0])
        dll.popleft()
    return True if dll.tolist() == cmplist else False
tests_to_run += [test_popleft_all]

def test_dll_true():
    dll = DLL()
    dll.append(Node(0))
    dbm.append(dll.__repr__())
    return dll
tests_to_run += [test_dll_true]

def test_dll_false():
    dll = DLL()
    return True if not dll else False
tests_to_run += [test_dll_false]

def test_dll_false_afterPops():
    dll = DLL()
    dll.append(Node(1))
    dll.appendLeft(Node(0))
    dll.pop()
    dll.popleft()
    return True if not dll else False
tests_to_run += [test_dll_false_afterPops]

def test_loop_over():
    dll = DLL()
    cmplist = []
    LOOP_COUNT = 5
    for i in range(LOOP_COUNT):
        dll.append(Node(i))
        cmplist.append(i)

    popedItems = []
    looped = 0
    while dll:
        popedItems.append(dll.popleft().data)
        looped += 1
        if looped > LOOP_COUNT:
            dbm.append("Infinite Loop :-|. Probably not setting some pointer somewhere properly b0ec49943e2d4d089fdde2d5d5128282")
            return False

    if popedItems != cmplist:
        dbm.append("Popped items aren't the same as expected list 59c90b99892640f881bf01717d5090d3")
        return False
    if dll.head is not None:
        dbm.append("Expected dll head to be None ead28943e01144c7b51257e2932736d2")
        return False
    if dll.tail is not None:
        dbm.append("Expected dll tail to be None e63493855ca64ed48b952a9909eeb465")
        return False
    if dll:
        dbm.append("dll is should be False. 23e682fbbd894ddabf0552f26700d86b")
        return False
    else:
        return True
tests_to_run += [test_loop_over]

def test_mergeWith():
    dll1 = DLL()
    dll2 = DLL()
    cmplist = []
    for i in [0,1,2]:
        dll1.append(Node(i))
        cmplist.append(i)
    for i in [3,4,5]:
        dll2.append(Node(i))
        cmplist.append(i)
    dll1.mergeWith(dll2)
    return True if dll1.tolist() == cmplist else False
tests_to_run += [test_mergeWith]

if BASIC_TESTS:
    # Test Runner:
    test_count = 0
    fail_count = 0
    import sys

    def print_stderr(msg):
        sys.stderr.write(msg + "\n")

    def dbm_print():
        if dbm:
            print_stderr(" ---Debug messages: ")
            for m in dbm:
                print_stderr(str(" " + str(m)))
            print_stderr(" ---Debug END")


    # Run quick tests:
    for test in tests_to_run:
        dbm = []
        if test():
            test_count += 1
        else:
            sys.stderr.write("Failed test:" + test.__name__ + "\n")
            dbm_print()
            fail_count += 1
    sys.stderr.write("Summary: Passed: " + str(test_count) + "  Failed:" + str(fail_count) + "\n")





if RANDOM_TESTS:
    dbm = []
    #### The below is a mechanism to test DLL with random data using random operations for a large number of operations.
    ####  The goal is to arbitrarily try to find corner cases that break stuff.
    import random
    def test_random_tests():
        print("Starting random testing. This can take a few seconds.")
        from collections import deque

        def op_append(datastructure, value):
            if isinstance(datastructure, deque):
                datastructure.append(value)
            elif isinstance(datastructure, DLL):
                datastructure.append(Node(value))
            else:
                raise TypeError("Expected deque or DLL type. Got: " + type(datastructure))

        def op_appendleft(ds, value):
            if isinstance(ds, deque):
                ds.appendleft(value)
            elif isinstance(ds, DLL):
                ds.appendLeft(Node(value))
            else:
                raise TypeError("Expected deque or DLL type. Got: " + type(ds))

        def op_pop(ds):
            if isinstance(ds, deque):
                return ds.pop()
            elif isinstance(ds, DLL):
                node = ds.pop()
                return node.data
            else:
                raise TypeError("Expected deque or DLL type. Got: " + type(ds))

        def op_popleft(ds):
            if isinstance(ds, deque):
                return ds.popleft()
            elif isinstance(ds, DLL):
                node = ds.popleft()
                return node.data
            else:
                raise TypeError("Expected deque or DLL type. Got: " + type(ds))

        def op_merge(ds1,ds2):
            if isinstance(ds1, deque):
                assert isinstance(ds2, deque), "Other ds should also be deque d67f012a873e46678e58473e2944228a"
                for i in ds2:
                    ds1.append(i)
            elif isinstance(ds1, DLL):
                assert isinstance(ds2, DLL), "Other ds should also be DLL a1c5d8ace0bc482da5d98ebd9fe73510"
                ds1.mergeWith(ds2)
            else:
                raise TypeError("Expected deque or DLL type. Got: " + type(ds1))

        ops = [op_append, op_appendleft, op_pop, op_popleft, op_merge]

        def random_test():
            dbm = []
            dll = DLL()
            dq = deque()

            OP_COUNT = 50
            for i in range(OP_COUNT):
                op = ops[random.randint(0, len(ops) -1)]

                if (dll and not dq) or (not dll and dq):
                    dbm.append("ERROR: dll and dq have different truth values. 459f66409fad4374b67481734856e860")
                    dbm.append("   dll:" + str(bool(dll)))
                    dbm.append("   dq:" + str(bool(dq)))
                    return False

                # print i, op
                if op in [op_append, op_appendleft]:
                    val = random.randint(0,100)
                    dbm.append((op, val))
                    op(dll, val)
                    op(dq, val)
                elif op in [op_pop, op_popleft]:
                    if dq:
                        if not dll:
                            dbm.append("Hmm, dll should be true here. 68deb0b98dbf44d5accc771190e32206")
                            return False

                        dbm.append(op)
                        val1 = op(dll)
                        val2 = op(dq)
                        if val1 != val2:
                            dbm.append("Val1: {} doesn't match val2: {}".format(val1, val2))
                            return False
                if dll.tolist() != list(dq):
                    dbm.append("Lists don't match. {} {}", dll, dq)
                    return False

                # Merge op increases testing runtime significantly.
                elif op == op_merge:
                    if dq:
                        if not dll:
                            dbm.append("Hmm, dll should be true here. a367624155ae40ba828355686cf20450")
                            return False

                        dbm.append(op)
                        random_list = [random.randint(0, 20) for _ in range(50)]
                        dq2 = deque()
                        dll2 = DLL()
                        for i in random_list:
                            dq2.append(i)
                            dll2.append(Node(i))
                        op_merge(dq, dq2)
                        op_merge(dll, dll2)
                        if dll.tolist() != list(dq):
                            dbm.append("Lists don't match after merge {} {}", dll, dq)
                            return False

            return True

        RANDOM_TEST_COUNT = 1000
        for i in range(RANDOM_TEST_COUNT):
            if not random_test():
                return False

        return True



        # merge dev code.
        # dll = DLL()
        # dll2 = DLL()
        # op_append(dll, 1)
        # op_append(dll, 2)
        # op_append(dll2, 3)
        # op_append(dll2, 4)
        # op_merge(dll, dll2)
        # print "dll merged: ", dll
        #
        # dq = deque()
        # dq2 = deque()
        # op_append(dq, 1)
        # op_append(dq, 2)
        # op_append(dq2, 3)
        # op_append(dq2, 4)
        # op_merge(dq, dq2)
        # print "deque merged: ", dq

    # Run advanced tests
    if test_random_tests():
        print("All is the moist")
    else:
        print("Random tests failed...")
        for line in dbm:
            print(line)