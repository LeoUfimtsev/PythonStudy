"""
date(2019,7,20) - Originally Started with Python2, then migrated to python3 via 2to3-3.7 tool. Possible py2->3 bugs.

--------------------
Python 2.7 Quickref sheet:
http://www.astro.up.pt/~sousasag/Python_For_Astronomers/Python_qr.pdf

Good References:
https://www.geeksforgeeks.org/python-programming-language/

Practice:
https://www.hackerrank.com
https://practice.geeksforgeeks.org/
https://leetcode.com/problems/two-sum/


Data Structures Snippets.
https://www.tutorialspoint.com/python/python_data_structure.htm
- Arrays, Lists, Dictionaries, Tuples, Matrix, Sets
- Linked Lists, Queue, Dequeue, HAshTable, Binary Tree, Search, Heaps, Graphs, algo..

"""
from functools import reduce

def Wisdom():
    """
    Double, Triple, check question input/constraint/output.
    - Re-word in your code as comments before solving after reading the full question. Ensure code meets those objectives.
    - You often miss-read/assume during first read due to not having read the later part.
        e.g: "The maximum length of the extension is 3", Means *Up to 3*, not exactly 3.

    Testing:
    - Before submission, always check the broken test case.
    - Before submission, test with a bunch of test cases. [working,broken,corner cases]
    - Usually best to implement a function and test against some custom granular inupt, rather than test a full stdin
       This makes it easier to write mini-testing frameworks for when a problem can have many little corner cases.
       E.g hackerrank_problem_Regex__Validate_email_address
            Here instead of hard-coding logic that does everything in a single function, we break the function that
            validates emails into a separate function.
            We then test this a small test framework against many different email addresses.
      - It's a good idea to implement a little testing framework if input can have many corner cases.

    Mistakes I tend to make:
    - Regex:
        - Forget to make it an exact match.  ^...$    See:regex_pattern_exact_match
        - Use match instead of search.
    - Syntax:
        a == b    # instead of a = b
    - Data Structures:
        - Try to use the logically 'correct' datastructure for each job. Don't try to hack it with anohter one.
        - E.g: if you need a Stack, use a stack (list). If you need a queue use a deque.
          Try not to use a stack(list) as a dequeue item and hacking by adding/poping items in reverse as this leads
          to subtle bugs.
          See problem: Trees__top_view
        Indexing:
        - I make mistakes if I use  arra[index] for nodes/edges.
             Sol: -> append i to index values. E.g  nodei edgei
        -  Label offsets if you use multiple times to avoid typos/errors.

    Efficient algorithms:
    - If it's a tough algo:
        try to write dumb/naive algo,
        generate lots of random tests,
        test new algo against olg algo with random tests. E.g: hackerrank_DataStructures_Stacks__LargestRect

    Note about Recursion:
    - Not efficient in python, but if can't solve iterarativley, increase recursion limit. sys.setrecursionlimit(100000)
    - Converting iterative to recursion is not systematic, each algo requires thuoght.
    - I found that sometimes recursion simplifies solution a lot.
      e.g Graph problem (and maybe backgtracking/dynamic problems):
      https://www.quora.com/How-often-is-recursion-brought-up-in-coding-interviews-at-larger-companies-like-Google-Facebook-Amazon-etc

    Approaching large & complex problems: (hours/days kinda problems)
    - First implement the naive and ineffective solution & get it working. E.g implement recursively or via a O(n^2) algo.
      This ensures that you have a good grasp on the problem.
      You should be quick enough to code things via simple/naive solution.
    - Implement unit tests.
    - Then optimize solution (or parts of solution) as you go along and validate via unit tests.
      Often it's much easier to build the more efficient solution if you have a working naive solution.

    """
    def whiteboarding():
        """
        - To try:
            [] Problem: Need to move segments of code.
                Try: Work landscape, main function has sub-function calls.
                Try: Think further ahead, keep code in mind. Try scrap code & expand after.
        """

        pass
    pass


def notations_short_hands_for_examples():
    pass
_ = None
n = 123
L = [1,2,3,4]
S = set([1,2,3])
D = dict()

def Py2vs3():
    def division():
        # Division Produces floats
        # 2: 5 / 2 = 2
        # 3: 5 / 2 = 2.5      -> 5 // 2 = 2
        pass


        # [10,4,  2, 1, 5]
        # [40,100,200, ]
        #
        # [5,10,3 ]  18-3
        # [13,8,15]



    def unicode():
        # Unicode
        # 2: str is bytes (ascii)
        # 3: str is unicode
        pass

    def range_map_filter():
        # 2: xrange = iter, range = list
        # 3: xrange doesn't exist.  range = iter.

        # map -> iter  alt: use: [func(x) for x in list]
        # filter -> iter
        #    p2   x = filter(lambda x : x % 2 == 0, [1,2,3,4])           #-> [2, 4]
        #    p3   x = list(filter(lambda x: x % 2 == 0, [1,2,3,4]))
        #    ++   x = [x for x in [1,2,3,4] if x % 2 == 0] #-> [2,4]
        # dictonary's .keys(), .values(), .items() -> iter  (yay)
        # reduce -> Na. Moved to: 'from functools import reduce'. & returns list still.
        pass

    def tool_2to3():
        # there is a 2to3 (and 2to3-7) tool to automatically port py2 to py3.
        pass

    def input_func():
        # 2: X = raw_input ("enter some values)
        # 3: X = input ("enter some values")

        # 3->  X = eval(input("expr)")
        pass

    def print_func():
        # 2: print "hello"   # sys.stdout.write("stuff")
        # 3: print("msg", [end=""])
        pass

    def sorting():
        # todo -research a bit more:
        pass
        # builtin.sorted() and list.sort() no longer accept the cmp argument providing a comparison function.
        # Use the key argument instead. N.B. the key and reverse arguments are now “keyword-only”.
        #
        # The cmp() function should be treated as gone, and the __cmp__() special method is no longer supported.
        #  Use __lt__() for sorting, __eq__() with __hash__(), and other rich comparisons as needed.
        #  (If you really need the cmp() functionality, you could use the expression (a > b) - (a < b) as the equivalent for cmp(a, b).)
        # https://docs.python.org/3/whatsnew/3.0.html

    ############# NEW in python 3:

    def set_dict_comprehension():
        s = {i for i in range(10)}  # -> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        d = {k: v for k, v in [[1, 'val1'], [2, 'val2']]}  # -> {1: 'val1', 2: 'val2'}

    # Refs
    # https://docs.python.org/3/whatsnew/3.0.html       Overview ref.
    # https://python-future.org/compatible_idioms.html  Cheatsheet

    def typing():
        # Cheatsheet: https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html
        pass



def print_func():
    print("Hello")
    print("Hello", "World!") #-> 'Hello World!'  # no new line, but with a space.

    # Printing multi-Lines:
    print("Hello\nWorld")
    # Hello
    # World
    print(["Hello\nWorld"])  # Putting multiline strings into a list prints them on a single line. Shows special chars.
    ['Hello\nWorld']

    # Handle new-line manually:
    import sys
    sys.stdout.write(str(123))  # write a string w/o newline and w/o space. https://www.hackerrank.com/challenges/python-print/problem
                                # alternative is to use __future__ and python 3's prting with print("geeks", end =""), see https://www.geeksforgeeks.org/print-without-newline-python/

    # Alternative, construct a string (" ".joint(iterable)) & print on single line
    # You can use python3's print function if:
    #from __future__ import print_function  # is first line in script.
    #print("End2", end='')

def function_example(a, b, myarg="default_value*1"):  #LTAG__default_arguments_paramaters
    print((a + b))
    # call:  function_example(2, 4, myarg=bob)
    # [1] Note, default value is static across all function calls. See: gotcha_default_args_evaluated_only_once()

    # Note, default values are evaluated only once, so if dealing with objects
    # it can introduce bugs. See: gotcha_default_args_evaluated_only_once

outside_var = "helloworld"
def func_nesting():
    print(outside_var)  # Can access outside var.
    # outside_var = "goodbye world"  # ERROR. Python2: Cannot re-bind outside variable. Use dic:  d["x"] = val.
                    # Ref: https://stackoverflow.com/questions/3190706/nonlocal-keyword-in-python-2-x



def multiple_return_values():
    a, b = 1, 2   # comma separated. Returns tuple, can auto-unzip it.

def Branching():
    def if_elif_else():
        # Selection
        # if (bool_expr):
        #   stmt ...
        # [elif (bool_expr):
        #   stmt
        # [else:
        #   stmt
        pass

    def single_line_if():
        if True: return "end"  # works on single line, but not recommended as per pep8.
        # src: https://stackoverflow.com/questions/18669836/is-it-possible-to-write-single-line-return-statement-with-if-statement

    def tenary_if():
        a, b = 5, 2
        min = a if a < b else b  #https://www.geeksforgeeks.org/ternary-operator-in-python/
        print(("bob" if 1 == 2 else "jack"))

    def member_in_list():
        if 2 in [1, 2, 3]:
            print ("2 is in 1,2,3")


def boolean_logic():
    # not and or           # words rather than signs
    # (EXPR) op (EXPR)     # Brackets can be deeply nested.
    if (not (False or (True and True))) or True:
        print(True)

    # Morgan:
    # Boolen: not (A or B)  == not A and not B      # Mnemonic: not -> A == Not a.   not-> or == and.
    #         not (A and B) == not A or not B

    # Set   : not (A | B)  == not A & not B
    #         not (A & B)  == not A | not B


def primitive_datatypes():
    # int, 32 bit             10
    # long Intiger > 32bits   10L
    # float                   10.0
    # complex  1.2j
    # bool     True, False
    # str      "mystr"
    # tuple (immutable sequence)   (2,4,7)
    # list  (mutable sequence)     [2,x,3.1]
    # dict  (Mapping)              {x:2, y:2}
    pass

def math_ops():
    # a OP b, OP in +, -, *, **, %, >, <=, >=, !=, ==
    # Note fractions:
    4 / 3  # Py3: 1.33   Py2: 1
    4.0 / 3.0  # = 1.3333  # float division   float(4) / float(3)r
    4.0 // 3.0  # = 1.0     # integer division

    # print (10.0 / 4.0)  # = 2.5
    # print (2 ** 3)      # = 8  (exponential, to power of)
    # print (11 % 5)      # = 1  (mod)

    def incrementing():
        # You can't i++, instead:
        i = 0
        i += 1  # ref: https://stackoverflow.com/questions/2632677/python-integer-incrementing-with/2632687#2632687

    def fractions():
        from fractions import Fraction
        f = Fraction(10, 5)  # Fraction(2, 1)
        # f = f1 OP f2     OP + * etc..
        f.numerator
        f.denominator
        f.conjugate()
        # Ref: https://docs.python.org/2/library/fractions.html



    # Getting median value from a sorted array
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

    # Test:
    for i in [1, 2, 3, 4, 5]:
        L = list(range(i))
        med = median(L)
        print(L, med)

def Loops():
    # (for | while) can contain (break | continue)

    def while_loop():
        while True:
            print("infLoop")
        pass

    def for_loops():
        # for VAR in <ITERABLE>:
        #    stmt
        for i in [1, 2, 3, 4]:
            print((i * i))

    def for_loop_reverse():

        # To traverse a list in reverse, can use negative range
        L = [1, 2, 3, 4]
        for i in range(len(L) - 1, -1, -1):
            print(L[i])

    def for_loop_with_else():
        # "else" part in a for loop is executed if a break never occured. As a "last" entry.
        mystr = "helloMoto123"
        for s in mystr:
            if s.isdigit():
                print(s)
                break
        else:
            print("no digit found")
        # -> 1

    def for_enumerate():
        for i, w in enumerate(['a', 'b', 'c'], 1):  # starting at 1 instead of 0
            print(i, w)
        # 0 a
        # 1 b
        # 2 c

    def while_else_break():
        # When to use 'else' in a while loop?

        # Else is executed if while loop did not break.
        # I kinda like to think of it with a 'runner' metaphor.
        # The "else" is like crossing the finish line, irrelevant of whether you started at the beginning or end of the track. "else" is only not executed if you break somewhere in between.

        runner_at = 0  # or 10 makes no difference, if unlucky_sector is not 0-10
        unlucky_sector = 6
        while runner_at < 10:
            print("Runner at: ", runner_at)
            if runner_at == unlucky_sector:
                print("Runner fell and broke his foot. Will not reach finish.")
                break
            runner_at += 1
        else:
            print("Runner has finished the race!")  # Not executed if runner broke his foot.

        # ref: https://stackoverflow.com/questions/3295938/else-clause-on-python-while-statement/57247169#57247169


        # E.g breaking out of a nested while ≤loop   # LTAG
        for i in [1, 2, 3]:
            for j in ['a', 'unlucky', 'c']:
                print(i, j)
                if j == 'unlucky':
                    break
            else:
                continue  # Only executed if inner loop didn't break.
            break  # This is only reached if inner loop 'breaked' out since continue didn't run.

        print("Finished")
        # 1 a
        # 1 unlucky
        # Finished

def lambda_map_filter_reduce() :
    numbers = [1, 2, 3, 4]

    # Lambda Map
    double = lambda x: x + x
    numbers_doubled = list(map(double, numbers))  # -> [2, 4, 6, 8]
    # Problem: https://www.hackerrank.com/challenges/map-and-lambda-expression/problem

    # Pass lambda around to functions
    def pass_lambda(lambdaFunc, i):
        return lambdaFunc(i)

    print(pass_lambda(lambda x: x + 1, 1))

    # Lambda filter
    keep_even = lambda x: x % 2 == 0
    numbers_filtered = list(filter(keep_even, numbers))       # ref: https://stackabuse.com/lambda-functions-in-python/


    # Reduce
    # Concept: Reduce to a single value. Cumulatively in succession, left to right, reduce.
    # In python 3, reduce needs to be imported.
    reduce(lambda x, y: x + y, [1, 2, 3])  #->6
    reduce(lambda x, y: x + y, [1, 2, 3], -1)  # -> 5   # default value.

class String_Theory:
    def string_comparison(self):
        "leo" == "leo"  # Equality. Preferred.
        "leo" is "leo"  # identity. id("leo"). Work, but equality generally preffered.
        # https://stackoverflow.com/questions/2988017/string-comparison-in-python-is-vs

    def string_multiline(self):
        # -) For multi-line strings, indentation is as-is. Yon can use '\' for early line breaks thou.
        ml = """\
line
hello
world\
"""
        # -) Or manually join:
        var = ("implicitly\n"
               "Joined\n"
               "string\n")
        # ref: https://stackoverflow.com/questions/2504411/proper-indentation-for-python-multiline-strings

    def string_splicing(self):
        myString = "RedHat"
        # Menonic: str[from:till)
        myString[1:]  # edHat
        myString[:2]  # Re
        myString[0]   # R  #index starts at 0?

    def spring_splitting(self):
        # Splitting & Joining:
        # Note: Delimiter matters. By default, spaces are removed.
        # "A   B".split()    #-> ["A","B"]
        # "A   B".split(" ") #-> ["A", "", "", "B"]
        mystr = "hello world how are you"
        mySplitStrList = mystr.split(" ")  # delimiter. -> ["hello", "world" ...
        concatinated_by_delimiter = "-".join(mySplitStrList) # -> "hello-world-how-....    #ref: https://www.programiz.com/python-programming/methods/string/join

        # See also re.split()

    def string_reversal(self):
        "leo"[::-1]  #-> "ih"     #reverser string #anki-todo
        # ==
        "".join(reversed(list("leo")))

    def raw_strings(self):
        # prefix 'r' to ignore escape sequences.
        print("hello\nworld")
        #hello
        #world
        print(r"hello\nworld")
        #hello\nworld

        # \n new line       \<STUFF> is hazardous.
        # \t tab
        # \\ backslash
        # \' \"
        # ref: https://docs.python.org/2.0/ref/strings.html


    def string_concatination(self):
        # " ".join(..) has effective run-time.   http://blog.mclemon.io/python-efficient-string-concatenation-in-python-2016-edition
        # delimiter.join([str1,str2,str3])
        " ".join(["hello", "world", "war", "Z"])   #'hello world war Z'
        "".join(["hello ", "world ", "war ", "Z"]) #'hello world war Z'
        " ".join(map(str, [1,2,3]))   # useful to print data structures.

        # For building some pretty output, consider
        self.string_formatting(self)

    def string_formatting(self):
        # 2 methods:
        # -  .format() >= python 2.6.  Better support for native Data structures (dict etc..) use for new code.
        # -  %  == Old C-Printf style. Discouraged due to quirky behaviour but not deprecated.

        # Manual or Automatic field numbering:
        "Welcome {} {}!".format("Bob", "Young")

        # "{arg_index} | {arg_name}".format([args])
        "{0} {0} {1} {2} {named}".format("hello", "world", 99, named="NamedArg")
        'hello hello world 99 NamedArg'

        # (!) DANGER: padding with None can throw.
        "{}".format(None)  # OK.
        "{:<5}".format(None)  # (!) Throws TypeError: unsupported format string passed to NoneType.__format__
        "{:<5}".format(str(None))  # OK

        # Padding:     #   {: Filler_Char | > | ^ | <   NUMS}
        "{:>10}".format("Hello")   # {:>10} Pad Right 10  Note {:XX} == {:>XX}
        '     Hello'
        "{:<10}".format("Hello")   # {:<10} Pad Left 10
        'Hello     '
        "{:_>10}".format("Hello")  # Pad with character.
        '_____Hello'

        "{:_^10}".format("Hello")  # Center Align.
        '__Hello___'

        # Truncating ".len"   like str[:2]
        "{:.2}".format("Hello")
        'He'

        # Numbers:
        "Number:{:d}".format(42)  # d = int.  f = float.
        'Number:42'

        # Numbers: Float, truncate last digits
        "{:.1f}".format(1.134) #-> 1.1

        # Numbers  with padding:  (min, but prints full number if len(str) > min)
        "Lucky Number:{:3d}".format(42)
        'Lucky Number: 42'
        "{:<10.2f}".format(233.146)  # Truncate by 2 and pad 10.  {
        '233.15    '
        # For signs & signs + padding, see ref 1.

        # Dictionary
        d = {"leo": 31, "bob": 55}
        "{d[bob]} {d[leo]}".format(d=d)
        '55 31'
        # List
        "{l[0]} {l[1]}".format(l=['a','b'])
        'a b'

        # Parametrized Format:
        "{:{padding}.{truncate}f}".format(3.14632, padding=10, truncate=2)
        '      3.15'

        # Also: DateTime formatting 1*  class *1

        # Refs:
        # [1] Lots of examples: https://pyformat.info/
        # [1] Which String Formatting method to use?: https://realpython.com/python-string-formatting

    def string_update_char(self):
        # Strings are immutable. To update one:
        mystr = "absde"

        # 1) Turn into a list. Manipulate list.
        strlist = list(mystr)
        strlist[2] = "c"
        mystr2 = "".join(strlist)
        print(mystr2)
        # Src: https://stackoverflow.com/questions/19926089/python-equivalent-of-java-stringbuffer
        # ExProb: https://www.hackerrank.com/challenges/swap-case/problem

        # 2) Slice
        mystr3 = mystr[:2] + "c" + mystr[3:]
        print(mystr3)

        #lref_pystr_index  Anki-x    # HackerR: https://www.hackerrank.com/challenges/python-mutations/problem

    def string_validation(self):
        "a-z,A-Z,0-9".isalnum()  # Alpha Numeric.   (!= regex '\w')
        "a-Z".isalpha()          # Alphabetical.   # Not quite the same as regex '\w' since \w contains '_'
        "123".isdigit()
        "a-z123".islower()
        "ABC123".isupper()
        # Example problem: https://www.hackerrank.com/challenges/string-validators/problem

    def string_constants____kwds_lower_upper_letters_alphabet(self):
        import string
        string.ascii_lowercase
        # 'abcdefghijklmnopqrstuvwxyz'
        string.ascii_uppercase
        # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        string.ascii_letters
        # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def string_alinghment(self):
        "Leo".ljust(20, "#")
        #'Leo#################'
        "Leo".center(20, "#")
        #'########Leo#########'
        "Leo".rjust(20, "#")
        #'#################Leo'
        #ex: https://www.hackerrank.com/challenges/text-alignment/problem
        #ex: https://www.hackerrank.com/challenges/designer-door-mat/problem

    def string_wrapping(self):
        import textwrap
        print(textwrap.wrap("HelloWorld", 5))
        #Hello
        #World

    def string_stripping(self):
        "00005".lstrip("0")  # 5
        "50000".rstrip("0")  # 5

    def string_numerical_values____kwds_unicode(self):
        ord('a')  # 97  # return unicode value.  #ascii being subset of unicode.
        chr(97)   # 'a'     input: [0 ... 255]
        chr(97) #u'a'    input: [0 ... 65535] anki-done (
        # ref: https://www.geeksforgeeks.org/ord-function-python/
        # prob: https://www.hac5535kerrank.com/challenges/most-commons/problem

    def string_substring_counting(self):
        # Count substring
        ":-|  ^_^ :-)  -_-  :-)  *_*  :-)".count(":-)") #3
        "aaaa".count("a", 1, 3) # 2    [start]/[end]

    def string_index_of_a_letter(self):
        "abcd".index("b") # 1


    def every_substring(s):  # e.g abba -> ['a', 'b'...,'bba', 'abba']
        ss = []
        for l in range(1, len(s) + 1):  # l = 1..4
            for i in range(0, len(s) - l + 1): # l=1 -> i=(0, 4-1+1-1=3)
                ss.append(s[i:i + l])  # e.g 0:1 > a  0:2 -> ab       # for generator: yield s[i:i+l]
        return ss
    print(every_substring("abba"))  # ['a', 'b', 'b', 'a', 'ab', 'bb', 'ba', 'abb', 'bba', 'abba']
    n = len("abba")
    substring_count = (n * (n + 1)) // 2


def List_():
    L = [1, 4, 2, 1, 2, "A"]   # [] for empty list.

    # Append:
    L.append(99)
    L += ["abc"]    # Note, += <Iterable>.   if you += "abc" you get  ["a","b","c"] rather than ["abc].
                    # Note1: += [i] is 2x slower than .append()
    # Insert:
    L.insert(0, 5) # insert 5 at start of list.

    # Index:
    L[0] # first element
    L[-1] # last element
    L[-3] # 3rd last.

    # Remove
    L.remove(5)

    # Pop (read & remove from right).
    L.pop()

    # Ref: https://www.geeksforgeeks.org/python-list/

    # List has .count() function to count # of items inside it:
    [1, 1, 5].count(1)
    #2

    # Empty list is False. 1+ is True.
    L = []
    if not L:
        print("List is empty.")
    # Useful if you want to set default value in functions if function did not do anything with list.
    # L2 = L or ["default]

    def multi_dimensional_lists():
        # List of lists..
        L = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        L[0][1]
        #2

    def deduplicating_list():
        uniquearr = list(dict.fromkeys(L))  #anki-todo
        uniquearr.sort(reverse=True)
        print(uniquearr)

    def list_comprehension():
        # Single level
        nums = [1, 2, 3, 4, 5, 6]
        double_evens = [n * 2 for n in nums if n % 2 == 0]
        print("Single level ", double_evens)  # [4, 8, 12]

        # Nested
        first_list = [2, 4, 6]
        second_list = [1, 2, 3]
        product_first_second = [a * b for a in first_list for b in second_list]  # "if a EXPR b" filter possible.
        print("Nested", product_first_second)  # [2, 4, 6,      4, 8, 12,     6, 12, 18]
        #  First loop, 2nd             3rd

        indent_list_a = [1, 2, 3]
        indent_list_b = [10, 20, 30]
        added_list = [a + b
                      for a in indent_list_a
                      for b in indent_list_b
                      if a % 2 != 0]  # [11, 21, 31, 13, 23, 33]
        print("Multi-Line", added_list)

        # Practice Problem: https://www.hackerrank.com/challenges/list-comprehensions/problem
        # Reference: https://hackernoon.com/list-comprehension-in-python-8895a785550b

        def mapping_func_to_iter():
            # map (func, iter1 [, iter 2])        # function has 1 arg per iter.
            numbers = (1, 2, 3, 4)
            result = [x + x for x in numbers]   # apply function (or lamdba) to an iterable item. RETURN: list
            print((list(result)))
            # src: https://www.geeksforgeeks.org/python-map-function/
            # See src for multi-arg, string looping etc..

        def bisect_insert_sorted():  # TAG__List TAG__Insertion TAG__Sorted binary insert
            # Useful to insert an element such that list stays sorted.
            # Bisect is O(log n), but insertion is O(n).
            from bisect import insort
            L = [1, 3]
            insort(L, 2)
            #L == [1, 2, 3]

        def bisect_bisect():
            from bisect import bisect
            # Bisect gives position of where to insert item. (i+1)
            L = [1, 3, 5, 7]
            to_ins = 4
            L.insert(bisect(L, to_ins), to_ins)
            # [1, 3, 4, 5, 7]

def generator_expressions():
    # Create generator to iterate over.   (yieling_func)
    gen = (i * 2 for i in range(10))
    for i in gen:
        print(i)
    # 1 2 3  ..

    # src: https://dbader.org/blog/python-generator-expressions

    L = list(range(10))
    L_iter = (-L[i] for i in range(len(L)))
    for i in L_iter:
        print(i)
    # 0, -1, -2, -3 ....


def sorting():  # 828a4bad40234324ba24bd02f6595334 -> CheatSheet.
    L = [2, 1, 3]

    # Opt 1) In-Place:
    # myList.sort([key=lambda x: EXPR][, reverse=False]):
    L.sort()
    # Prob: https://www.hackerrank.com/challenges/python-sort-sort/problem

    # Opt 2) Return a new sorted list:
    # sorted(<Iterable>, [key=lambda x: EXPR], [reverse=False])
    # new_sorted_list = sorted([obj1,obj2,obj3], [key=lambda x: x.attribute], reverse=True)

    # Dealing with situation where you need to sort an on index
    L = [[2,"a"], [1, "b"]]
    L.sort(key=lambda x:x[0])  # -> [[1, 'b'], [2, 'a']]   # sort by number at index 0.
    L.sort(key=lambda x:x[1])  # -> [[2, 'a'], [1, 'b']]   # sort by number at index 1


    # ref: https://docs.python.org/2/library/functions.html#sorted

    # Note: in-place sort can't sort string. sorted(s) can. To sort a string:
    "".join(sorted("bac"))  # anki-todo

def dict_examples():
    mydic = {'jack': 999, 'jennie': 111}
    mydic['leo'] = 123
    mydic['bob'] = 234
    mydic['leo']       # accessing
    mydic.pop('bob')   # deleting
    print(mydic)
    # src: https://www.geeksforgeeks.org/python-dictionary/

    # See also OrderedDict in collections.

def set_examples():
    L = [1,2,3]
    B = [2,3,4]

    # set is an undordered collection of items.
    # To modify an item, remove(item) & add(item).
    myset = set()  # ref: https://www.geeksforgeeks.org/python-sets/
    myset = set([1,2,3])
    myset = {8,9}
    myset.add(1)
    myset.add(2)
    myset.add(3)

    myset.update([1,2,3]) # #like add for a list.

    myset.intersection_update(set()) # Modifies myset.
    myset.difference_update(set())
    myset.symmetric_difference_update(set())

    myset.discard(999)   # works even if e not in set.
    myset.remove(2)      # Raises KeyError if e doesn't exist.
    myset.pop()          # Raises KeyError if e doesn't exist.

    if 2 in myset:
        print("2 is in set!")

    len(myset)

    # Return new sets:
    # Set Theory:
    setA, setB = set(), set()
    setA.union(setB)  #(A | B)  -> returns new set...
    setA.intersection(setB) # (A & B)    # ref: https://www.programiz.com/python-programming/set
    setA.difference(setB)   # (A - B) # values that exist in A but not in B.
    setA.symmetric_difference(setB) # (A ^ B) # in either a or b, but not both.
    # Diagrams: https://www.hackerrank.com/challenges/py-set-symmetric-difference-operation/problem

    # setA.issubset(t)   s <= b
    # setA.issuperset(t) s >= b

    # ref: https://docs.python.org/2/library/sets.html



def heaps_with_heapq():
    # 1ed485d0f6614735afbf9a7efc834caf

    # Methods that take an Array as arguments:
    from heapq import heapify, heappush, heappop, heapreplace
    heap = []
    heapify(list)  # augments the list.  O(n)
    heappush(heap, int)   # O(log n)
    heappop(int)          # O(log n)
    heap[0]        # peak.

    # heapq is a min heap.
    # For max heap, negate input & output  "heappush(h, -val)" "-heappop(h)"

    # Small classes to wrap min/max heaps are good/clean way to deal with heaps.
    class maxheap:
        def __init__(self):
            self.h = []
            self.push = lambda x: heappush(self.h, -x)
            self.pop = lambda: -heappop(self.h)
            self.peak = lambda: -self.h[0]
            self.len = lambda: len(self.h)

    class minhheap:
        def __init__(self):
            self.h = []
            self.push = lambda x: heappush(self.h, x)
            self.pop = lambda: heappop(self.h)
            self.peak = lambda: self.h[0]
            self.len = lambda: len(self.h)

    def kth_min_max():
        from heapq import nsmallest, nlargest
        # kTh smallest : Keep a min heap of size k.  O(n log k)
        h, k = [], 3
        for i in [5, 2, 1, 7, 4, 2, 8, 10, 2]:
            if len(h) < k:
                heappush(h, i)
            else:
                if i > k:  heapreplace(h, k)
            print("3rd smallest: ", h[0])  # Ex: 65fdd808fcad43b2b2726062aeaa108d

        # Build in support for k-min/max in O(n log k) time:
        L = [1,2,3,2,5,6,8]
        nsmallest(2, L)  # ,key=lambda x:x) -> ]
        nlargest(2, L)   # ,key=lambda x:x) -> [8, 6]

        # Performance wise, you get O(n log k) as expected compared to sort:
        # L = rand_list(10000000)
        # timeit(lambda: sorted(L)[0:6], number=50)
        # 44.241248495000036
        # timeit(lambda: heapq.nsmallest(6, L), number=50)
        # 14.27249390999998
        # Time complexity, indirect uses nlargest ref: https://stackoverflow.com/questions/29240807/python-collections-counter-most-common-complexity/29240949


    def heapq_with_classes():
        # In general,
        import heapq
        class Person:
            def __init__(self, name, age, height):
                self.name = name
                self.age = age
                self.height = height

            def __lt__(self, other):
                return self.age < other.age  # set to >  for max heap.

            def __repr__(self):
                return "{} {} {}".format(self.name, self.age, self.height)


        people = [
            Person("Leo", 31, 168),
            Person("Valarie", 19, 157),
            Person("Jane", 20, 150),
            Person("Bob", 40, 170),
            Person("M.Jordon", 45, 210)
        ]

        print(heapq.nlargest(2, people, key=lambda x: x.height)[1].name)  # bob.

        heapq.heapify(people)
        while people:
            print(heapq.heappop(people))
            # Bob
            # Valarie 19 157
            # Jane 20 150
            # Leo 31 168
            # Bob 40 170
            # M.Jordon 45 210


def runtime_complexities_O(n):
    # Ref: Text
    # https://www.ics.uci.edu/~pattis/ICS-33/lectures/complexitypython.txt
    # Ref:
    # https://wiki.python.org/moin/TimeComplexity

    # Algorithm Cheat Sheet:
    # http://bigocheatsheet.com/

    # List:
        # index  l[i]        O(1)
        # store  l[i]=1      O(1)
        # len    len(l)      O(1)
        #        l.append(1) O(1)   *avrg/amortized.
        # sort   l.sort()    O(n)

    # Set:
        #      av    worste
        # add O(1)
        # in  O(1)  O(n)?
        # pop O(1)

    pass

def bit_wise_ops():
    # https://wiki.python.org/moin/BitwiseOperators
    # A << 1   # move bits left by 1.
    # B >> 1
    # A & B    A | B     A ^ B    ~A
    pass


def module_import():
    # Module import
    # import module_name
    # from module_name import name , .. | *
    pass

def asterisk_usage():
    # Has many useages.
    # * list, position args,
    # ** dist, keyword args,

    # in funcs, expand args to list/dict
    # in args to func, unpack list/dict :   myFunc(*[1,2,3])  -> myFunc(1,2,3)

    # in assignment, dynamically assign to list/dict

    # https://medium.com/understand-the-python/understanding-the-asterisk-of-python-8b9daaa4a558

    # Zip/Unzip used for unpackaging/packaging lists.
    list(zip([1, 2, 3], ["a", "b", "c"]))
    #[(1, 'a'), (2, 'b'), (3, 'c')]
    # Ex: https://www.hackerrank.com/challenges/zipped/problem
    pass


def stdin_console_input():
    # myvar = input()  # python 2: input expression.   python 3: input string. Use eval(..)
    # print (myvar) # 1 + 1 -> 2
    # Problem: https://www.hackerrank.com/challenges/input/problem

    # myvar2 = raw_input()  # Python 2: input string.  python 3: gone.
    # print (myvar2)  # 1 + 1 -> '1 + 1'

    # Splitting input into list/dic:
    # Problem: https://www.hackerrank.com/challenges/finding-the-percentage/problem

    pass

def Classes():
    # c88fb2fff01f444ca67d892a202078c3 -> cheatcheet

    class myClass:
        def __init__(self, val):
            self.val = val
        def __repr__(self):
            return self.val

    # Augment classes on the fly:
    mc = myClass("MyVal")
    hasattr(mc, "myAttr")  # False
    mc.myAttr = "hello"
    hasattr(mc, "myAttr")  # True
    # ref: https://stackoverflow.com/questions/610883/how-to-know-if-an-object-has-an-attribute-in-python

    def class_basic():
        # class MyClass:
        #     #someVar = 5  #Instance var sample.
        #
        # my = MyClass();
        # print (my.x)  # 5
        # my.x = 10
        # print (my.x) # 10
        pass


    class class_for_data:  # Example of how to use a class to store data. See also namedtuple
        def __init__(self, *arg_list, **kwargs):
            self.a = "a"
            self.b = "b"
            self.args = arg_list
            self.kwargs = kwargs

    # d = class_for_data(1,2,3, leo=":-)")
    # print d.a, d.b, d.args, d.kwargs


    def class_with_constructor():
        pass
        # # Class with constructor
        # class Person:
        #     def __init__(self, name, age):
        #         self.name = name
        #         self.age = age
        #
        #
        # p = Person("Leo", 31)
        # print (p.name)
        # print (p.age)

    def class_repr_tostr():
        # Proper printing of class when called with print str(myClass)
        class myClass:
            def __repr(self):
                return "tostr like"


    def class_inheritence():
        # Ref: https://www.w3schools.com/python/python_inheritance.asp
        class Car:
            def __init__(self, name):
                self.name = name

            def __hash__(self):
                return hash(self.name)

            def __eq__(self, other):
                return self.name == other.name


        class Audi(Car):
            def __init__(self, name, ring_count):
                super().__init__(name)               # Note, super() with brackets.
                self.ring_count = ring_count

            def __repr__(self):
                return "Audi(name={}, ring_count={}".format(self.name, self.ring_count)


        class BMW(Car):
            def __init__(self, name, coolness_factor):
                super().__init__(name)
                self.coolness_factor = coolness_factor

            def __repr__(self):
                return "BMW(name={}, coolness_factor={}".format(self.name, self.coolness_factor)


        a = Audi("A4", 5)
        a2 = Audi("A4", 3)
        b = BMW("Model 5", 42)

        s = set()
        s.add(a)
        s.add(a2)
        s.add(b)
        print(s)  # {BMW(name=Model 5, coolness_factor=42, Audi(name=A4, ring_count=5}


def hashing_of_classes():
    # Default Object implementation: (Pseudo code)
    # class object:
    #    def __eq___(self, other):
    #        return id(self) == id(other)   # Default __eq__ compares by id. Thus default class __eq__ only true on self-id comparison
    #    def __hash__(self):
    #        return hash(id(self)) / 16  # ish... Default __hash__ compares by hash value of id to guarantee uniqueness.

    # For dict/set to find things like d[a], it uses hash values. So the following must hold:
    # if a == b -> hash(a) == hash(b)

    # Thus if we override __eq__, we must also implement __hash__
    # Ref/Article: https://hynek.me/articles/hashes-and-equality/  # when hashing breaks...

    # Example:
    # Node class that is unique on val & color. But we can change visited state without breaking set/dict & comparison.
    # (Useful for graph traversals)
    class node:
        def __init__(self, val, color):
            self.val = val  # Immutable
            self.color = color  # Immutable
            self.visited = False  # Mutable

        def __repr__(self):
            return "Node: {} {} {}".format(self.val, self.color, self.visited)

        # eq Must be implemented for set()/dict to determine duplicates. Otherwise obj id used for eq. Obj id is mostly unique.
        # I.e otherwise node(1) == node(1) -> False.
        def __eq__(self, other):
            return self.val == other.val and self.color == other.color

        # Must be implemented if __eq__ is overridden so that property holds:   if a==b, hash(a)==hash(b)
        # (Otherwise you get "TypeError: unhashable type:" in python3.python2 allows)
        # hash value must not change in lifetime of object.
        def __hash__(self):
            return hash((self.val, self.color))  # Unique data in a tuple, since tuples are immutable.
            # Ref: https://stackoverflow.com/questions/2909106/whats-a-correct-and-good-way-to-implement-hash

    see = dataclases() # for similar implementation with annotation.

    a = node(1, "Red")
    b = node(2, "Blue")
    aa = node(1, "Red")
    aa.visited = True
    print("a == aa", a == aa)  # -> True.   (w/o eq/hash, -> False).
    s = set([a, b, aa])  # -> Node 1 & 2.   (w/o eq/hash, -> Node 1,1,2)
    print(s)


def dataclases():
    # Py >= 3.7
    from dataclasses import dataclass, field
    from typing import Any, List, Set, Dict, Deque, DefaultDict, Tuple   # for primitives, just use int/float/bool/str/bytes


    @dataclass()
    class Node:
        name: str      # must pass when creating. Node("mynode", set())
        edges: Set
        edges2: Set['Node']  # recursive structure reference. Auto-predict works better.
        generic: Any   # can be any field.

        # Default for primitives
        visited: bool = False  # type annotation with default value.
        name: str = "marry"   # w/ default.
        value = 1  # default value w/o type hints.

        # Default for objects must use default_factory=FUNCTION to initialize
        edges: Set = field(default_factory=set)  # set/list/dict.  [1]
        file_perm: List = field(default_factory=lambda: ['r', 'w', 'x'])

    # @dataclases() arg reference:
        eq=True # Generate __eq__ method.
        unsafe_hash=False  # Can be overriden to provide hashable func.
        frozen=False  # Immutable, adds hashfunction.
        order=False  # generate __lt__ __le__ __gt__ __ge__. Tuple made out of attributes for cmp.
        repr=True    # should __repr__ method be created?

        # ## Recursive self structure reference. (e.g node in linked list/Tree). -> Put into quotes.

    # Field reference:
    # field(
    #     default=        #If provided, sets default value
    #     default_factory=FUNCTION  IF provided, called during instantiation
    #     compare=True,   # should field be used in comparison?
    #     hash=None       # Default: inherits compare. True/False if hash to be computed.
    #     repr=True       # Referenced by __repr__ method?
    # )

    # Readings:
    # Basics/overview:
    # https://realpython.com/python-data-classes/
    #
    # Basics, comparisons, frozen, inheritence:
    # https://medium.com/mindorks/understanding-python-dataclasses-part-1-c3ccd4355c34
    # default_factory, excluding fields from comparison, exclude from representation, omit from init.
    # https://medium.com/mindorks/understanding-python-dataclasses-part-2-660ecc11c9b8

    # References
    #     https://docs.python.org/3/library/dataclasses.html
    # [1] https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
    #     https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    # ########################
    # ## For use in Sets/Dicts (making them hashable)
    # ########################
    # 1) Make Immutable
    @dataclass(frozen=True)  #-> Immutable == Hashable
    class node:
        val: int

    # 2) Override eq/hash manually.
    from dataclasses import dataclass
    @dataclass()
    class node:
        val: int
        color: int
        visited: bool = False

        def __eq__(self, other):
            return self.val == other.val and self.color == other.color

        def __hash__(self):
            return hash((self.val, self.color))

    # 3) Use unsafe_hash=True and compare=Flase
    # To hash based on some values, use unsafe_hash=True and manually exclude
    from dataclasses import dataclass, field
    @dataclass(unsafe_hash=True)
    class node:
        x: int
        visit_count: int = field(default=10, compare=False)  # hash inherits compare setting. So valid.
        # visit_count: int = field(default=False, hash=False)   # also valid. Arguably easier to read, but can break some compare code.
        # visit_count: int = False   # if mutated, hashing breaks. (3* printed)


    s = set()
    n = node(1)
    s.add(n)
    if n in s: print("1* n in s")
    n.visit_count = 11
    if n in s:
        print("2* n still in s")
    else:
        print("3* n is lost to the void because hashing broke.")


    # ########################
    # ## Post init actions:  (maybe use proper class at this point.
    # ########################
    from math import sqrt
    @dataclass()
    class point:
        x: int
        y: int

        def __post_init__(self):     # Override this guy.
            self.dist = sqrt(self.x ** 2 + self.y ** 2)
    # p = point(2, 2)
    # p.dist
    # #2.8284271247461903


def exception_handling():
    # # Exception handing
    try:
         1 / 0
    except (ValueError, ZeroDivisionError) as e:
        print(e)
    except Exception as e:  # Catch all exceptions.
        print(e)
        # raise Exception(e)  # to re-throw.
    except:   # Shorthand
        pass
    finally:
        print("Optionally, at the end.")

    raise ValueError("meh")

    # Ref: https://docs.python.org/2/tutorial/errors.html#defining-clean-up-actions
    # Prob: https://www.hackerrank.com/challenges/incorrect-regex/problem

def build_in_functions():
    type(str())  # -> type of obj. -> <type 'str'>
    isinstance("Hello", type(str()))

    # Any All
    any([True, False, False]) # Check if any item in list are true.
    all([True, True, True])   # They short-circuit: https://stackoverflow.com/questions/2580136/does-python-support-short-circuiting/14892812
    # Example: Check that all numbers are positive:
    all(i > 0 for i in [1,2,3,-9])

    # Example: Check if there is a string in a lis
    t = type(str(()))
    #print any(isinstance(item, t) for item in [ue,"mystring"])


    x = 1
    s = "string"
    abs()
    max()
    min()
    pow(2,3)  # 8

    round(1.123, 2) # 1.12
    sum()

    dict() # empty dict. d = dict()
    list() # empty list. l = list()
    tuple() # of items.

    float(x) # int or string to ..
    int(x)   # float or str  to ..
    str()

    len(s)   # number of items in sequence

    obj = list()
    id(obj) # memory addr of obj

    open("file") # for input     # see writing_and_reading_files

    list(range(x))       # [0, 1, 2, ..., x-1]
    list(range(2, 4))    # [2, 3]    #Mnemonic [...)
    list(range(0,6, 2))  # [0,2,4]  #increment by 2.
    list(range(4,-2,-2)) # [4,2,0]
    list(range(5, -1, -1)) # [5, 4, 3, 2, 1, 0]  # For (i=5;i>=1;--)


    import math
    print(math.floor(1.2))  # 1.0
    print(math.ceil(1.2))  # 1.0

def exiting_quitting():
    # Opt 1) Exit with status (usually 0-128)
    import sys
    sys.exit(127)

    # Opt 2) One-liner:
    raise SystemExit

def random():
    def random_choice():
        import random
        L = list(range(5))
        R = [random.choice(L) for _ in range(10)]  # -> [3,1,0,....]

        L = "abc"
        R = "".join(random.choice(L) for _ in range(10))  # -> "

    def random_numbers():
        import random
        five_random_numbers_between_0_10 = [random.randint(0, 10) for _ in range(5)]   # list comprehension.
        print(five_random_numbers_between_0_10)

        # more: https://realpython.com/python-random/]

        # Shuffling a list:
        L = [1,2,3]
        random.shuffle(L)  #-> L = [


def Regular_Expressions_regex():
    # Regex Tool: https://regex101.com/

    # Ref: https://docs.python.org/2/library/re.html
    # Ref: https://www.regular-expressions.info/python.html
    # Summary:
    # . = any
    # ^ = start      $ = end
    #
    # Quantifiers:
    # * = 0+   == {0,}
    # + = 1+   == {1,}
    # ? = 0 or 1  == {0,1}
    # {min,max}   instances of... {2} == 2 instances.  {2,}  means 2+,  ref: https://www.regular-expressions.info/repeat.html#limit

    # \w = alphanumeric(*1) and _    == [a-zA-Z0-9_]   # *1 includes '_', which is not alphanumeric.
    #                                                    Use isalnum() [doesn't contain '_'] or [a-zA-Z0-9] instead.
    #                                                    e.g prob: https://www.hackerrank.com/challenges/re-group-groups/problem

    # (?<=...) previous without consume  (?<!...) not previous
    # (?=...)  next without consume.     (?!...)  not next.

    # (\w+) group, e.g a word.
    # (...)\1  group & reference to group.
    # (P<GroupName)(P=GroupName)   group & named reference to group.

    def regex_func_split():
        import re
        #re.split(r"REG-EXPR", STR)
        re.split(r"@", "hello@world.com")     # ['hello', 'world.com']
        re.split(r"@|\.", "hello@world.com")  # ['hello', 'world', 'com']  # (@) or (escaped dot)
        # prob: https://www.hackerrank.com/challenges/re-split/problem

    def regex_func_search():
        # Find first *anywhere* in string.   (!) 're.match') is find from start
        import re
        r = re.search(r"ag[eo]", "the age of Ultron is the age of power")
        bool(r)
        r.group()   #-> age

        r.start()  #  4
        r.end()    #  7

        s = re.search(r'\w{2}', "helloworld")
        s.group() #-> 'he'

    def regex_func_match():
        # Match only from start of string    (!) 're.search' is find anywhere)
        # re.match(REGEX, PATTERN, [flags = re...]
        import re
        bool(re.match(r"hello", "Helloworld", flags=re.IGNORECASE))
        # See also grouping.

    def regex_func_findall_and_finditer():
        import re
        r2 = re.findall(r"a.?e", "the age of Ultron is the age of power", )
        # ['age', 'age']

        for w in re.finditer(r'\w+', "The age of Empires"):
            print(">" + w.group() + "<")  # if using a group in regex, reference group here also.
        #>The<
        #>age<
        #>of<
        #>Empires<

        # No easy way to tell if findIter() has no matches: https://stackoverflow.com/questions/56050201/in-python-how-to-if-finditer-has-no-matches
        import re
        matches, m = re.finditer("\w", "$$$%%%"), None
        for m in matches:
            print(m.group())
        if not m:
            print("Empty")

    def regex_func_substitution():
        import re
        s = "Hello || world || how are || you?"
        re.sub(re.escape(r" || "), "_", s)     # escapes: \|\|
        'Hello_world_how are_you?'
        # Problem: https://www.hackerrank.com/challenges/re-sub-regex-substitution/problem

        # To reference groups: \g<1>
        re.sub("(age) (of) (empires)",r"\g<3> \g<2> \g<1>", "age of empires")
        #'empires of age'


    def regex_pattern_syntax():
        import re
        bool(re.match(r"\+", "+bob"))  # -> True.   ## escape special symbols

        # {PATTERN}COUNT
        re.findall("x{3}", "Hello my xxx but not my xx and my xxxx")  # ['xxx', 'xxx']

        # [ABC] = sets, means A or B or C.  [a-z] means range.
        print(re.findall("[aieou]", "hello world. How are you?")) # Find vowewls.  -> ['e', 'o', 'o', 'o', 'a', 'e', 'o', 'u']
        print(re.findall(r"[a-e]", "hello world. How are you?"))  # Find a-e  -> ['e', 'd', 'a', 'e']
        # [+]	In sets, +, *, ., |, (), $,{} has no special meaning, so [+] means: return a match for any + character in the string
        print(re.findall(r"[+-.]bob", "+bob -bob .bob notbob")) # -> ['+bob', '-bob', '.bob']

        # Problem: Match floating: https://www.hackerrank.com/challenges/introduction-to-regex/problem
        # Ref: https://www.w3schools.com/python/python_regex.asp

    def regex_pattern_exact_match():
        import re
        # E.g you want to check if string is an exact match rather than partial match.
        bool(re.match("a{2}", "aaaa"))      #-> True
        bool(re.match("^a{2}$", "aaaa"))    #-> False
        # Problem: Phone number match: https://www.hackerrank.com/challenges/validating-the-phone-number/problem
        # Problem: Valid Credit Card: hackerrank_problem_Regex__validate_credit_card()

    def regex_pattern_greedy_vs_lazy():
        # Default regex is greedy (tries to match as much as possible). Use "?" to make it lazy:
        import re
        re.findall("<.*>", "<...>___>")
        ['<...>___>']
        re.findall("<.*?>", "<...>___>")
        ['<...>']
        # Ref: https://developers.google.com/edu/python/regular-expressions

    def regex_pattern_lookahead():
        """Without Look ahead/behind, matching string is consumed, which can interfer with multiple matches."""
        import re
        # (?<=...)  match only if preceedeb by ..     (?! ...) not by.   # Don't consume.
        # (?=...) match only if next item                                # Don't consume.
        re.findall("(?<=0)a(?=0)", "0a0a0xa0")
        ['a', 'a']
        # Problem: https://www.hackerrank.com/challenges/re-sub-regex-substitution/problem

    def regex_caseSensitivity_shorthand():
        import re
        re.findall("a(?i)", "Hai Alfred")  #(?i) shorthand for flag = i, == case insensitive. See re flags.
       # ['a', 'A']

    def regex_group():
        # Reference a matched group in a regex.
        import re
        m = re.match(r'(\w+)@(\w+)\.(\w+)', "user@domain.extension")
        m.group(0) # | group()   #  Entire match.
        'user@domain.extension'
        m.groups() # list of all matched groups.
        ('user', 'domain', 'extension')

        m.group(1)
        'user'
        m.group(2)
        'domain'
        m.group(3)
        'extension'
        m.group(2, 3)
        ('domain', 'extension')

        # Works for various regex functions, E.g for search:
        s = re.search(r'(@\w+).(\w+)', "Hello@domain.com")
        s.group()
        '@domain.com'
        s.groups(0)
        ('@domain', 'com')

        # groupdic() -> define key/value.
        # Problem, Tutorial: https://www.hackerrank.com/challenges/re-group-groups/problem
        import string

    def regex_group_selects():
        import re
        # Using a group can determine what is selected for output.
        # E.g, find numbers 1 or 2, consequitve 2+ times, surrounded by a or b's.
        re.findall(r'[ab]([12]{2,})[ab]', "aaaaa1111bbbb1222aaa")
        #['1111', '1222']
        # Problem: https://www.hackerrank.com/challenges/re-findall-re-finditer/problem

        for m in re.finditer(r'[ab]([12]{2,})[ab]', "aaaaa1111bbbb1222aaa"):
            print(m.group(1))
        #1111
        #1222

    def regex_group_backreference():
        import re
        # In general, you reference groups. Either by index(starting at 1), or by group name.

        # Numerical
        f = re.search(r'([a-z])\1', "%%% hellllo")   # Match two letters.
        f.group()
        # 'll'
            # Problem: https://www.hackerrank.com/challenges/re-group-groups/problem

        # Named Groups
        #P<groupName) to create group.  P=groupName to reference.
        f = re.search(r'(?P<groupName>\w)(?P=groupName)', "%%% hellllo booooob")
        f.group()
        #'ll'

    def reger_group_qunantifiers():
        # Quantifiers can be applied to groups for matching.
        import re
        # Ex 1:
        bool(re.match(r"^(a|b|c){3}$", "acb")) #-> True

        # Ex 2:
        matches = re.finditer(r"(a|b|c){3}(?i)", "zAAAzBBBzABCz")
        for m in matches:
            print(m.group())
        # AAA
        # BBB
        # ABC

    def regexObject__compile_pattern():
        # Mechanism:
        # - Compile pattern to regexObject.
        # - use regexObject.<func>(input_string)
        # Advantage: Functions have start(pos)/ endpos paramaters. Faster if pattern re-used.

        # Example:
        import re
        regexObject = re.compile("\w+")  # Pattern compiled to RegexObject.
        match = regexObject.search("The Age of Empire", 0)
        if not match:
            print("No match")
        while match:
            print(match.start(), match.end(), match.group())
            start_pos = match.end()
            match = regexObject.search("The Age of Empires", start_pos)

        # 0 3 The
        # 4 7 Age
        # 8 10 of
        # 11 18 Empires
        # https://www.hackerrank.com/challenges/re-start-re-end/problem

    def regex_pattern_escaping():
        regex_func_substitution()


def HackerRank_specific():
    """Code only relevant to Hacker Rank"""
    def input_line():
        # Read one line at a time. (Python 2)
        s = input()
        strings = input().split() #  "a b c" -> ['a','b','c']

    def input_ines():
        """
        Option 1: Read From sys.stdin. Each line has \n at the end.
                    Generally only useful if you want the full input as a single string.

        Option 2: Read via raw_input() (py2). Omits '\n' from return value.
                    Versatile, flexible, general purpose, jack-of-all-trades.
        """

        def option_1__sys_stdin():
            # ShortHand:   # Favorite: Short enough to memorize.
            import sys
            full_stdin = "".join([line for line in sys.stdin])

            # Then iterate over it:
            for line in full_stdin.splitlines():
                print(line)

            # Long Hand:
            import sys
            L = []
            for line in sys.stdin:
                L += [line]
            full_stdin = "".join(L)

        def option_2__raw_input():
            """If We're given line count:"""
            # Longhand:
            L = []
            for _ in range(int(input())):
                L += [input()]
            full_stdin = "\n".join(L)

            # Shorthand:
            full_stdin = "\n".join([input() for _ in range(int(input()))])

            """If not:"""
            try:
                while True:
                    print(input())
            except EOFError:
                pass


    # Method to (probably) determine if your code runs on HackerRank
    import os
    if "USER" not in os.environ:
        print("Running on HackerRank")
        raise SystemExit # so no else needed for non-Hacker Rank code below. Allows you to easily copy & paste code between IDE/HackerRank.

    # Repo with good solutions + explanations:
    # https://github.com/clfm/HackerRank



def IterTools____Cartesian_Permutation_combination(): # TAG_Iterators

    # 807293e9da914ce69ea32f34e5ffd1f7  (cheatsheet link)
    def itertools_isslice():
        # Iterate over a section of an array in a memory friendly way.
        from itertools import islice
        L = [1,2,3,4]
        for i in islice(L, 1, 3):  # itertools.isslice(Iter, [start], stop [, step])
            print(i)  # 2, 3

    def itertools_chain():
        # Iterate over connected iterables in a memory friendly way   chain(*args)
        from itertools import chain
        L1 = [1,2,3]
        L2 = [4,5,6]
        L3 = [7,8,9]
        for i in chain(L1, L2, L3):
            print(i)  # 1 ... 9

    from itertools import combinations, permutations
    def cartesian_product():
        from itertools import product
        list(product([1,2], ["a", "b"])) # [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')] # https://www.hackerrank.com/challenges/itertools-product/problem

    def premutations():
        from itertools import permutations
        # permutations(<ITER>, select_count)
        list(permutations("HACK", 2)) # [('H', 'A'), ('H', 'C'),  .... https://www.hackerrank.com/challenges/itertools-permutations/problem

    def combinations():
        from itertools import combinations
        list(combinations("ABC", 2))
        # [('A', 'B'), ('A', 'C'), ('B', 'C')]

    def combinations_vs_permutations():
        # Permutaiton order matters.   (mnemonic: 1) pad lock is permutation lock. Lottery is permutation.
        # Combination order doesn't matter.
        from itertools import combinations, permutations
        list(combinations("ABC", 2))
        # [('A', 'B'), ('A', 'C'), ('B', 'C')]
        list(permutations("ABC", 2))  # n!
        # [('A', 'B'), ('B', 'A'),  ...  ('A', 'C'),('B', 'C'), ('C', 'A'), ('C', 'B')]
        # https://betterexplained.com/articles/easy-permutations-and-combinations/

    def combination_with_replacement():
        # Combinations with Replacement:  https://www.hackerrank.com/challenges/itertools-combinations-with-replacement/problem
        from itertools import combinations_with_replacement
        list(combinations_with_replacement([1, 2], 2))
        #[(1, 1), (1, 2), (2, 2)]
        list(combinations([1, 2], 2))
        #[(1, 2)]

    from itertools import groupby
    # GroupBy           # https://www.hackerrank.com/challenges/compress-the-string/problem
    for i in groupby("99995599"):
        print((int(i[0]), len(list(i[1]))))

    #(9, 4)   9 occured 4 times...
    #(5, 2)
    #(9, 2)

    # Useful iterators to reduce  in loops:
    # https://docs.python.org/2/library/itertools.html#itertools.imap

def Misc__things_learned():

    def none_comparison_via_is():
        obj = None
        if obj == None:
            print("Incorrect")

        if obj is None:  #is not
            print("Correct")

        # is  -> Object Identity
        # ==  Equality between two objects   #Height of two friends.
        # 'is' is a bit faster since '==' does a dictionary lookup.
        # Ref: http://jaredgrubb.blogspot.com/2009/04/python-is-none-vs-none.html

        # E.g Problem: https://www.hackerrank.com/challenges/find-the-merge-point-of-two-joined-linked-lists/problem?h_l=interview&playlist_slugs%5B%5D%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D%5B%5D=linked-lists


    def gotcha_default_args_evaluated_only_once():
        # https://docs.python-guide.org/writing/gotchas/

        # ex:
        def caching_square_func(i, cached=dict()):
            if i in cached:
                print(i, "already cached.")
                return cached[i]
            else:
                print(i, "will be cached")
                isqr = i * i
                cached[i] = isqr
                return isqr

        # for i in [2,3,2,3]:
        #    caching_square_func(i)


    def gotcha_late_binding_closure():
        i = 5
        f = lambda x: x + i   # i is 'read/evaluated' at function call time rather than define time.
        i = 10
        f(1)
        #11

        # Hack/solution. Bind arg at def time:
        j = 5
        g = lambda x, j=j: x + j  # << j=j
        j = 10
        g(1)
        # 6

    # More advanced: https://docs.python-guide.org/writing/gotchas/

    def evaluated_first_assigned_after():
        a = 1
        b = 2
        c = 3
        a, b, c = b, c, 4
        a, b, c
        #(2, 3, 4)



    def benchmarking_and_timing_and_memory_profiling():
        #### Manual:
        from time import time
        t0 = time()
        # < do something.
        print(('%10s %5.3f seconds' % ("test", time() - t0))) # elapsed


        #### Via function
        import timeit
        def func():
            max([1, 2, 3])  # code to benchmark

        timeit.timeit(func, number=10000)  # test it 10,000 times. Default number is 1000000

        # from memory_profiler import profile  # add '@profile' to method and run.
        # cd ~/git/study/python2 && mprof run problemPy2.py && mprof plot

    def recursion_limited_to_991():

        # Python's recursion is limited to under 1000 items. Thus use data structures instead.
        def goDeep(x):
            print(x)
            goDeep(x + 1)
        goDeep(1)

        #989
        #990
        #991
        #..
        # RuntimeError: maximum recursion depth exceeded



class COLLECTIONS():
    def Counter(self):
        # Count items in an iterable.
        #   Sorted by count & order encountered.
        from collections import Counter #note capital C
        c = Counter(['a', 'a', 'a', 'b', 'a', 'b', 'c']) # -> Counter({'a': 4, 'b': 2, 'c': 1})  #key, value
        list(c.items())  # [('a', 4), ('c', 1), ('b', 2)]
        list(c)   # get a list of unique keys.  ['a,'b','c']


        list(c.keys())    # if X in c.keys()
        list(c.values())  # less useful.
        c.update([9,8]) # add keys & count them to counter.
        c.most_common()[:1] # get X number of most common keys. O(n log k)  [:1] = most common. [-1:] least common.
        # complexity ref: https://stackoverflow.com/questions/29240807/python-collections-counter-most-common-complexity/29240949
        c.clear()
        c[1] -= 1  # dercement index 1 by 1.
        c[1]  # print value of 1.
        del c[1]
        # prob (easy): https://www.hackerrank.com/challenges/collections-counter/problem
        # prob (medium): https://www.hackerrank.com/challenges/most-commons/problem
        # ref: https://docs.python.org/2/library/collections.html#collections.Counter


    def defaultdic(self):
        # TAG_optional
        # Like dictionary, except provides default value if item not in list.
        from collections import defaultdict
        # d = defaultdict(CALLABLE)   Ex: lambda:-1, int, list, set,   # int -> 0

        # E.g 1:
        seen = defaultdict(lambda: 0)
        seen['yellow'] += 1

        # E.g 2:
        d = defaultdict(list)
        d['key'].append(1)
        d['key'].append(2)
        d['leo'].append(42)
        list(d.items())

        #[('key', [1, 2]), ('leo', [42])]
        # Problem: https://www.hackerrank.com/challenges/defaultdict-tutorial/problem
        # Ref: https://docs.python.org/2/library/collections.html#collections.Counter


    def namedtuple(self):
        # Useful for storing named values.
        from collections import namedtuple
        Point = namedtuple('Point', ['x', 'y'])
        p = Point(1, y=3)  # pass args via args or kwargs
        print(p.x, p.y)  # 1 # 3

        Person = namedtuple("Person", 'name,age')  # comma separated.
        leo = Person("leo", 31)
        # Person(name='leo', age=31)

        Car = namedtuple("Car", "model age  mph") # space separated.
        # ..
        # Problem: https://www.hackerrank.com/challenges/py-collections-namedtuple/problem

    def OrderedDic(self):
        # Ordered Dictionary:
        from collections import OrderedDict
        od = OrderedDict()
        od[2] = 'a'
        od[1] = 'b'
        od[3] = 'c'
        # -> OrderedDict([(2, 'a'), (1, 'b'), (3, 'c')])
        list(od)  # [(2, 'a'), (1, 'b'), (3, 'c')]
        od.move_to_end(1,last=False)  # last=False moves to front.
        next(iter(od.items()))  # first item.
        od.popitem(last=False)  # pop first/last item.



    def OrderedCounter_custom(self):
        # Combination of Counter and OrderedDict
        # Properties:d
        # - Provides count of each key.
        # - remembers order in which keys were added.
        # - application: consider pre-sorting input.
        from collections import Counter, OrderedDict
        class OrderedCounter(Counter, OrderedDict):
            pass

        o = OrderedCounter(sorted("zzaabbbiiiiiiiiiiff"))
        # OrderedCounter({'i': 10, 'b': 3, 'a': 2, 'f': 2, 'z': 2})
        o.most_common()[:3]
        # [('i', 10), ('b', 3), ('a', 2)]
        # Prob: https://www.hackerrank.com/challenges/most-commons/problem
        # ref:  https://www.hackerrank.com/challenges/most-commons/forum/comments/220882
        # ref2: https://codefisher.org/catch/blog/2015/06/16/how-create-ordered-counter-class-python/

    def dequeue(self):
        # Double sided queue. Efficient left/right append/poping.
        from collections import deque
        d = deque([1,2,3])
        d.append(4)
        d.appendleft(0)
        d.pop()
        d.popleft()
        len(d) # ..
        # Prob (easy):   https://www.hackerrank.com/challenges/py-collections-deque/problem
        # Prob (medium): https://www.hackerrank.com/challenges/piling-up/problem

############## RAW
class Algorithms_Sort:
    def BubbleSort(self):

        # Leo's Attempt:   (If in doubt, can re-implement)
        # Mnemonic: Keep moving items till no swap occured anymore.
        # 3 4 2 1
        # 3 2 4 1
        # 2 3 4 1
        # 2 3 1 4
        # 2 1 3 4
        # 1 2 3 4
        def leo_bubble_sort(l):
            swap_performed = True
            while swap_performed:
                swap_performed = False
                for i in range(len(l) - 1):
                    if l[i] > l[i + 1]:
                        l[i], l[i + 1] = l[i + 1], l[i]
                        swap_performed = True
            return l

        # Better approach
        # Mnemonic:
        #   Bubble sort "puts" elements gradually into their place.
        #   After every loop, last element(s) are their place, no need to revisit them.
        # 3 4 2 1

        # 3 2 4 1   # Loop 1. Last item is in it's place.
        # 3 2 1 4

        # 3 1 2 4   # Loop 2. Last 2 item in it's place.

        # 1 3 2 4   # Loop 3. Last 3 items in it's place.
        # 1 2 3 4
        def bubble_sort(l):
            for i in range(len(l) - 1):
                for j in range(len(l) - 1 - i):
                    if l[j] > l[j + 1]:
                        l[j], l[j + 1] = l[j + 1], l[j]
            return l
        # Ref tutorial: https://www.youtube.com/watch?v=YHm_4bVOe1s
        # Ref code: https://github.com/joeyajames/Python/blob/master/Sorting%20Algorithms/bubble_sort.py

def with_statement():
    # Used to kinda like try/finally, to do setup op and teardown op.
    # https://effbot.org/zone/python-with-statement.htm
    with open("x.txt") as f:
        data = f.read()
        #do something with data
        # (To read/write files, see
        writing_and_reading_files()

    class open:
        def __enter__(self):  # Called at start of with.
            # ...
            return 123  # something to with for use by 'f' var.
        def __exit___(self):  # Called at the end of with.
            pass

def writing_and_reading_files():
    # w = write
    # r = read
    # a = append. If file doesn't exist, creates it.
    # + = create if doesn't exist. Open file for reading and writing(updating)
    #--- Also:
    # x - Creates new file. If file already exists, the op fails.
    # t - text mode (default)
    # b - binary mode.

    # E.g write to file, creating it if it doesn't exist.
    with open("file.txt", "w+") as f:
        f.write("hello\n")

    # E.g read file content:
    f = open("file.txt", "r")
    if f.mode == "r":
        contents = f.read()
    f.close()

    # read line by line
    f = open("text.txt", "r")
    f_lines = f.readlines() # -> List of lines.

    # Ref: https://www.guru99.com/reading-and-writing-files-in-python.html

def Language__meta():
    def removing_recusion_limit():
        # Normally recursion has a limit of ~1000. To override:

        import sys
        sys.setrecursionlimit(5000)

        # test with:
        def deep(i):
            print(i)
            deep(i+1)



def date_and_time():
    # Date:
    from datetime import date, timedelta
    some_day = date(2019, 7, 7)
    today = date.today()
    current_week_number = date.today().isocalendar()[1]  # I think uses system first day of week.

    # Deltas:
    yesterday = today - timedelta(days=1)  # ref: http://www.pressthered.com/adding_dates_and_times_in_python/
    last_year = date(today.year - 1, today.month, today.day)


    # Time:
    import datetime
    curr_time = datetime.datetime.now()

def SOMEDAY_MAYBE():
    pass

def INBOX():
    pass

def dectorators():
    pass
    # https://www.datacamp.com/community/tutorials/decorators-python
    # http://simeonfranklin.com/blog/2012/jul/1/python-decorators-in-12-steps/
    # Hacker Rank -> python -> Closures and decorators.

    def debug_decorator():
        # Add decorator to function to get debug info when it's called.
        # ref: https://realpython.com/primer-on-python-decorators/
        import functools
        def debug(func):
            """Print the function signature and return value"""
            @functools.wraps(func)
            def wrapper_debug(*args, **kwargs):
                args_repr = [repr(a) for a in args]                      # 1
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
                signature = ", ".join(args_repr + kwargs_repr)           # 3
                print(f"Calling {func.__name__}({signature})")
                value = func(*args, **kwargs)
                print(f"{func.__name__!r} returned {value!r}")           # 4
                return value
            return wrapper_debug

        @debug
        def myFunc(a,b,c):   # When called, prints: Calling myFunc(1,2,3) .... 'myFunc' returned -1
            return -1


def caching():
    see = function_caching()

def function_caching():
    see = caching()
    # 9781debae32d4592bd6379f2a5c1e068
    from functools import lru_cache  # Least recently used.
    @lru_cache(maxsize=None)  # w/o cache: 10.625341928 sec w/ cache: 0.051493941 sec
    def n_fact(n):
        if n == 1:
            return 1
        else:
            return n * n_fact(n - 1)

    import timeit
    import sys
    sys.setrecursionlimit(20000)
    print(timeit.timeit(lambda: n_fact(9000), number=500))

def Memory_profiling():
    # Doc
    # https://pypi.org/project/memory-profiler/

    # *) Install:
    # pip install -U memory_profiler

    # *) Print line-by line memory information of a function:
    # from memory_profiler import profile

    # @profile
    # def func():
    #     pass

    # *) Daw a memory Plot
    # cd ~/git/study/python2 && mprof run problemPy2.py && mprof plot
    pass

def LinkedList():
    # Most basic:
    class LikedList:
        def __init__(self):
            self.head = None

    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None

def SortedContainers():
    # Python doesn't have sorted containers :-(. But has heaps for tracking min/max.
    # Implementations are available via pip thou. E.g:
    # http://www.grantjenks.com/docs/sortedcontainers/
    # blist also has some sortedList implementations:
    # http://stutzbachenterprises.com/blist/

    # SortedDict - lg(n) for insert. O(1) for retrieval.

    pass

def LIB__REQUESTS_http_REST_consume_and_call():
    # todo_someday - consolidate note sets 1 & 2.
    def NOTE__SET1():
        def requests__get_data():
            # Sample json api's to query:
            # https://jsonplaceholder.typicode.com/guide.html

            # pip install requests   # using pip3, check pip --version
            import json
            import requests
            import base64

            '''
            requests.
                get(url, params=None, **kwargs):    e.g arg: arg auth_values
                post(url, data=None, json=None, **kwargs):
                put(url, data=None, **kwargs):
                delete(url, **kwargs):
                patch(url, data=None, **kwargs):
    
                Security:
                    user = "mike.preston@rubrik.us"
                    passwd = "SuperSecret"
                    auth_values = (user, passwd)
            '''

            # Example 1: JsonDict { ... } to python dict:
            # https://jsonplaceholder.typicode.com/posts/1
            # {
            #   "userId": 1,
            #   "id": 1,
            #   "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",
            #   "body": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto"
            # }
            data_json = requests.get("https://jsonplaceholder.typicode.com/posts/1").json()
            # data_json -> {dict}
            print(data_json["userId"], data_json["id"], data_json["title"], data_json["title"])

            # Example 2: Json List of Dict's -> python dict:
            # https://jsonplaceholder.typicode.com/users/1/todos
            # [
            #   {
            #     "userId": 1,
            #     "id": 1,
            #     "title": "delectus aut autem",
            #     "completed": false
            #   },
            #   {
            #     "userId": 1,
            #     "id": 2,
            #     "title": "quis ut nam facilis et officia qui",
            #     "completed": false
            #   },
            # ...

            data_json = requests.get("https://jsonplaceholder.typicode.com/users/1/todos").json()
            entry0 = data_json[0]
            print(entry0["id"], entry0["completed"], entry0["title"])

            # Checking for valid response:
            resp = data_json = requests.get("https://jsonplaceholder.typicode.com/users/1/todos")
            if resp.status_code != 200:
                print("error")
            else:
                print("no error")

            # Sources and Further readings:
            # https://realpython.com/api-integration-in-python/



            ## OTHER:
            # - Parsing: Beautiful soup
            # - Requests-html (newer request)

        def request__post_put_data():
            # Place to test post calls:
            # http://httpbin.org/#/HTTP_Methods/post_post
            # https://docs.postman-echo.com/?version=latest
            # See also for more: https://stackoverflow.com/questions/5725430/http-test-server-accepting-get-post-requests

            import requests
            # http://httpbin.org/#/HTTP_Methods/post_post:
            rp = requests.post("http://httpbin.org/post", data={"hello": "world"})
            rp.json()
            # {'args': {}, 'data': '', 'files': {}, 'form': {'hello': 'world'}, 'headers': {'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate', 'Content-Length': '11', 'Content-Type': 'application/x-www-form-urlencoded', 'Host': 'httpbin.org', 'User-Agent': 'python-requests/2.22.0'}, 'json': None, 'origin': '99.239.145.120, 99.239.145.120', 'url': 'https://httpbin.org/post'}

    def NOTE__SET1():
        # pip install requests
        import requests
        import json
        # tutorial: https://www.youtube.com/watch?v=tb8gHvYlCFs&t=120s

        def read_text():
            r = requests.get("https://xkcd.com/353/")
            # dir(r) # list of methods
            # help(r) # help about Response object

            r.text  # raw text returned by object.

            # How to get image?
            # https://imgs.xkcd.com/comics/python.png

        def read_binary():
            r = requests.get("https://imgs.xkcd.com/comics/python.png")
            # r_image.content  # b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x0 ...

            r.status_code  # 200 = OK.
            r.ok  # True/False
            r.headers  # {'Server': 'nginx', 'Content-Type': 'image/png', 'Last-Modified': 'Mon, 01 Feb 2010 13:07:49 GMT', 'ETag': '"4b66d225-162d3"', 'Expires': 'Mon, 02 Sep 2019 23:16:36 GMT'
            with open('comic.png', 'wb') as f:  # wb = write binary
                f.write(r.content)  # comic.png contains our commic.

            # http://httpbin.org/#/   < test queries. Written by the guy who wrote requests library.

        def get_stuff():
            # Instead of appending args to URL  .....org/get?page=2&count=25 , we pass dictonary:
            payload = {'page': 2, 'count': 25}
            r = requests.get("https://httpbin.org/get", params=payload)
            r.text
            # {
            #   "args": {
            #     "count": "25",
            #     "page": "2"
            #   }, .....

        def post_stuff():
            payload = {'username': 'corey', 'password': 'testing'}
            r = requests.post("https://httpbin.org/post", data=payload)  # or put.
            r.text
            # { ...
            #   "form": {
            #     "password": "testing",
            #     "username": "corey"
            #   },
            r_dict = r.json()  # convert to python dictonary.

        def authenticate():
            # http://httpbin.org/basic-auth/lufimtse/wetter      user/pass prompt.
            pass
            r = requests.get("http://httpbin.org/basic-auth/lufimtse/wetter", auth=("lufimtse", 'wetter'))  # timeout=3   (seconds)   ..org/delay/5
            r.text  # '{\n  "authenticated": true, \n  "user": "lufimtse"\n}\n'  or '' if wrong creds.
            r  # 401 invalid. 200 valid.

        def stream_lines():
            # steaming continious data.
            r = requests.get('https://httpbin.org/stream/20', stream=True)
            for line in r.iter_lines():
                # filter out keep-alive new lines
                if line:
                    decoded_line = line.decode('utf-8')
                    pydict = json.loads(decoded_line)
                    print(str(pydict['id']) + " ", end="")
                    # 0, 1, 2 ... 19
            print()
            # Ref: https://2.python-requests.org/en/master/user/advanced/

        def stream_binary_or_lines():
            uri = "http://stream.meetup.com/2/rsvps"
            response = requests.get(uri, stream=True)

            # -- Read 50 bytes at a time.
            # for chunk in response.iter_content(chunk_size=50):
            #     if chunk:
            #         print(chunk)

            # -- Read a line at a time.
            # for chunk in response.iter_lines():
            #     # print(str(chunk))
            #     if chunk:
            #         rsvp = json.loads(chunk)
            #         print(rsvp)

            # -- Read a single character at a time. Split by '\n' new line char.
            buffer = ""
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                # print(chunk)
                if str(chunk)[-1] == '\n':
                    buffer += str(chunk)
                    pydict = json.loads(buffer)  #
                    print(str("rsvp_id: ") + str(pydict["rsvp_id"]))  # rsvp_id: 1804738166
                    buffer = ""
                else:
                    buffer += str(chunk)
            # rsvp_id: 1804738188
            # rsvp_id: 1804738190 ...
            # src: https://markhneedham.com/blog/2015/11/28/python-parsing-a-json-http-chunking-stream/

        # Further tutorials:
        # BeautifulSoup Tutorial - https://youtu.be/ng2o98k983k  # Parse HTML
        # File Objects Tutorial - https://youtu.be/Uh2ebFW8OYM

