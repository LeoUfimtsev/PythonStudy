# Add this: Settings -> build -> console -> Python Console -> Start script:
# from console_helpers import *

from datetime import date
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter, OrderedDict
import random
import string
import heapq
from timeit import timeit
import requests

def line_split_to_str_list(line):
    """E.g: 1 2 3 -> [1, 2, 3]   Useful to convert numbers into a python list."""
    return "[" + ", ".join(line.split()) + "]"

def line_split_to_str_list(line):
    """E.g: 1 2 3 -> [1, 2, 3]   Useful to convert numbers into a python list."""
    return ("[" + ", ".join(line.split()) + "]")

def lprint(L):
    """List print. Useful for when items are very large."""
    print("List: ")
    for i in L:
        print("  " + str(i))

# Commonly used aliases
t = True
f = False
p = print


def rand_list(n=10):
    return [random.randint(0,10) for _ in range(n)]

def prime_generator():  # pg = prime_generator(); next(pg)
    """First 100 primes are retrieved, subsequent once are computed"""
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