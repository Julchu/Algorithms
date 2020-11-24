# typing module needed for dictionaries (Lists)
from typing import List

[1, n][2, n][3, n][4, n][5, None]

prev = head [1, n]
n = head.next [2, n]
prev.next = None [1, None]

Loop:
    head = n [2, n]
    n = n.next [3, n]
    head.n = prev [2, n][1, None]
    prev = head [2, n]

Loop:
    head = n [3, n]
    n = n.next [4, n]
    head.next = prev [3, n][2, n][1, None]
    prev = head [3, n]

Loop:
    head = n [4, n]
    n = n.next[5, n]
    head.next = prev [4, n][3, n][2, n][1, None]
    prev = head [4, n]

