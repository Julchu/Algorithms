# typing module needed for dictionaries (Lists)
from collections import defaultdict, deque
from typing import List

class ListNode:
    def __init__(self):
        self.inDegrees = 0
        self.next = []

numCourse = 7
prerequisites = [[1,0],[0,3],[0,2],[3,2],[2,5],[4,5],[5,6],[2,4]]

values = []
nodes = {}
valuesExists = {}
totalEdges = 0
for list in prerequisites:
    current, prereq = list[0], list[1]
    if current not in values:
        values.append(current)
        valuesExists[current] = True
    if current in nodes:
        nodes[current].next.append(prereq)
    else:
        nodes[current] = ListNode()
        nodes[current].next.append(prereq)
    if prereq in nodes:
            nodes[prereq].inDegrees += 1
    else:
        nodes[prereq] = ListNode()
        nodes[prereq].inDegrees += 1
    totalEdges += 1

noReq = []
for value in values:
    if nodes[value].inDegrees == 0:
        noReq.append(value)
removedEdges = 0
while noReq:
    course = noReq.pop()
    for prereq in nodes[course].next:
        nodes[prereq].inDegrees -= 1
        removedEdges += 1
        if nodes[prereq].inDegrees == 0:
            noReq.append(prereq)

if removedEdges == totalEdges:
    # return True
    print(True)
else:
    # return False
    print(False)
