# typing module needed for dictionaries (Lists)
from typing import List

class ListNode:
    def __init__(self, val, next=None, visited=False):
        self.val = val
        self.next = [next]
        self.visited = visited

    def __str__(self):
        return "val: " + str(self.val) + " next: " + str(self.next) + " visited: " + str(self.visited)

numCourses = 7
lists = [[1, 0], [0, 3], [0, 2], [3, 2], [2, 5], [4, 5], [5, 6], [2, 4]]
print()

# def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
values = []
nodes = {}

for list in lists:
    if list[0] in nodes:
        nodes[list[0]].next.append(list[1])
    else:
        values.append(list[0])
        nodes[list[0]] = ListNode(list[0], list[1])

for list in lists:
    stack = []
    stack.append(nodes[list[0]])
    while stack:
        node = stack.pop()
        print(node)
        if node.visited:
            print("")
            break
        node.visited = True
        for n in node.next:
            if n in nodes:
                stack.append(nodes[n])
    for value in values:
        # print(nodes[value])
        nodes[value].visited = False
        # print(nodes[value])
# return True

def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        from collections import defaultdict, deque
        # key: index of node; value: GNode
        graph = defaultdict(GNode)

        totalDeps = 0
        for relation in prerequisites:
            nextCourse, prevCourse = relation[0], relation[1]
            graph[prevCourse].outNodes.append(nextCourse)
            graph[nextCourse].inDegrees += 1
            totalDeps += 1

        # we start from courses that have no prerequisites.
        # we could use either set, stack or queue to keep track of courses with no dependence.
        nodepCourses = deque()
        for index, node in graph.items():
            if node.inDegrees == 0:
                nodepCourses.append(index)

        removedEdges = 0
        while nodepCourses:
            # pop out course without dependency
            course = nodepCourses.pop()

            # remove its outgoing edges one by one
            for nextCourse in graph[course].outNodes:
                graph[nextCourse].inDegrees -= 1
                removedEdges += 1
                # while removing edges, we might discover new courses with prerequisites removed, i.e. new courses without prerequisites.
                if graph[nextCourse].inDegrees == 0:
                    nodepCourses.append(nextCourse)

        if removedEdges == totalDeps:
            return True
        else:
            # if there are still some edges left, then there exist some cycles
            # Due to the dead-lock (dependencies), we cannot remove the cyclic edges
            return False