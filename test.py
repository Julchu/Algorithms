# typing module needed for dictionaries (Lists)
from typing import List

# class Node:
#     def __init__(self, value, next=None):
#         self.value = value
#         self.next = next

# def reverseList(self, head: ListNode) -> ListNode:
#     if head:
#         prev = head
#         n = head.next
#         prev.next = None

#         while n is not None:
#             head = n
#             n = n.next
#             head.next = prev
#             prev = head
#         return head

# node5 = Node(5)
# node4 = Node(4, node5)
# node3 = Node(3, node4)
# node2 = Node(2, node3)
# node1 = Node(1, node2)
# print(node5.next.val)
# reverseList(node1)
# print(node5.next.val)


# def travel(node: Node) -> Node:
#     prev = node
#     node = node.next
#     aux(node, prev)
#     prev.next = None

# def aux(node: Node, prev: Node) -> Node:
#     if node is None:
#         return
#     else:
#         node.next = prev
#         aux(prev, prev.next)
        

# travel(node1)

# a = {"a": 1, "c": 2}

# letter = "b"
# if letter in a:
#     a[letter] += 1
# else: 
#     a[letter] = 1


# a.pop("c")
# print(a)