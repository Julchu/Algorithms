# typing module needed for dictionaries (Lists)
from typing import List

class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

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

'''
[]

[[]]

[[], [], []]

[[1], [], []]

[[1, 2], [2], []]
'''

# l6 = ListNode(6, None)
l6 = None

l5 = ListNode(5, None)
l4 = ListNode(4, l5)

l3 = ListNode(3, None)
l2 = ListNode(2, l3)
l1 = ListNode(1, l2)

lists = [l1, l3, l6]

# for list in lists:
# 	if list:
# 		if list.val:
# 			print(list.val)

for list in lists:
	print(list)