# typing module needed for dictionaries (Lists)
from typing import List

class ListNode:
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next

def reverseList(head: ListNode) -> ListNode:
	stack = []
	while head:
		stack.append(head)
		head = head.next
	
	newHead = stack.pop()
	returnHead = newHead
	while stack:
		next = stack.pop()
		newHead.next = next
		newHead = newHead.next
	newHead.next = None

	return returnHead

n4 = ListNode(2, None)
n3 = ListNode(8, n4)
n2 = ListNode(5, n3)
n1 = ListNode(3, n2)
head = ListNode(1, n1)

print(reverseList(head))