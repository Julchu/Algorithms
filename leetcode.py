# 153: my attempt
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return
        elif len(nums) == 1:
            return nums[0]
        else:
            left = 0
            right = len(nums)-1
            return self.findMinAux(nums, left, right)
    
    def findMinAux(self, nums: List[int], left, right) -> int:
        if nums[left] <= nums[right]:
            return nums[left]
        else:
            mid = (left + right) // 2
            left_min = self.findMinAux(nums, left, mid)
            right_min = self.findMinAux(nums, mid+1, right)
            
            if left_min <= right_min:
                return left_min
            else:
                return right_min

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
    def reverseList(self, head: ListNode) -> ListNode:
        if head:
            prev = head
            n = head.next
            prev.next = None

            while n is not None:
                head = n
                n = n.next
                head.next = prev
                prev = head
            return head