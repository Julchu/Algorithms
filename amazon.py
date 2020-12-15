from typing import List
import collections
import math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

class Solution:
    """
    Online Assessment Q1
    Old IDs are palindromes
    New IDs are the same as old ID but breaking palindrome, and lowest alphabetical order

    Ex: "aabbaa" -> "aaabaa"
    Ex: "bab" -> "aab"
    
    Test: 
        s = "bab"
        breakPalindrome(s)
    """
    def breakPalindrome(self, oldID: int) -> int:
        newID = ""
        length = len(oldID)
        mid = length // 2
        allAs = True
        done = False
        index = 0
        
        while index <= mid:
            if not done and oldID[index] != "a":
                newID += "a"
                allAs = False
                done = True
            else: 
                newID += oldID[index]
            
            index += 1
        while index < length:
            newID += oldID[index]
            index += 1

        if allAs:
            newID = "IMPOSSIBLE"

        return newID

    """
    Online Assessment Q2
    Find shortest Euclidian distance between robot positions
    numRobots: number of robots
    position X and position Y: equal length arrays of X, Y positions
    Euclidian distance: (X1-X2)^2 + (Y1-Y2)^2

    Example inputs: 
        numRobots = 5
        positionX = [0, 10, 15, 20, 25]
        positionY = [0, 10, 15, 20, 25]
    
    Test: 
        closestSquaredDistance(numRobots, positionX, positionY)
    """
    def closestSquaredDistance(self, numRobots, positionX, positionY) -> int:
        possibleCombinations = int(math.comb(numRobots, 2))
        
        count = 0
        left = 0
        right = 1

        shortestDistance = (positionX[right] - positionX[left])**2 + (positionY[right] - positionY[left])**2
        
        while count < possibleCombinations:

            euclid = (positionX[right] - positionX[left])**2 + (positionY[right] - positionY[left])**2
            if euclid < shortestDistance:
                shortestDistance = euclid

            right += 1
            if right == numRobots:
                left += 1
                right = left + 1
            
            count += 1
            
        return shortestDistance
    """
    Arrays and Strings
    """
    # 1. Two Sum problem (EASY)
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numsSet = {}
        for index in range(len(nums)):
            numsSet[nums[index]] = index
        for index in range(len(nums)):
            difference = target - nums[index]
            if difference in numsSet and index is not numsSet[difference]:
                return [index, numsSet[difference]]
          
    # 3. Longest Substring Without Repeating Characters (MED)
    def lengthOfLongestSubstring(self, s: str) -> int:
        maxLength = 0
        letters = {}
        left = 0
        right = 0
        length = len(s)

        while right < length:
            letter = s[right]

            if letter in letters:
                maxLength = max(maxLength, right - left)
                left = max(left, letters[letter] + 1)
                
            letters[letter] = right
            right += 1
        
        maxLength = max(maxLength, right - left)
        return maxLength

    #11. Container With Most Water (MED)
        # Official solution
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        maxArea = 0

        while left != right:
            area = min(height[left], height[right]) * (right - left)
            if area > maxArea:
                maxArea = area

            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return maxArea

    # 49. Group Anagrams (MED)
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        strsDict = {}
        groupCombined = []
        groups = []

        for str in strs:
            combinedLetters = ""
            for letter in str:
                combinedLetters += letter

            combinedLetters = "".join(sorted(combinedLetters)) # n & log n

            if combinedLetters in strsDict:
                strsDict[combinedLetters].append(str)
            else:
                strsDict[combinedLetters] = [str]
                groupCombined.append(combinedLetters)

        for str in groupCombined:
            groups.append(strsDict[str])
        return groups

    # 42. Trapping Rain Water (HARD)

    # 273. Integer to English Words (HARD)
    # Unfinished, but conceptually working
    def numberToWords(self, num: int) -> str:
        cases = { 
            0: "", 
            1: "One", 
            2: "Two", 
            3: "Three", 
            4: "Four", 
            5: "Five", 
            6: "Six", 
            7: "Seven", 
            8: "Eight", 
            9: "Nine", 
            10: "Ten", 
            11: "Eleven", 
            12: "Twelve", 
            13: "Thirteen", 
            14: "Fourteen", 
            15: "Fifteen", 
            16: "Sixteen", 
            17: "Seventeen", 
            18: "Eighteen", 
            19: "Nineteen", 
            20: "Twenty", 
            30: "Thirty", 
            40: "Forty", 
            50: "Fifty", 
            60: "Sixty", 
            70: "Seventy", 
            80: "Eighty", 
            90: "Ninety", 
            100: "Hundred", 
            1000: "Thousand", 
            1000000: "Million", 
            1000000000: "Billion" 
        }

        words = ""
        places = [1000, 1000000, 1000000000]
        counter = 0
        if num == 0:
            return "Zero"
        while num > 0:
            # 1000
            if num > 0:
                # 10[00]
                words += self.remainder(num % 100, cases) + " "
                num //= 100
            # 10
            if num > 0:
                words += "Hundred"
            if num > 0:
                words +=  " " + cases[num % 10]
                num //= 10

            if num > 0: 
                words += " " + cases[places[counter]] + " "
                counter += 1
        
        reversedWords = words.split(" ")
        if reversedWords[-1] == "":
            reversedWords.pop()
        reversedWords = reversedWords[::-1]
        if reversedWords[-1] == "":
            reversedWords.pop()
        words = ""
        for i in range(len(reversedWords) - 1):
            words += reversedWords[i] + " "
        words += reversedWords[-1]
        
        return words

    def remainder(self, num: int, cases: dict) -> str:
        words = ""
        if num >= 21:
            words += cases[num % 10] + " " + cases[num // 10 * 10]
        else: 
            words = cases[num]
        return words
    
    # 387. First Unique Character in a String (EASY)
    def firstUniqChar(self, s: str) -> int:
        letters = {}
        for char in s:
            if char not in letters:
                letters[char] = 1
            else:
                letters[char] += 1
        for i in range(len(s)):
            if letters[s[i]] == 1:
                return i
        return -1

    """
    # Linked Lists
    """
    # 2. Add Two Numbers (MED)
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        stack = []
        carry = 0
        
        while l1 is not None or l2 is not None:
            sum = carry
            carry = 0
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2: 
                sum += l2.val
                l2 = l2.next
            if sum >= 10:
                sum -= 10
                carry = 1
            stack.append(ListNode(sum, None))
        if carry == 1:
            stack.append(ListNode(carry, None))
        for i in range(len(stack)-1):
            stack[i].next = stack[i+1]
        return stack[0]

    # 21. Merge Two Sorted Lists (EASY)
    # Recursive
    def mergeTwoListsRecursive(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 and l2:
            if l1.val < l2.val:
                return ListNode(min(l1.val, l2.val), self.mergeTwoListsRecursive(l1.next, l2))
            else: 
                return ListNode(min(l1.val, l2.val), self.mergeTwoListsRecursive(l1, l2.next))
        elif l1 and not l2:
            return ListNode(l1.val, self.mergeTwoListsRecursive(l1.next, None))
        elif not l1 and l2:
            return ListNode(l2.val, self.mergeTwoListsRecursive(None, l2.next))
        else: 
            return None
    # Iterative
    def mergeTwoListsIterative(self, l1: ListNode, l2: ListNode) -> ListNode:
        first = None
        if l1 or l2:
            if l1 and l2:
                if l1.val < l2.val:
                    first = l1
                    l1 = l1.next
                else:
                    first = l2
                    l2 = l2.next
            elif l1:
                first = l1
                l1 = l1.next
            elif l2:
                first = l2
                l2 = l2.next
        last = first

        while l1 or l2:
            if l1 and l2:
                if l1.val < l2.val:
                    last.next = ListNode(l1.val, None)
                    l1 = l1.next
                else:
                    last.next = ListNode(l2.val, None)
                    l2 = l2.next
            elif l1:
                last.next = ListNode(l1.val, None)
                l1 = l1.next
            elif l2:
                last.next = ListNode(l2.val, None)
                l2 = l2.next
            last = last.next
        return first

    # 138. Copy List with Random Pointer (MED)
    def copyRandomList(self, head: 'Node') -> 'Node':
        node = None
        firstHead = head
        visited = {}
        if head:
            node = Node(head.val, None, None)
            visited[head] = node
            head = head.next
        firstNode = node
        while head:
            visited[head] = Node(head.val, None, None)
            node.next = visited[head]
            node = node.next
            head = head.next
        
        head = firstHead
        node = firstNode

        while head:
            if head.random:
                node.random = visited[head.random]
            node = node.next
            head = head.next
            
        return firstNode

    # 206. Reverse Linked List (EASY)
    """
    A linked list can be reversed either iteratively or recursively. Could you implement both?
    """
    # Iterative
    def reverseListIterative(self, head: ListNode) -> ListNode:
        if head:
            prev = head
            next = head.next
            head.next = None
            while next:
                head = next
                next = head.next
                head.next = prev
                prev = head
        return head

    # Recursive (official solution))
    def reverseListRecursive(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        prev = self.reverseListRecursive(head.next)
        head.next.next = head
        head.next = None
        return prev

    # 23. Merge k Sorted Lists (HARD)
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        queue = []
        min = 0
        if lists:
            if lists[0]:
                min = lists[0].val
                minNode = lists[0]
        else:
            return lists
        
        count = 0
        current = 0
        while lists:
            for list in lists:
                if list:
                    if list.val < min:
                        minNode = list
                        min = list.val
                        list = list.next
            queue.append(minNode)
            list[current] = list[current].next
            count = 0
            current = count


        return

    # 25. Reverse Nodes in k-Group (HARD)
    """
    Could you solve the problem in O(1) extra memory space?
    You may not alter the values in the list's nodes, only nodes itself may be changed.
    """
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        return

solution = Solution()

l8 = ListNode(6, None)
l7 = ListNode(2, l8)

l6 = ListNode(4, None)
l5 = ListNode(3, l6)
l4 = ListNode(1, l5)

l3 = ListNode(5, None)
l2 = ListNode(4, l3)
l1 = ListNode(1, l2)

lists = [l1, l2, l3]

print(solution.mergeKLists(lists))