import random
from typing import List
from collections import defaultdict, deque
import math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        return str(self.val)

class RandomNode:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

    def __str__(self):
        return str(self.val)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.val)

class Course:
    def __init__(self):
        self.inDegrees = 0
        self.next = []

class DNode:
    def __init__(self, key, value, left, right):
        self.key = key
        self.value = value
        self.left = left       
        self.right = right

    def __str__(self) -> str:
        return str(self.key) + ": " + str(self.value) 

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.first = None
        self.last = None
    
    def __str__(self) -> str:
        return str(self.cache)

    def get(self, key: int) -> int:
        # If key in cache
        if key in self.cache:
            self.moveToBack(key)
            return self.cache[key].value

        # Else:
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        # Key exists
        if key in self.cache:
            self.moveToBack(key)
            self.cache[key].value = value

        # New key
        else:
            # Maxed
            if len(self.cache) == self.capacity:
                # Create new node and put into cache
                self.cache[key] = DNode(key, value, self.cache[self.last], self.cache[self.first].right)

                # Set last to new node
                self.cache[self.last].right = self.cache[key]
                
                # Set first.right'left to new node (last)
                self.cache[self.first].right.left = self.cache[key]

                # Remove first
                first = self.cache.pop(self.first)

                # Set first to first.right
                self.first = first.right.key

            # Not maxed
            else:
                # Empty
                if len(self.cache) == 0:
                    self.first = key
                    self.cache[key] = DNode(key, value, None, None)

                # Not empty
                else:
                    self.cache[key] = DNode(key, value, self.cache[self.last], self.cache[self.first])
                    self.cache[self.first].left = self.cache[key]
                    self.cache[self.last].right = self.cache[key]
            self.last = key

    def moveToBack(self, key) -> None:
        self.cache[self.first].left = self.cache[self.last].right = self.cache[key]
        self.cache[key].left.right, self.cache[key].right.left = self.cache[key].right, self.cache[key].left
        self.cache[key].right = self.cache[self.first]
        self.cache[key].left = self.cache[self.last]
        self.last = key

        # If key was also first
        if self.first == key:
            self.first = self.cache[self.first].right.key
    '''
    lRUCache = LRUCache(2)
    print(lRUCache.put(1, 1))
    print(lRUCache.put(2, 2))
    print(lRUCache.get(1))
    print(lRUCache.put(3, 3))
    print(lRUCache.get(2))
    print(lRUCache.put(4, 4))
    print(lRUCache.get(1))
    print(lRUCache.get(3))
    print(lRUCache.get(4))

    None
    None
    1
    None
    -1
    None
    -1
    3
    4
    '''

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

    # 42. Trapping Rain Water (HARD) Official solution
    def trap(self, height: List[int]) -> int:
        ans = current = 0
        stack = []
        while current < len(height):
            top = stack.pop()
            while stack and height[current] > top:
                if not stack:
                    break
                distance = current - stack[-1] - 1
                bounded_height = min(height[current], height[stack[-1]]) - height[top]
                ans += distance * bounded_height
            current += 1
            stack.append(current)
        
        return ans


    # 238. Product of Array Except Self (MED) Official solution
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        length = len(nums)
        
        L, R, answer = [0]*length, [0]*length, [0]*length

        L[0] = 1
        for i in range(1, length):
            
            L[i] = nums[i - 1] * L[i - 1]
        
        R[length - 1] = 1
        for i in reversed(range(length - 1)):
            R[i] = nums[i + 1] * R[i + 1]
        
        # Constructing the answer array
        for i in range(length):

            answer[i] = L[i] * R[i]
        
        return answer

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
    def copyRandomList(self, head: 'RandomNode') -> 'RandomNode':
        node = None
        firstHead = head
        visited = {}
        if head:
            node = RandomNode(head.val, None, None)
            visited[head] = node
            head = head.next
        firstNode = node
        while head:
            visited[head] = RandomNode(head.val, None, None)
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
        min = None
        firstNode = None
        currentNode = None
        done = False
        n = len(lists)
        if n > 0:
            while not done:
                empty = True
                
                count = 0
                current = count
                
                while count < n-1 and not lists[count]:
                    count += 1
                if lists[count]:
                    min = lists[count].val
                    current = count
                
                count = 0

                for list in lists:
                    if list:
                        if list.val < min:
                            min = list.val
                            current = count
                        empty = False
                    count += 1
                if lists[current]:
                    if not firstNode:
                        firstNode = lists[current]
                        currentNode = firstNode
                    else: 
                        currentNode.next = lists[current]
                        currentNode = currentNode.next

                    lists[current] = lists[current].next
                    
                done = empty

        return firstNode

    """
    Trees and Graphs
    """
    def BFS(self, root: TreeNode):
        queue = []
        queue.append(root)
        while queue:
            node = queue.pop(0)
            print(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return

    # Iterative
    def DFSIterative(self, root: TreeNode):
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            print(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return

    def DFSRecursive(self, root: TreeNode):
        if root:
            print(root.val)
            self.DFSRecursive(root.left)
            self.DFSRecursive(root.right)

    '''
    def DFSAux(self, root: TreeNode, visited: dict):
        print(root.val)
        visited[root] = True
        if root.left: 
            self.DFSAux(root.left, visited)
        if root.right:
            self.DFSAux(root.right, visited)
    '''

    '''
        #         3
        #    9         20
        # N    N    15    7

        In order: 9, 3, 15, 20, 7
        Pre order: 3, 9, 20, 15, 7
        Post order: 9, 15, 7, 20, 3

        #         4
        #    2         6
        # 1    3    5    7

        In order: 1, 2, 3, 4, 5, 6, 7
        Pre order: 4, 2, 1, 3, 6, 5, 7
        Post order: 1, 3, 2, 5, 7, 6, 4
    '''

    def preOrderTraversal(self, root: TreeNode):
        if root:
            print(root)
            self.preOrderTraversal(root.left)
            self.preOrderTraversal(root.right)

    def inOrderTraversal(self, root: TreeNode): 
        if root:
            self.inOrderTraversal(root.left)
            print(root)
            self.inOrderTraversal(root.right)
    
    def postOrderTraversal(self, root: TreeNode): 
        if root:
            self.postOrderTraversal(root.left)
            self.postOrderTraversal(root.right)
            print(root)


    # 101. Symmetric Tree (EASY)
        # Recursively and iteratively
    def isSymmetric(self, root: TreeNode) -> bool:
        rows = {}
        height = 0
        q = []
        if root:
            q.append(root)
            rows[height] = [root]
            height += 1
        while q:
            for _ in range(len(q)):
                n = q.pop(0)
                if n.left:
                    q.append(n.left)
                if n.right:
                    q.append(n.right)
                if height not in rows:
                    rows[height] = [n.left]
                else:
                    rows[height].append(n.left)
                rows[height].append(n.right)
            height += 1

        i = 0
        while i < len(rows):
            n = len(rows[i])
            for j in range(n//2):
                if rows[i][j] and rows[i][n-1-j]:
                    if rows[i][j].val != rows[i][n-1-j].val:
                        return False
                elif (not rows[i][j] and rows[i][n-1-j]) or (rows[i][j] and not rows[i][n-1-j]):
                    return False
            i += 1
        return True

    # 733. Flood Fill (EASY)
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        visited = []
        for i in range(len(image)):
            visited.append([False] * len(image[i]))

        color = image[sr][sc]
        queue = []
        queue.append([sr, sc])
        image[sr][sc] = newColor

        while queue:
            coords = queue.pop(0) # [1, 1]
            row = coords[0]
            col = coords[1]
            visited[row][col] = True
            
            # left
            if 0 < col:
                if image[row][col-1] == color and visited[row][col-1] == False:
                    image[row][col-1] = newColor
                    queue.append([row, col-1])

            # right
            if col < len(image[0])-1 and visited[row][col+1] == False:
                if image[row][col+1] == color:
                    image[row][col+1] = newColor
                    queue.append([row, col+1])

            # top
            if 0 < row:
                if image[row-1][col] == color and visited[row-1][col] == False:
                    image[row-1][col] = newColor
                    queue.append([row-1, col])

            # bot
            if row < len(image)-1:
                if image[row+1][col] == color and visited[row+1][col] == False:
                    image[row+1][col] = newColor
                    queue.append([row+1, col])
        return image

    # 98. Validate Binary Search Tree (MED)
    def isValidBST(self, root: TreeNode) -> bool:
        numbers = []
        if root:
            self.isValidBSTAux(root, numbers)
        if len(numbers) > 0:
            min = numbers[0]
            for i in range(1, len(numbers), 1):
                if numbers[i] <= min:
                    return False
                min = numbers[i]
        return True

    def isValidBSTAux(self, root: TreeNode, numbers) -> bool:
        if root:
            self.isValidBSTAux(root.left, numbers)
            numbers.append(root.val)
            self.isValidBSTAux(root.right, numbers)

    # 102. Binary Tree Level Order Traversal (MED)
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        rows = []
        queue = []
        if root:
            queue.append(root)
        while queue:
            newRow = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                newRow.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            rows.append(newRow)
        return rows

    # 103. Binary Tree Zigzag Level Order Traversal (MED)
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        reversed = False
        rows = []
        queue = []
        if root:
            queue.append(root)
        while queue:
            newRow = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                newRow.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if reversed:
                newRow = newRow[::-1]
                reversed = False
            else:
                reversed = True
            rows.append(newRow)
        return rows

    # 200. Number of Islands (MED)
    def numIslands(self, grid: List[List[str]]) -> int:
        islands = 0
        visited = []

        for i in range(len(grid)):
            visited.append([False] * len(grid[i]))

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == "1" and visited[i][j] == False:
                    islands += 1
                    visited[i][j] = True
                    self.BFSIslands(i, j, grid, visited)
        return islands

    def BFSIslands(self, i: int, j: int, grid: list, visited: list) -> None:
        queue = []
        queue.append([i, j])
        while queue:
            for _ in range(len(queue)):
                coords = queue.pop(0)
                row = coords[0]
                col = coords[1]
                # left
                if col > 0:
                    if grid[row][col-1] == "1" and visited[row][col-1] == False:
                        visited[row][col-1] = True
                        queue.append([row, col-1])
                # right
                if col < len(grid[0]) - 1:
                    if grid[row][col+1] == "1" and visited[row][col+1] == False:
                        visited[row][col+1] = True
                        queue.append([row, col+1])
                # top
                if row > 0:
                    if grid[row-1][col] == "1" and visited[row-1][col] == False:
                        visited[row-1][col] = True
                        queue.append([row-1, col])
                # bot
                if row < len(grid) - 1:
                    if grid[row+1][col] == "1" and visited[row+1][col] == False:
                        visited[row+1][col] = True
                        queue.append([row+1, col])
        return

    # 207. Course Schedule (MED)
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
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
                nodes[current] = Course()
                nodes[current].next.append(prereq)
            if prereq in nodes:
                    nodes[prereq].inDegrees += 1
            else:
                nodes[prereq] = Course()
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
            return True
        else:
            return False

    # 236. Lowest Common Ancestor of a Binary Tree (MED)
    

    # 543. Diameter of Binary Tree (EASY)
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        diameter = self.diameterOfBinaryTreeAux(root, 0)
        return diameter - 1

    def diameterOfBinaryTreeAux(self, root: TreeNode, diameter: int) -> int:
        if not root:
            return 1
        l = self.diameterOfBinaryTreeAux(root.left, diameter)
        r = self.diameterOfBinaryTreeAux(root.right, diameter)
        diameter = max(l + r + 1, diameter)
        return max(l, r) + 1

    """
Sorting and Searching
    """
    
    # 33. Search in Rotated Sorted Array (MED)
    def search(self, nums: List[int], target: int) -> int:
        return

    # 56. Merge Intervals (MED)
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        return

    # 167. Two Sum II - Input array is sorted (EASY)
    def twoSum2(self, numbers: List[int], target: int) -> List[int]:
        return

    # 253. Meeting Rooms II (MED)
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        return

    # 347. Top K Frequent Elements (MED)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        return

    # 973. K Closest Points to Origin (MED)
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        return

    # 4. Median of Two Sorted Arrays
        # The overall run time complexity should be O(log (m+n)) (HARD)
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:

        # Merge arrays into deque
        merged = deque([], maxlen=(len(nums1) + len(nums2)))
        while nums1 or nums2:
            if nums1 and nums2:
                if nums1[-1] > nums2[-1]:
                    merged.appendleft(nums1.pop())
                else:
                    merged.appendleft(nums2.pop())
            elif nums1:
                merged.appendleft(nums1.pop())
            else:
                merged.appendleft(nums2.pop())

        # Find median
            # if even number: take mid 2 and average
            # else odd number: takase middle 
        median = float()
        n = len(merged)
        if n % 2 == 0:
            median = (merged[n//2] + merged[n//2-1]) / 2
        else:
            median = merged[n//2]
        return median

    # 215. Kth Largest Element in an Array (MED) method 1 (merge sort: O(n log n)
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # Sort nums w/ merge sort
        nums = self.mergeSort(nums)
        # Loop backwards through k to find k-th largest element
        return nums[-k]

    def mergeSort(self, nums: List[int]) -> List[int]:
        # Split
        m = len(nums)//2
        l = nums[:m]
        r = nums[m:]

        if len(nums) > 1:
            self.mergeSort(l)
            self.mergeSort(r)

            # Merge
            self.mergeSortMerge(l, r, nums)
        return nums

    def mergeSortMerge(self, l: List[int], r: List[int], nums: List[int]) -> List[int]:
        i = j = k = 0
        while i < len(l) and j < len(r):
            if l[i] < r[j]:
                nums[k] = l[i]
                i += 1
            else:
                nums[k] = r[j]
                j += 1
            k += 1
        while i < len(l):
            nums[k] = l[i]
            i += 1
            k += 1
        while j < len(r):
            nums[k] = r[j]
            j += 1
            k += 1
        return nums

    # 215. Kth Largest Element in an Array (MED) method 2 (selection sort: O(k * n))
    def findKthLargest2(self, nums: List[int], k: int) -> int:
        max = 0
        for _ in range(k):
            current = 0
            if nums:
                max = nums[0]
            for i in range(len(nums)):
                if nums[i] > max:
                    max = nums[i]
                    current = i
            nums.pop(current)
        return max

    def findKthLargest3(self, nums: List[int], k: int) -> int:
        kthSmallest = self.quickSelect(nums, 0, len(nums)-1, len(nums)-k)
        return kthSmallest
    
    def quickSort(self, nums: List[int], low: int, high: int) -> List[int]:
        if low < high:
            # Divide
            index = self.partition(nums, low, high)

            # Conquer
            self.quickSort(nums, low, index-1)
            self.quickSort(nums, index+1, high)

        return nums
    
    def quickSelect(self, nums: List[int], low: int, high: int, k: int):
        index = self.partition(nums, low, high)
        
        if index == k:
            return nums[index]
        elif index < k:
            return self.quickSelect(nums, low, index-1, k)
        else:
            return self.quickSelect(nums, index+1, high, k)

    def partition(self, nums, low, high) -> int:
        # pivot = nums[high]

        index = (low + high)//2
        # index = random.randint(low, high)
        
        pivot = nums[index]
        nums[index], nums[index] = nums[index], nums[index]

        index = low
        for i in range(low, high):
            if nums[i] <= pivot:
                nums[index], nums[i] = nums[i], nums[index]
                index += 1
        nums[index], nums[high] = nums[high], nums[index]

        return index

    def orangesRotting(self, grid: List[List[int]]) -> int:
        visited = []
        timer = 0
        for i in range(len(grid)):
            visited.append([False] * len(grid[i]))
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if visited[i][j] == False and grid[i][j] == 2:
                    visited[i][j] = True
                    timer = self.BFSOranges(grid, visited, i, j, timer)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 1:
                    return -1
        print(grid)
        return timer
    
    def BFSOranges(self, grid, visited, i, j, timer):
        queue = []
        queue.append([i, j])
        
        while queue:
            rotten = False
            coords = queue.pop(0)
            row = coords[0]
            col = coords[1]
            
            
            # Left
            if col > 0:
                if grid[row][col-1] == 1 and visited[row][col-1] == False:
                    grid[row][col-1] = 2
                    visited[row][col-1] = True
                    queue.append([row, col-1])
                    rotten = True
            if col < len(grid[0]) - 1:
                # Right
                if grid[row][col+1] == 1 and visited[row][col+1] == False:
                    grid[row][col+1] = 2
                    visited[row][col+1] = True
                    queue.append([row, col+1])
                    rotten = True
            
            if row > 0:
                # Top
                if grid[row-1][col] == 1 and visited[row-1][col] == False:
                    grid[row-1][col] = 2
                    visited[row-1][col] = True
                    queue.append([row-1, col])
                    rotten = True
            if row < len(grid) - 1:
                # Bot
                if grid[row+1][col] == 1 and visited[row+1][col] == False:
                    grid[row+1][col] = 2
                    visited[row+1][col] = True
                    queue.append([row+1, col])
                    rotten = True

            if rotten:
                timer += 1
                rotten = False
        return timer


solution = Solution()
# grid = [[2,1,1],[1,1,0],[0,1,1]]
# grid = [[2,1,1],[0,1,1],[1,0,1]]
# grid = [[0,2]]
grid = [[2,1,1],[1,1,1],[0,1,2]]
k = 5

print(solution.orangesRotting(grid))