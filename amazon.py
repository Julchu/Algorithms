from typing import List
import collections
import math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        return str(self.val)

class Node:
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
            if root.left: 
                self.DFSRecursive(root.left)
            if root.right:
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

    # 124. Binary Tree Maximum Path Sum (HARD)

    # 127. Word Ladder (MED)

    # Word Ladder 2

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

    # 236. Lowest Common Ancestor of a Binary Tree (MED)

    # 543. Diameter of Binary Tree (EASY)
    def diameterOfBinaryTree(self, root: TreeNode) -> int:


        return

    # 675. Cut Off Trees for Golf Event (HARD)

solution = Solution()

'''
#          1
#      2          3
#   4    5     6     7
# 8  9 10 11 12 13 14 15
'''
rrr = TreeNode(15, None, None)
rrl = TreeNode(14, None, None)
rlr = TreeNode(13, None, None)
rll = TreeNode(12, None, None)
lrr = TreeNode(11, None, None)
lrl = TreeNode(10, None, None)
llr = TreeNode(9, None, None)
lll = TreeNode(8, None, None)
rr = TreeNode(7, rrl, rrr)
rl = TreeNode(6, rll, rlr)
lr = TreeNode(5, lrl, lrr)
ll = TreeNode(4, lll, llr)
r = TreeNode(3, rl, rr)
l = TreeNode(2, ll, lr)
root = TreeNode(1, l, r)

print("BFS: ")
print(solution.BFS(root))
print("\nDFS: ")
print(solution.DFSRecursive(root))
print("\nIn Order: ")
print(solution.inOrderTraversal(root))
print("\nPre Order: ")
print(solution.preOrderTraversal(root))
print("\nPost Order: ")
print(solution.postOrderTraversal(root))


# grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]

# grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]

# print(solution.numIslands(grid))
