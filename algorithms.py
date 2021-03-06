from typing import List

# a = [10, 8, 7, 5, 3]
# b = [10, 7, 3, 2, 1, -5]

def sortedMerge(a, b):
    c = []

    while a and b:
        if a[-1] <= b[-1]:
            c.insert(0, a.pop())
        else: 
            c.insert(0, b.pop())
    if a:
        c = a + c
    else:
        c = b + c
    return c

# print(sortedMerge(a, b))

# graph = {}
# graph[0] = [1]
# graph[0].append(2)
# graph[1] = [2]
# graph[2] = [0]
# graph[2].append(3)
# graph[3] = [3]

def BFS(graph, s):
        visited = [False] * (len(graph))
        queue = []
        queue.append(s) 
        visited[s] = True
  
        while queue:
            s = queue.pop(0) 
            print (s, end = " ")
            for i in graph[s]: 
                if visited[i] == False: 
                    queue.append(i) 
                    visited[i] = True
        print()
# BFS(graph, 2)

def DFSUtil(self, v, visited): 
    visited[v] = True
    print(v, end = ' ') 

    for i in self.graph[v]: 
        if visited[i] == False: 
            self.DFSUtil(i, visited) 

def DFS(self, v): 
    visited = [False] * (len(self.graph)) 
    self.DFSUtil(v, visited) 



# DFS(graph, start_node, end_node):
#	frontier = new Stack()
#	frontier.push(start_node)
#	explored = new Set()    while frontier is not empty:
#		current_node = frontier.pop()
#		if current_node in explored: continue
#		if current_node == end_node: return success
        
#		for neighbor in graph.get_neigbhors(current_node):
#			frontier.push(neighbor)
# 			explored.add(current_node)

# graph = {}
# graph[0] = [1]
# graph[0].append(2)
# graph[1] = [2]
# graph[2] = [0]
# graph[2].append(3)
# graph[3] = [3]

# Leetcode

# 1. Two Sum
""" 
    Given array of integers, return indices of the two numbers such that they add up to a specific target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
"""
def twoSum(nums: List[int], target: int) -> List[int]:
	diff = {}
	for i in range(len(nums)):
		if diff.get(target - nums[i]):
			return [diff.get(target - nums[i]), i]
		diff[nums[i]] =  i

# 3 Longest Substring Without Repeating Characters
def lengthOfLongestSubstring(s: str) -> int:
	substrings = []
	substringLengths = []
	exists = {}
	a = ""
	output = 0

	for i in s:
		if not exists.get(i):
			exists[i] = "yes"
			a = a + i
		else:
			substrings.append(a)
			substringLengths.append(len(a))
			a = i
	substrings.append(a)
	substringLengths.append(len(a))
	min = 0
	index = 0
	if len(substringLengths) > 0:
		for i in range(len(substringLengths)):
			if substringLengths[i] > min:
				min = substringLengths[i]
				index = i
		output = len(substrings[index])
	return output

# 20. Valid Parentheses
def isValid(s: str) -> bool:
	current = []
	n = len(s)
	# Odd # of characters
	if n % 2 != 0:
		return False

	for i in s:
		if i == "(" or i == "{" or i == "[":
			current.append(i)
		elif len(current) > 0: 
			if current[-1] == "(" and i == ")":
				current.pop()
			elif current[-1] == "{" and i == "}":
				current.pop()
			elif current[-1] == "[" and i == "]":
				current.pop()
	if len(current) > 0:
		return False
	return True

# 104. Maximum Depth of Binary Tree
# def maxDepth(root: TreeNode) -> int:
#     if root:
#         if root.left is not None and root.right is not None:
#             return max(self.maxDepth(root.left) + 1, self.maxDepth(root.right) + 1)
#         elif root.left is not None:
#             return self.maxDepth(root.left) + 1
#         elif root.right is not None:
#             return self.maxDepth(root.right) + 1
#         else:
#             return 1
#     return 0

# 125. Valid Palindrome
def palindrome():
    newString = ""
    l = []
    r = []
        
    for i in s:
        if i.isalnum():
            newString = newString + i
    
    n = len(newString)

    for i in range(n//2):
        l.append(newString[i].lower())
        r.append(newString[n-i-1].lower())
    for i in range(len(l)):
        if l[i] != r[i]:
            return False
    return True

    
# def lengthOfLongestSubstring(s: str) -> int:
# 	n = len(s)
# 	ans = 0
# 	index = 0
# 	j = 0
# 	if n > 0:
# 		max = s[j]
# 	for i in range(n):
		
		
# 		i
		
# 		if ans <  j - i + 1:
# 		ans = j - i + 1
# 		a


# public int lengthOfLongestSubstring(String s) {
# 	int n = s.length(), ans = 0;
# 	int[] index = new int[128]; // current index of character
# 	// try to extend the range [i, j]
# 	for (int j = 0, i = 0; j < n; j++) {
# 		i = Math.max(index[s.charAt(j)], i);
# 		ans = Math.max(ans, j - i + 1);
# 		index[s.charAt(j)] = j + 1;
# 	}
# 	return ans;
# }

# 121. Best Time to Buy and Sell Stock
def maxProfit(prices: List[int]) -> int:
	min = 9999999999999999
	max = 0
	for i in range(len(prices)):
		if prices[i] < min:
			min = prices[i]
		elif (prices[i] - min > max):
			max = prices[i] - min
	return max

# 206. Reverse Linked List
def reverseLinkedList():
	stack = []
	while head:
		stack.append(head)
		head = head.next
	returnHead = None
	if stack:
		newHead = stack.pop()
		returnHead = newHead
		while stack:
			next = stack.pop()
			newHead.next = next
			newHead = newHead.next
		newHead.next = None

	return returnHead


# 217 Contains Duplicates
def containsDuplicate(self, nums: List[int]) -> bool:
	# Somewhat optimized: 2 for loops, no nesting; dictionary
	numsDict = {}
	for i in range(len(nums)):
		if numsDict[nums[i]]:
			return False
		else: 
			numsDict[nums[i]] = 1

# 238. Product of Array Except Self
# With division and O(n)
def productExceptSelf(self, nums: List[int]) -> List[int]:
	product = 1
	n = len(nums)
	for i in range(n): 
		product *= nums[i]
	output = [product] * n
	for i in range(n):
		output[i] = output[i] // nums[i]
	return output

# Without division and O(n)
def productExceptSelf(nums: List[int]) -> List[int]:
	n = len(nums)
	L = [0] * n
	R = [0] * n
	output = [0] * n

	L[0] = 1
	R[n-1] = 1
	for i in range(1, n):
		L[i] = nums[i - 1] * L[i - 1]
	
	for i in range(n-2, -1, -1):
		R[i] = nums[i + 1] * R[i + 1]
	
	for i in range(n):
		output[i] = L[i] * R[i]
	return output

	"""
	# Visualization:
		L = [1, 1, 2, 6]
		R = [24, 12, 4, 1]
		Output: [24, 12, 8, 6]

	Explanation:
		Cannot reduce (divide), so want to start small that are multiplying together
		O(n) means precomputing, rather than computing like O(n^2)
			Precomputing would indicate multiplying subparts beforehand (usually 2): subarrays
			Subarrays indicate having a left and right (since multiplying 2 things together)
			Ex: [1, 2, 3, 4]; 
				First subarray of left of 1 doesn't exist so stay at 1; first subarray of right of 1 would be [2, 3, 4] 
				Second subarray of left of 2 would be 1; right subarray of right of 2 would be [3, 4]
				Third subarray of left of 3 would be [1, 2]; right subarray of right of 3 would be [4]
				Fourth subarray of left of 4 would be [1, 2, 3]; right subarray of right of 4 doesn't exist so stay at 1
				L = [1, 1, 2, 6]
				R = [24, 12, 4, 1]

				Output = [24, 12, 8, 6]
	"""
            
def productExceptSelf2(nums: List[int]) -> List[int]:
	n = len(nums)
	
	answer = [0] * n
	answer[0] = 1
	
	for i in range(1, n):
		answer[i] = nums[i - 1] * answer[i - 1]

	R = 1
	for i in reversed(range(n)):
		answer[i] = answer[i] * R
		R *= nums[i]
	
	return answer

# 53. Maximum Subarray

# 997.  Find the Town Judge
def findJudge(N: int, trust: List[List[int]]) -> int:
	judge = -1
	if N == 1 and len(trust) == 0:
		judge = N
	if len(trust) > 0:
		people = {}
		for i in trust:
			if people.get(i[1]):
				people[i[1]] += 1
			else:
				people[i[1]] = 1
		found = False
		for key in people:
			if people[key] == N-1:
				judge = key
		
		for i in trust:
			if i[0] == judge:
				judge = -1
	return judge

# TopHat Interview
def isAnagram(string1, string2):
    exists = {}
    for i in string1:
        if exists.get(i):
            exists[i] += 1
        else:
            exists[i] = 1
    
    for i in string2:
        if exists.get(i):
            if exists.get(i, 0) > 1:
                exists[i] -= 1
            elif exists.get(i) == 1:
                exists.pop(i)
        else:
            return False
    return True

def numFields(fields):
    counter = 0
    for i in range(len(fields)):
        for j in range(len(fields[i])):
            if fields[i][j] == True:
                counter += 1
                visited(i, j)

def visited(i, j):
    fields[i][j] = False
    if j < len(fields[0]) and fields[i][j+1] == True:
        visited(i, j+1)
    if j > 0 and fields[i][j-1] == True:
        visited(i, j-1)
    if i < len(fields) and fields[i+1][j] == True:
        visited(i+1, j)
    if i > 0 and fields[i-1][j] == True:
        visited(i-1, j)

# fields=	[[True, False, True, False], 
#         [False, True, True, False], 
#         [False, False, False, True]]

def reverseString(s: List[str]) -> None:
    def aux(i):
        if i == len(s)//2:
            return
        s[i], s[len(s) - 1 - i] = s[len(s) - 1 - i], s[i]
        aux(i+1)
    aux(0)
    return s

# s = ["h","e","l","l","o"]
# print(reverseString(s))


"""
Recursion:
    base case, then base case + 1

"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# class Solution:
# def swapPairs(self, head: ListNode) -> ListNode:
# 	if head is None or head.next is None:
# 		return head
# 	temp = head.next    
	
# 	head.next = self.swapPairs(temp.next)
	
# 	temp.next = head
	
# 	# self.swapPairs(temp.next)
	
# 	return temp

def maxProduct(nums: List[int]) -> int:
	max = float('-inf')
	negative_max = 1
	current_product = 1
	
	for i in nums:
		current_product *= i
		negative_max *= i
		if max < current_product:
			max = current_product
		print("max: ", max)
		print("current: ", current_product)
		print("negative: ", negative_max)

		if current_product == 0:
			current_product = 1
	
	return max

# Google Phone Screen
	# Print 2-D array using only 1 for loop
	a = []

	i = 0
	j = 0
	m = len(a)
	# Explain why for loop makes no sense (non-generic sizes)
	# while True:
	#     if i < m:
	#         n = len(a[i])
	#     if i == m:
	#         break
	#     if j < n:
	#         print(a[i][j])
	#         j += 1
	#     else:
	#         j = 0
	#         i += 1
		
	# Find top K elements - if given input is 1 trillion numbers
	"""
		BST -> want inserts for checking each time new element from input is looked at, to ensure that current K elements are always top
			AVL tree: better for lookup
			Red Black tree: better for inserts

		https://stackoverflow.com/questions/27478298/time-complexity-of-deletion-in-binary-search-tree/27485440#27485440
	"""