from typing import List

a = [10, 8, 7, 5, 3]
b = [10, 7, 3, 2, 1, -5]

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

graph = {}
graph[0] = [1]
graph[0].append(2)
graph[1] = [2]
graph[2] = [0]
graph[2].append(3)
graph[3] = [3]

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

graph = {}
graph[0] = [1]
graph[0].append(2)
graph[1] = [2]
graph[2] = [0]
graph[2].append(3)
graph[3] = [3]

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
			
	def productExceptSelf2(nums: List[int]) -> List[int]:
		n = len(nums)
        
        answer = [0] * n
        answer[0] = 1
		
        for i in range(1, n):
            answer[i] = nums[i - 1] * answer[i - 1]

        R = 1;
        for i in reversed(range(n)):
            answer[i] = answer[i] * R
            R *= nums[i]
        
        return answer