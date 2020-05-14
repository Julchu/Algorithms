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
#     frontier = new Stack()
#     frontier.push(start_node)
#     explored = new Set()    while frontier is not empty:
#         current_node = frontier.pop()
#         if current_node in explored: continue
#         if current_node == end_node: return success
		
#         for neighbor in graph.get_neigbhors(current_node):
#             frontier.push(neighbor)        explored.add(current_node)

graph = {}
g[0] = [1]
g[0].append(2)
g[1] = [2]
g[2] = [0]
g[2].append(3)
g[3] = [3]


from typing import List
#1 Two Sum
def twoSum1(self, nums: List[int], target: int) -> List[int]:
	n = len(nums)
	if n < 1:
		return
	elif n == 1:
		if nums[0] + nums[1] == target:
			return [0, 1]
	else:
		i = 0
		j = 1
		while i < n - 2:
			while j < n-1:
				if nums[i] + nums[j] == target:
					return [i, j]
				else:
					j += 1
			i += 1
	return

def twoSum2(self, nums: List[int], target: int) -> List[int]:
	nums3 = {}
	for n in nums:
		if not nums3.get(3):
			nums3[target - n] = 0
	for n in nums:
		if nums3[target-n]:
			return [n, target-n]

def twoSum(self, nums, target):
	d = {}
	solution = []
	for n in nums:
		d[n] = target-n
	n = 0
	while not solution:
		if d.get(d[n]):
			solution = [n, d[n]]
		else:
			n += 1
	return solution

# 217 Contains Duplicates
	def containsDuplicate(self, nums: List[int]) -> bool:
		# Brute Force: 1 inner for loops
		# for i in range(len(nums)):
		#     count = 0
		#     for j in range(len(nums)):
		#         if nums[i] == nums[k]:
		#             count += 1
		# if count > 1:
		#     return true

		# Somewhat optimized: 2 for loops, no nesting; dictionary
		numsDict = {}
		for i in range(len(nums)):
			if numsDict[nums[i]]:
				return False
			else: 
				numsDict[nums[i]] = 1
			
				