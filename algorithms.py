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

# print(graph)

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

nums = [10, 2, 2, 6, 4, 4]
target = 5
def twoSum(nums, target):
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

print(twoSum(nums, target))

# {-5: 10, 3: 2, 0: 5, -1: 6, 2: 3, -3: 8}
# {10: -5, 2: 3, 5: 0, 6: -1, 3: 2, 8: -3}

# a = {5: 3}

# if not a.get(3):
# 	print("no")
# if 3 in a.values():
# 	print("yes")

from typing import List
class Solution:
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
        # nums2 = {}
        nums3 = {}
        # for n in nums:
        #     nums2[n] = 0
        for n in nums:
            if not nums3.get(3):
            # if not nums3.get[target - n]:
                nums3[target - n] = 0
        for n in nums:
            if nums3[target-n]:
                return [n, target-n]

nums = [2, 7, 11, 15]
# S = Solution()

# print(S.twoSum1(nums, 9))
# print(S.twoSum2(nums, 9))

# a = {}
# a[2] = 3
# print(a)
# if not a.get(3):
#     print("no")
