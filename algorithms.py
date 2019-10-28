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