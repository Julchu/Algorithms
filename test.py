# typing module needed for dictionaries (Lists)
from typing import List

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

print(findJudge(N, trust))