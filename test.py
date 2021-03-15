a = [2, 1, 3, 5, 3, 2]
def test(a):
	minVal = -1
	minIndex = len(a)
	for i in range(len(a)):
		for j in range(i+1, len(a), 1):
			if a[i] == a[j] and j < minIndex:
				minVal = a[j]
				minIndex = j
	if minIndex < len(a):
		return minVal
	return -1
print(test(a))