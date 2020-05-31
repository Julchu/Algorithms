# typing module needed for dictionaries (Lists)
from typing import List

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
nums = [1, 2, 3, 4]
print(productExceptSelf(nums))

