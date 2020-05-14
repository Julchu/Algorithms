from typing import List
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
		
			