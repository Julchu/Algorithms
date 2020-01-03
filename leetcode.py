from typing import List
# 1. Two Sum
def twoSum(nums: List[int], target: int) -> List[int]:
    """ 
        Given array of integers, return indices of the two numbers such that they add up to a specific target.
        You may assume that each input would have exactly one solution, and you may not use the same element twice.
    """
    diff = {}
    for i in range(len(nums)):
        if diff.get(target - nums[i]):
            return [diff.get(target - nums[i]), i]
        diff[nums[i]] =  i
    
"""
    Test cases:
        [2, 7, 11, 15], 9
        [3, 2, 4], 6
        [3, 3], 6
"""
given_nums = [2, 7, 11, 15]
target = 9
print(twoSum(given_nums, target))
