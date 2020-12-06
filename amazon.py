from typing import List
import math
class Solution:
    #1. Two Sum problem 
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numsSet = {}
        for index in range(len(nums)):
            numsSet[nums[index]] = index
        for index in range(len(nums)):
            difference = target - nums[index]
            if difference in numsSet and index is not numsSet[difference]:
                return [index, numsSet[difference]]

    """
    Old IDs are palindromes
    New IDs are the same as old ID but breaking palindrome, and lowest alphabetical order

    Ex: "aabbaa" -> "aaabaa"
    Ex: "bab" -> "aab"
    
    Test: 
        s = "bab"
        breakPalindrome(s)
    """
    def breakPalindrome(self, oldID: int) -> int:
        newID = ""
        length = len(oldID)
        mid = length // 2
        allAs = True
        done = False
        index = 0
        
        while index <= mid:
            if not done and oldID[index] != "a":
                newID += "a"
                allAs = False
                done = True
            else: 
                newID += oldID[index]
            
            index += 1
        while index < length:
            newID += oldID[index]
            index += 1

        if allAs:
            newID = "IMPOSSIBLE"

        return newID

    """
    Find shortest Euclidian distance between robot positions
    numRobots: number of robots
    position X and position Y: equal length arrays of X, Y positions
    Euclidian distance: (X1-X2)^2 + (Y1-Y2)^2

    Example inputs: 
        numRobots = 5
        positionX = [0, 10, 15, 20, 25]
        positionY = [0, 10, 15, 20, 25]
    
    Test: 
        closestSquaredDistance(numRobots, positionX, positionY)
    """
    def closestSquaredDistance(self, numRobots, positionX, positionY) -> int:
        possibleCombinations = int(math.comb(numRobots, 2))
        
        count = 0
        left = 0
        right = 1

        shortestDistance = (positionX[right] - positionX[left])**2 + (positionY[right] - positionY[left])**2
        
        while count < possibleCombinations:

            euclid = (positionX[right] - positionX[left])**2 + (positionY[right] - positionY[left])**2
            if euclid < shortestDistance:
                shortestDistance = euclid

            right += 1
            if right == numRobots:
                left += 1
                right = left + 1
            
            count += 1
            
        return shortestDistance
          
    #3. Longest Substring Without Repeating Characters
        # s = "ab"
        # s = "abcabcbb"
        # s = "pwwkew"
        # s = " "
        # s = "abba"
        # s = "aab"
    def lengthOfLongestSubstring(self, s: str) -> int:
        maxLength = 0
        letters = {}
        left = 0
        right = 0
        length = len(s)

        while right < length:
            letter = s[right]

            if letter in letters:
                maxLength = max(maxLength, right - left)
                left = max(left, letters[letter] + 1)
                
            letters[letter] = right
            right += 1
        
        maxLength = max(maxLength, right - left)
        return maxLength

    #11. Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        maxArea = 0

        while left != right:
            area = min(height[left], height[right]) * (right - left)
            if area > maxArea:
                maxArea = area

            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return maxArea

    #49. Group Anagrams
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = []
        groupIndex = 0
        
        while strs:
            str1 = strs.pop()
            groups.append([str1])

            i = 0
            n = len(strs)

            while i < n:
                str2 = strs[i]
                if self.checkAnagram(str1, str2):
                    groups[groupIndex].append(str2)
                    strs.pop(i)
                    i -= 1
                    n -= 1
                i += 1
            groupIndex += 1
        return groups

    def checkAnagram(self, str1: str, str2: str) -> bool:
        letters = {}
        for letter in str1:
            if letter in letters:
                letters[letter] += 1
            else:
                letters[letter] = 1
        length = len(str2)
        index = 0
        while index < length:
            letter = str2[index]
            if letter in letters:
                if letters[letter] == 1:
                    letters.pop(letter)
                else:
                    letters[letter] -= 1
                index += 1
            else:
                return False
        return True


solution = Solution()

# strs = ["eat","tea","tan","ate","nat","bat"]
strs = [["b",""]]
# print(solution.checkAnagram(strs[0], strs[2]))
print(solution.groupAnagrams(strs))