from typing import List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    #1 Two Sum problem 
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
            if oldID[index] is not "a" and not done:
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
        possibleCombinations = int(factorial(numRobots) / factorial (2) / factorial(numRobots - 2))
        
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
        
    def factorial(self, num):
        fact = 1
        for i in range(1, num+1):
            fact *= i
        return fact
          
    #3 Longest Substring Without Repeating Characters
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
        if len(s) > 1:
            if len(s) == 2:
                if s[0] == s[1]:
                    maxLength = 1
                else:
                    maxLength = 2
            else:
                while right < len(s):
                    letter = s[right]
                    if (right - left) > maxLength:
                        maxLength = right - left
                    if letter in letters:
                        left = letters[letter] + 1
                    letters[letter] = right
                    right += 1
        else: 
            maxLength = len(s)
        return maxLength

    #11 Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        possibleCombinations = int(factorial(numRobots) / factorial (2) / factorial(numRobots - 2))
        
        count = 0
        left = 0
        right = 1

        volume = min(n[right], n[left]) * (right - left)
 
        # while count < possibleCombinations:
            



solution = Solution()
