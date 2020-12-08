from typing import List
import collections
import math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    """
    Online Assessment Q1
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
    Online Assessment Q2
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
    """
    Arrays and Strings
    """
    # 1. Two Sum problem 
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numsSet = {}
        for index in range(len(nums)):
            numsSet[nums[index]] = index
        for index in range(len(nums)):
            difference = target - nums[index]
            if difference in numsSet and index is not numsSet[difference]:
                return [index, numsSet[difference]]
          
    # 3. Longest Substring Without Repeating Characters
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

    #11. Container With Most Water (MED)
        # Official solution
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

    # 49. Group Anagrams (MED)
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        strsDict = {}
        groupCombined = []
        groups = []

        for str in strs:
            combinedLetters = ""
            for letter in str:
                combinedLetters += letter

            combinedLetters = "".join(sorted(combinedLetters)) # n & log n

            if combinedLetters in strsDict:
                strsDict[combinedLetters].append(str)
            else:
                strsDict[combinedLetters] = [str]
                groupCombined.append(combinedLetters)

        for str in groupCombined:
            groups.append(strsDict[str])
        return groups

    # 42. Trapping Rain Water (HARD)

    # 273. Integer to English Words (HARD)
    # Unfinished, but conceptually working
    def numberToWords(self, num: int) -> str:
        cases = { 
            0: "", 
            1: "One", 
            2: "Two", 
            3: "Three", 
            4: "Four", 
            5: "Five", 
            6: "Six", 
            7: "Seven", 
            8: "Eight", 
            9: "Nine", 
            10: "Ten", 
            11: "Eleven", 
            12: "Twelve", 
            13: "Thirteen", 
            14: "Fourteen", 
            15: "Fifteen", 
            16: "Sixteen", 
            17: "Seventeen", 
            18: "Eighteen", 
            19: "Nineteen", 
            20: "Twenty", 
            30: "Thirty", 
            40: "Forty", 
            50: "Fifty", 
            60: "Sixty", 
            70: "Seventy", 
            80: "Eighty", 
            90: "Ninety", 
            100: "Hundred", 
            1000: "Thousand", 
            1000000: "Million", 
            1000000000: "Billion" 
        }

        words = ""
        places = [1000, 1000000, 1000000000]
        counter = 0
        if num == 0:
            return "Zero"
        while num > 0:
            # 1000
            if num > 0:
                # 10[00]
                words += self.remainder(num % 100, cases) + " "
                num //= 100
            # 10
            if num > 0:
                words += "Hundred"
            if num > 0:
                words +=  " " + cases[num % 10]
                num //= 10

            if num > 0: 
                words += " " + cases[places[counter]] + " "
                counter += 1
        
        reversedWords = words.split(" ")
        if reversedWords[-1] == "":
            reversedWords.pop()
        reversedWords = reversedWords[::-1]
        if reversedWords[-1] == "":
            reversedWords.pop()
        words = ""
        for i in range(len(reversedWords) - 1):
            words += reversedWords[i] + " "
        words += reversedWords[-1]
        
        return words

    def remainder(self, num: int, cases: dict) -> str:
        words = ""
        if num >= 21:
            words += cases[num % 10] + " " + cases[num // 10 * 10]
        else: 
            words = cases[num]
        return words
    
    """
    Linked Lists
    """
    # 2. Add Two Numbers
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:



solution = Solution()
