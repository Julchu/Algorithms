# typing module needed for dictionaries (Lists)
from typing import List

def lengthOfLongestSubstring(s: str) -> int:
	substrings = []
	substringLengths = []
	exists = {}
	a = ""
	output = 0

	for i in s:
		if not exists.get(i):
			exists[i] = "yes"
			a = a + i
		else:
			substrings.append(a)
			substringLengths.append(len(a))
			a = i
	substrings.append(a)
	substringLengths.append(len(a))
	min = 0
	index = 0
	if len(substringLengths) > 0:
		for i in range(len(substringLengths)):
			if substringLengths[i] > min:
				min = substringLengths[i]
				index = i
		output = len(substrings[index])
	return output

print(lengthOfLongestSubstring(s))


# def lengthOfLongestSubstring(s: str) -> int:
# 	n = len(s)
# 	ans = 0
# 	index = 0
# 	j = 0
# 	if n > 0:
# 		max = s[j]
# 	for i in range(n):
		
		
# 		i
		
# 		if ans <  j - i + 1:
# 		ans = j - i + 1
# 		a


# public int lengthOfLongestSubstring(String s) {
# 	int n = s.length(), ans = 0;
# 	int[] index = new int[128]; // current index of character
# 	// try to extend the range [i, j]
# 	for (int j = 0, i = 0; j < n; j++) {
# 		i = Math.max(index[s.charAt(j)], i);
# 		ans = Math.max(ans, j - i + 1);
# 		index[s.charAt(j)] = j + 1;
# 	}
# 	return ans;
# }