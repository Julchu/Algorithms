# You are given N numbers in the form of array. 
# You have to select K (K <= N) numbers from those numbers. 
# You can select numbers from end only (either front or last). 
# After selection the number gets erased from the array. 
# Find the max sum

# A = [1, 9, 3, 4]
# K = 2

# Possible: 
# 1, 9 -> 10
# 1, 4
# 4, 1
# 4, 3

# Hint he gave later: 
# 1, 9 -> 10
# 1, 4 -> (1 + 9) - 9 + 4
# 4, 1
# 4, 3 -> (1 + 4) - 1 + 3

# 9, 3 NOT Possible

# Algorithm hint 1:

# max_sum = 0
# current_sum = sum(A[0:k])
# max_sum = current_sum
# current_sum = current_sum - A[k-1] + A[n-1]
# max_sum = current_sum

# Algorithm hint 2:

# max_sum = current_sum = sum(A[0:k])
# for i in range(0, k):
# 	current_sum = current_sum - A[k-1-i] + A[n-1-i]
# 	max_sum = max(max_sum, current_sum)

# He then just asked what the time and space complexity of this would be
