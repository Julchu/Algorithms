# typing module needed for dictionaries (Lists)
from typing import List

class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1
sc = 1
newColor = 2

'''
sr: 1, sc: 1
1 1 1
1 1 0
1 0 1
'''

color = image[sr][sc]
queue = []
queue.append([sr, sc])

while queue:
    coords = queue.pop(0) # [1, 1]
    row = coords[0]
    col = coords[1]
    
    # left
    if 0 < col:
        if image[row][col-1] == color:
            image[row][col-1] = newColor
            queue.append([row, col-1])
            print([row, col-1])

    # right
    if col < len(image[0])-1:
        if image[row][col+1] == color:
            image[row][col+1] = newColor
            queue.append([row, col+1])
            print([row, col+1])

    # top
    if 0 < row:
        if image[row-1][col] == color:
            image[row-1][col] = newColor
            queue.append([row-1, col])
            print([row-1, col])

    # bot
    if row < len(image)-1:
        if image[row+1][col] == color:
            image[row+1][col] = newColor
            queue.append([row+1, col])
            print([row+1, col])

array1 = [[1,1,1],[1,1,0],[1,0,1]]

array2 = array1
for i in range(len(array1)):
    array2[i] = [False] * len(array1[i])