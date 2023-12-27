            
# Different ways of coding the same: 
a_list = [1,2,3]
a_reverse = a_list[::-1]

if i < len(a_reverse):
    digit_a = int(a_reverse[i]) 
else:
    digit_a = 0 

            
digit_a = int(a_reverse[i]) if i < len(a_reverse) else 0


# Write a function to find the longest common prefix string amongst an array of strings. 
# If there is no common prefix, return an empty string "".
class Solution(object):
    def longestCommonPrefix(self, strs):
        prefix = ''

        min_length =  min(len(s) for s in strs)
        
        for i in range(min_length):
            current_char = strs[0][i]
            
            if all(s[i] == current_char for s in strs):
                prefix = prefix + current_char  
            
            else:
                break  

        return prefix

# code to test solution
sol = Solution()
print(sol.longestCommonPrefix['text', 'textbook'])

# good to understand methods like sort, pop
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        # initializations
        indices_with_val = []
        k = 0

        # sort the lists such that pop has no impact on moving numbers
        nums.sort() #ascending

        # get the indices of where val occurs in nums
        for i, num in enumerate(nums):
            if num == val:
                indices_with_val.append(i)
            else:
                k += 1

        # sort the lists such that pop has no impact on moving numbers        
        indices_with_val.sort(reverse = True) #descending

        # pop indexes where nums = val. in-place so use pop instead of remove
        for index in indices_with_val:
            nums.pop(index)

        return k
    
# good to understand pop that it also returns while deleting and a dictionary and a stack
class Solution(object):
    def isValid(self, s):
        # initiate stack and dictionary
        stack = []
        dictionary = {')':'(', ']':'[', '}':'{'}

        # loop over string
        for char in s:
            if char in dictionary:
                if not stack or stack.pop() != dictionary[char]:
                    return False
            else:
                stack.append(char)

        return not stack
        """
        :type s: str
        :rtype: bool
        """
#dictionary
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # dictionary to save the frequencies of the different items
        dictionary = {}
        max_frequency = 1
        majority_element = 0

        # base case: if list only contains 1 number
        if len(nums) == 1:
            return nums[0]

        # loop over elements and add to dictionary with their frequency
        for num in nums:
            if num in dictionary:
                dictionary[num] = dictionary[num] + 1
                if dictionary[num] > max_frequency:
                    max_frequency = dictionary[num]
                    majority_element = num
            else:
                dictionary[num] = 1 # frequency is 1 when you add it
        
        return majority_element

# sqrt definition using binary search instead of brute force
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0:
            return 0
        if x == 1:
            return 1
        
        left, right = 0, x // 2  # Initialize the search range
        
        while left <= right:
            mid = left + (right - left) // 2  # Calculate the midpoint
            
            if mid * mid == x:
                return mid  # Found the square root
            elif mid * mid < x:
                left = mid + 1  # Adjust the left boundary
            else:
                right = mid - 1  # Adjust the right boundary
        
        return right + 1 if (right + 1) * (right + 1) <= x else right  # Return the rounded square root

# brute force vs binary search
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        ''' BRUTE FORCE
        for i, num in enumerate(nums):
            if num == target:
                return i
        
        return -1
        '''

        index_left = 0
        index_right = len(nums) - 1

        while index_right - index_left >= 0:
            index_mid = index_left + (index_right - index_left)//2
            if nums[index_mid] == target:
                return index_mid
            if nums[index_mid] < target:
                index_left = index_mid + 1
            else:
                index_right = index_mid - 1
        
        return -1

'''

x--------------x-------------x
left          mid            right

'''

# stairs is with fibonacci: and a way to fill a list with empty values
ways = [0 for _ in range(n)]

# binary tree (94)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []

        def traverse(node):
            if node:
                # Recursively traverse the left subtree
                traverse(node.left)
                # Visit the current node and add its value to the result
                result.append(node.val)
                # Recursively traverse the right subtree
                traverse(node.right)

        # Start the traversal from the root node
        traverse(root)

        return result


# binary tree
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """

        def traverse(node1, node2):
            # Check if both nodes are None
            if not node1 and not node2:
                return True

            # If only one of them is None, they are not the same
            if not node1 or not node2:
                return False

            # Check if the values of the current nodes are equal
            if node1.val != node2.val:
                return False

            # Recursively compare the left and right subtrees
            return traverse(node1.left, node2.left) and traverse(node1.right, node2.right)

        # Start the traversal from the root nodes of both trees
        return traverse(p, q)
    
# symmetric binary tree
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def isMirror(left, right):
            # If both nodes are None, they are symmetric
            if not left and not right:
                return True
            # If only one of them is None, they are not symmetric
            if not left or not right:
                return False
            # Check if the values are the same and if the subtrees are mirrors
            return (left.val == right.val) and \
                isMirror(left.left, right.right) and \
                isMirror(left.right, right.left)

        # Start the symmetry check from the root's left and right subtrees
        return isMirror(root, root)
    
# binary tree maxDepth using recursive nested function or self.
class Solution(object):
    def maxDepth(self, root):
        # Base case
        # If the subtree is empty i.e. root is NULL, return depth as 0
        if root == None:
            return 0
        # if root is not NULL, call the same function recursively for its left child and right child...
        # When the two child function return its depth...
        # Pick the maximum out of these two subtrees and return this value after adding 1 to it
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        depth = max(left_depth, right_depth) + 1 # Adding 1 is the current node 
        return depth
    
class Solution(object):
    def maxDepth(self, root):
        def findMaxDepth(node):
            if node is None:
                return 0
            left_depth = findMaxDepth(node.left)
            right_depth = findMaxDepth(node.right)
            return max(left_depth, right_depth) + 1

        # Call the nested function to compute the maximum depth
        return findMaxDepth(root)
    
#binary tree good explanations
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """

        ## Base case: if the root is None, there's no path, so return False
        if root is None:
            return False
        
        # Subtract the current node's value from the targetSum
        targetSum -= root.val
        
        # If the current node is a leaf and targetSum is 0, we found a valid path
        if not root.left and not root.right:
            return targetSum == 0
        
        # Recursively check left and right subtrees for a valid path
        left_result = self.hasPathSum(root.left, targetSum)
        right_result = self.hasPathSum(root.right, targetSum)
        
        # Return True if either the left or right subtree has a valid path
        return left_result or right_result

# character operations palindrome
Class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """

        # convert uppercase letters into lowercase letters
        s = s.lower()

        # remove special characters
        s= ''.join(letter for letter in s if letter.isalnum())

        # check if string is the same as reversed string --> if yes, palindrome
        return s == s[::-1]
        
'''
or with .replace
# initializing bad_chars_list
bad_chars = [';', ':', '!', "*", " "]

# remove bad_chars
for i in bad_chars:
    test_string = test_string.replace(i, '')
'''


# .remove(value) .pop(index) .append(value) --> not needed to save as ... = 
Class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        stack = []

        for i in nums:
            if i in stack:
                stack.remove(i)
            else:
                stack.append(i)
        
        return stack[0]

# interesting join at the end of a list to a string
class Solution(object):
    def convertToTitle(self, columnNumber):
        """
        :type columnNumber: int
        :rtype: str
        """
        # initialize the alphabet
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        # could also just do letters = 'abcdefghijklmnopqrstuvwxyz'
        
        result = []
        # for numbers within the first alphabet
        while columnNumber > 0:
            remainder = columnNumber % 26
            letter = letters[remainder -1]
            result.append(letter)
            columnNumber = (columnNumber-1) // 26

        return ''.join(result[::-1])


# merge DF & join SQL
# pandas
import pandas as pd

# Merge the two DataFrames using a left join
result = pd.merge(Person, Address, on='personID', how='left')
or with left_on and right_on
# Select the desired columns
result = result[['firstname', 'lastname', 'city', 'state']]

# SQL
SELECT a.firstname, a.lastname, b.city, b.state
FROM Person a LEFT JOIN Address b 
ON a.personID = b.personID

# SQL who earns more than his manager (derived table)
SELECT derivedTable.name as Employee
FROM 
    (
    SELECT a.name, a.salary, b.managerId, b.salary as managersalary
    FROM employee a iNNER JOIN employee b
    ON a.managerId = b.id
    ) as derivedTable
WHERE salary > managersalary

# sets and replacing values like this [0] * nr_colums
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        nr_rows, nr_columns = len(matrix), len(matrix[0])
        zero_rows = set()
        zero_columns = set()

        for i, row in enumerate(matrix):
            for j, columnvalue in enumerate(row):
                if columnvalue == 0:
                    zero_rows.add(i)
                    zero_columns.add(j)

        for row in zero_rows:
            matrix[row] = [0] * nr_columns
        
        for column in zero_columns:
            for i in range(nr_rows):
                matrix[i][column] = 0

# backward range:
for i in range(final_column, first_column - 1, -1):

# insert in list without replacing
fruits = ['apple', 'banana', 'cherry']

fruits.insert(1, "orange")

print(fruits)

# minstack functions
class MinStack(object):

    def __init__(self):
        self.stack = []

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.stack.append(val)

    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop(-1)

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return min(self.stack)


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()



# zip
class Solution(object):
    def carFleet(self, target, position, speed):
        """
        :type target: int
        :type position: List[int]
        :type speed: List[int]
        :rtype: int
        """
        pair = [[p,s] for p, s in zip(position, speed)]
        pair.sort()

        stack = []

        for p, s in pair[::-1]:
            time = (target - p)/s
            stack.append(time)
            
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()
        
        return len(stack)