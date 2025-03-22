#47. Count Inversions
def count_inversions(arr):
    def merge_and_count(arr, temp_arr, left, mid, right):
        i, j, k = left, mid + 1, left
        inv_count = 0

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                i += 1
            else:
                temp_arr[k] = arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1

        while i <= mid:
            temp_arr[k] = arr[i]
            i += 1
            k += 1

        while j <= right:
            temp_arr[k] = arr[j]
            j += 1
            k += 1

        for i in range(left, right + 1):
            arr[i] = temp_arr[i]
        return inv_count

    def merge_sort_and_count(arr, temp_arr, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += merge_sort_and_count(arr, temp_arr, left, mid)
            inv_count += merge_sort_and_count(arr, temp_arr, mid + 1, right)
            inv_count += merge_and_count(arr, temp_arr, left, mid, right)
        return inv_count

    return merge_sort_and_count(arr, [0] * len(arr), 0, len(arr) - 1)

arr = [1, 20, 6, 4, 5]
print("Number of Inversions:", count_inversions(arr))

#48. Find the Longest Palindromic Substring
def longest_palindromic_substring(s):
    def expand_from_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    longest = ""
    for i in range(len(s)):
        odd_palindrome = expand_from_center(i, i)
        even_palindrome = expand_from_center(i, i + 1)

        if len(odd_palindrome) > len(longest):
            longest = odd_palindrome
        if len(even_palindrome) > len(longest):
            longest = even_palindrome
    return longest

s = "babad"
print("Longest Palindromic Substring:", longest_palindromic_substring(s))

#49. Traveling Salesman Problem (TSP)
from itertools import permutations

def tsp(graph, start):
    n = len(graph)
    vertices = list(range(n))
    vertices.remove(start)

    min_cost = float('inf')
    best_route = None

    for perm in permutations(vertices):
        current_cost = 0
        k = start
        for i in perm:
            current_cost += graph[k][i]
            k = i
        current_cost += graph[k][start]

        if current_cost < min_cost:
            min_cost = current_cost
            best_route = (start,) + perm + (start,)
    return best_route, min_cost

graph = [
    [0, 29, 20, 21],
    [29, 0, 15, 17],
    [20, 15, 0, 28],
    [21, 17, 28, 0]
]
start = 0
route, cost = tsp(graph, start)
print("Shortest Route:", route)
print("Minimum Cost:", cost)

#50. Graph Cycle Detection
def has_cycle(graph, node, visited, parent):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            if has_cycle(graph, neighbor, visited, node):
                return True
        elif parent != neighbor:
            return True
    return False

def is_cyclic(graph, n):
    visited = [False] * n
    for node in range(n):
        if not visited[node]:
            if has_cycle(graph, node, visited, -1):
                return True
    return False

graph = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1, 3],
    3: [2]
}
print("Graph contains cycle:", is_cyclic(graph, 4))

#51. Longest Substring Without Repeating Characters
def length_of_longest_substring(s):
    char_index = {}
    left = max_length = 0

    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        char_index[char] = right
        max_length = max(max_length, right - left + 1)
    return max_length
    
s = "abcabcbb"
print("Length of Longest Substring:", length_of_longest_substring(s))

#52. Find All Valid Parentheses Combinations
def generate_parentheses(n):
    def backtrack(s, left, right):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + "(", left + 1, right)
        if right < left:
            backtrack(s + ")", right + 1, left)
    result = []
    backtrack("", 0, 0)
    return result

n = 3
print("Valid Parentheses Combinations:", generate_parentheses(n))

#53. Zigzag Level Order Traversal of Binary Tree
from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def zigzag_level_order(root):
    if not root:
        return []

    result, queue, left_to_right = [], deque([root]), True

    while queue:
        level, size = [], len(queue)
        for _ in range(size):
            node = queue.popleft()
            if left_to_right:
                level.append(node.val)
            else:
                level.insert(0, node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)
        left_to_right = not left_to_right
    return result

root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

print("Zigzag Level Order Traversal:", zigzag_level_order(root))

#54. Palindrome Partitioning
def is_palindrome(s, left, right):
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

def partition_palindrome(s):
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return

        for end in range(start, len(s)):
            if is_palindrome(s, start, end):
                backtrack(end + 1, path + [s[start:end + 1]])
    result = []
    backtrack(0, [])
    return result

s = "aab"
print("Palindrome Partitions:", partition_palindrome(s))
