# 1) [2, 7, 4, 1, 3, 6]
def count_pairs_with_sum(lst, target_sum):
    count = 0
    seen = set()
    for num in lst:
        complement = target_sum - num
        if complement in seen:
            count += 1
        seen.add(num)
    return count
# 2)
def calculate_range(lst):
    if len(lst) < 3:
        return "Range determination not possible"
    return max(lst) - min(lst)
# 3)
def matrix_power(matrix, m):
    def multiply_matrices(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
                for j in range(len(b[0]))] for i in range(len(a))]
   
    result = matrix
    for _ in range(m-1):
        result = multiply_matrices(result, matrix)
    return result
# 4)
def highest_occurring_char(text):
    letters = [c.lower() for c in text if c.isalpha()]
    if not letters:
        return None, 0
    freq = {}
    for ch in letters:
        if ch in freq:
            freq[ch] += 1
        else:
            freq[ch] = 1
    max_char = None
    max_count = 0
    for ch in freq:
        if freq[ch] > max_count:
            max_char = ch
            max_count = freq[ch]
    return max_char, max_count


if __name__ == "__main__":
    nums = [2, 7, 4, 1, 3, 6]
    print(f"Pairs which give 10 oin adding: {count_pairs_with_sum(nums, 10)}")
   
    nums_range = [5, 3, 8, 1, 0, 4]
    print(f"Range of list: {calculate_range(nums_range)}")
   
    matrix = [[1, 2], [3, 4]]
    print(f"Matrix squared: {matrix_power(matrix, 2)}")

    test_str = "hippopotamus"
    char, count = highest_occurring_char(test_str)
    print(f"Most frequent character: '{char}' (appears {count} times)")
