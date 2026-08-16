from typing import List

def next_permute(n: List[int]) -> List[int]:
    for i in range(len(n)-2, -1, -1):
        if n[i] < n[i+1]:
            break
    if i < 0:
        return n

    # find j that is bigger than n[i]
    for j in range(len(n)-1, i, -1):
        if n[j] > n[i]:
            tmp = n[j]
            n[j] = n[i]
            n[i] = tmp
            return n[:i+1] + n[i+1:][::-1]

    assert False


assert next_permute([1, 2]) == [2, 1]
assert next_permute([3,1,5]) == [3,5,1]
assert next_permute([5,8,3]) == [8,3,5], "{}".format(next_permute([5,8,3]))
assert next_permute([1,2,3,8,9]) == [1,2,3,9,8]
assert next_permute([3,4,7,2,2,6,4,1]) == [3,4,7,2,4,1,2,6]

