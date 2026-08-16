from __future__ import annotations

# 198757573965186

def depth_sum(a, depth=1):
    direct = 0
    recursive = 0
    for item in a:
        if isinstance(item, int):
            direct += item
        else:
            recursive += depth_sum(item, depth=depth+1)
    return recursive + direct*depth

# without recursion, because this is tail recursive
def depth_sum2(a):
    depth = 0
    sum = 0
    remaining = a
    while remaining:
        depth += 1
        left = []
        for item in remaining:
            if isinstance(item, int):
                sum += item*depth
            else:
                left.extend(item)
        remaining = left
    return sum
        
if __name__ == '__main__':

    assert depth_sum([1, [2, 3]]) == 11
    assert depth_sum([8, 4, [5, [9], 3], 6]) == 61

    assert depth_sum2([1, [2, 3]]) == 11
    assert depth_sum2([8, 4, [5, [9], 3], 6]) == 61
