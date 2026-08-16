
def sum_to_integer_pointer(a, target):
    if not a:
        return False
    
    i = 0
    j = 0
    sum = a[0]

    while True:
        # print(i, j, sum)
        if sum < target:
            j += 1
            if j >= len(a):
                return False
            sum += a[j]
        elif sum > target:
            sum -= a[i]
            i += 1
        else:
            return True

def sum_to_integer_hash(a, target):
    if not a:
        return False

    sums = set()
    sum = 0
    for i in a:
        sum += i
        if sum == target or (sum-target) in sums:
            return True
        sums.add(sum)

    return False

if __name__ == '__main__':

    assert sum_to_integer_pointer([1,3,1,4,23], 8)
    assert not sum_to_integer_pointer([1, 3, 1, 4, 23], 7)

    assert sum_to_integer_hash([1,3,1,4,23], 8)
    assert not sum_to_integer_hash([1, 3, 1, 4, 23], 7)

    assert sum_to_integer_hash([1, 0, 3], 0)
    assert sum_to_integer_pointer([1, 0, 3], 0)
