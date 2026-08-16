import sys
total = int(sys.stdin.readline())
for i in range(total):
  N = int(sys.stdin.readline())
  nums = [int(j) for j in sys.stdin.readline().split(' ')]
  max_num = max(nums)
  appears = [N]*(max_num+1)
  for j in range(N):
    appears[nums[j]] = min(appears[nums[j]], j)
  smaller_appears = [N] * (max_num+1)
  cur_min = N
  for j in range(max_num+1):
    cur_min = min(cur_min, appears[j])
    smaller_appears[j] = cur_min

  cur_max = 0
  for j in range(N):
    cur_max = max(cur_max, j-smaller_appears[nums[j]])

  print(cur_max)
