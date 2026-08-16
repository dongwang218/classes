# Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.

# 1 -1 longest substring where sum keep >=0, end at 0
# scan left to right, keep track of current valid subsequence
# if two sequence has same count, then no need to keep the shorter one.

class Sequence(object):
  def __init__(self, start):
    self.start = start
    self.end = None # the position that it ends
    self.balance = 1

  def add(self):
    self.balance += 1

  def subtract(self, pos):
    self.balance -= 1
    if self.balance < 0:
      return False
    else:
      if self.balance == 0:
        self.end = pos
      return True
  def valid(self):
    return self.end is not None

class Solution(object):
  def longestValidParentheses(self, s):
    """
    :type s: str
    :rtype: int
      """
    open = []
    closed = []
    for index, c in enumerate(list(s)):
      if c == '(':
        for i in range(len(open)-1, -1, -1):
          open[i].add()
        open.append(Sequence(index))
      else:
        assert(c == ')')
        for i in range(len(open)-1, -1, -1):
          status = open[i].subtract(index)
          if not status:
            done = open.pop(i)
            if done.valid():
              closed.append(done)

    for done in open:
      if done.valid():
        closed.append(done)

    if closed:
      best = sorted(closed, key = lambda x: x.end - x.start)[-1]
      print('best is ', s[best.start:(best.end+1)])
      return best.end-best.start+1
    else:
      return 0

if __name__ == '__main__':
  sln = Solution()
  tests = ['', '(', ')', '()', ')(', '(()', '()(', '()())', '((((())))()(()))']
  for t in tests:
    print('input %s, result %d' % (t, sln.longestValidParentheses(t)))



































# https://leetcode.com/problems/longest-valid-parentheses/description/
