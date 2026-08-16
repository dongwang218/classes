# '.' Matches any single character.
# '*' Matches zero or more of the preceding element.
# s could be empty and contains only lowercase letters a-z.
# p could be empty and contains only lowercase letters a-z, and characters like . or *.

class Solution(object):
  def isMatch(self, s, p):
    """
    :type s: str
    :type p: str
    :rtype: bool
    """
    if len(s) == 0 and len(p) == 0:
      return True
    if len(p) > 1 and p[1] == '*':
      if self.isMatch(s, p[2:]): return True # zero

      i = 0
      while (s[i] == p[0] or p[0] == '.') and i < len(s):
        i += 1
        if self.isMatch(s[i:], p): return True
      return self.isMatch(s[i:], p[2:])
    elif len(s) > 0 and len(p) == 0:
      return False
    elif len(s) == 0 and len(p) > 0:
      return False
    elif s[0] == p[0] or p[0] == '.':
      return self.isMatch(s[1:], p[1:])
    else:
      return False


if __name__ == '__main__':
  tests = [('aa', 'a', False),
           ("aa", "a*", True),
           ('ab', '.*', True),
           ('aab', 'c*a*b', True),
           ('mississippi', 'mis*is*p*.', False)
  ]
  sln = Solution()
  for s, p, r in tests:
    assert(sln.isMatch(s, p) == r)


























# https://leetcode.com/problems/regular-expression-matching/description/
