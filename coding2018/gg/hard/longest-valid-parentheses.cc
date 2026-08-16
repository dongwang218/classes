/*
https://practice.geeksforgeeks.org/problems/longest-valid-parentheses/0
Longest valid Parentheses Submissions: 15932   Accuracy: 24.1%   Difficulty: Hard

Given a string S consisting of opening and closing parenthesis '(' and ')'. Find length of the longest valid parenthesis substring.

Input:
First line contains number of test cases T.  Each test case have one line string S of character '(' and ')' of length  N.

Constraints:
1 <= T <= 500
1 <= N <= 105

Example:
Input:
2
((()
)()())

Output:
2
4

Explanation:
Testcase 1: Longest valid balanced parantheses is () and its length is 2.


 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
using namespace std;

int main() {
  int total;
  cin >> total;
  for (int i = 0; i < total; ++i) {
    string line;
    cin >> line;
    stack<int> pos;
    int cur_max = 0;
    for (int j = 0; j < line.size(); ++j) {
      if (line[j] == ')' && !pos.empty() && line[pos.top()] == '(') {
        pos.pop();
        int left = !pos.empty() ? pos.top() : -1;
        cur_max = max(cur_max, j - left);
      } else {
        pos.push(j);
      }
    }
    cout << cur_max << endl;
  }
}
