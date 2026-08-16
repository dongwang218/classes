/*
https://practice.geeksforgeeks.org/problems/find-median-in-a-stream/0
Find median in a stream Submissions: 28271   Accuracy: 30.57%   Difficulty: Hard

Given an input stream of n integers the task is to insert integers to stream and print the median of the new stream formed by each insertion of x to the stream.

Example

Flow in stream : 5, 15, 1, 3
5 goes to stream --> median 5 (5)
15 goes to stream --> median 10 (5, 15)
1 goes to stream --> median 5 (5, 15, 1)
3 goes to stream --> median 4 (5, 15, 1, 3)

Input:
The first line of input contains an integer N denoting the no of elements of the stream. Then the next N lines contains integer x denoting the no to be inserted to the stream.

Output:
For each element added to the stream print the floor of the new median in a new line.

Constraints:
1<=N<=105 + 7
1<=x<=105 + 7

Example:
Input:
4
5
15
1
3
Output:
5
10
5
4
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;

int main() {
  priority_queue<int, vector<int>, std::greater<int>> upper;
  priority_queue<int> lower;

  int total, n;
  cin >> total;
  for (int i = 0; i < total; ++i) {
    cin >> n;
    if (lower.empty() || n < lower.top()) lower.push(n);
    else upper.push(n);

    if (lower.size() > upper.size() + 1) {
      n = lower.top();
      lower.pop();
      upper.push(n);
    }

    if (upper.size() > lower.size() + 1) {
      n = upper.top();
      upper.pop();
      lower.push(n);
    }

    int mid;
    if (lower.size() == upper.size()) {
      mid = (lower.top() + upper.top()) / 2;
    } else if (lower.size() < upper.size()) {
      mid = upper.top();
    } else {
      mid = lower.top();
    }
    cout << mid << endl;
  }
}
