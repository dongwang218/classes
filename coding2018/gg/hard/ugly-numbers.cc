/*
https://practice.geeksforgeeks.org/problems/ugly-numbers/0
Ugly Numbers Submissions: 7982   Accuracy: 36.84%   Difficulty: Hard

Ugly numbers are numbers whose only prime factors are 2, 3 or 5. The sequence 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, … shows the first 11 ugly numbers. By convention, 1 is included. Write a program to find Nth Ugly Number.

Input:
The first line of input contains an integer T denoting the number of test cases. T testcases follow. For each testcase there is one line of input that contains the number N.

Output:
Print the Nth Ugly Number.

Constraints:
1 ≤ T ≤ 104
1 ≤ N ≤ 104

Example:
Input:
2
10
4
Output:
12
4

  2 3 5
2 1 0 0
3 0 1 0
4 2 0 0
5 0 0 1
12 2 1 0
15 0 1 1
18 1 2 0
20 2 0 1
24 3 1 0

solution: https://www.geeksforgeeks.org/ugly-numbers/
merge three array of ugly numbers
*/

#include <vector>
#include <iostream>
using namespace std;

typedef unsigned long long ll;
int main(int argc, char** argv) {
  int total;
  cin >> total;
  vector<int> requests(total);
  int max_N = 0;
  for (int i = 0; i < total; ++i) {
    cin >> requests[i];
    max_N = max(max_N, requests[i]);
  }

  vector<ll> num(max_N);
  num[0] = 1;
  int i2 = 0, i3 = 0, i5 = 0;
  ll u2 = 2, u3 = 3, u5 = 5;
  for (int i = 1; i < max_N; ++i) {
    ll n = min(u2, min(u3, u5));
    num[i] = n;
    if (n == u2) {
      i2 ++;
      u2 = 2 * num[i2];
    }
    if (n == u3) {
      i3 ++;
      u3 = 3 * num[i3];
    }
    if (n == u5) {
      i5 ++;
      u5 = 5 * num[i5];
    }
  }
  for (int i = 0; i < total; ++i) {
    cout << num[requests[i]-1] << endl;
  }
}
