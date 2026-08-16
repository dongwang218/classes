/*
https://practice.geeksforgeeks.org/problems/egg-dropping-puzzle/0
https://code.oursky.com/famous-egg-dropping-puzzle-in-combinatorics/
Egg Dropping Puzzle Submissions: 15134   Accuracy: 41.45%   Difficulty: Medium

Suppose you have N eggs and you want to determine from which floor in a K-floor building you can drop an egg such that it doesn't break. You have to determine the minimum number of attempts you need in order find the critical floor in the worst case while using the best strategy.There are few rules given below.

An egg that survives a fall can be used again.
A broken egg must be discarded.
The effect of a fall is the same for all eggs.
If the egg doesn't break at a certain floor, it will not break at any floor below.
If the eggs breaks at a certain floor, it will break at any floor above.
For more description on this problem see wiki page

Input:
The first line of input is  T denoting the number of testcases.Then each of the T lines contains two positive integer N and K where 'N' is the number of eggs and 'K' is number of floor in building.

Output:
For each test case, print a single line containing one integer the minimum number of attempt you need in order find the critical floor.

Constraints:
1<=T<=30
1<=N<=10
1<=K<=50


Example:
Input:
1
2 10

Output:
4

try 4, if broken, 1, 2, 3; if not, then 7, if borken, 5 6; then 7+2=9, if broken, 8

the num_attempts m should satisfy 1 C m + ... + min(N,m) C m >= K

num[K][N] then maintain the prefix sum to test, num of attempt much smaller than K, so
fill the array for attempt from 1, 2, 3, until the sum is bigger than K.
 */

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void printPascal(const vector<vector<int>> pascal) {
  for (int i = 0; i < pascal.size(); ++i) {
    for (int j = 0; j < pascal[i].size(); ++j) cout << pascal[i][j] << ',';
    cout << endl;
  }
}
int main() {
  int total;
  cin >> total;
  for (int i = 0; i < total; ++i) {
    int N, K; // N eggs, K floors
    cin >> N >> K;
    vector<vector<int>> pascal(K+1, vector<int>(N+1, 1));
    bool found = false;
    for (int m = 1; m <= K && !found; ++m) {
      int sum = 0;
      for (int j = 1; j <= min(m, N); ++j) {
        pascal[m][j] = pascal[m-1][j-1] + (m-1 >= j ? pascal[m-1][j] : 0);
        sum += pascal[m][j];
        if (sum >= K) {
          found = true;
          cout << m << endl;
          //cout << "found" << endl;
          break;
        }
      }
    }
    //printPascal(pascal);
  }
}
