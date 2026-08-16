/*
https://practice.geeksforgeeks.org/problems/x-total-shapes/0
X Total Shapes Submissions: 6763   Accuracy: 53.81%   Difficulty: Hard

Given N * M string array of O's and X's
Return the number of 'X' total shapes. 'X' shape consists of one or more adjacent X's (diagonals not included).

Example (1):

OOOXOOO
OXXXXXO
OXOOOXO

answer is 1 , shapes are  :
(i)     X
    X X X X
    X     X


Example (2):

XXX
OOO
XXX

answer is 2, shapes are
(i)  XXX

(ii) XXX

Input:
The first line of input takes the number of test cases, T.
Then T test cases follow. Each of the T test cases takes 2 input lines.
The first line of each test case have two integers N and M.The second line of N space separated strings follow which indicate the elements in the array.

Output:

Print number of shapes.

Constraints:

1<=T<=100

1<=N,M<=50

Example:

Input:
2
4 7
OOOOXXO OXOXOOX XXXXOXO OXXXOOO
10 3
XXO OOX OXO OOO XOX XOX OXO XXO XXX OOO

Output:
4
6

dfs count connect components
 */

#include <vector>
#include <iostream>
#include <string>
using namespace std;

void print(const vector<string>& shape) {
  for (int i = 0; i < shape.size(); ++i) cout << shape[i] << endl;
  cout << endl;
}

void dfs(const vector<string>& shape, vector<vector<bool>>& visited, int i, int j) {
  //cout << i << ' ' << j << endl;
  visited[i][j] = true;
  for (int m = -1; m <= 1; m++) {
    for (int n = -1; n <= 1; n++) {
      if (m == 0 && n ==0) continue;
      if (m != 0 && n != 0) continue;
      int ni = i+m, nj = n+j;
      if (ni >= 0 && ni < shape.size() && nj >= 0 && nj < shape[0].size() && !visited[ni][nj] && shape[i][j] == 'X') {
        dfs(shape, visited, ni, nj);
      }
    }
  }
}

int main(int argc, char** argv) {
  int total;
  cin >> total;
  while(--total >= 0) {
    int m, n;
    cin >> m >> n;
    vector<string> shape(m);
    for (int i = 0; i < m; ++i) cin >> shape[i];
    //print(shape);
    int count = 0;
    vector<vector<bool>>visited(m, vector<bool>(n, false));
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        if (shape[i][j] == 'X' && !visited[i][j]) {
          //cout << "new component" << endl;
          count++;
          dfs(shape, visited, i, j);
        }
      }
    }
    cout << count << endl;
  }
}
