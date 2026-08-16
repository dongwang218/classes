/*
https://practice.geeksforgeeks.org/problems/allocate-minimum-number-of-pages/0

Allocate minimum number of pages Submissions: 2548   Accuracy: 30.65%   Difficulty: Hard

You are given N number of books. Every ith book has Pi number of pages.
You have to allocate books to M number of students. There can be many ways or permutations to do so. In each permutation one of the M students will be allocated the maximum number of pages. Out of all these permutations, the task is to find that particular permutation in which the maximum number of pages allocated to a student is minimum of those in all the other permutations, and print this minimum value.

Each book will be allocated to exactly one student. Each student has to be allocated atleast one book.

Note: Return -1 if a valid assignment is not possible, and allotment should be in contiguous order (see explanation for better understanding).

Input:

The first line contains 'T' denoting the number of testcases. Then follows description of T testcases:
Each case begins with a single positive integer N denoting the number of books.
The second line contains N space separated positive integers denoting the pages of each book.
And the third line contains another integer M, denoting the number of students.


Output:
For each test case, output a single line containing minimum number of pages each student has to read for corresponding test case.


Constraints:
1<= T <=70
1<= N <=50
1<= A [ i ] <=250
1<= M <=50

Example:
Input:
1
4
12 34 67 90
2

Output:
113

Explaination: Allocation can be done in following ways:

{12} and {34, 67, 90}     Maximum Pages = 191

{12, 34} and {67, 90}     Maximum Pages = 157

{12, 34, 67} and {90}        Maximum Pages = 113

Therefore, the minimum of these cases is 113, which is selected as output.

two ways to do it:
dp: f[n][m] compute minum of assign first n books to m people,
  f[n][m] = min(max(f[n-1][i], sum(i+1, ..., m)) for i in n-1 to m)
complexity is n * m * m.
binary search for the minimum num, complexity O(n * log(sum(pages))), use binary search
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

// only return false when there are books left
bool isFeasible(const vector<int>& pages, int N, int M, int max_pages) {
  int remaining_pp = M;
  int cur_sum = 0;
  for (int j = 0; j < N; ++j) {
    if (cur_sum+pages[j] <= max_pages) {
      cur_sum += pages[j];
    } else {
      cur_sum = pages[j];
      remaining_pp--;
    }
  }
  return remaining_pp > 0;
}
int main() {
  int total;
  cin >> total;
  for (int i = 0; i < total; ++i) {
    int N, M; // N books, M people
    cin >> N;
    vector<int> pages(N);
    for (int j = 0; j < N; ++j) cin >> pages[j];
    cin >> M;
    int right = accumulate(pages.begin(), pages.end(), 0);
    int left = *max_element(pages.begin(), pages.end());

    int result = -1;
    while (left <= right && N >= M) {
      int mid = (left+right)/2;
      //cout << "left " << left << " right " << right << " mid " << mid << endl;
      if (isFeasible(pages, N, M, mid)) {
        //cout << "yes\n";
        result = mid;
        right = mid-1;
      } else {
        //cout << "no\n";
        left = mid+1;
      }
    }
    cout << result << endl;
  }
}
