/*
https://practice.geeksforgeeks.org/problems/overlapping-intervals/0
Overlapping Intervals Submissions: 9224   Accuracy: 34.55%   Difficulty: Medium

Given a collection of Intervals,merge all the overlapping Intervals.
For example:

Given [1,3], [2,6], [8,10], [15,18],

return [1,6], [8,10], [15,18].

Make sure the returned intervals are sorted.



Input:

The first line contains an integer T, depicting total number of test cases.
Then following T lines contains an integer N depicting the number of Intervals and next line followed by the intervals starting and ending positions 'x' and 'y' respectively.


Output:

Print the intervals after overlapping in sorted manner.  There should be a newline at the end of output of every test case.

Constraints:

1 ≤ T ≤ 50
1 ≤ N ≤ 100
0 ≤ x ≤ y ≤ 100


Example:

Input
2
4
1 3 2 4 6 8 9 10
4
6 8 1 9 2 4 4 7

Output
1 4 6 8 9 10
1 9
 */

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
  int total;
  cin >> total;
  for (int i = 0; i < total; ++i) {
    //cout << "working on " << i << " out of " << total << endl;
    int num_interval;
    cin >> num_interval;
    vector<pair<int, int> > intervals;
    for (int j = 0; j < num_interval*2; j += 2) {
      int left, right;
      cin >> left >> right;
      intervals.push_back({left, right});
    }

    sort(intervals.begin(), intervals.end(), [](const pair<int,int>& one, const pair<int,int>& two) {
        return one.first < two.first;
      });
    int left = intervals[0].first, right = intervals[0].second;
    bool first = true;
    for (int j = 1; j < num_interval; ++j) {
      if (intervals[j].first <= right) right = max(right, intervals[j].second);
      else {
        if (first) {
          first = false;
        } else cout << ' ';

        cout << left << ' ' << right;
        left = intervals[j].first;
        right = intervals[j].second;
      }
    }
    if (first) {
      first = false;
    } else cout << ' ';
    cout << left << ' ' << right << endl;
  }
  return 0;
}
