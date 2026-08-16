/*
https://practice.geeksforgeeks.org/problems/maximum-index/0
Given an array A of integers, find the maximum of j - i subjected to the constraint of A[i] <= A[j].

Example :

A : [3 5 4 2]

Output : 2
for the pair (3, 4)



Input:

The first line contains an integer T, depicting total number of test cases.
Then following T lines contains an integer N depicting the size of array and next line followed by the value of array.


Output:

Print the maximum difference of the indexes i and j in a separtate line.


Constraints:

1 ≤ T ≤ 100
1 ≤ N ≤ 1000
0 ≤ A[i] ≤ 100


Example:

Input
1
2
1 10
Output
1

keep track of descreasing sequence.
when we have a long decreasing, then a hug bar, then we can't get rid of previous ones, so following bar will has to scan this backward again. so not linear

A[i] < 100, so we could fill firstappear to be the smallest index where i appear, then
compute smallerappears, then scan N from right to left and j - smallerappears[N[j]], linear time.
turn out this is not the case the input could be huge
File "/home/a51abda00f5c17e6dbb5930b9ce705a3.py", line 7, in <module>
    assert max_num >= 0 and max_num <= 100, "too big %s" % max_num
AssertionError: too big 2147469842

*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;

int main_smalll() {
  int total;
  cin >> total;
  for (int i = 0; i < total; ++i) {
    int N;
    cin >> N;
    vector<int> nums(N);
    for (int j = 0; j < N; ++j) cin >> nums.at(j);

    int max_num = *max_element(nums.begin(), nums.end());
    vector<int> appears(max_num+1, N);
    for (int j = 0; j < N; ++j) appears.at(nums.at(j)) = min(appears.at(nums.at(j)), j);
    vector<int>smaller_appears(max_num+1, N);
    int cur_min = N;
    for (int j = 0; j <= max_num; ++j) {
      cur_min = min(cur_min, appears.at(j));
      smaller_appears.at(j) = cur_min;
    }
    int cur_max = 0;
    for (int j = N-1; j >= 0; --j) {
      cur_max = max(cur_max, j-smaller_appears.at(nums.at(j)));
    }
    cout << cur_max << endl;
  }
}

int main_nlogn() {
  int total;
  cin >> total;
  for (int i = 0; i < total; ++i) {
    int N;
    cin >> N;
    vector<int> nums(N);
    map<int, int> appears; // map value to where it appears first
    for (int j = 0; j < N; ++j) {
      cin >> nums.at(j);
      if (appears.find(nums[j]) == appears.end()) {
        appears[nums[j]] = j;
      }
    }
    map<int, int> smaller_appears; // smallest index for all values <= v
    int cur_min = N;
    for (auto it = appears.begin(); it != appears.end(); ++it) {
      cur_min = min(cur_min, it->second);
      smaller_appears[it->first] = cur_min;
    }

    int cur_max = 0;
    for (int j = N-1; j >= 0; --j) {
      cur_max = max(cur_max, j-smaller_appears[nums.at(j)]);
    }
    cout << cur_max << endl;
  }
}

// https://www.geeksforgeeks.org/given-an-array-arr-find-the-maximum-j-i-such-that-arrj-arri/
// create a decreasing sequencde (the smallest elment so far)and a increasing sequence from right (the largest element so far).
int main() {
  int total;
  cin >> total;
  for (int i = 0; i < total; ++i) {
    int N;
    cin >> N;
    vector<int> nums(N);
    for (int j = 0; j < N; ++j) {
      cin >> nums.at(j);
    }
    vector<int> smallest(N), largest(N);
    smallest[0] = nums[0]; largest[N-1] = nums[N-1];
    for (int j = 1; j < N; ++j) {
      smallest[j] = min(smallest[j-1], nums[j]);
    }
    for (int j = N-2; j >= 0; --j) {
      largest[j] = max(largest[j+1], nums[j]);
    }
    int cur_max = 0;
    for (int i = 0, j = 0; i < N && j < N; ) {
      if (smallest[i] <= largest[j]) {
        cur_max = max(cur_max, j-i);
        j++;
      } else i++;
    }
    cout << cur_max << endl;
  }
}
