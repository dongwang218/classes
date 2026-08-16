/*
https://practice.geeksforgeeks.org/problems/word-break-part-2/0
Word Break - Part 2 Submissions: 3375   Accuracy: 32.66%   Difficulty: Hard

Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.

Return all such possible sentences.

For example, given
s = "snakesandladder",
dict = ["snake", "snakes", "and", "sand", "ladder"].

A solution is ["snakes and ladder",
           "snake sand ladder"].

Input:
The first line contains an integer T, denoting the number of test cases.
Every test case has 3 lines.
The first line contains an integer N, number of words in the dictionary.
The second line contains N strings denoting the words of the dictionary.
The third line contains a string s.

Output:
For each test case, print all possible strings in one line. Each string is enclosed in (). See Example.
If no string possible print "Empty" (without quotes).

Constraints:
1<=T<=100
1<=N<=20
1<=Length of each word in dictionary <=15
1<=Length of s<=1000


Note: Make sure the strings are sorted in your result.

Exapmle:
Input:
1
5
lr m lrm hcdar wk
hcdarlrm

Output:
(hcdar lr m)(hcdar lrm)

use dp https://www.youtube.com/watch?v=UPO-moDOYlo, store the places that works
 */

#include <string>
#include <vector>
#include <iostream>
using namespace std;

// 0 to 1 is first character
bool contains(const vector<string>& dict, const string& str, int from, int to) {
  string sub = str.substr(from, to-from);
  for (const auto& one: dict) {
    if (sub == one) return true;
  }
  return false;
}

void getString(const vector<vector<int>>& valid, const string& str, int at, vector<string>& results, string prefix) {
  if (at == 0) results.push_back(prefix);
  else {
    for (const auto& j: valid[at]) {
      getString(valid, str, j, results,
                str.substr(j, at-j) + (prefix.empty() ? "" : (" " + prefix)));
    }
  }
}

int main(int argc, char** argv) {
  int total;
  cin >> total;
  while (--total >= 0) {
    int n;
    cin >> n;
    vector<string> dict(n);
    for (int i = 0; i < n; ++i) cin >> dict[i];
    string str;
    cin >> str;
    int len = str.size();
    vector<vector<int> > valid(len+1);
    for (int i = 1; i <= len; ++i) {
      for (int j = i-1; j >= 0; --j) {
        if ((j == 0 || !valid[j].empty()) && contains(dict, str, j, i)) {
          valid[i].push_back(j);
        }
      }
    }
    if (valid[len].empty()) cout << "Empty";
    else {
      vector<string> results;
      getString(valid, str, len, results, "");
      for (const auto& r: results) {
        cout << "(" << r << ")";
      }
    }
    cout << endl;
  }
}
