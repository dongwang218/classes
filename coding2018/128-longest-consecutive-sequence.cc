/*Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Your algorithm should run in O(n) complexity.

Example:

Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
Seen this question in a real interview before?
Difficulty:Hard
Total Accepted:161.1K
Total Submissions:409.6K
Contributor:LeetCode
Subscribe to see which companies asked this question.

Related Topics

Similar Questions
Binary Tree Longest Consecutive Sequence
C++
*/

#include <unordered_map>
#include <vector>

using namespace std;

// put into hash_map
// go through data again
//   check if +1 exists and -1 does not exists, then remove it from hash_map while continue +1 check
// return the max sequence length
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
      std::unordered_map<int, bool> visited;
      for (auto num: nums) {
        visited[num] = true;
      }
      int maxLength = 0;
      for (auto num: nums) {
        if (visited.find(num-1) == visited.end()) {
          int start = num;
          while (visited.find(num+1) != visited.end()) {
            num++;
          }
          maxLength = std::max(maxLength, num-start+1);
        }
      }
      return maxLength;
    }
};


































// https://leetcode.com/problems/longest-consecutive-sequence/description/
