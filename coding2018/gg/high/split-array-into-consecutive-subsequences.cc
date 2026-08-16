/*
https://leetcode.com/problems/split-array-into-consecutive-subsequences/description/

659. Split Array into Consecutive Subsequences
DescriptionHintsSubmissionsDiscussSolution
You are given an integer array sorted in ascending order (may contain duplicates), you need to split them into several subsequences, where each subsequences consist of at least 3 consecutive integers. Return whether you can make such a split.

Example 1:
Input: [1,2,3,3,4,5]
Output: True
Explanation:
You can split them into two consecutive subsequences :
1, 2, 3
3, 4, 5
Example 2:
Input: [1,2,3,3,4,4,5,5]
Output: True
Explanation:
You can split them into two consecutive subsequences :
1, 2, 3, 4, 5
3, 4, 5
Example 3:
Input: [1,2,3,4,4,5]
Output: False
Note:
The length of the input is in range of [1, 10000]
Seen this question in a real interview before?
Difficulty:Medium
Total Accepted:12.5K
Total Submissions:33.2K
Contributor:fallcreek
Subscribe to see which companies asked this question.

Related Topics

Similar Questions
Top K Frequent Elements
C++

simple try all prefix of consecutive number and recursion.
dp: for a prefix, loop through its prefix and make sure suffix is contsecutive >=3.
The issue is there will be repeated nums left. 12334455
*/
class Solution {

public:

    bool isPossible(vector<int>& nums) {



    }
};
/*
Call a chain a sequence of 3 or more consecutive numbers.

Considering numbers x from left to right, if x can be added to a current chain, it's at least as good to add x to that chain first, rather than to start a new chain.

Why? If we started with numbers x and greater from the beginning, the shorter chains starting from x could be concatenated with the chains ending before x, possibly helping us if there was a "chain" from x that was only length 1 or 2.

Algorithm

Say we have a count of each number, and let tails[x] be the number of chains ending right before x.

Now let's process each number. If there's a chain ending before x, then add it to that chain. Otherwise, if we can start a new chain, do so.

It's worth noting that our solution can be amended to take only O(1)O(1) additional space, since we could do our counts similar to Approach #1, and we only need to know the last 3 counts at a time.


http://www.cnblogs.com/grandyang/p/7525821.html

first all greedy works, no need to explore all prefix. the need is used to extend previous chain. the freq is the remaining counts. Only start a new chain if it is not an extension.
*/
class Solution {
public:
    bool isPossible(vector<int>& nums) {
        unordered_map<int, int> freq, need;
        for (int num : nums) ++freq[num];
        for (int num : nums) {
            if (freq[num] == 0) continue;
            else if (need[num] > 0) {
                --need[num];
                ++need[num + 1];
            } else if (freq[num + 1] > 0 && freq[num + 2] > 0) {
                --freq[num + 1];
                --freq[num + 2];
                ++need[num + 3];
            } else return false;
            --freq[num];
        }
        return true;
    }
};
