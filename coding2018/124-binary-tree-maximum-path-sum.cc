/*Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

Example 1:

Input: [1,2,3]

       1
      / \
     2   3

Output: 6
Example 2:

Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42
*/

// 11 -10 9 20 15 7
// scan from left to right if cur sum is positive continue, otherwise stop and start when it become positive
// this is incorrect, the found subsequence may not be a tree

/*if node maintain the max sum path to itself, then we easily get max path to it, also get maximal updown path maintain a path through it. a node can decide to pick left, right, or alone to start the path. for the updown path only need to maintain the max path length below (could be a, could be ba, ac, or bac)*/

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cassert>

using namespace std;
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

class Solution {
public:

  // return the length of max path to node, and update the maxSum
  int maxPathSumRecursion(TreeNode* node, int& maxSum) {
    if (node == NULL) return INT_MIN;
    int leftSum = maxPathSumRecursion(node->left, maxSum);
    int rightSum = maxPathSumRecursion(node->right, maxSum);

    int result = node->val;
    result += max(max(leftSum, rightSum), 0);

    int thisSum = node->val;
    thisSum += max(leftSum, 0);
    thisSum += max(rightSum, 0);

    if (thisSum > maxSum) maxSum = thisSum;
    return result;
  }

    int maxPathSum(TreeNode* root) {
      int maxSum = INT_MIN;
      maxPathSumRecursion(root, maxSum);
      return maxSum;
    }
};
































// https://leetcode.com/problems/binary-tree-maximum-path-sum/description/
//https://www.geeksforgeeks.org/find-maximum-path-sum-two-leaves-binary-tree/
