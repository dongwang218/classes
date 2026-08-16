/*
https://practice.geeksforgeeks.org/problems/travelling-salesman-problem/0
Travelling Salesman Problem Submissions: 1947   Accuracy: 39.13%   Difficulty: Hard

Given a matrix M of size N where M[i][j] denotes the cost of moving from city i to city j. Your task is to complete a tour from the city 0 (0 based index) to all other cities such that you visit each city atmost once and then at the end come back to city 0 in min cost.

Input:
The first line of input contains an integer T denoting the no of test cases. Then T test cases follow. Each test case contains an integer N denoting the size of the matrix then in the next line are N*N space separated values of the matrix M.

Output:
For each test case print the required result denoting the min cost of the tour in a new line.

Constraints:
1<=T<=15
1<=N<=12
1<=M[][]<=10000

Example:
Input:
2
2
0 111
112 0
3
0 1000 5000
5000 0 1000
1000  5000  0
Output:
223
3000


 */
