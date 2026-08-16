/*
https://leetcode.com/problems/implement-magic-dictionary/description/

676. Implement Magic Dictionary
DescriptionHintsSubmissionsDiscussSolution
Implement a magic directory with buildDict, and search methods.

For the method buildDict, you'll be given a list of non-repetitive words to build a dictionary.

For the method search, you'll be given a word, and judge whether if you modify exactly one character into another character in this word, the modified word is in the dictionary you just built.

Example 1:
Input: buildDict(["hello", "leetcode"]), Output: Null
Input: search("hello"), Output: False
Input: search("hhllo"), Output: True
Input: search("hell"), Output: False
Input: search("leetcoded"), Output: False
Note:
You may assume that all the inputs are consist of lowercase letters a-z.
For contest purpose, the test data is rather small by now. You could think about highly efficient algorithm after the contest.
Please remember to RESET your class variables declared in class MagicDictionary, as static/class variables are persisted across multiple test cases. Please see here for more details.
Seen this question in a real interview before?
Difficulty:Medium
Total Accepted:17.9K
Total Submissions:36.2K
Contributor:just_not_over_12
Subscribe to see which companies asked this question.

Related Topics

Similar Questions
Implement Trie (Prefix Tree)Longest Word in Dictionary
C++

the solution build large dictionary where each word and position is replace with a *.

A more space efficent, but slow trie implemtnation is
https://leetcode.com/problems/implement-magic-dictionary/discuss/107453/Easiest-JAVA-with-Trie-no-need-to-count-the-number-of-changes
*/
class MagicDictionary {
2
public:
3
    /** Initialize your data structure here. */
4
    MagicDictionary() {
5

6
    }
7

8
    /** Build a dictionary through a list of words */
9
    void buildDict(vector<string> dict) {
10

11
    }
12

13
    /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
14
    bool search(string word) {
15

16
    }
17
};
18

19
/**
20
 * Your MagicDictionary object will be instantiated and called as such:
21
 * MagicDictionary obj = new MagicDictionary();
22
 * obj.buildDict(dict);
23
 * bool param_2 = obj.search(word);
24
 */
