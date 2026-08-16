/*
This problem is an interactive problem new to the LeetCode platform.

We are given a word list of unique words, each word is 6 letters long, and one word in this list is chosen as secret.

You may call master.guess(word) to guess a word.  The guessed word should have type string and must be from the original list with 6 lowercase letters.

This function returns an integer type, representing the number of exact matches (value and position) of your guess to the secret word.  Also, if your guess is not in the given wordlist, it will return -1 instead.

For each test case, you have 10 guesses to guess the word. At the end of any number of calls, if you have made 10 or less calls to master.guess and at least one of these guesses was the secret, you pass the testcase.

Besides the example test case below, there will be 5 additional test cases, each with 100 words in the word list.  The letters of each word in those testcases were chosen independently at random from 'a' to 'z', such that every word in the given word lists is unique.

Example 1:
Input: secret = "acckzz", wordlist = ["acckzz","ccbazz","eiowzz","abcczz"]

Explanation:

master.guess("aaaaaa") returns -1, because "aaaaaa" is not in wordlist.
master.guess("acckzz") returns 6, because "acckzz" is secret and has all 6 matches.
master.guess("ccbazz") returns 3, because "ccbazz" has 3 matches.
master.guess("eiowzz") returns 2, because "eiowzz" has 2 matches.
master.guess("abcczz") returns 4, because "abcczz" has 4 matches.

We made 5 calls to master.guess and one of them was the secret, so we pass the test case.
Note:  Any solutions that attempt to circumvent the judge will result in disqualification.

Seen this question in a real interview before?
Difficulty:Hard
Total Accepted:6.2K
Total Submissions:16.7K
Contributor:1337c0d3r
Subscribe to see which companies asked this question.

Related Topics

C++

1
*/
/**
2
 * // This is the Master's API interface.
3
 * // You should not implement it, or speculate about its implementation
4
 * class Master {
5
 *   public:
6
 *     int guess(string word);
7
 * };
8
 */

class Solution {

public:

    void findSecretWord(vector<string>& wordlist, Master& master) {



    }
};

Intuition

We can guess that having less words in the word list is generally better. If the data is random, we can reason this is often the case.

Now let's use the strategy of making the guess that minimizes the maximum possible size of the resulting word list. If we started with NN words in our word list, we can iterate through all possibilities for what the secret could be.

Algorithm

Store H[i][j] as the number of matches of wordlist[i] and wordlist[j]. For each guess that hasn't been guessed before, do a minimax as described above, taking the guess that gives us the smallest group that might occur.


class Solution(object):
    def findSecretWord(self, wordlist, master):
        N = len(wordlist)
        self.H = [[sum(a==b for a,b in itertools.izip(wordlist[i], wordlist[j]))
                   for j in xrange(N)] for i in xrange(N)]

        possible, path = range(N), ()
        while possible:
            guess = self.solve(possible, path)
            matches = master.guess(wordlist[guess])
            if matches == len(wordlist[0]): return
            possible = [j for j in possible if self.H[guess][j] == matches]
            path = path + (guess,)

    def solve(self, possible, path = ()):
        if len(possible) <= 2: return possible[0]

        ansgrp, ansguess = possible, None
        for guess, row in enumerate(self.H):
            if guess not in path:
                groups = [[] for _ in xrange(7)]
                for j in possible:
                    if j != guess:
                        groups[row[j]].append(j) # breaks possible into groups by num of matching char
                maxgroup = max(groups, key = len) # the max reamining words with same matching chars
                if len(maxgroup) < len(ansgrp):
                    ansgrp, ansguess = maxgroup, guess

        return ansguess
