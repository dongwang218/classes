// 890 https://leetcode.com/problems/find-and-replace-pattern/description/

#include <string>
#include <vector>
#include <unordered_map>
using namespace std;

class Solution {
public:
    vector<string> findAndReplacePattern(vector<string>& words, string pattern) {
      vector<string> results;
      for (const auto& word: words) {
        unordered_map<char, char> perm, reverse;
        if (word.size() != pattern.size()) continue;
        int i = 0;
        for (; i < pattern.size(); ++i) {
          char p = pattern[i], w = word[i];
          auto it = perm.find(p);
          if (it != perm.end() && w != it->second) break;
          it = reverse.find(w);
          if (it != reverse.end() && p != it->second) break;
          perm[p] = w;
          reverse[w] = p;
        }
        if (i == pattern.size()) results.push_back(word);
      }
      return results;
    }
};
