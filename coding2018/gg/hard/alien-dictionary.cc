/*
https://practice.geeksforgeeks.org/problems/alien-dictionary/1
Alien Dictionary Submissions: 3688   Accuracy: 34.53%   Difficulty: Hard

Given a sorted dictionary of an alien language having N words and k starting alphabets of standard dictionary the task is to complete the function which returns a string denoting the order of characters in the language..

Note: Many orders may be possible for a particular test case, thus you may return any valid order.


Examples:

Input:  Dict[] = { "baa", "abcd", "abca", "cab", "cad" }, k = 4
Output: Function returns "bdac"
Here order of characters is 'b', 'd', 'a', 'c'
Note that words are sorted and in the given language "baa"
comes before "abcd", therefore 'b' is before 'a' in output.
Similarly we can find other orders.

Input:  Dict[] = { "caa", "aaa", "aab" }, k = 3
Output: Function returns "cab"

Input:
The first line of input contains an integer T denoting the no of test cases. Then T test cases follow. Each test case contains an integer N and k denoting the size of the dictionary. Then in the next line are sorted space separated values of the alien dictionary.

Output:
For each test case in a new line output will be 1 if the order of string returned by the function is correct else 0 denoting incorrect string returned.

Constraints:
1<=T<=200
1<=N<=100
1<=k<=26
1<=Length of words <=1000

Example:
Input:
2
5 4
baa abcd abca cab cad
3 3
caa aaa aab

Output:
1
1

how to represent the precedence relation? we can process each pair of string and find the first different character and add an edge. Afterwards, look at all edges and select one that is not in the tail. It can be the first remove all edges that where this appears, and repeat to select the next. This works but it is not nice. There is also a full blown topilogical sort procedure using stack.
 */


//Initial Template for C++
/* script for test case http://code.geeksforgeeks.org/PetjYq */
#include<bits/stdc++.h>
#include <string>
using namespace std;
string printOrder(string [], int , int );
string order;
bool f(string a,string b)
{
	int p1=0;int p2=0;
	for(int i=0;i<min(a.size(),b.size()) and p1 ==p2;i++ )
	{
		p1 = order.find(a[i]);
		p2 = order.find(b[i]);
	//	cout<<p1<<" "<<p2<<endl;
	}

	if(p1 == p2 and a.size()!=b.size())
	return a.size()<b.size();

	return p1<p2;
}

// Driver program to test above functions
int main()
{
    int t;
    cin>>t;
    while(t--)
    {
        int n,k;
        cin>>n>>k;
    string *s = new string[n];
    for(int i=0;i<n;i++)cin>>s[i];

        string ss = printOrder(s,n,k);
        order="";
        for(int i=0;i<ss.size();i++)
        order+=ss[i];

        string *temp = new string[n];
        std::copy(s, s + n, temp);
        sort(temp,temp+n,f);

        bool f= true;
        for(int i=0;i<n;i++)
        if(s[i]!=temp[i])f=false;

        cout<<f;
        cout<<endl;

    }
	return 0;
}

void dfs(const vector<vector<int>>& graph, int node, vector<bool>& visited, stack<int>& order) {
  visited[node] = true;
  for (const auto& next: graph[node]) {
    if (!visited[next]) {
      dfs(graph, next, visited, order);
    }
  }
  order.push(node);
}

/*Please note that it's Function problem i.e.
you need to write your solution in the form of Function(s) only.
Driver Code to call/invoke your function is mentioned above.*/

//User function Template for C++
/*The function should return a string denoting the
order of the words in the Alien Dictionary */
// N string, K voc size
string printOrder(string dict[], int N, int K)
{
  //cout << endl << endl;
  vector<vector<int>> graph(K);
  // only need to compare adjacent words
  for (int i = 0; i < N-1; ++i) {
    for (int j = 0; j < min(dict[i].size(), dict[i+1].size()); ++j) {
      if (dict[i][j] != dict[i+1][j]) {
        int id1 = dict[i][j]-'a', id2 = dict[i+1][j]-'a';
        graph[id1].push_back(id2);
        break;
        //cout << "edge " << char(id1+'a') << ',' << char(id2+'a') << endl;
      }
    }
  }
  stack<int> order;
  vector<bool> visited(K, false);
  for (int i = 0; i < K; ++i) {
    if (!visited[i]) {
      dfs(graph, i, visited, order);
    }
  }
  string r;
  while (!order.empty()) {
    int node = order.top();
    order.pop();
    r += char(node+'a');
  }
  //cout << "order is " << r << endl;
  return r;
}
