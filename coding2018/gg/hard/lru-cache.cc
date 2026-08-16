/*
https://practice.geeksforgeeks.org/problems/lru-cache/1
LRU Cache Submissions: 14822   Accuracy: 21.64%   Difficulty: Hard

The task is to design and implement methods of an LRU cache. The class has two methods get and set which are defined as follows.
get(x)   : Gets the value of the key x if the key exists in the cache otherwise returns -1
set(x,y) : inserts the value if the key x is not already present. If the cache reaches its capacity it should invalidate the least recently used item before inserting the new item.
In the constructor of the class the size of the cache should be intitialized.

Input:
The first line of input contain an integer T denoting the no of test cases. Then T test case follow. Each test case contains 3 lines. The first line of input contains an integer N denoting the capacity of the cache and then in the next line is an integer Q denoting the no of queries Then Q queries follow. A Query can be of two types
1. SET x y : sets the value of the key x with value y
2. GET x : gets the key of x if present else returns -1.

Output:
For each test case in a new line output will be space separated values of the query.

Constraints:
1<=T<=100
1<=N<=10
1<=Q<=100

Example(To be used only for expected output):
Input
2
2
2
SET 1 2 GET 1
2
7
SET 1 2 SET 2 3 SET 1 5 SET 4 5 SET 6 7 GET 4 GET 1

Output
2
5 -1

Note:The Input/Output format and Example given are used for system's internal purpose, and should be used by a user for Expected Output only. As it is a function problem, hence a user should not read any input from stdin/console. The task is to complete the function specified, and not to write the full code.


 */


#include<bits/stdc++.h>
using namespace std;
class LRUCache
{
public:
    LRUCache(int );

    int get(int );

    void set(int , int );
};
int main()
{
    int t;
    cin>>t;
    while(t--)
    {
    int n;
    cin>>n;
    LRUCache *l  = new LRUCache(n);
    int q;
    cin>>q;
    for(int i=0;i<q;i++)
    {
        string a;
        cin>>a;
        if(a=="SET")
        {
            int aa,bb;
            cin>>aa>>bb;
            l->set(aa,bb);
        }else if(a=="GET")
        {
            int aa;
            cin>>aa;
            cout<<l->get(aa)<<" ";
        }
    }
    cout<<endl;
    }
}



/*Please note that it's Function problem i.e.
you need to write your solution in the form of Function(s) only.
Driver Code to call/invoke your function is mentioned above.*/

/*The structure of the class is as follows
class LRUCache
{
public:
    LRUCache(int );
    int get(int );
    void set(int , int );
};*/
/*You are required to complete below methods */
/*Inititalize an LRU cache with size N */
int cap;
unordered_map<int, pair<list<int>::iterator, int> > cache;
list<int> lru; // keys

LRUCache::LRUCache(int N)
{
     //Your code here
  cap = N;
  cache.clear();
  lru.clear();
}
/*Sets the key x with value y in the LRU cache */
void LRUCache::set(int x, int y)
{
     //Your code here
  auto it = cache.find(x);
  if (it != cache.end()) {
    lru.erase(it->second.first);
    cache.erase(it);
  }
  if (cache.size() == cap) {
    auto& key = lru.front();
    cache.erase(cache.find(key));
    lru.pop_front();
  }
  lru.push_back(x);
  cache.insert({x, {--lru.end(), y}});
}
/*Returns the value of the key x if
present else returns -1 */
int LRUCache::get(int x)
{
    //Your code here
  auto it = cache.find(x);
  if (it != cache.end()) {
    lru.erase(it->second.first);
    lru.push_back(x);
    it->second.first = --lru.end();
    return it->second.second;
  } else {
    return -1;
  }
}
