// Input: [0,1,0,2,1,0,1,3,2,1,2,1]
// Output: 6
#include<iostream>
#include<vector>
#include<stdlib.h>
#include<cassert>

using namespace std;
// start condition: drop
// end condition: get back to same level
// increase trap amount
// remove included
class Trap {
public:
  Trap(int start, int height) {
    this->start = start;
    this->end = start;
    this->height = height;
    this->minHeight = height;
    this->capacity = 0;
  }

  bool addNext(int x, int y) {
    this->capacity += max(0, this->height - y);
    this->end = x;
    this->minHeight = min(this->minHeight, y);
    return(y >= this->height);
  }

  void normalize() {
    if (end < start) {
      int x = start;
      start = end;
      end = x;
    }
  }

  friend ostream& operator<<(ostream& os, const Trap& trap);

  int start;
  int minHeight;
  int end;
  int height;
  int capacity;
};

ostream& operator<<(ostream& os, const Trap& trap)
{
  os << "Trap(" << trap.start << ", " << trap.end << ", " << trap.capacity << ")";
  return os;
}

class Solution {
public:
  int trap(vector<int>& height) {
    vector<Trap> open, closed;
    for (int i = 0; i < height.size(); ++i) {
      for (auto it = open.begin(); it != open.end(); ) {
        if (it->addNext(i, height[i])) {
          closed.push_back(*it);
          it = open.erase(it);
        } else {
          ++it;
        }

      }
      if (i < height.size() - 1 && height[i] > height[i-1]) {
        open.emplace_back(Trap(i, height[i]));
      }
    }

    // do the same with the other direction
    open.clear();
    for (int i = height.size()-1; i >= 0; --i) {
      for (auto it = open.begin(); it != open.end(); ) {
        if (it->addNext(i, height[i])) {
          it->normalize();
          closed.push_back(*it);
          it = open.erase(it);
        } else {
          ++it;
        }

      }
      if (i > 0 &&  height[i-1] < height[i]) {
        open.emplace_back(i, height[i]);
      }
    }

    // remove included
    std::sort(closed.begin(), closed.end(), [](const Trap& a, const Trap& b) -> bool {
        if (a.start < b.start) return true;
        else if (a.start == b.start) return a.end > b.end;
        else return false;
      });

    int start = 0, end = 0;
    int cap = 0;
    for (int i = 0; i < closed.size(); ++i) {
      auto trap = closed[i];
      if (trap.start >= end) {
        cap += trap.capacity;
        start = trap.start;
        end = trap.end;
        cout << trap << endl;
      } else {
        // included
        assert(trap.end <= end);
      }
    }
    return cap;
  }
};


int main(int argc, char** argv) {
  Solution sln;
  vector<pair<vector<int>, int>> heights {
    make_pair<vector<int>, int>({0,1,0,2,1,0,1,3,2,1,2,1}, 6),
      make_pair<vector<int>, int>({4, 0, 3, 2, 0, 1}, 3+1)

        };

  for (auto x: heights) {
    int cap = sln.trap(x.first);
    cout << endl;
    std::for_each(x.first.begin(), x.first.end(), [](const int& h) {
        cout << h << ", ";
      });
    cout << " ";
    cout << "capacity " << cap << endl;
    assert(cap == x.second);
  }

  return 0;
}








































// https://leetcode.com/problems/trapping-rain-water/description/
