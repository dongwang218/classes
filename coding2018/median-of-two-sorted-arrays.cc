// O(log (m+n))
// -std=c++11

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cassert>

using namespace std;

class Solution {
public:
  // -1, 0, 1, less, equal, bigger than median
  int testMedian(int med, int index, vector<int>& nums) {
    if (index < 0) return 1;
    else if (index >= nums.size()) return -1;
    if (index > 0 && med < nums[index-1]) return -1;
    else if (med > nums[index]) return 1;
    else return 0;
  }

  // Note: if total is even, need to test 2 position
  int isMedian(int med, int index, vector<int>& nums, bool even) {
    int test1 = testMedian(med, index, nums);
    if (test1 == 0) return 0;
    if (even && index > 0) {
      int test2 = testMedian(med, index-1, nums);
      if (test2 == 0) return 0;
      assert(test1 == test2);
      return test1;
    } else {
      return test1;
    }
  }

  // there could be two medians next to each other in one
  int testNeighbour(vector<int>& one, vector<int>& another, int mid) {
    int total = one.size() + another.size();
    int rank = (int)(total/2);
    bool even = (total % 2) == 0;
    if (mid > 0) {
      if (isMedian(one[mid-1], rank-(mid-1), another, even) == 0) return(mid-1);
    }
    if (mid < one.size()-1) {
      if (isMedian(one[mid+1], rank-(mid+1), another, even) == 0) return(mid+1);
    }
    return -1;
  }

  double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    vector<int> candidates;
    int mid1 = findOneInAnother(nums1, nums2);
    int mid2 = findOneInAnother(nums2, nums1);

    if (mid1 >= 0) {
      candidates.push_back(nums1[mid1]);
      int mid2 = testNeighbour(nums1, nums2, mid1);
      if (mid2 >= 0) {
        candidates.push_back(nums1[mid2]);
      }
    }

    if (mid2 >= 0) {
      candidates.push_back(nums2[mid2]);
      int mid1 = testNeighbour(nums2, nums1, mid2);
      if (mid1 >= 0) {
        candidates.push_back(nums2[mid1]);
      }
    }
    std::sort(candidates.begin(), candidates.end());
    auto last = std::unique(candidates.begin(), candidates.end());
    candidates.erase(last, candidates.end());
    cout << endl << "candidates: ";
    std::for_each (candidates.begin(), candidates.end(), [](const int d) {
        cout << d << ' ';
      });
    assert(candidates.size() <= 2);

    double accum = 0.0;
    std::for_each (candidates.begin(), candidates.end(), [&](const int d) {
        accum += d;
      });
    return ((double)accum / candidates.size());
  }

  // binary search to test for median from array one
  int findOneInAnother(vector<int>& one, vector<int>& another) {
    int total = one.size() + another.size();
    int rank = (int)(total / 2);
    bool even = (rank % 2) == 0;
    int left = 0, right = one.size()-1;
    int mid = -1;
    while (left <= right) {
      mid = (int)(left+right)/2;
      int med = one[mid];

      int test = isMedian(med, rank-mid, another, even);
      if (test == 0) return mid;
      else if (test < 0) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }
    return(-1);
  }

};

int main(int argc, char** argv) {
  int m = 100;
  int n = 21;

  Solution sln;
  vector<int> nums1, nums2;
  for (int i = 0; i < m; i++) {
    nums1.push_back(rand() % 100);
  }
  for (int i = 0; i < n; i++) {
    nums2.push_back(rand() % 100);
  }
  std::sort(nums1.begin(), nums1.end());
  std::sort(nums2.begin(), nums2.end());
  cout << "array 1" << endl;
  std::for_each(nums1.cbegin(), nums1.cend(),
                [] (const int c) {std::cout << c << " ";}
                );

  cout << endl << "array 2" << endl;
  std::for_each(nums2.cbegin(), nums2.cend(),
                [] (const int c) {std::cout << c << " ";}
                );

  double median = sln.findMedianSortedArrays(nums1, nums2);

  for (vector<int>::iterator iter = nums2.begin(); iter != nums2.end(); ++iter) {
    nums1.push_back(*iter);
  }
  std::sort(nums1.begin(), nums1.end());
  int len = (m+n);
  double expected = (len % 2 == 0) ? (nums1[len/2-1] + nums1[len/2]) / 2.0 : nums1[(int)(len/2)];

  cout << endl << "merged array" << endl;
  std::for_each(nums1.cbegin(), nums1.cend(),
                [] (const int c) {std::cout << c << " ";}
                );


  cout << "actual" << median << " expected " << expected << endl;
}
















// https://leetcode.com/problems/median-of-two-sorted-arrays/description/
