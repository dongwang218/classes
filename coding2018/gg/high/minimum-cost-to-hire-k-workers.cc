/*
857. Minimum Cost to Hire K Workers
DescriptionHintsSubmissionsDiscussSolution
There are N workers.  The i-th worker has a quality[i] and a minimum wage expectation wage[i].

Now we want to hire exactly K workers to form a paid group.  When hiring a group of K workers, we must pay them according to the following rules:

Every worker in the paid group should be paid in the ratio of their quality compared to other workers in the paid group.
Every worker in the paid group must be paid at least their minimum wage expectation.
Return the least amount of money needed to form a paid group satisfying the above conditions.



Example 1:

Input: quality = [10,20,5], wage = [70,50,30], K = 2
Output: 105.00000
Explanation: We pay 70 to 0-th worker and 35 to 2-th worker.
Example 2:

Input: quality = [3,1,10,10,1], wage = [4,8,2,2,7], K = 3
Output: 30.66667
Explanation: We pay 4 to 0-th worker, 13.33333 to 2-th and 3-th workers seperately.


Note:

1 <= K <= N <= 10000, where N = quality.length = wage.length
1 <= quality[i] <= 10000
1 <= wage[i] <= 10000
Answers within 10^-5 of the correct answer will be considered correct.
Seen this question in a real interview before?
Difficulty:Hard
Total Accepted:5.6K
Total Submissions:12.9K
Contributor:mj_coder
Subscribe to see which companies asked this question.

Related Topics

C++
the difficulty is due to the ratio adding a new one affects the existing.

https://leetcode.com/problems/minimum-cost-to-hire-k-workers/solution/
The greedy soln tries to set each one as the minimum expectation guy and then include others with adjusted salary. then pick the smallest k-1 as the total cost.
*/
class Solution {

public:

    double mincostToHireWorkers(vector<int>& quality, vector<int>& wage, int K) {



    }
};


Intuition

As in Approach #1, at least one worker is paid their minimum wage expectation.

Additionally, every worker has some minimum ratio of dollars to quality that they demand. For example, if wage[0] = 100 and quality[0] = 20, then the ratio for worker 0 is 5.0.

The key insight is to iterate over the ratio. Let's say we hire workers with a ratio R or lower. Then, we would want to know the K workers with the lowest quality, and the sum of that quality. We can use a heap to maintain these variables.

Algorithm

Maintain a max heap of quality. (We're using a minheap, with negative values.) We'll also maintain sumq, the sum of this heap.

For each worker in order of ratio, we know all currently considered workers have lower ratio. (This worker will be the 'captain', as described in Approach #1.) We calculate the candidate answer as this ratio times the sum of the smallest K workers in quality.

a better explaination is at https://buptwc.github.io/2018/06/26/Leetcode-857-Minimum-Cost-to-Hire-K-Workers/
basically the critial person has the hightest wage/quality. So if we add workers in increasing order, then the cost is the smallest quality k-1 people.

    def mincostToHireWorkers(self, quality, wage, K):
        from fractions import Fraction
        workers = sorted((Fraction(w, q), q, w)
                         for q, w in zip(quality, wage))

        ans = float('inf')
        pool = []
        sumq = 0
        for ratio, q, w in workers:
            heapq.heappush(pool, -q)
            sumq += q

            if len(pool) > K:
                sumq += heapq.heappop(pool)

            if len(pool) == K:
                ans = min(ans, ratio * sumq)

        return float(ans)
