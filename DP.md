动态规划就是缓存中间结果来加速计算，对计算顺序有严格的要求

本文中收录的题目都是有严格位置依赖的，也就是画表格填数，毫无语法难度

#### [1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/)

```java
class Solution {
    public int maxDistance(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dup = Arrays.copyOf(grid, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) continue;
                dup[i][j] = 201;
                if (i > 0) {
                    dup[i][j] = Math.min(dup[i][j], dup[i - 1][j] + 1);
                }
                if (j > 0) {
                    dup[i][j] = Math.min(dup[i][j], dup[i][j - 1] + 1);
                }
            }
        }
        int res = 0;
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (grid[i][j] == 1) continue;
                if (i < n - 1) {
                    dup[i][j] = Math.min(dup[i][j], dup[i + 1][j] + 1);
                }
                if (j < m - 1) {
                    dup[i][j] = Math.min(dup[i][j], dup[i][j + 1] + 1);
                }
                res = Math.max(res, dup[i][j]);
            }
        }
        return res == 201 ? -1 : res - 1;
    }
}
```

#### 1035. Uncrossed Lines

2D dp

```java
class Solution {  
    public int maxUncrossedLines(int[] nums1, int[] nums2) {  
        int n = nums2.length;  
        int[] dp = new int[n + 1];  
        for (int num : nums1) {  
            for (int j = n - 1; j >= 0; j--) {  
                if (num == nums2[j]) {  
                    // compare with left-up  
                    dp[j + 1] = dp[j] + 1;  
                }  
            }  
            for (int j = 0; j < n; j++) {  
                //  compare with left and up  
                dp[j + 1] = Math.max(dp[j + 1], dp[j]);  
            }  
        }  
        return dp[n];  
    }  
}
```

#### [97. Interleaving String](https://leetcode.com/problems/interleaving-string/)

java

```java
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length(), n = s2.length();
        if (s3.length() != m + n) {
            return false;
        }
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i < m; i++) {
            dp[i + 1][0] = dp[i][0] && s1.charAt(i) == s3.charAt(i);
        }
        for (int j = 0; j < n; j++) {
            dp[0][j + 1] = dp[0][j] && s2.charAt(j) == s3.charAt(j);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                boolean fromI = dp[i][j + 1] && s1.charAt(i) == s3.charAt(i + j + 1);
                boolean fromJ = dp[i + 1][j] && s2.charAt(j) == s3.charAt(i + j + 1);
                dp[i + 1][j + 1] = fromI || fromJ;
            }
        }
        return dp[m][n];
    }
}
```

#### [740. Delete and Earn](https://leetcode.com/problems/delete-and-earn/)

java

```java
class Solution {
    public int deleteAndEarn(int[] nums) {
        int bound = (int) 1e4 + 1;
        int[] bucket = new int[bound];
        for (int num : nums) {
            bucket[num] += num;
        }
        int[] dp = new int[bound];
        dp[0] = bucket[0];
        dp[1] = bucket[1];
        for (int i = 2; i < bound; i++) {
            dp[i] = Math.max(bucket[i] + dp[i - 2], dp[i - 1]);
        }
        return dp[bound - 1];
    }
}
```

#### [576. Out of Boundary Paths](https://leetcode.com/problems/out-of-boundary-paths/)

java

```java
class Solution {
    public int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
        int[][][] dp = new int[2][m][n];
        dp[0][startRow][startColumn] = 1;
        int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        int mod = (int) (1e9 + 7);
        int res = 0;
        for (int k = 0; k < maxMove; k++) {
            dp[(k + 1) & 1] = new int[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (dp[k & 1][i][j] == 0) {
                        continue;
                    }
                    for (int[] dir : dirs) {
                        int x = i + dir[0];
                        int y = j + dir[1];
                        if (x >= 0 && y >= 0 && x < m && y < n) {
                            dp[(k + 1) & 1][x][y] = (dp[(k + 1) & 1][x][y] + dp[k & 1][i][j]) % mod;
                        } else {
                            res = (res + dp[k & 1][i][j]) % mod;
                        }
                    }
                }
            }
        }
        return res;
    }
}
```

#### 894. All Possible Full Binary Trees

java

```java
class Solution {
    public List<TreeNode> allPossibleFBT(int n) {
        return dfs(n, new HashMap<>());
    }

    List<TreeNode> dfs(int n, HashMap<Integer, List<TreeNode>> cache) {
        List<TreeNode> res = new ArrayList<>();
        if ((n & 1) != 1) {
            return res;
        }
        if (n == 1) {
            res.add(new TreeNode(0));
            return res;
        }
        if (cache.containsKey(n)) {
            return cache.get(n);
        }
        for (int i = 1; i < n - 1; i += 2) {
            List<TreeNode> lefts = dfs(i, cache);
            List<TreeNode> rights = dfs(n - 1 - i, cache);
            for (TreeNode left : lefts) {
                for (TreeNode right : rights) {
                    TreeNode curr = new TreeNode(0);
                    curr.left = left;
                    curr.right = right;
                    res.add(curr);
                }
            }
        }
        cache.put(n, res);
        return res;
    }
}
```

#### 583. Delete Operation for Two Strings

Go

```go
func minDistance(word1 string, word2 string) int {  
   m, n := len(word1), len(word2)  
   dp := make([][]int, m+1)  
   for i := range dp {  
      dp[i] = make([]int, n+1)  
      dp[i][0] = i  
   }  
   for i := range dp[0] {  
      dp[0][i] = i  
   }  
   for i := range word1 {  
      for j := range word2 {  
         if word1[i] == word2[j] {  
            dp[i+1][j+1] = dp[i][j]  
         } else {  
            dp[i+1][j+1] = min(dp[i+1][j], dp[i][j+1]) + 1  
         }  
      }  
   }  
   return dp[m][n]  
}  
```

#### 712. Minimum ASCII Delete Sum for Two Strings

java

```java
class Solution {
    public int minimumDeleteSum(String s1, String s2) {
        int m = s1.length(), n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i < n; i++) {
            dp[0][i + 1] = s2.charAt(i) + dp[0][i];
        }
        for (int i = 0; i < m; i++) {
            dp[i + 1][0] = s1.charAt(i) + dp[i][0];
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (s1.charAt(i) == s2.charAt(j)) {
                    dp[i + 1][j + 1] = dp[i][j];
                } else {
                    dp[i + 1][j + 1] = Math.min(
                            dp[i + 1][j] + s2.charAt(j),
                            dp[i][j + 1] + s1.charAt(i)
                    );
                }
            }
        }
        return dp[m][n];
    }
}
```

#### 5. Longest Palindromic Substring

```go
func longestPalindrome(s string) string {  
   n := len(s)  
   dp := make([][]bool, n)  
   left, right := 0, 0  
   for i := range dp {  
      dp[i] = make([]bool, n)  
   }  
   for i := n - 1; i >= 0; i-- {  
      for j := i; j < n; j++ {  
         dp[i][j] = s[i] == s[j] && (j-i < 3 || dp[i+1][j-1])  
         if dp[i][j] && j+1-i > right+1-left {  
            left, right = i, j  
         }  
      }  
   }  
   return s[left : right+1]  
}
```

#### 10. Regular Expression Matching

compress

需要把要更新的那一行预先涂为false 并不是DP中的每个点都会被loop扫描，更不是每个点都更新

```go
func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	dp := [2][]bool{}
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	for j := range p {
		if p[j] == '*' && dp[0][j-1] {
			dp[0][j+1] = true
		}
	}
	for i := range s {
		dp[(i+1)&1] = make([]bool, n+1)
		for j := range p {
			if p[j] == '.' || p[j] == s[i] {
				dp[(i+1)&1][j+1] = dp[i&1][j]
				continue
			}
			if p[j] == '*' {
				if p[j-1] == s[i] || p[j-1] == '.' {
					dp[(i+1)&1][j+1] = dp[(i+1)&1][j-1] || dp[i&1][j] || dp[i&1][j+1]
				} else {
					// empty
					dp[(i+1)&1][j+1] = dp[(i+1)&1][j-1]
				}
			}
		}
	}
	return dp[m&1][n]
}
```

#### 32. Longest Valid Parentheses

go dp

https://leetcode.com/problems/longest-valid-parentheses/discuss/14126/My-O(n)-solution-using-a-stack   dennyrong

```go
func longestValidParentheses(s string) int {
    dp := make([]int, len(s))
    longest, open := 0, 0
    for i := range s {
        if s[i] == '(' {
            open++
        } else if open > 0 {
            dp[i] = 2 + dp[i-1]
            if i-dp[i] > 0 {
                dp[i] += dp[i-dp[i]]
            }
            open--
            longest = max(longest, dp[i])
        }
    }
    return longest
}
```

#### 44. Wildcard Matching

compres

```go
func isMatch(s string, p string) bool {
	m, n := len(s), len(p)
	dp := [2][]bool{}
	for i := range dp {
		dp[i] = make([]bool, n+1)
	}
	dp[0][0] = true
	for j := 0; j < n && p[j] == '*'; j++ {
		dp[0][j+1] = true
	}
	for i := range s {
		dp[(i+1)&1] = make([]bool, n+1)
		for j := range p {
			if s[i] == p[j] || p[j] == '?' {
				dp[(i+1)&1][j+1] = dp[i&1][j]
				continue
			}
			if p[j] == '*' {
				dp[(i+1)&1][j+1] = dp[(i+1)&1][j] || dp[i&1][j+1]
			}
		}
	}
	return dp[m&1][n]
}
```

#### 64. Minimum Path Sum

conpress

```go
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	dp := [2][]int{}
	for i := range dp {
		dp[i] = make([]int, n)
	}
	dp[0][0] = grid[0][0]
	for j := 1; j < n; j++ {
		dp[0][j] = dp[0][j-1] + grid[0][j]
	}
	for i := 1; i < m; i++ {
		dp[i&1][0] = dp[(i-1)&1][0] + grid[i][0]
		for j := 1; j < n; j++ {
			dp[i&1][j] = min(dp[(i-1)&1][j], dp[i&1][j-1]) + grid[i][j]
		}
	}
	return dp[(m-1)&1][n-1]
}
```

#### 89. Gray Code

```go
func grayCode(n int) []int {  
   res := []int{0}  
   for i := 0; i < n; i++ {  
      for j := len(res) - 1; j >= 0; j-- {  
         res = append(res, res[j]|(1<<i))  
      }  
   }  
   return res  
}
```

#### 91. Decode Ways

O(1) space

```go
func numDecodings(s string) int {
   N := len(s)
   prev1, dp, prev2 := 1, 0, 0
   for i := N - 1; i >= 0; i-- {
      if s[i] != '0' {
         dp = prev1
      } else {
         dp = 0
      }
      if i+1 < N && (s[i] == '1' || s[i] == '2' && s[i+1] < '7') {
         dp += prev2
      }
      prev1, prev2 = dp, prev1
   }
   return prev1
}
```

#### 115. Distinct Subsequences

compress

```go
func numDistinct(s string, t string) int {  
   m, n := len(s), len(t)  
   dp := [2][]int{}  
   for i := range dp {  
      dp[i] = make([]int, n+1)  
   }  
   dp[0][0] = 1  
   for i := range s {  
      dp[(i+1)&1][0] = 1  
      for j := range t {  
         dp[(i+1)&1][j+1] = dp[i&1][j+1]  
         if s[i] == t[j] {  
            dp[(i+1)&1][j+1] += dp[i&1][j]  
         }  
      }  
   }  
   return dp[m&1][n]  
}
```

#### 118. Pascal's Triangle

```go
func generate(n int) [][]int {  
   res := make([][]int, n)  
   for i := 0; i < n; i++ {  
      res[i] = make([]int, i+1)  
      res[i][0] = 1  
      res[i][i] = 1  
      for j := 1; j < i; j++ {  
         res[i][j] = res[i-1][j-1] + res[i-1][j]  
      }  
   }  
   return res  
}
```

#### 119. Pascal's Triangle II

```go
func getRow(n int) []int {  
   res := make([]int, n+1)  
   res[0] = 1  
   for i := 1; i < n+1; i++ {  
      for j := i; j > 0; j-- {  
         res[j] += res[j-1]  
      }  
   }  
   return res  
}
```

#### 132. Palindrome Partitioning II

go

```go
func minCut(s string) int {
	n := len(s)
	dp := make([]int, n+1) // number of cuts for the first k characters
	for i := range dp {
		dp[i] = i - 1
	}
	for i := range s {
		for j := 0; i-j >= 0 && i+j < n && s[i-j] == s[i+j]; j++ {
			dp[i+j+1] = min(dp[i+j+1], 1+dp[i-j]) //  dp[i-j] corresponds to s[0, i-j-1]
		}
		for j := 1; i-j+1 >= 0 && i+j < n && s[i-j+1] == s[i+j]; j++ {
			dp[i+j+1] = min(dp[i+j+1], 1+dp[i-j+1])
		}
	}
	return dp[n]
}
```

#### 135. Candy

```go
func candy(ratings []int) int {  
   n := len(ratings)  
   dp := make([]int, n)  
   for i := range dp {  
      dp[i] = 1  
   }  
   for i := 1; i < n; i++ {  
      if ratings[i] > ratings[i-1] {  
         dp[i] = dp[i-1] + 1  
      }  
   }  
   for i := n - 2; i >= 0; i-- {  
      if ratings[i] > ratings[i+1] {  
         dp[i] = max(dp[i+1]+1, dp[i])  
      }  
   }  
   res := 0  
   for _, v := range dp {  
      res += v  
   }  
   return res  
}  
```

#### 174. Dungeon Game

```go
func calculateMinimumHP(dungeon [][]int) int {
   m, n := len(dungeon), len(dungeon[0])
   dp := make([][]int, m+1)
   for i := range dp {
      dp[i] = make([]int, n+1)
      for j := range dp[i] {
         dp[i][j] = math.MaxInt
      }
   }
   dp[m][n-1] = 1
   dp[m-1][n] = 1
   for i := m - 1; i >= 0; i-- {
      for j := n - 1; j >= 0; j-- {
         need := min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j]
         if need <= 0 {
            dp[i][j] = 1
         } else {
            dp[i][j] = need
         }
      }
   }
   return dp[0][0]
}
```

#### 256. Paint House

rollong array 

```java
class Solution {
    public int minCost(int[][] costs) {
        int n = costs.length;
        if (n == 0) return 0;
        int[][] dp = new int[2][3];
        dp[0] = Arrays.copyOf(costs[0], 3);
        for (int i = 1; i < n; i++) {
            Arrays.fill(dp[i & 1], Integer.MAX_VALUE);
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    if (j == k) continue;
                    dp[i & 1][j] = Math.min(dp[i & 1][j], dp[i - 1 & 1][k] + costs[i][j]);
                }
            }
        }
        return Arrays.stream(dp[n - 1 & 1]).min().orElse(0);
    }
}
```

#### 312. Burst Balloons

最后打爆k坐标的气球

go

```go
func maxCoins(nums []int) int {  
   n := len(nums)  
   nums = append(append([]int{1}, nums...), 1)  
   dp := make([][]int, n+2)  
   for i := range dp {  
      dp[i] = make([]int, n+2)  
   }  
   for i := n; i >= 0; i-- {  
      for j := i + 2; j < n+2; j++ {  
         for k := i + 1; k < j; k++ {  
            dp[i][j] = max(dp[i][j], nums[i]*nums[j]*nums[k]+dp[i][k]+dp[k][j])  
         }  
      }  
   }  
   return dp[0][n+1]  
}  
```

#### 514. Freedom Trail

Go

```go
func findRotateSteps(ring string, key string) int {
	pos := make(map[byte][]int)
	for i := range ring {
		r := ring[i]
		if _, ok := pos[r]; ok {
			pos[r] = append(pos[r], i)
		} else {
			pos[r] = []int{i}
		}
	}
	state := map[int]int{0: 0}
	for i := range key {
		nextState := make(map[int]int)
		for _, target := range pos[key[i]] { // every possible target position
			nextState[target] = math.MaxInt
			for start := range state { // every possible start position
				nextState[target] = min(nextState[target], state[start]+distance(target, start, ring))
			}
		}
		state = nextState
	}
	res := math.MaxInt
	for _, v := range state {
		if res > v {
			res = v
		}
	}
	return res + len(key)
}
func distance(i, j int, ring string) int {
	return min(abs(i-j), len(ring)-abs(i-j))
}
func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
```

#### 516. Longest Palindromic Subsequence

```go
func longestPalindromeSubseq(s string) int {
	n := len(s)
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	for i := n - 1; i >= 0; i-- {
		dp[i][i] = 1
		for j := i + 1; j < n; j++ {
			if s[i] == s[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = max(dp[i+1][j], dp[i][j-1])
			}
		}
	}
	return dp[0][n-1]
}
```

#### 718. Maximum Length of Repeated Subarray

```go
func findLength(nums1 []int, nums2 []int) int {  
   n := len(nums2)  
   res := 0  
   dp := make([]int, n+1)  
   for i := range nums1 {  
      for j := n - 1; j >= 0; j-- {  
         if nums1[i] == nums2[j] {  
            dp[j+1] = dp[j] + 1  
         } else {  
            dp[j+1] = 0  
         }  
         res = max(res, dp[j+1])  
      }  
   }  
   return res  
}  
```

#### 837. New 21 Game

https://leetcode.com/problems/new-21-game/discuss/132334/One-Pass-DP-O(N)

```go
func new21Game(n int, k int, maxPts int) float64 {  
   if k == 0 || n >= k+maxPts {  
      return 1  
   }  
   dp := make([]float64, n+1)  
   dp[0] = 1  
   res, prev := 0.0, 1.0  
   for i := 1; i < n+1; i++ {  
      dp[i] = prev / float64(maxPts)  
      if i < k {  
         prev += dp[i]  
      } else {  
         res += dp[i]  
      }  
      if i-maxPts >= 0 {  
         prev -= dp[i-maxPts]  
      }  
   }  
   return res  
}
```

#### 873. Length of Longest Fibonacci Subsequence

https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/solutions/152343/c-java-python-check-pair/?orderBy=most_votes

go

```go
func lenLongestFibSubseq(arr []int) int {
	n := len(arr)
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	hashmap := make(map[int]int)
	// dp[a, b] = dp[b-a, a] + 1 or 2
	res := 0
	for end, b := range arr {
		hashmap[b] = end
		for start, a := range arr[:end] {
			if prev, ok := hashmap[b-a]; ok && b-a < a {
				dp[start][end] = dp[prev][start] + 1
			}
			res = max(res, dp[start][end]+2)
		}
	}
	if res < 3 {
		res = 0
	}
	return res
}
```

#### 877. Stone Game

```go
func stoneGame(piles []int) bool {  
   n := len(piles)  
   dp := make([][]int, n)  
   for i := range dp {  
      dp[i] = make([]int, n)  
      dp[i][i] = piles[i]  
   }  
   for i := 1; i < n; i++ {  
      for j := 0; j < n-i; j++ {  
         dp[j][i+j] = max(piles[j]-dp[j+1][i+j], piles[i+j]-dp[i][i+j-1])  
      }  
   }  
   return dp[0][n-1] > 0  
}  
```

#### 887. Super Egg Drop

https://leetcode.com/problems/super-egg-drop/discuss/158974/C%2B%2BJavaPython-2D-and-1D-DP-O(KlogN)

```go
func superEggDrop(k int, n int) int {
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, k+1)
	}
	i := 0
	for dp[i][k] < n {
		i++
		for j := 1; j < k+1; j++ {
			dp[i][j] = dp[i-1][j-1] + dp[i-1][j] + 1
		}
	}
	return i
}
```

#### 918. Maximum Sum Circular Subarray

[https://leetcode.com/problems/maximum-sum-circular-subarray/solutions/178422/One-Pass/](https://leetcode.com/problems/maximum-sum-circular-subarray/solutions/178422/One-Pass/)

```go
func maxSubarraySumCircular(nums []int) int {
	maxSum, minSum := nums[0], nums[0]
	total, currMax, currMin := 0, 0, 0
	for _, num := range nums {
		currMax = max(num, currMax+num)
		currMin = min(num, currMin+num)
		maxSum = max(maxSum, currMax)
		minSum = min(minSum, currMin)
		total += num
	}
	if maxSum > 0 {
		return max(maxSum, total-minSum)
	}
	return maxSum
}
```

#### 935. Knight Dialer

Go

```go
func knightDialer(n int) int {  
   x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 := 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  
   mod := int(math.Pow(10, 9)) + 7  
   for i := 0; i < n-1; i++ {  
      x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 =  
         (x4+x6)%mod,  
         (x6+x8)%mod,  
         (x7+x9)%mod,  
         (x4+x8)%mod,  
         (x3+x9+x0)%mod,  
         0,  
         (x1+x7+x0)%mod,  
         (x2+x6)%mod,  
         (x1+x3)%mod,  
         (x4+x2)%mod  
   }  
   return (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) % mod  
}
```

#### 940. Distinct Subsequences II

go

```go
func distinctSubseqII(s string) int {
	endWith := [26]int64{}
	mod := int64(1e9 + 7)
	for i := range s {
		endWith[s[i]-'a'] = sum(endWith, mod) + 1
	}
	return int(sum(endWith, mod))
}
func sum(arr [26]int64, mod int64) int64 {
	var res int64 = 0
	for _, v := range arr {
		res = (res + v) % mod
	}
	return res
}
```

#### 2327. Number of People Aware of a Secret

Go

```go
func peopleAwareOfSecret(n int, delay int, forget int) int {
	dp := make([]int, n+1)
	dp[1] = 1
	mod := int(1e9 + 7)
	share := 0
	for i := 2; i < n+1; i++ {
		if i >= delay {
			share = (share + dp[i-delay]) % mod
		}
		if i >= forget {
			share = (share - dp[i-forget] + mod) % mod
		}
		dp[i] = share
	}
	res := 0
	for i := n + 1 - forget; i < n+1; i++ {
		res = (res + dp[i]) % mod
	}
	return res
}
```

#### 1000. Minimum Cost to Merge Stones

有许多堆石头，已知每一堆的重量，规定每次恰好能将连续的`k`堆石头合并为1堆 ，合并的花费为这些石头重量之和，如何计算将所有石头最终合并为1堆的最小花费？最多有30堆石头，如果不能最终合并为1堆，记作-1

java

依然是最终处理分界点两端的两块

每次消去k块石头再返回1块，每次其实减少k-1

n-1能否整除k-1标志着最终能否到达1块石头

```java
class Solution {
    public int mergeStones(int[] stones, int k) {
        int n = stones.length;
        if (n == 1) return 0;
        if ((n - 1) % (k - 1) != 0) return -1;
        int[] prefixSum = new int[n + 1];
        for (int i = 0; i < n; i++) {
            prefixSum[i + 1] = prefixSum[i] + stones[i];
        }
        int[][] dp = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + k - 1; j < n; j++) {
                dp[i][j] = Integer.MAX_VALUE;
                for (int l = i; l < j; l += k - 1) {
                    dp[i][j] = Math.min(dp[i][j], dp[i][l] + dp[l + 1][j]);
                }
                int len = j - i + 1;
                if ((len - 1) % (k - 1) == 0) dp[i][j] += prefixSum[j + 1] - prefixSum[i];
            }
        }
        return dp[0][n - 1];
    }
}
```

#### 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance

java

```java
class Solution {  
    public int findTheCity(int n, int[][] edges, int distanceThreshold) {  
        int[][] dp = new int[n][n];  
        for (int[] row : dp) {  
            Arrays.fill(row, 10001);  
        }  
        for (int[] e : edges) {  
            dp[e[0]][e[1]] = e[2];  
            dp[e[1]][e[0]] = e[2];  
        }  
        for (int i = 0; i < n; i++) {  
            dp[i][i] = 0;  
        }  
        for (int k = 0; k < n; k++) {  
            for (int i = 0; i < n; i++) {  
                for (int j = 0; j < n; j++) {  
                    dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k][j]);  
                }  
            }  
        }  
        int small = n, res = 0;  
        for (int i = 0; i < n; i++) {  
            int count = 0;  
            for (int j = 0; j < n; j++) {  
                if (dp[i][j] <= distanceThreshold) count++;  
            }  
            if (count <= small) {  
                small = count;  
                res = i;  
            }  
        }  
        return res;  
    }  
}
```

#### [1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

go space compress

```go
func longestCommonSubsequence(text1 string, text2 string) int {
	m, n := len(text1), len(text2)
	dp := [2][]int{}
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := range text1 {
		for j := range text2 {
			if text1[i] == text2[j] {
				dp[(i+1)&1][j+1] = 1 + dp[i&1][j]
			} else {
				dp[(i+1)&1][j+1] = max(dp[(i+1)&1][j], dp[i&1][j+1])
			}
		}
	}
	return dp[m&1][n]
}
```

#### [72. Edit Distance](https://leetcode.com/problems/edit-distance/)

space compress

```go
func minDistance(word1 string, word2 string) int {
	m, n := len(word1), len(word2)
	dp := [2][]int{}
	for i := range dp {
		dp[i] = make([]int, n+1)
	}
	for i := range dp[0] {
		dp[0][i] = i
	}
	for i := range word1 {
		dp[(i+1)&1][0] = i + 1
		for j := range word2 {
			if word1[i] == word2[j] {
				dp[(i+1)&1][j+1] = dp[i&1][j]
			} else {
				dp[(i+1)&1][j+1] = 1 + min(dp[(i+1)&1][j], dp[i&1][j+1], dp[i&1][j])
			}
		}
	}
	return dp[m&1][n]
}
```

#### [688. Knight Probability in Chessboard](https://leetcode.com/problems/knight-probability-in-chessboard/)

2D dp

空间压缩在本题也适用，从第1层开始DP，仅保存前一层的数据

java

```java
public class Solution {
    public double knightProbability(int n, int k, int r, int c) {
        int[][] dirs = new int[][]{{-2, -1}, {-1, -2}, {1, -2}, {2, -1}, {2, 1}, {1, 2}, {-1, 2}, {-2, 1}};
        float[][][] dp = new float[2][n][n];
        fill(dp[0], 1);
        for (int l = 1; l < k + 1; l++) {
            fill(dp[l & 1], 0);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int[] dir : dirs) {
                        int x = i + dir[0];
                        int y = j + dir[1];
                        if (x >= n || x < 0 || y >= n || y < 0) continue;
                        dp[l & 1][i][j] += dp[l - 1 & 1][x][y] / 8;
                    }
                }
            }
        }
        return dp[k & 1][r][c];
    }

    void fill(float[][] curr, float val) {
        for (float[] floats : curr) {
            Arrays.fill(floats, val);
        }
    }
}
```

#### [1269. Number of Ways to Stay in the Same Place After Some Steps](https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

space compression

```java
class Solution {
    public int numWays(int steps, int arrLen) {
        int n = Math.min(steps / 2 + 1, arrLen);
        int mod = (int) 1e9 + 7;
        long[][] dp = new long[2][n];
        dp[0][0] = 1;
        for (int i = 1; i < steps + 1; i++) {
            for (int j = 0; j < n; j++) {
                dp[i & 1][j] = (dp[i - 1 & 1][j] + (j + 1 < n ? dp[i - 1 & 1][j + 1] : 0) + (j > 0 ? dp[i - 1 & 1][j - 1] : 0)) % mod;
            }
        }
        return (int) dp[steps & 1][0];
    }
}
```

#### 377. Combination Sum IV

顺序不同视作不同的组合 有顺序的01背包

```go
func combinationSum4(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 1; i < len(dp); i++ {
		for _, num := range nums {
			if i >= num {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[target]
}
```

#### 473. Matchsticks to Square

go

```go
func makesquare(nums []int) bool {
	sum := 0
	for _, stick := range nums {
		sum += stick
	}
	if sum%4 != 0 {
		return false
	}
	per := sum / 4
	n := len(nums)
	dp := make([]bool, 1<<n)
	total := make([]int, 1<<n)
	dp[0] = true
	for i := range dp {
		if dp[i] {
			for j, num := range nums {
				bit := i | (1 << j)
				if bit != i && num+total[i]%per <= per {
					dp[bit] = true
					total[bit] = total[i] + num
				}
			}
		}
	}
	return dp[1<<n-1]
}
```

#### 698. Partition to K Equal Sum Subsets

go

```go
func canPartitionKSubsets(nums []int, k int) bool {
	all := 0
	for _, num := range nums {
		all += num
	}
	if all%k != 0 {
		return false
	}
	target := all / k
	n := len(nums)
	dp := make([]bool, 1<<n)
	total := make([]int, 1<<n)
	dp[0] = true
	for i := range dp {
		if dp[i] {
			for j, num := range nums {
				next := i | (1 << j)
				if next != i {
					//running sum + nums[j] <= target sum required
					if num+total[i]%target <= target {
						dp[next] = true
						total[next] = total[i] + num
					} else {
						break
					}
				}
			}
		}
	}
	return dp[1<<n-1]
}
```

#### 139. Word Break

Go

```go
func wordBreak(s string, wordDict []string) bool {
	n := len(s)
	dp := make([]bool, n+1)
	dp[0] = true
	for i := 1; i < n+1; i++ {
		for _, word := range wordDict {
			L := len(word)
			if i >= L && word == s[i-L:i] {
				dp[i] = dp[i] || dp[i-L]
			}
		}
	}
	return dp[n]
}
```

#### 691. Stickers to Spell Word

The big idea is to use number from `0` to `2^n-1` as bitmap to represent every `subset` of `target`

https://leetcode.com/problems/stickers-to-spell-word/solutions/108333/rewrite-of-contest-winner-s-solution/?orderBy=most_votes

go

```go
func minStickers(stickers []string, target string) int {
	m := 1 << len(target)
	dp := make([]int, m)
	for i := range dp {
		dp[i] = math.MaxInt32
	}
	dp[0] = 0
	for i := range dp {
		if dp[i] == math.MaxInt32 {
			continue
		}
		for _, sticker := range stickers {
			curr := i
			for j := range sticker {
				for r := range target {
					if sticker[j] == target[r] && curr&(1<<r) == 0 {
						curr |= 1 << r
						break
					}
				}
			}
			dp[curr] = min(dp[curr], dp[i]+1)
		}
	}
	if dp[m-1] == math.MaxInt32 {
		return -1
	}
	return dp[m-1]
}
```

#### 198. House Robber

compress

```go
func rob(nums []int) int {  
   prev, curr := 0, 0  
   for _, num := range nums {  
      prev, curr = curr, max(curr, prev+num)  
   }  
   return curr  
}  
```

#### 213. House Robber II

go

```go
func rob(nums []int) int {
	n := len(nums)
	if n < 2 {
		return nums[0]
	}
	return max(dfs(nums[:n-1]), dfs(nums[1:]))
}
func dfs(nums []int) int {
	prev, curr := 0, 0
	for _, num := range nums {
		prev, curr = curr, max(curr, prev+num)
	}
	return curr
}
```

#### 337. House Robber III

```go
type profit struct {
	enter    int
	notEnter int
}

func rob(root *TreeNode) int {
	res := dfs(root)
	return max(res.enter, res.notEnter)
}
func dfs(root *TreeNode) profit {
	if root == nil {
		return profit{0, 0}
	}
	left := dfs(root.Left)
	right := dfs(root.Right)
	enter := left.notEnter + right.notEnter + root.Val
	notEnter := max(left.enter, left.notEnter) + max(right.enter, right.notEnter)
	return profit{enter, notEnter}
}
```

#### 121. Best Time to Buy and Sell Stock

只能买卖一次

```go
func maxProfit(prices []int) int {
    profit, buyIn := 0, prices[0]
    for _, p := range prices{
        if profit < p-buyIn {
            profit = p-buyIn
        }
        if buyIn > p{
            buyIn = p
        }
    }
    return profit
}
```

#### 122. Best Time to Buy and Sell Stock II

不能同时参与多比交易。买卖次数不限

```go
func maxProfit(prices []int) int {
    res := 0
    for k := range prices {
        if k > 0 && prices[k] > prices[k-1] {
            prof := prices[k] - prices[k-1]
            res += prof
        }
    }
    return res
}
```

#### 309. Best Time to Buy and Sell Stock with Cooldown

```go
func maxProfit(prices []int) int {  
   n := len(prices)  
   s0 := make([]int, n) //can buy  
   s1 := make([]int, n) //can sell  
   s2 := make([]int, n) //take a rest  
   s0[0] = 0  
   s1[0] = -prices[0]  
   s2[0] = math.MinInt32  
   for i := 1; i < len(prices); i++ {  
      s0[i] = max(s0[i-1], s2[i-1])           // stay at s0 or rest from s2  
      s1[i] = max(s1[i-1], s0[i-1]-prices[i]) // stay at s1 or buy from s0  
      s2[i] = s1[i-1] + prices[i]             // sell from s1 and take a rest  
   }  
   return max(s0[n-1], s2[n-1])  
}  
```

#### 714. Best Time to Buy and Sell Stock with Transaction Fee

go

```go
func maxProfit(prices []int, fee int) int {
   sale, hold := 0, (-1)*prices[0]
   for i := 1; i < len(prices); i++ {
      prev := hold
      hold = max(hold, sale-prices[i])
      sale = max(sale, prev+prices[i]-fee)
   }
   return sale
}
```