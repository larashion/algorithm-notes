#### 300. Longest Increasing Subsequence

binary search
[https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation](https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation)

python

```python
class Solution:  
    def lengthOfLIS(self, nums: List[int]) -> int:  
        LIS = []  
        for num in nums:  
            i = self.binary_search(LIS, num)  
            if i == len(LIS):  
                LIS.append(num)  
            else:  
                LIS[i] = num  
        return len(LIS)  
  
    def binary_search(self, LIS: List[int], val: int) -> int:  
        left, right = 0, len(LIS)  
        while left < right:  
            mid = (left + right) // 2  
            if LIS[mid] < val:  
                left = mid + 1  
            else:  
                right = mid  
        return left
```

go

```go
func lengthOfLIS(nums []int) int {
   LIS := make([]int, 0, len(nums))
   for _, n := range nums {
      i := sort.SearchInts(LIS, n)
      if i == len(LIS) {
         LIS = append(LIS, n)
      } else {
         LIS[i] = n
      }
   }
   return len(LIS)
}
```

java

```java
class Solution {
    int lengthOfLIS(int[] nums) {
        ArrayList<Integer> LIS = new ArrayList<>();
        for (int num : nums) {
            int i = Collections.binarySearch(LIS, num);
            if (i >= 0) continue;
            i = -i - 1;
            if (i == LIS.size()) {
                LIS.add(num);
            } else {
                LIS.set(i, num);
            }
        }
        return LIS.size();
    }
}
```

rust

```rust
impl Solution {
    fn length_of_lis(nums: Vec<i32>) -> i32 {
        let lis = nums.into_iter().fold(vec![], |mut acc, num| {
            let i = acc.partition_point(|&x| x < num);
            if i == acc.len() {
                acc.push(num);
            } else {
                acc[i] = num;
            }
            acc
        });
        lis.len() as i32
    }
}
```

#### 354. Russian Doll Envelopes

java

```java
class Solution {  
    public int maxEnvelopes(int[][] envelopes) {  
        Arrays.sort(envelopes, (a, b) -> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);  
        int n = envelopes.length;  
        int[] heights = new int[n];  
        for (int i = 0; i < n; i++) {  
            heights[i] = envelopes[i][1];  
        }  
        return lengthOfLIS(heights);  
    }  
  
    int lengthOfLIS(int[] nums) {
        ArrayList<Integer> LIS = new ArrayList<>();
        for (int num : nums) {
            int i = Collections.binarySearch(LIS, num);
            if (i >= 0) continue;
            i = -i - 1;
            if (i == LIS.size()) {
                LIS.add(num);
            } else {
                LIS.set(i, num);
            }
        }
        return LIS.size();
    }
}
```

go

```go
func maxEnvelopes(envelopes [][]int) int {  
   sort.Slice(envelopes, func(i, j int) bool {  
      if envelopes[i][0] == envelopes[j][0] {  
         return envelopes[i][1] > envelopes[j][1]  
      }  
      return envelopes[i][0] < envelopes[j][0]  
   })  
   n := len(envelopes)  
   heights := make([]int, n)  
   for i := range envelopes {  
      heights[i] = envelopes[i][1]  
   }  
   return lengthOfLIS(heights)  
}  
func lengthOfLIS(nums []int) int {  
   LIS := make([]int, 0, len(nums))  
   for _, n := range nums {  
      i := sort.SearchInts(LIS, n)  
      if i == len(LIS) {  
         LIS = append(LIS, n)  
      } else {  
         LIS[i] = n  
      }  
   }  
   return len(LIS)  
}
```

rust

```rust
use std::cmp::Reverse;  
  
impl Solution {  
    pub fn max_envelopes(envs: Vec<Vec<i32>>) -> i32 {  
        let mut envs: Vec<_> = envs  
            .iter()  
            .map(|envs| (envs[0], Reverse(envs[1])))  
            .collect();  
        envs.sort_unstable();  
  
        fn length_of_lis(nums: Vec<i32>) -> i32 {  
            let lis = nums.into_iter().fold(vec![], |mut acc, num| {  
                let i = acc.partition_point(|&x| x < num);  
                if i == acc.len() {  
                    acc.push(num);  
                } else {  
                    acc[i] = num;  
                }  
                acc  
            });  
            lis.len() as i32  
        }  
          
        let envs: Vec<_> = envs.into_iter().map(|(a, Reverse(b))| b).collect();  
        length_of_lis(envs)  
    }  
}
```

#### 673. Number of Longest Increasing Subsequence

```go
func findNumberOfLIS(nums []int) int {
	n, maxLen, res := len(nums), 0, 0
	dp, count := make([]int, n), make([]int, n)
	for i := range nums {
		dp[i] = 1
		count[i] = 1
		for j := range nums[:i] {
			if nums[i] > nums[j] {
				if dp[i] == dp[j]+1 {
					count[i] += count[j]
				}
				if dp[i] < dp[j]+1 {
					dp[i] = dp[j] + 1
					count[i] = count[j]
				}
			}
		}
		if dp[i] == maxLen {
			res += count[i]
		}
		if dp[i] > maxLen {
			maxLen = dp[i]
			res = count[i]
		}
	}
	return res
}
```
