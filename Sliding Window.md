在数组上用指针框定窗口范围，对区间求和、求最大、求最小

当窗口长度固定，只需1个指针，否则需要2个

#### [1031. Maximum Sum of Two Non-Overlapping Subarrays](https://leetcode.com/problems/maximum-sum-of-two-non-overlapping-subarrays/)

java

```java
class Solution {
    public int maxSumTwoNoOverlap(int[] nums, int fir, int sec) {
        return Math.max(maxSum(nums, fir, sec), maxSum(nums, sec, fir));
    }

    private int maxSum(int[] nums, int l, int m) {
        // initialize
        int sumL = 0, sumM = 0;
        for (int i = 0; i < l + m; i++) {
            if (i < l) sumL += nums[i];
            else sumM += nums[i];
        }
        // mark the max sum of the L-length subarray, L and M don't need to be adjacent.
        int maxL = sumL;
        // there is no variable 'maxM', otherwise it will cause overlap
        int res = maxL + sumM;
        for (int i = l + m; i < nums.length; i++) {
            sumL += nums[i - m] - nums[i - m - l]; // compute the sum of every L-length subarray
            sumM += nums[i] - nums[i - m];
            maxL = Math.max(maxL, sumL);
            res = Math.max(res, maxL + sumM);
        }
        return res;
    }
}
```

#### [1208. Get Equal Substrings Within Budget](https://leetcode.com/problems/get-equal-substrings-within-budget/)

java

```java
class Solution {
    public int equalSubstring(String s, String t, int maxCost) {
        int i = 0, cost = 0, res = 0;
        for (int j = 0; j < s.length(); j++) {
            cost += Math.abs(s.charAt(j) - t.charAt(j));
            while (cost > maxCost) {
                cost -= Math.abs(s.charAt(i) - t.charAt(i));
                i++;
            }
            res = Math.max(res, j - i + 1);
        }
        return res;
    }
}    
```

#### 2090. K Radius Subarray Averages

java

```java
class Solution {
    public int[] getAverages(int[] nums, int k) {
        int len = 2 * k + 1, n = nums.length;
        int[] ans = new int[n];
        Arrays.fill(ans, -1);
        long sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += nums[i];
            if (i >= len) sum -= nums[i - len];
            if (i >= len - 1) ans[i - k] = (int) (sum / len);
        }
        return ans;
    }
}
```

#### 3. Longest Substring Without Repeating Characters

go

```go
func lengthOfLongestSubstring(s string) int {
	counter := [256]int{}
	i, res := 0, 0
	for j := range s {
		counter[s[j]]++
		for ; counter[s[j]] > 1; i++ {
			counter[s[i]]--
		}
		res = max(res, j-i+1)
	}
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

Use hashset

java

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        HashSet<Character> set = new HashSet<>();
        int i = 0, res = 0, n = s.length();
        char[] chars = s.toCharArray();
        for (int j = 0; j < n; j++) {
            while (set.contains(chars[j])) {
                set.remove(chars[i++]);
            }
            set.add(chars[j]);
            res = Math.max(res, set.size());
        }
        return res;
    }
}
```

rust

```rust
use std::cmp::max;
use std::collections::HashSet;

impl Solution {
    pub fn length_of_longest_substring(s: String) -> i32 {
        let mut set: HashSet<char> = HashSet::new();
        let mut res = 0;
        let mut i = 0;
        let s: Vec<char> = s.chars().collect();
        for j in 0..s.len() {
            while set.contains(&s[j]) {
                set.remove(&s[i]);
                i += 1;
            }
            set.insert(s[j]);
            res = max(res, set.len());
        }
        res as i32
    }
}
```

#### 30. Substring with Concatenation of All Words

可以将起点根据 当前下标与单词长度的取余结果 进行分类

rust

```rust
use std::collections::HashMap;

impl Solution {
    pub fn find_substring(s: String, words: Vec<String>) -> Vec<i32> {
        let n = s.len();
        let m = words.len();
        let w = words[0].len();

        let need = words.iter().fold(HashMap::new(), |mut acc, word| {
            *acc.entry(&word[..]).or_insert(0) += 1;
            acc
        });

        let mut res = vec![];
        // offset
        for i in 0..w {
            let mut need = need.clone();
            let mut valid = 0;
            // build the window word by word
            for j in (i..n - w + 1).step_by(w) {
                // j is the left bound of the word, right bound of the window
                let curr = &s[j..j + w];
                if let Some(val) = need.get(curr) {
                    if *val > 0 { valid += 1; }
                }
                *need.entry(curr).or_default() -= 1;
                // shrink the window
                if j >= m * w {
                    let idx = j - m * w;
                    let prev = &s[idx..idx + w];
                    let num = need.entry(prev).or_default();
                    *num += 1;
                    if *num > 0 { valid -= 1; }
                }
                if valid == m {
                    res.push((j + w - m * w) as _)
                }
            }
        }
        res
    }
}
```

java

```java
class Solution {
    List<Integer> findSubstring(String s, String[] words) {
        int n = s.length(), m = words.length, w = words[0].length();
        Map<String, Integer> target = new HashMap<>();
        for (String str : words)
            target.put(str, target.getOrDefault(str, 0) + 1);
        List<Integer> ans = new ArrayList<>();
        // offset
        for (int i = 0; i < w; i++) {
            Map<String, Integer> cnt = new HashMap<>(target);
            int valid = 0;
            for (int j = i; j < n - w + 1; j += w) {
                String curr = s.substring(j, j + w);
                if (cnt.containsKey(curr) && cnt.get(curr) > 0) valid++;
                cnt.put(curr, cnt.getOrDefault(curr, 0) - 1);
                // shrink the window
                if (j >= m * w) {
                    int idx = j - m * w;
                    String prev = s.substring(idx, idx + w);
                    cnt.put(prev, cnt.get(prev) + 1);
                    if (cnt.get(prev) > 0) valid--;
                }
                if (valid == m) ans.add(j + w - m * w);
            }
        }
        return ans;
    }
}
```

#### 76. Minimum Window Substring

Go

```go
func minWindow(s string, t string) string {
	need := make([]int, 128)
	for _, ch := range t {
		need[ch]++
	}
	i, res, valid := 0, 0, 0
	d := math.MaxInt32
	for j := range s {
		if need[s[j]] > 0 {
			valid++
		}
		need[s[j]]--
		for ; valid == len(t); i++ {
			if d > j-i+1 {
				d = j - i + 1
				res = i
			}
			need[s[i]]++
			if need[s[i]] > 0 {
				valid--
			}
		}
	}
	if d == math.MaxInt32 {
		return ""
	}
	return s[res : res+d]
}

```

java

```java
class Solution {
    public String minWindow(String source, String target) {
        int valid = 0, d = Integer.MAX_VALUE, res = 0;
        int[] need = new int[128];
        for (char c : target.toCharArray())
            need[c]++;
        char[] cs = source.toCharArray();
        for (int i = 0, j = 0; j < source.length(); j++) {
            if (need[cs[j]] > 0) valid++;
            need[cs[j]]--;
            for (; valid == target.length(); i++) {
                if (j - i + 1 < d) {
                    d = j - i + 1;
                    res = i;
                }
                need[cs[i]]++;
                if (need[cs[i]] > 0) valid--;
            }
        }
        return d == Integer.MAX_VALUE ? "" : source.substring(res, res + d);
    }
}
```

rust

```rust
impl Solution {
    pub fn min_window(s: String, t: String) -> String {
        let mut d = usize::MAX;
        let mut i = 0;
        let mut res = 0;
        let mut need = t.as_bytes().iter().fold(vec![0; 128], |mut acc, x| {
            acc[*x as usize] += 1;
            acc
        });
        let mut valid = 0;
        let arr = s.as_bytes().iter().map(|&x| x as usize).collect::<Vec<usize>>();
        for (j, &right) in arr.iter().enumerate() {
            if need[right] > 0 { valid += 1; }
            need[right] -= 1;
            while valid == t.len() {
                if d > j - i + 1 {
                    d = j - i + 1;
                    res = i;
                }
                let left = arr[i];
                need[left] += 1;
                if need[left] > 0 { valid -= 1; }
                i += 1;
            }
        }
        if d == usize::MAX { "".into() } else { s[res..res + d].into() }
    }
}
```

#### 209. Minimum Size Subarray Sum

python

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left, window_sum = 0, 0
        res = sys.maxsize
        for right in range(len(nums)):
            window_sum += nums[right]
            while window_sum >= target:
                res = min(res, right - left + 1)
                window_sum -= nums[left]
                left += 1
        return res if res != sys.maxsize else 0
```

go

```go
func minSubArrayLen(target int, nums []int) int {
	res, sum, i := math.MaxInt32, 0, 0
	for j := range nums {
		sum += nums[j]
		for ; sum >= target; i++ {
			sum -= nums[i]
			res = min(j-i+1, res)
		}
	}
	if res == math.MaxInt32 {
		return 0
	}
	return res
}
func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
```

#### 220. Contains Duplicate III

go

```go
func containsNearbyAlmostDuplicate(nums []int, indexDiff int, valueDiff int) bool {
	if indexDiff <= 0 || valueDiff < 0 || len(nums) < 2 {
		return false
	}
	buckets := make(map[int]int)
	for i := range nums {
		key := nums[i] / (valueDiff + 1)
		if nums[i] < 0 {
			key--
		}
		if _, ok := buckets[key]; ok {
			return true
		}
		if v, ok := buckets[key-1]; ok && nums[i]-v <= valueDiff {
			return true
		}
		if v, ok := buckets[key+1]; ok && v-nums[i] <= valueDiff {
			return true
		}
		if len(buckets) >= indexDiff {
			delete(buckets, nums[i-indexDiff]/(valueDiff+1))
		}
		buckets[key] = nums[i]
	}
	return false
}
```

python

```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        if k <= 0 or t < 0 or len(nums) < 2:
            return False
        buckets = {}
        for i in range(len(nums)):
            key = nums[i] // (t + 1)
            if key in buckets:
                return True
            if (key - 1) in buckets and nums[i] - buckets[key - 1] <= t:
                return True
            if (key + 1) in buckets and buckets[key + 1] - nums[i] <= t:
                return True
            if len(buckets) >= k:
                del buckets[nums[i - k] // (t + 1)]
            buckets[key] = nums[i]
        return False
```

#### 239. Sliding Window Maximum

rust 

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn max_sliding_window(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut res = vec![];
        let k = k as usize;
        let mut deque: VecDeque<usize> = VecDeque::new();
        for (i, &v) in nums.iter().enumerate() {
            if i > k - 1 && deque[0] == i - k {
                deque.pop_front();
            }
            while let Some(&top) = deque.back() {
                if v > nums[top] {
                    deque.pop_back();
                } else {
                    break;
                }
            }
            deque.push_back(i);
            if i >= k - 1 {
                res.push(nums[deque[0]])
            }
        }
        res
    }
}
```

java

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] res = new int[n - k + 1];
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (!deque.isEmpty() && deque.peek() == i - k)
                deque.poll();
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();
            }
            deque.offer(i);
            if (i >= k - 1)
                res[i - k + 1] = nums[deque.peek()];
        }
        return res;
    }
}
```

Go deque

```go
func maxSlidingWindow(nums []int, k int) []int {
	n := len(nums)
	res := make([]int, n-k+1)
	dq := make([]int, 0, k+1)
	for i, v := range nums {
		if i > k-1 && dq[0] == i-k {
			dq = dq[1:]
		}
		for len(dq) > 0 && v > nums[dq[len(dq)-1]] {
			dq = dq[:len(dq)-1]
		}
		dq = append(dq, i)
		if i >= k-1 {
			res[i-k+1] = nums[dq[0]]
		}
	}
	return res
}

```

Python deque

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res, dq = [], deque()
        for i, v in enumerate(nums):
            if i > k-1 and dq[0] < i - k + 1:
                dq.popleft()
            while dq and v > nums[dq[-1]]:
                dq.pop()
            dq += i,
            if i > k-2:
                res += nums[dq[0]],
        return res
```

divide & conquer 

```go
func maxSlidingWindow(nums []int, k int) []int {
    res := make([]int, len(nums)-k+1)
    left, right := make([]int, len(nums)), make([]int, len(nums))
    for i := range nums {
        if i%k == 0 {
            left[i] = nums[i]
        } else {
            left[i] = max(nums[i], left[i-1])
        }
        j := len(nums) - 1 - i
        if j%k == k-1 || j == len(nums)-1 {
            right[j] = nums[j]
        } else {
            right[j] = max(nums[j], right[j+1])
        }
    }
    for m := 0; m < len(nums)-k+1; m++ {
        res[m] = max(right[m], left[m+k-1])
    }
    return res
}
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

divide & conquer

```python
class Solution:  
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:  
        n = len(nums)  
        res, left, right = [0] * (n - k + 1), [0] * n, [0] * n  
        for i in range(n):  
            if i % k == 0:  
                left[i] = nums[i]  
            else:  
                left[i] = max(nums[i], left[i - 1])  
            j = n - 1 - i  
            if j % k == k - 1 or j == n - 1:  
                right[j] = nums[j]  
            else:  
                right[j] = max(nums[j], right[j + 1])  
        for i in range(n - k + 1):  
            res[i] = max(right[i], left[i + k - 1])  
        return res
```

#### 395. Longest Substring with At Least K Repeating Characters

java

```java
class Solution {
    public int longestSubstring(String s, int k) {
        int res = 0;
        for (int p = 1; p <= 26; p++) {
            int[] cnt = new int[26];
            for (int i = 0, j = 0, distinct = 0, valid = 0; j < s.length(); j++) {
                int J = s.charAt(j) - 'a';
                cnt[J]++;
                if (cnt[J] == 1) distinct++;
                if (cnt[J] == k) valid++;
                while (distinct > p) {
                    int I = s.charAt(i++) - 'a';
                    cnt[I]--;
                    if (cnt[I] == 0) distinct--;
                    if (cnt[I] == k - 1) valid--;
                }
                if (distinct == valid) res = Math.max(res, j - i + 1);
            }
        }
        return res;
    }
}
```

rust

```rust
use std::cmp::max;  
  
impl Solution {  
    pub fn longest_substring(s: String, k: i32) -> i32 {  
        let s = s.as_bytes();  
        (1..=26).fold(0, |mut acc, p| {  
            let mut cnt = [0; 26];  
            let mut i = 0;  
            let mut distinct = 0;  
            let mut valid = 0;  
            for j in 0..s.len() {  
                let right = (s[j] - b'a') as usize;  
                cnt[right] += 1;  
                if cnt[right] == 1 { distinct += 1 }  
                if cnt[right] == k { valid += 1 }  
                while distinct > p {  
                    let left = (s[i] - b'a') as usize;  
                    cnt[left] -= 1;  
                    if cnt[left] == 0 { distinct -= 1 }  
                    if cnt[left] == k - 1 { valid -= 1 }  
                    i += 1;  
                }  
                if distinct == valid { acc = max(acc, (j - i + 1) as i32) }  
            }  
            acc  
        })  
    }  
}
```

Go

```go
func longestSubstring(s string, k int) int {
	res := 0
	for p := 1; p < 27; p++ {
		counter := [26]int{}
		i, valid, distinct := 0, 0, 0
		for j := range s {
			J := s[j] - 'a'
			counter[J]++
			if counter[J] == 1 {
				distinct++
			}
			if counter[J] == k {
				valid++
			}
			for ; distinct > p; i++ {
				I := s[i] - 'a'
				counter[I]--
				if counter[I] == 0 {
					distinct--
				}
				if counter[I] == k-1 {
					valid--
				}
			}
			if distinct == valid {
				res = max(res, j-i+1)
			}
		}
	}
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

#### 424. Longest Repeating Character Replacement

go

```go
func characterReplacement(s string, k int) int {
	count := [26]int{}
	i, maxCount, res := 0, 0, 0
	for j := range s {
		J := s[j] - 'A'
		count[J]++
		// largest count of a single, unique character in the current window
		maxCount = max(maxCount, count[J])
		//too many to replace, shrink the window
		for j-i+1-maxCount > k {
			count[s[i]-'A']--
			i++
		}
		res = max(res, j-i+1)
	}
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

java

```java
class Solution {
    public int characterReplacement(String s, int k) {
        int i = 0, maxCount = 0, res = 0, n = s.length();
        int[] cnt = new int[n];
        for (int j = 0; j < n; j++) {
            int J = s.charAt(j) - 'A';
            maxCount = Math.max(maxCount, ++cnt[J]);
            while (j - i + 1 - maxCount > k) {
                cnt[s.charAt(i++) - 'A']--;
            }
            res = Math.max(res, j - i + 1);
        }
        return res;
    }
}
```

#### 438. Find All Anagrams in a String

一个很大的误区是明明窗口长度是固定的，还用俩指针判断来判断去

need 描述与目标的差距，所以窗口扩充时，need计数减少，反之增加

java

```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        int w = p.length();
        int[] need = new int[26];
        for (char c : p.toCharArray())
            need[c - 'a']++;
        ArrayList<Integer> res = new ArrayList<>();
        int valid = 0;
        char[] cs = s.toCharArray();
        for (int i = 0; i < s.length(); i++) {
            int curr = cs[i] - 'a';
            if (need[curr] > 0) {
                valid++;
            }
            need[curr]--;
            if (i > w - 1) {
                int out = cs[i - w] - 'a';
                need[out]++;
                if (need[out] > 0) valid--;
            }
            if (i >= w - 1 && valid == w) res.add(i - w + 1);
        }
        return res;
    }
}
```

go

```go
func findAnagrams(s string, p string) []int {  
   w := len(p)  
   need := [26]int{}  
   for i := range p {  
      need[p[i]-'a']++  
   }  
   var res []int  
   valid := 0  
   for i := range s {  
      if need[s[i]-'a'] > 0 {  
         valid++  
      }  
      need[s[i]-'a']--  
      if i > w-1 {  
         need[s[i-w]-'a']++  
         if need[s[i-w]-'a'] > 0 {  
            valid--  
         }  
      }  
      if i >= w-1 && valid == w {  
         res = append(res, i-w+1)  
      }  
   }  
   return res  
}
```

#### 567. Permutation in String

和438题一模一样

java

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int[] need = new int[26];
        for (char c : s1.toCharArray()) {
            need[c - 'a']++;
        }
        int valid = 0, w = s1.length();
        char[] s = s2.toCharArray();
        for (int i = 0; i < s2.length(); i++) {
            int curr = s[i] - 'a';
            if (need[curr] > 0) valid++;
            need[curr]--;
            if (i > w - 1) {
                int out = s[i - w] - 'a';
                need[out]++;
                if (need[out] > 0) valid--;
            }
            if (i >= w - 1 && valid == w) return true;
        }
        return false;
    }
}
```

go

```go
func checkInclusion(s1 string, s2 string) bool {
	need := [26]int{}
	for i := range s1 {
		need[s1[i]-'a']++
	}
	w := len(s1)
	valid := 0
	for i := range s2 {
		curr := s2[i] - 'a'
		if need[curr] > 0 {
			valid++
		}
		need[curr]--
		if i > w-1 {
			out := s2[i-w] - 'a'
			need[out]++
			if need[out] > 0 {
				valid--
			}
		}
		if i >= w-1 && valid == w {
			return true
		}
	}
	return false
}
```

#### 643. Maximum Average Subarray I

java

```java
class Solution {
    public double findMaxAverage(int[] nums, int k) {
        int sum = 0, res = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (i > k - 1) sum -= nums[i - k];
            if (i >= k - 1) res = Math.max(res, sum);
        }
        return res / 1.0 / k;
    }
}
```

go

```go
func findMaxAverage(nums []int, k int) float64 {  
   res, sum := math.MinInt32, 0  
   for i := range nums {  
      sum += nums[i]  
      if i > k-1 {  
         sum -= nums[i-k]  
      }  
      if i >= k-1 {  
         res = max(res, sum)  
      }  
   }  
   return float64(res) / float64(k)  
}  
  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}
```

#### 713. Subarray Product Less Than K

求的是subarray的数量

如果新加入一个元素，新的subarray怎么求

增加了j结尾的、剔除了i开头的，剩下的j-i+1个元素开头的subarray

java

```java
class Solution {  
    public int numSubarrayProductLessThanK(int[] nums, int k) {  
        int prod = 1, res = 0;  
        for (int i = 0, j = 0; j < nums.length; j++) {  
            prod *= nums[j];  
            while (i <= j && prod >= k) {  
                prod /= nums[i++];  
            }  
            res += j - i + 1;  
        }  
        return res;  
    }  
}
```

go

```go
func numSubarrayProductLessThanK(nums []int, k int) int {  
   res, prod, i := 0, 1, 0  
   for j := range nums {  
      prod *= nums[j]  
      for ; i <= j && prod >= k; i++ {  
         prod /= nums[i]  
      }  
      res += j - i + 1  
   }  
   return res  
}
```

#### 862. Shortest Subarray with Sum at Least K

go

```go
func prefix(nums []int) []int {  
   res := make([]int, len(nums)+1)  
   for i := range nums {  
      res[i+1] = nums[i] + res[i]  
   }  
   return res  
}  
func shortestSubarray(nums []int, k int) int {  
   n := len(nums)  
   pref := prefix(nums)  
   res := n + 1  
   var deque []int  
   for i := range pref {  
      for len(deque) > 0 && pref[i]-pref[deque[0]] >= k {  
         res = min(res, i-deque[0])  
         deque = deque[1:]  
      }  
      for len(deque) > 0 && pref[deque[len(deque)-1]] >= pref[i] {  
         deque = deque[:len(deque)-1]  
      }  
      deque = append(deque, i)  
   }  
   if res <= n {  
      return res  
   }  
   return -1  
}  
func min(a, b int) int {  
   if a > b {  
      return b  
   }  
   return a  
}
```

java

```java
class Solution {  
  public int shortestSubarray(int[] nums, int k) {  
    int n = nums.length;  
    int res = n + 1;  
    long[] pre = new long[n + 1]; // long  
    for (int i = 0; i < nums.length; i++) {  
      pre[i + 1] = pre[i] + nums[i];  
    }  
    ArrayDeque<Integer> dq = new ArrayDeque<>();  
    for (int i = 0; i < pre.length; i++) {  
      while (!dq.isEmpty() && pre[i] - pre[dq.getFirst()] >= k) {  
        res = Math.min(res, i - dq.pollFirst());  
      }  
      while (!dq.isEmpty() && pre[dq.peekLast()] >= pre[i]) {  
        dq.pollLast();  
      }  
      dq.add(i);  
    }  
    return res < n + 1 ? res : -1;  
  }  
}
```

python

```python
class Solution:  
    def shortestSubarray(self, nums: List[int], k: int) -> int:  
        n = len(nums)  
        pre = [0] * (n + 1)  
        for i in range(n):  
            pre[i + 1] = pre[i] + nums[i]  
        dq = deque()  
        res = n + 1  
        for i in range(n + 1):  
            while dq and pre[i] - pre[dq[0]] >= k:  
                res = min(res, i - dq.popleft())  
            while dq and pre[dq[-1]] >= pre[i]:  
                dq.pop()  
            dq.append(i)  
        return res if res <= n else -1
```

#### 904. Fruit Into Baskets

java

```java
class Solution {
    public int totalFruit(int[] fruits) {
        HashMap<Integer, Integer> cnt = new HashMap<>();
        int res = 0;
        for (int i = 0, j = 0; j < fruits.length; j++) {
            cnt.put(fruits[j], cnt.getOrDefault(fruits[j], 0) + 1);
            while (cnt.size() > 2) {
                cnt.put(fruits[i], cnt.get(fruits[i]) - 1);
                cnt.remove(fruits[i++], 0);
            }
            res = Math.max(res, j - i + 1);
        }
        return res;
    }
}
```

go

```go
func totalFruit(fruits []int) int {
	i := 0
	cnt := make(map[int]int)
	res := 0
	for j := range fruits {
		cnt[fruits[j]]++
		for ; len(cnt) > 2; i++ {
			cnt[fruits[i]]--
			if cnt[fruits[i]] == 0 {
				delete(cnt, fruits[i])
			}
		}
		res = max(res, j-i+1)
	}
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

rust

```rust
use std::collections::HashMap;  
  
impl Solution {  
    pub fn total_fruit(fruits: Vec<i32>) -> i32 {  
        let mut cnt = HashMap::new();  
        let i = fruits.iter().fold(0, |mut acc, fruit| {  
            *cnt.entry(fruit).or_insert(0) += 1;  
            if cnt.len() > 2 {  
                let i = &fruits[acc];  
                *cnt.entry(i).or_default() -= 1;  
                if cnt[i] == 0 { cnt.remove(i); }  
                acc += 1;  
            }  
            acc  
        });  
        (fruits.len() - i) as i32  
    }  
}
```

#### 930. Binary Subarrays With Sum

go

```go
func numSubarraysWithSum(nums []int, goal int) int {
	return atMost(nums, goal) - atMost(nums, goal-1)
}
func atMost(nums []int, goal int) int {
	if goal < 0 {
		return 0
	}
	res, i, sum := 0, 0, 0
	for j, num := range nums {
		sum += num
		for ; sum > goal; i++ {
			sum -= nums[i]
		}
		res += j - i + 1
	}
	return res
}
```

#### 992. Subarrays with K Different Integers

go

```go
func subarraysWithKDistinct(nums []int, k int) int {
	return atMost(nums, k) - atMost(nums, k-1)
}
func atMost(nums []int, k int) int {
	res, i, cnt := 0, 0, make(map[int]int)
	for j, num := range nums {
		cnt[num]++
		for ; len(cnt) > k; i++ {
			cnt[nums[i]]--
			if cnt[nums[i]] == 0 {
				delete(cnt, nums[i])
			}
		}
		res += j - i + 1
	}
	return res
}
```

java

```java
class Solution {
    public int subarraysWithKDistinct(int[] nums, int K) {
        return atMostK(nums, K) - atMostK(nums, K - 1);
    }

    int atMostK(int[] nums, int k) {
        int i = 0, res = 0;
        HashMap<Integer, Integer> cnt = new HashMap<>();
        for (int j = 0; j < nums.length; j++) {
            cnt.put(nums[j], cnt.getOrDefault(nums[j], 0) + 1);
            while (cnt.size() > k) {
                cnt.put(nums[i], cnt.get(nums[i]) - 1);
                cnt.remove(nums[i++], 0);
            }
            res += j - i + 1;
        }
        return res;
    }
}
```

#### 1004. Max Consecutive Ones III

java

```java
class Solution {  
    public int longestOnes(int[] nums, int k) {  
        int res = 0, cnt = 0;  
        for (int i = 0, j = 0; j < nums.length; j++) {  
            if (nums[j] == 0) cnt++;  
            while (cnt > k && nums[i++] == 0) {  
                cnt--;  
            }  
            res = Math.max(j - i + 1, res);  
        }  
        return res;  
    }  
}
```

go

```go
func longestOnes(nums []int, k int) int {  
   res, i, cnt := 0, 0, 0  
   for j, num := range nums {  
      if num == 0 {  
         cnt++  
      }  
      for ; cnt > k; i++ {  
         if nums[i] == 0 {  
            cnt--  
         }  
      }  
      res = max(res, j-i+1)  
   }  
   return res  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}
```

#### 1052. Grumpy Bookstore Owner

java

```java
class Solution {
    public int maxSatisfied(int[] customers, int[] grumpy, int minutes) {
        int window = 0, res1 = 0, res2 = 0;
        for (int i = 0; i < customers.length; i++) {
            if (grumpy[i] == 0) res1 += customers[i];
            else window += customers[i];
            if (i >= minutes) {
                int left = i - minutes;
                window -= grumpy[left] * customers[left];
            }
            res2 = Math.max(res2, window);
        }
        return res1 + res2;
    }
}
```

Go

```go
func maxSatisfied(customers []int, grumpy []int, minutes int) int {
	res1, res2, window := 0, 0, 0
	for i := range customers {
		if grumpy[i] == 0 {
			res1 += customers[i]
		} else {
			window += customers[i]
		}
		if i >= minutes {
			window -= grumpy[i-minutes] * customers[i-minutes]
		}
		res2 = max(res2, window)
	}
	return res1 + res2
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

#### 1234. Replace the Substring for Balanced String

java

```java
class Solution {
    public int balancedString(String s) {
        int i = 0, n = s.length(), k = n / 4, res = n;
        int[] cnt = new int[128];
        for (int j = 0; j < n; j++) {
            cnt[s.charAt(j)]++;
        }
        for (int j = 0; j < n; j++) {
            // erase elements in the window
            cnt[s.charAt(j)]--;
            while (i < n && cnt['Q'] <= k && cnt['W'] <= k && cnt['E'] <= k && cnt['R'] <= k) {
                res = Math.min(j - i + 1, res);
                cnt[s.charAt(i++)]++;
            }
        }
        return res;
    }
}
```

#### 1248. Count Number of Nice Subarrays

go

```go
func numberOfSubarrays(nums []int, k int) int {
	return atMost(nums, k) - atMost(nums, k-1)
}
func atMost(nums []int, goal int) int {
	res, i, odd := 0, 0, 0
	for j, num := range nums {
		odd += num & 1
		for ; odd > goal; i++ {
			odd -= nums[i] & 1
		}
		res += j - i + 1
	}
	return res
}
```

java

```java
class Solution {
    public int numberOfSubarrays(int[] nums, int k) {
        return atMost(nums, k) - atMost(nums, k - 1);
    }

    int atMost(int[] nums, int k) {
        int cnt = 0, i = 0, res = 0;
        for (int j = 0; j < nums.length; j++) {
            cnt += nums[j] & 1;
            while (cnt > k) {
                cnt -= nums[i++] & 1;
            }
            res += j - i + 1;
        }
        return res;
    }
}
```

#### 1358. Number of Substrings Containing All Three Characters

go

```go
func numberOfSubstrings(s string) int {  
   cnt := [3]int{}  
   i := 0  
   res := 0  
   for j := range s {  
      cnt[s[j]-'a']++  
      for cnt[0] > 0 && cnt[1] > 0 && cnt[2] > 0 {  
         cnt[s[i]-'a']--  
         i++  
      }  
      res += i  
   }  
   return res  
}
```

java

```java
class Solution {  
    public int numberOfSubstrings(String s) {  
        int[] count = {0, 0, 0};  
        int res = 0, i = 0;  
        for (int j = 0; j < s.length(); j++) {  
            count[s.charAt(j) - 'a']++;  
            while (count[0] > 0 && count[1] > 0 && count[2] > 0) count[s.charAt(i++) - 'a']--;  
            res += i;  
        }  
        return res;  
    }  
}
```

#### 1425. Constrained Subsequence Sum

java

```java
class Solution {
    public int constrainedSubsetSum(int[] nums, int k) {
        int n = nums.length, res = nums[0];
        int[] cp = Arrays.copyOf(nums, n);
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (!deque.isEmpty()) cp[i] += deque.peek();
            res = Math.max(res, cp[i]);
            while (!deque.isEmpty() && deque.peekLast() < cp[i]) {
                deque.pollLast();
            }
            if (cp[i] > 0) deque.offer(cp[i]);
            if (!deque.isEmpty() && i >= k && deque.peek() == cp[i - k]) {
                deque.pollFirst();
            }
        }
        return res;
    }
}
```

go

```go
func constrainedSubsetSum(nums []int, k int) int {
	var deque []int
	cp := make([]int, len(nums))
	copy(cp, nums)
	for i := range cp {
		if len(deque) > 0 {
			cp[i] += deque[0]
		}
		for len(deque) > 0 && cp[i] > deque[len(deque)-1] {
			deque = deque[:len(deque)-1]
		}
		if cp[i] > 0 {
			deque = append(deque, cp[i])
		}
		if i >= k && len(deque) > 0 && deque[0] == cp[i-k] {
			deque = deque[1:]
		}
	}
	return max(cp)
}
func max(nums []int) int {
	res := nums[0]
	for _, v := range nums {
		if v > res {
			res = v
		}
	}
	return res
}
```

#### 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

go

```go
func longestSubarray(nums []int, limit int) int {
    var (
        maxQ []int
        minQ []int
    )
    i := 0
    for _, num := range nums {
        for len(maxQ) > 0 && maxQ[len(maxQ)-1] < num {
            maxQ = maxQ[:len(maxQ)-1]
        }
        for len(minQ) > 0 && minQ[len(minQ)-1] > num {
            minQ = minQ[:len(minQ)-1]
        }
        maxQ = append(maxQ, num)
        minQ = append(minQ, num)
        if maxQ[0]-minQ[0] > limit {
            if maxQ[0] == nums[i] {
                maxQ = maxQ[1:]
            }
            if minQ[0] == nums[i] {
                minQ = minQ[1:]
            }
            i++
        }
    }
    return len(nums) - i
}
```

java

```java
    int longestSubarray(int[] nums, int limit) {
        ArrayDeque<Integer> maxQueue = new ArrayDeque<>();
        ArrayDeque<Integer> minQueue = new ArrayDeque<>();
        int i = 0;
        for (int num : nums) {
            while (!maxQueue.isEmpty() && maxQueue.peekLast() < num) maxQueue.pollLast();
            while (!minQueue.isEmpty() && minQueue.peekLast() > num) minQueue.pollLast();
            maxQueue.offer(num);
            minQueue.offer(num);
            if (maxQueue.peek() - minQueue.peek() > limit) {
                if (maxQueue.peek() == nums[i]) maxQueue.poll();
                if (minQueue.peek() == nums[i]) minQueue.poll();
                i++;
            }
        }
        return nums.length - i;
    }
```

#### 1456. Maximum Number of Vowels in a Substring of Given Length

go

```go
func maxVowels(s string, k int) int {
	count := 0
	res := 0
	for i := range s {
		if isVowel(s[i]) {
			count++
		}
		if i > k-1 && isVowel(s[i-k]) {
			count--
		}
		res = max(count, res)
	}
	return res
}
func isVowel(b byte) bool {
	switch b {
	case 'a', 'e', 'i', 'o', 'u':
		return true
	default:
		return false
	}
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

java

```java
class Solution {
    public int maxVowels(String s, int k) {
        int count = 0, res = 0;
        for (int i = 0; i < s.length(); i++) {
            if (isVowel(s.charAt(i))) {
                count++;
            }
            if (i > k - 1 && isVowel(s.charAt(i - k))) {
                count--;
            }
            res = Math.max(res, count);
        }
        return res;
    }

    boolean isVowel(char b) {
        switch (b) {
            case 'a', 'e', 'i', 'o', 'u' -> {
                return true;
            }
        }
        return false;
    }
}
```

#### 2134. Minimum Swaps to Group All 1's Together II

java

```java
class Solution {
    public int minSwaps(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        int count = 0;
        int n = nums.length;
        int res = sum;
        for (int i = 0; i < 2 * n; i++) {
            if (nums[i % n] == 0) count++;
            if (i > sum - 1 && nums[(i - sum) % n] == 0) count--;
            if (i >= sum - 1) res = Math.min(res, count);
        }
        return res;
    }
}
```