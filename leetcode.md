#### [165. Compare Version Numbers](https://leetcode.com/problems/compare-version-numbers/)

java

```java
class Solution {
    public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        int len = Math.max(v1.length, v2.length);
        for (int i = 0; i < len; i++) {
            int num1 = (i < v1.length) ? Integer.parseInt(v1[i]) : 0;
            int num2 = (i < v2.length) ? Integer.parseInt(v2[i]) : 0;
            if (num1 == num2) continue;
            return Integer.compare(num1, num2);
        }
        return 0;
    }
}
```

python

```python
class Solution:  
    def compareVersion(self, version1: str, version2: str) -> int:  
        v1 = version1.split(".")  
        v2 = version2.split(".")  
        m, n = len(v1), len(v2)  
        for i in range(max(m, n)):  
            num1 = int(v1[i]) if i < m else 0  
            num2 = int(v2[i]) if i < n else 0  
            if num1 == num2:  
                continue  
            return -1 if num1 < num2 else 1  
        return 0
```
#### [1222. Queens That Can Attack the King](https://leetcode.com/problems/queens-that-can-attack-the-king/)


```java
class Solution {
    public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
        HashSet<List<Integer>> set = new HashSet<>();
        for (int[] queen : queens) {
            set.add(List.of(queen[0], queen[1]));
        }
        ArrayList<List<Integer>> res = new ArrayList<>();
        int[] dir = {-1, 0, 1};
        for (int i : dir) {
            for (int j : dir) {
                for (int k = 1; k < 8; k++) {
                    int x = i * k + king[0];
                    int y = j * k + king[1];
                    List<Integer> coordinate = List.of(x, y);
                    if (set.contains(coordinate)) {
                        res.add(coordinate);
                        break;
                    }
                }
            }
        }
        return res;
    }
}
```


#### 989. Add to Array-Form of Integer

java

```java
class Solution {
    public List<Integer> addToArrayForm(int[] num, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = num.length - 1; i >= 0; i--) {
            res.add((num[i] + k) % 10);
            k = (num[i] + k) / 10;
        }
        while (k > 0) {
            res.add(k % 10);
            k /= 10;
        }
        Collections.reverse(res);
        return res;
    }
}
```

#### [1462. Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/)

java

```java
class Solution {
    // Floyd algorithm
    public List<Boolean> checkIfPrerequisite(int n, int[][] prerequisites, int[][] queries) {
        boolean[][] connected = new boolean[n][n];
        for (int[] pre : prerequisites) {
            connected[pre[0]][pre[1]] = true;
        }
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    connected[i][j] |= connected[i][k] && connected[k][j];
                }
            }
        }
        ArrayList<Boolean> res = new ArrayList<>();
        for (int[] query : queries) {
            res.add(connected[query[0]][query[1]]);
        }
        return res;
    }
}
```
#### 1557. Minimum Number of Vertices to Reach All Nodes

java

```java
class Solution {
    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
        boolean[] reachable = new boolean[n];
        for (List<Integer> edge : edges) {
            reachable[edge.get(1)] = true;
        }
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (reachable[i]) {
                continue;
            }
            res.add(i);
        }
        return res;
    }
}
```

#### 352. Data Stream as Disjoint Intervals

java

```java
class SummaryRanges {
    TreeMap<Integer, int[]> treeMap;

    public SummaryRanges() {
        treeMap = new TreeMap<>();
    }

    public void addNum(int val) {
        if (treeMap.containsKey(val)) {
            return;
        }
        Integer lowerKey = treeMap.lowerKey(val);
        Integer higherKey = treeMap.higherKey(val);
        if (lowerKey != null && higherKey != null && val == treeMap.get(lowerKey)[1] + 1 && val == treeMap.get(higherKey)[0] - 1) {
            treeMap.get(lowerKey)[1] = treeMap.get(higherKey)[1];
            treeMap.remove(higherKey);
            return;
        }
        if (lowerKey != null && val <= treeMap.get(lowerKey)[1] + 1) {
            treeMap.get(lowerKey)[1] = Math.max(val, treeMap.get(lowerKey)[1]);
            return;
        }
        if (higherKey != null && val == treeMap.get(higherKey)[0] - 1) {
            treeMap.put(val, new int[]{val, treeMap.get(higherKey)[1]});
            treeMap.remove(higherKey);
            return;
        }
        treeMap.put(val, new int[]{val, val});
    }

    public int[][] getIntervals() {
        return treeMap
                .values()
                .stream()
                .sorted(Comparator.comparingInt(a -> a[0]))
                .toArray(int[][]::new);
    }
}
```

#### 1267. Count Servers that Communicate

java

```java
class Solution {
    public int countServers(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[] rows = new int[m], cols = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    continue;
                }
                rows[i]++;
                cols[j]++;
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    continue;
                }
                if (rows[i] > 1 || cols[j] > 1) {
                    res++;
                }
            }
        }
        return res;
    }
}
```

#### 709. To Lower Case

```go
func toLowerCase(s string) string {
	arr := []byte(s)
	for i := range arr {
		if arr[i] >= 'A' && arr[i] <= 'Z' {
			arr[i] += -'A' + 'a'
		}
	}
	return string(arr)
}
```

#### 6. Zigzag Conversion

```go
func convert(s string, numRows int) string {
    matrix, down, up := make([][]rune, len(s)), 0, numRows-2
    t := []rune(s)
    for i := 0; i < len(s); {
        if down < numRows {
            matrix[down] = append(matrix[down], t[i])
            i++
            down++
        } else if up > 0 {
            matrix[up] = append(matrix[up], t[i])
            i++
            up--
        } else {
            down, up = 0, numRows-2
        }
    }
    solution := make([]rune, 0, len(s))
    for _, row := range matrix {
        for _, item := range row {
            solution = append(solution, item)
        }
    }
    return string(solution)
}
```

#### 7. Reverse Integer

```go
func reverse(x int) int {  
   res := 0  
   for x != 0 {  
      res = res*10 + x%10  
      x /= 10  
   }  
   if res > 1<<31-1 || res < -1<<31 {  
      return 0  
   }  
   return res  
}
```

#### 8. String to Integer (atoi)

Go

```go
func myAtoi(s string) int {
	idx, n := 0, len(s)
	for idx < n && s[idx] == ' ' {
		idx++
	}
	sign := 1
	if idx < n && (s[idx] == '-' || s[idx] == '+') {
		if s[idx] == '-' {
			sign = -1
		}
		idx++
	}
	res := 0
	for idx < n && s[idx] >= '0' && s[idx] <= '9' {
		val := int(s[idx] - '0')
		if res > (math.MaxInt32-val)/10 {
			if sign == 1 {
				return math.MaxInt32
			}
			return math.MinInt32
		}
		res = res*10 + val
		idx++
	}
	return res * sign
}
```

#### 9. Palindrome Number

```go
func isPalindrome(x int) bool {
    if x < 0 || (x > 0 && x % 10 == 0) {
        return false
    }
    inputNum := x
    newNum := 0
    for x != 0 {
        newNum = newNum*10 + x % 10
        x = x/10
    }
    return newNum == inputNum
}
```

#### 12. Integer to Roman

go

```go
func intToRoman(num int) string {
	M := []string{"", "M", "MM", "MMM"}
	C := []string{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"}
	X := []string{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"}
	I := []string{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}
	return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10]
}
```

#### 13. Roman to Integer

go

```go
func romanToInt(s string) int {  
   translate := map[byte]int{  
      'I': 1,  
      'V': 5,  
      'X': 10,  
      'L': 50,  
      'C': 100,  
      'D': 500,  
      'M': 1000,  
   }  
   s = strings.Replace(s, "IV", "IIII", -1)  
   s = strings.Replace(s, "IX", "VIIII", -1)  
   s = strings.Replace(s, "XL", "XXXX", -1)  
   s = strings.Replace(s, "XC", "LXXXX", -1)  
   s = strings.Replace(s, "CD", "CCCC", -1)  
   s = strings.Replace(s, "CM", "DCCCC", -1)  
   res := 0  
   for i := range s {  
      res += translate[s[i]]  
   }  
   return res  
}
```

#### 14. Longest Common Prefix

找出最长公共前缀

Vertical Scanning

```go
func longestCommonPrefix(strs []string) string {
	var char byte
	for i := 0; ; i++ {
		for k, str := range strs {
			if i == len(str) || (str[i] != char && k > 0) {
				return str[:i]
			}
			char = str[i]
		}
	}
}
```

java

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        char prev = 0;
        for (int i = 0; ; i++) {
            for (int j = 0; j < strs.length; j++) {
                String str = strs[j];
                if (i == str.length() || j > 0 && str.charAt(i) != prev) {
                    return str.substring(0, i);
                }
                prev = str.charAt(i);
            }
        }
    }
}
```


#### 18. 4Sum
go
```go
func fourSum(nums []int, target int) [][]int {
    res := make([][]int, 0)
    sort.Ints(nums)
    for a := range nums {
        if a > 0 && nums[a] == nums[a-1] {
            continue
        }
        if nums[a] > target && (nums[a] > 0 || target > 0) {
            return res
        }
        for b := a + 1; b < len(nums); b++ {
            if b > a+1 && nums[b] == nums[b-1] {
                continue
            }
            c, d := b+1, len(nums)-1
            for c < d {
                switch {
                case nums[a]+nums[b]+nums[c]+nums[d] == target:
                    res = append(res, []int{nums[a], nums[b], nums[c], nums[d]})
                    for c < d && nums[c] == nums[c+1] {
                        c++
                    }
                    for c < d && nums[d] == nums[d-1] {
                        d--
                    }
                    d--
                    c++
                case nums[a]+nums[b]+nums[c]+nums[d] < target:
                    c++
                default:
                    d--
                }
            }
        }
    }
    return res
}
```

#### 26. Remove Duplicates from Sorted Array

go

```go
func removeDuplicates(nums []int) int {  
   n := len(nums)  
   count := 0  
   for i := 1; i < n; i++ {  
      if nums[i] == nums[i-1] {  
         count++  
      } else {  
         nums[i-count] = nums[i]  
      }  
   }  
   return n - count  
}
```

go

```go
func removeDuplicates(nums []int) int {  
   i := 0  
   for _, v := range nums {  
      if i == 0 || v > nums[i-1] {  
         nums[i] = v  
         i++  
      }  
   }  
   return i  
}
```

java

```java
class Solution {  
  public int removeDuplicates(int[] nums) {  
    int i = 0;  
    for (int n : nums) {  
      if (i == 0 || n > nums[i - 1]) {  
        nums[i++] = n;  
      }  
    }  
    return i;  
  }  
}
```


```java
class Solution {  
  public int removeDuplicates(int[] nums) {  
    int count = 0, n = nums.length;  
    for (int i = 1; i < n; i++) {  
      if (nums[i] == nums[i - 1]) {  
        count++;  
      }else{  
        nums[i - count] = nums[i];  
      }  
    }  
    return n-count;  
  }  
}
```

#### 27. Remove Element
go

```go
func removeElement(nums []int, val int) int {
    i := 0
    for _, v := range nums {
        if v != val {
            nums[i] = v
            i++
        }
    }
    return i
}
```

https://leetcode.com/problems/remove-element/discuss/12584/6-line-Python-solution-48-ms

python

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        for x in nums:
            if x != val:
                nums[i] = x
                i += 1
        return i
```

#### 28. Implement strStr()

go

```go
func strStr(haystack string, needle string) int {
   for i := 0; i < len(haystack)-len(needle)+1; i++ {
      j := 0
      for ; j < len(needle); j++ {
         if needle[j] != haystack[i+j] {
            break
         }
      }
      if j == len(needle) {
         return i
      }
   }
   return -1
}
```

python

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for i in range(len(haystack)-len(needle)+1):
            j = 0
            while j < len(needle):
                if haystack[i+j] != needle[j]:
                    break
                j += 1
            if j == len(needle):
                return i
        return -1
```

#### 29. Divide Two Integers

```go
// go
func sig(dividend int, divisor int) int {
    if dividend > 0 && divisor > 0 || dividend < 0 && divisor < 0 {
        return 1
    }
    return -1
}
func divide(dividend int, divisor int) int {
    if dividend == math.MinInt32 && divisor == -1 {
        return math.MaxInt32
    }
    signal, res := sig(dividend, divisor), 0
    dividend, divisor = abs(dividend), abs(divisor)
    for i := 31; i >= 0; i-- {
        if (dividend >> i) >= divisor {
            res |= 1 << i
            dividend -= divisor << i
        }
    }
    return res * signal
}
func abs(a int) int {
    if a < 0 {
        return -a
    }
    return a
}
```



```python
# python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        if dividend == -(2**31) and divisor == -1:
            return 2**31-1
        if dividend < 0 and divisor < 0 or dividend > 0 and divisor > 0:
            sig = 1
        else:
            sig = -1
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        for i in range(31, -1, -1):
            if (dividend >> i) >= divisor:
                res |= 1<<i
                dividend -= divisor<<i
        return res*sig
```

#### 31. Next Permutation

```go
func nextPermutation(nums []int) {
    i := len(nums) - 2
    for i > -1 && nums[i] >= nums[i+1] {
        i--
    }
    if i == -1 {
        sort.Ints(nums)
        return
    }
    j := i + 2
    for j < len(nums) && nums[i] < nums[j] {
        j++
    }
    nums[i], nums[j-1] = nums[j-1], nums[i]
    start, end := i+1, len(nums)-1
    for start < end {
        nums[start], nums[end] = nums[end], nums[start]
        start++
        end--
    }
}
```

python

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = len(nums)-2
        while i >= 0 and nums[i] >= nums[i+1]:
            i -= 1
        if i == -1:
            nums.sort()
            return
        j = i + 2
        while j < len(nums) and nums[j] > nums[i]:
            j += 1
        nums[j-1], nums[i] = nums[i], nums[j-1]
        start, end = i+1, len(nums)-1
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
```


#### 41. First Missing Positive

```go
func firstMissingPositive(nums []int) int {
   for _, v := range nums {
      for v > 0 && v <= len(nums) && nums[v-1] != v {
         nums[v-1], v = v, nums[v-1]
      }
   }
   for i, v := range nums {
      if v != i+1 {
         return i + 1
      }
   }
   return len(nums) + 1
}
```

#### 45. Jump Game II

go

```go
func jump(nums []int) int {  
   if len(nums) == 1 {  
      return 0  
   }  
   curr, maxJump, steps := 0, 0, 0  
   for k, v := range nums {  
      maxJump = max(maxJump, k+v)  
      if k == curr {  
         curr = maxJump  
         steps++  
         if curr >= len(nums)-1 {  
            break  
         }  
      }  
   }  
   return steps  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}
```

python

```python
class Solution:  
    def jump(self, nums: List[int]) -> int:  
        if len(nums) == 1:  
            return 0  
        step, curr, max_jump = 0, 0, 0  
        for k, v in enumerate(nums):  
            max_jump = max(max_jump, k + v)  
            if k == curr:  
                curr = max_jump  
                step += 1  
            if curr >= len(nums) - 1:  
                return step
```

#### 48. Rotate Image

```go
func rotate(matrix [][]int) {
   for i := 0; i < len(matrix)/2; i++ {
      matrix[i], matrix[len(matrix)-1-i] = matrix[len(matrix)-1-i], matrix[i]
   }
   for i := 0; i < len(matrix); i++ {
      for j := i + 1; j < len(matrix); j++ {
         matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
      }
   }
}
```

#### 54. Spiral Matrix

```go
func valid(up, down, left, right int) bool {
    return up < down && left < right
}
func spiralOrder(matrix [][]int) (res []int) {
    up, down, left, right := 0, len(matrix), 0, len(matrix[0])
    var x, y int
    for up < down && left < right {
        for x, y = left, up; x < right && valid(up, down, left, right); x++ {
            res = append(res, matrix[y][x])
        }
        up++
        for x, y = right-1, up; y < down && valid(up, down, left, right); y++ {
            res = append(res, matrix[y][x])
        }
        right--
        for x, y = right-1, down-1; x >= left && valid(up, down, left, right); x-- {
            res = append(res, matrix[y][x])
        }
        down--
        for x, y = left, down-1; y >= up && valid(up, down, left, right); y-- {
            res = append(res, matrix[y][x])
        }
        left++
    }
    return res
}
```

python

```python
class Solution:
    def valid(self, up, down, left, right) -> bool:
        return up < down and left < right

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        up, down, left, right = 0, len(matrix), 0, len(matrix[0])
        res = []
        x, y = 0, 0
        while up < down and left < right:
            for x in range(left, right):
                y = up
                res += matrix[y][x],
            up += 1
            if not self.valid(up, down, left, right):
                break
            for y in range(up, down):
                x = right - 1
                res += matrix[y][x],
            right -= 1
            if not self.valid(up, down, left, right):
                break
            for x in range(right - 1, left - 1, -1):
                y = down - 1
                res += matrix[y][x],
            down -= 1
            if not self.valid(up, down, left, right):
                break
            for y in range(down - 1, up - 1, -1):
                x = left
                res += matrix[y][x],
            left += 1
            if not self.valid(up, down, left, right):
                break
        return res
```

#### 55. Jump Game

go

```go
func canJump(nums []int) bool {
    maxJump := nums[0]
    for k, v := range nums {
        if maxJump < k {
            return false
        }
        maxJump = max(k+v, maxJump)
        if maxJump >= len(nums)-1 {
            return true
        }
    }
    return false
}
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

python

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_jump = nums[0]
        for k, v in enumerate(nums):
            if max_jump < k:
                return False
            max_jump = max(max_jump,k+v)
            if max_jump >= len(nums)-1:
                return True
        return False
```

#### 56. Merge Intervals

go

```go
func merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	var result [][]int
	for _, interval := range intervals {
		left := interval[0]
		right := interval[1]
		if len(result) > 0 && left <= result[len(result)-1][1] {
			result[len(result)-1][1] = max(result[len(result)-1][1], right)
		} else {
			result = append(result, interval)
		}
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

python

```python
class Solution:

    def merge(self, intervals):
        out = []
        intervals.sort(key=lambda i: i[0])
        for i in intervals:
            if out and i[0] <= out[-1][1]:
                out[-1][1] = max(out[-1][1], i[1])
            else:
                out += i,
        return out
```

#### 57. Insert Interval

go

```go
func insert(intervals [][]int, newInterval []int) [][]int {
   ret := make([][]int, 0)
   i := 0
   for ; i < len(intervals) && intervals[i][1] < newInterval[0]; i++ {
      ret = append(ret, intervals[i])
   }
   for ; i < len(intervals) && intervals[i][0] <= newInterval[1]; i++ {
      newInterval = []int{min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])}
   }
   ret = append(ret, newInterval)
   for ; i < len(intervals); i++ {
      ret = append(ret, intervals[i])
   }
   return ret
}
func min(a, b int) int {
   if a < b {
      return a
   }
   return b
}
func max(a, b int) int {
   if a > b {
      return a
   }
   return b
}
```

python

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        ret = []
        i = 0
        while i < len(intervals) and intervals[i][1] < newInterval[0]:
            ret += intervals[i],
            i += 1
        while i < len(intervals) and intervals[i][0] <= newInterval[1]:
            newInterval = [min(intervals[i][0], newInterval[0]), max(intervals[i][1], newInterval[1])]
            i += 1
        ret += newInterval,
        while i < len(intervals):
            ret += intervals[i],
            i += 1
        return ret
```


#### 59. Spiral Matrix II

go

```go
func generateMatrix(n int) [][]int {
   grid := make([][]int, n)
   for i := range grid {
      grid[i] = make([]int, n)
   }
   left, right := 0, n
   up, down := 0, n
   count := 1
   for left < right && up < down {
      for x := left; x < right; x++ {
         grid[up][x] = count
         count++
      }
      up++
      for y := up; y < down; y++ {
         grid[y][right-1] = count
         count++
      }
      right--
      for x := right - 1; x >= left; x-- {
         grid[down-1][x] = count
         count++
      }
      down--
      for y := down - 1; y >= up; y-- {
         grid[y][left] = count
         count++
      }
      left++
   }
   return grid
}
```

python

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        grid = [[0 for _ in range(n)] for _ in range(n)]
        count = 1
        left, right, up, down = 0, n, 0, n
        while left < right and up < down:
            for x in range(left, right):
                grid[up][x] = count
                count += 1
            up += 1
            for y in range(up, down):
                grid[y][right - 1] = count
                count += 1
            right -= 1
            for x in range(right - 1, left - 1, -1):
                grid[down - 1][x] = count
                count += 1
            down -= 1
            for y in range(down - 1, up - 1, -1):
                grid[y][left] = count
                count += 1
            left += 1
        return grid
```


#### 62. Unique Paths

dp---go

```go
func uniquePaths(m int, n int) int {
   dp := make([]int, n)
   for k := range dp {
      dp[k] = 1
   }
   for i := 1; i < m; i++ {
      for j := 1; j < n; j++ {
         dp[j] += dp[j-1]
      }
   }
   return dp[n-1]
}
```

dp---python

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1 for _ in range(n)]
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[n - 1]
```

#### 63. Unique Paths II

go

```go
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
   dp := make([]int, len(obstacleGrid[0]))
   dp[0] = 1
   for i := range obstacleGrid {
      for j := range obstacleGrid[0] {
         if obstacleGrid[i][j] == 1 {
            dp[j] = 0
         } else if j > 0 {
            dp[j] += dp[j-1]
         }
      }
   }
   return dp[len(dp)-1]
}
```

python

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        dp = [0 for _ in obstacleGrid[0]]
        dp[0] = 1
        for i in range(len(obstacleGrid)):
            for j in range(len(obstacleGrid[0])):
                if obstacleGrid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:
                    dp[j] += dp[j-1]
        return dp[-1]
```


#### 65. Valid Number

```go
func isNumber(s string) bool {
    pointSeen, eSeen, numberSeen := false, false, false
    for i := range s {
        switch {
        case s[i] <= '9' && s[i] >= '0':
            numberSeen = true
        case s[i] == '.':
            if eSeen || pointSeen {
                return false
            }
            pointSeen = true
        case s[i] == 'e' || s[i] == 'E':
            if eSeen || !numberSeen {
                return false
            }
            numberSeen = false
            eSeen = true
        case s[i] == '-' || s[i] == '+':
            if i > 0 && s[i-1] != 'e' && s[i-1] != 'E' {
                return false
            }
        default:
            return false
        }
    }
    return numberSeen
}
```

#### 66. Plus One

Go

```go
func plusOne(digits []int) []int {
	n := len(digits)
	cp := make([]int, n)
	copy(cp, digits)
	for i := n - 1; i >= 0; i-- {
		cp[i]++
		if cp[i] < 10 {
			return cp
		}
		cp[i] = 0
	}
	cp = make([]int, n+1)
	cp[0] = 1
	return cp
}
```

#### 67. Add Binary

```go
func addBinary(a string, b string) string {
    res := make([]byte, 0)
    i, j, carry := len(a)-1, len(b)-1, 0
    for i >= 0 || j >= 0 || carry != 0 {
        sum := carry
        if i >= 0 {
            sum += int(a[i] - '0')
            i--
        }
        if j >= 0 {
            sum += int(b[j] - '0')
            j--
        }
        res = append(res, byte(sum%2+'0'))
        carry = sum / 2
    }
    reverse(res)
    return string(res)
}
func reverse(a []byte) {
    left, right := 0, len(a)-1
    for left < right {
        a[left], a[right] = a[right], a[left]
        left++
        right--
    }
}
```


#### 70. Climbing Stairs

```go
func climbStairs(n int) int {
    prev, res := 0, 1
    for i := 0; i < n; i++ {
        prev, res = res, prev+res
    }
    return res
}
```


#### 73. Set Matrix Zeroes

go

```go
func setZeroes(matrix [][]int) {  
   col0 := false  
   m, n := len(matrix), len(matrix[0])  
   for i := 0; i < m; i++ {  
      if matrix[i][0] == 0 {  
         col0 = true  
      }  
      for j := 1; j < n; j++ {  
         if matrix[i][j] == 0 {  
            matrix[i][0], matrix[0][j] = 0, 0  
         }  
      }  
   }  
   for i := m - 1; i >= 0; i-- {  
      for j := n - 1; j >= 1; j-- {  
         if matrix[i][0] == 0 || matrix[0][j] == 0 {  
            matrix[i][j] = 0  
         }  
      }  
      if col0 {  
         matrix[i][0] = 0  
      }  
   }  
}
```

java

```java
class Solution {  
    public void setZeroes(int[][] matrix) {  
        boolean col0 = false;  
        int m = matrix.length, n = matrix[0].length;  
        for (int i = 0; i < m; i++) {  
            if (matrix[i][0] == 0) {  
                col0 = true;  
            }  
            for (int j = 1; j < n; j++) {  
                if (matrix[i][j] == 0) {  
                    matrix[i][0] = 0;  
                    matrix[0][j] = 0;  
                }  
            }  
        }  
        for (int i = m - 1; i >= 0; i--) {  
            for (int j = n - 1; j >= 1; j--) {  
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {  
                    matrix[i][j] = 0;  
                }  
            }  
            if (col0) {  
                matrix[i][0] = 0;  
            }  
        }  
    }  
}
```

#### 75. Sort Colors

```go
func sortColors(nums []int) {
   red, white, blue := 0, 0, len(nums)-1
   for white <= blue {
      switch nums[white] {
      case 0:
         nums[white], nums[red] = nums[red], nums[white]
         red++
         white++
      case 1:
         white++
      case 2:
         nums[white], nums[blue] = nums[blue], nums[white]
         blue--
      }
   }
}
```


#### 80. Remove Duplicates from Sorted Array II

```go
func removeDuplicates(nums []int) int {
   i := 0
   for _, n := range nums {
      if i < 2 || n > nums[i-2] {
         nums[i] = n
         i++
      }
   }
   return i
}
```

#### 81. Search in Rotated Sorted Array II

https://leetcode.com/problems/search-in-rotated-sorted-array-ii/discuss/28195/Python-easy-to-understand-solution-(with-comments)

```go
func search(nums []int, target int) bool {
   left, right := 0, len(nums)
   for left < right {
      mid := (left + right) / 2
      if nums[mid] == target {
         return true
      }
      for left < mid && nums[left] == nums[mid] {
         left++
      }
      if nums[left] <= nums[mid] {
         if nums[left] <= target && target < nums[mid] {
            right = mid
         } else {
            left = mid + 1
         }
      } else {
         if nums[mid] < target && target <= nums[right-1] {
            left = mid + 1
         } else {
            right = mid
         }
      }
   }
   return false
}
```


#### 134. Gas Station

naive greedy

```go
func canCompleteCircuit(gas []int, cost []int) int {
    curr, minGas := 0, math.MaxInt
    for i := range gas {
        curr += gas[i] - cost[i]
        minGas = min(minGas, curr)
    }
    if curr < 0 {
        return -1
    }
    if minGas >= 0 {
        return 0
    }
    for i := len(gas) - 1; i >= 0; i-- {
        minGas += gas[i] - cost[i]
        if minGas >= 0 {
            return i
        }
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
#### 152. Maximum Product Subarray

```go
func maxProduct(nums []int) int {
   maxPro, minPro, result := nums[0], nums[0], nums[0]
   for i := 1; i < len(nums); i++ {
      if nums[i] < 0 {
         maxPro, minPro = minPro, maxPro
      }
      maxPro = max(maxPro*nums[i], nums[i])
      minPro = min(nums[i], minPro*nums[i])
      result = max(result, maxPro)
   }
   return result
}
```

#### 169. Majority Element

求众数，出现频率超过一半的数

排序，然后找到位于中间的数就是答案。

```go
func majorityElement(nums []int) int {
    sort.Ints(nums)
    return nums[len(nums)/2]
}
```

#### 172. Factorial Trailing Zeroes

python

```python
class Solution:  
    def trailingZeroes(self, n: int) -> int:  
        return 0 if n == 0 else n // 5 + self.trailingZeroes(n // 5)
```

#### 179. Largest Number

```go
func largestNumber(nums []int) string {
    sort.Slice(nums, func(i, j int) bool {
        v1, v2 := float64(nums[i]), float64(nums[j])
        if v1 == v2 || v1*v2 == 0 {
            return v1 > v2
        }
        lg1, lg2 := int(math.Log10(v1)), int(math.Log10(v2))
        return v2*math.Pow10(lg1+1)+v1 < v1*math.Pow10(lg2+1)+v2
    })
    res := ""
    for _, num := range nums {
        if len(res) > 0 && res[0] == '0' {
            continue
        }
        res += strconv.Itoa(num)
    }
    return res
}
```

#### 189. Rotate Array

给定一个数组，将数组中的元素向右移动 k 个位置，多出来的元素放左边

要求不使用额外空间。

冒冒失失的用 k 作为索引，一提交就会越界，而假设数组长度为7，k=7，相当于没有反转。

将题目要求的反转拆分成三步：

反转整个数组

反转从0到k的元素

反转从k到末位的元素

```go
func rotate(nums []int, k int) {
   k %= len(nums)
   reverse(nums)
   reverse(nums[:k])
   reverse(nums[k:])
}

func reverse(arr []int) {
   for i := 0; i < len(arr)/2; i++ {
      arr[i], arr[len(arr)-i-1] = arr[len(arr)-i-1], arr[i]
   }
}
```


#### 202. Happy Number

go

```go
func isHappy(n int) bool {
    mp := make(map[int]bool)
    for !mp[n] {
        mp[n] = true
        n = getNext(n)
    }
    return n == 1
}
func getNext(n int) int {
    s, res := strconv.Itoa(n), 0
    for i := range s {
        res += square(s[i] - 48)
    }
    return res
}

func square(n uint8) int {
    m := int(n)
    return m * m
}
```

python

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()
        while n not in seen:
            seen.add(n)
            n = sum(int(x) ** 2 for x in str(n))
        return n == 1
```


#### 204. Count Primes

go

```go
func countPrimes(n int) int {
    if n < 2 {
        return 0
    }
    prime, count := make([]bool, n), 0
    for i := 2; i < n; i++ {
        prime[i] = true
    }
    for i := 2; i < int(math.Sqrt(float64(n)))+1; i++ {
        if prime[i] {
            for j := 2; i*j < n; j++ {
                prime[i*j] = false
            }
        }
    }
    for i := 2; i < len(prime); i++ {
        if prime[i] {
            count++
        }
    }
    return count
}
```


#### 221. Maximal Square

python

```python
class Solution:

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m, n = len(matrix), len(matrix[0])
        res = 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '1':
                    dp[i + 1][j + 1] = min(dp[i][j], dp[i][j + 1], dp[i + 1][j]) + 1
                    res = max(res, dp[i + 1][j + 1])
        return res ** 2
```

#### 223. Rectangle Area

python

```python
class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        width = min(ax2, bx2) - max(ax1, bx1)
        height = min(ay2, by2) - max(ay1, by1)
        return (bx2 - bx1) * (by2 - by1) + (ax2 - ax1) * (ay2 - ay1) - max(width, 0) * max(height, 0)
```

#### 228. Summary Ranges

go

```go
func summaryRanges(nums []int) []string {  
   var res []string  
   for i := 0; i < len(nums); i++ {  
      left := i  
      for i+1 < len(nums) && nums[i+1] == nums[i]+1 {  
         i++  
      }  
      s := strconv.Itoa(nums[left])  
      if left != i {  
         s += "->" + strconv.Itoa(nums[i])  
      }  
      res = append(res, s)  
   }  
   return res  
}
```

java

```java
class Solution {  
    public List<String> summaryRanges(int[] nums) {  
        ArrayList<String> res = new ArrayList<>();  
        for (int i = 0; i < nums.length; i++) {  
            int left = i;  
            while (i + 1 < nums.length && nums[i + 1] == nums[i] + 1) {  
                i++;  
            }  
            if (left != i) {  
                res.add(nums[left] + "->" + nums[i]);  
            } else {  
                res.add(nums[left] + "");  
            }  
        }  
        return res;  
    }  
}
```

#### 233. Number of Digit One

the current digit being 0, 1 and >=2

go

```go
func countDigitOne(n int) int {  
   res := 0  
   for m := 1; m < n+1; m *= 10 {  
      res += (n/m + 8) / 10 * m
      if n/m%10 == 1 {  
         res += n%m + 1  
      }  
   }  
   return res  
}
```


#### 263. Ugly Number

go

```go
func isUgly(n int) bool {
   for _, p := range []int{2, 3, 5} {
      for n%p == 0 && n > 0 {
         n /= p
      }
   }
   return n == 1
}
```

java

```java
class Solution {  
    public boolean isUgly(int n) {  
        for (int i = 2; i < 6 && n > 0; i++) {  
            while (n % i == 0) {  
                n /= i;  
            }  
        }  
        return n == 1;  
    }  
}
```

#### 264. Ugly Number II

```go
func nthUglyNumber(n int) int {
   dp := make([]int, n+1)
   dp[1] = 1
   p2, p3, p5 := 1, 1, 1
   for i := 2; i < len(dp); i++ {
      t2, t3, t5 := dp[p2]*2, dp[p3]*3, dp[p5]*5
      dp[i] = minOne(t2, t3, t5)
      if t2 == dp[i] {
         p2++
      }
      if t3 == dp[i] {
         p3++
      }
      if t5 == dp[i] {
         p5++
      }
   }
   return dp[n]
}
func minOne(values ...int) int {
   res := values[0]
   for _, value := range values {
      if res > value {
         res = value
      }
   }
   return res
}
```


#### 283. Move Zeroes

```python
class Solution:  
    def moveZeroes(self, nums: List[int]) -> None:  
        throw = 0  
        for i in range(len(nums)):  
            if nums[i] == 0:  
                throw += 1  
            elif throw > 0:  
                nums[i], nums[i - throw] = nums[i - throw], nums[i]
```

```go
func moveZeroes(nums []int) {  
   throw := 0  
   for i := range nums {  
      if nums[i] == 0 {  
         throw++  
      } else if throw > 0 {  
         nums[i], nums[i-throw] = nums[i-throw], nums[i]  
      }  
   }  
}
```


#### 292. Nim Game

```python
class Solution:
    def canWinNim(self, n: int) -> bool:
        return n%4
```

#### 303. Range Sum Query - Immutable

go

```go
type NumArray struct {  
   prefix []int  
}  
  
func Constructor(nums []int) NumArray {  
   res := NumArray{make([]int, len(nums)+1)}  
   for i := range nums {  
      res.prefix[i+1] = res.prefix[i] + nums[i]  
   }  
   return res  
}  
  
func (numArray *NumArray) SumRange(left int, right int) int {  
   return numArray.prefix[right+1] - numArray.prefix[left]  
}
```

#### 304. Range Sum Query 2D - Immutable

go

```go
type NumMatrix struct {  
   prefixMatrix [][]int  
}  
  
func Constructor(matrix [][]int) NumMatrix {  
   m, n := len(matrix), len(matrix[0])  
   res := make([][]int, m+1)  
   for i := range res {  
      res[i] = make([]int, n+1)  
   }  
   for i := range matrix {  
      res[i+1] = make([]int, len(matrix[0])+1)  
      for j := range matrix[0] {  
         res[i+1][j+1] = matrix[i][j] + res[i][j+1] + res[i+1][j] - res[i][j]  
      }  
   }  
   return NumMatrix{prefixMatrix: res}  
}  
  
func (numMatrix *NumMatrix) SumRegion(row1 int, col1 int, row2 int, col2 int) int {  
   m := numMatrix.prefixMatrix  
   return m[row2+1][col2+1] + m[row1][col1] - m[row1][col2+1] - m[row2+1][col1]  
}
```

#### 319. Bulb Switcher

https://leetcode.com/problems/bulb-switcher/discuss/77104/Math-solution

go

```go
func bulbSwitch(n int) int {
   return int(math.Sqrt(float64(n)))
}
```


#### 343. Integer Break

go

```go
func integerBreak(n int) int {
    dp := make([]int, n+1)
    dp[1], dp[2] = 1, 1
    for i := 3; i < n+1; i++ {
        for j := 1; j <= i/2; j++ {
            dp[i] = max(dp[i], j*(i-j), j*dp[i-j])
        }
    }
    return dp[n]
}
func max(values ...int) int {
    res := values[0]
    for _, value := range values {
        if value > res {
            res = value
        }
    }
    return res
}
```

python

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        if n <= 3:
            return n - 1
        if n % 3 == 0:
            return 3 ** (n // 3)
        if n % 3 == 1:
            return 3 ** (n // 3 - 1) * 4
        return 3 ** (n // 3) * 2
```

#### 344. Reverse String

go

```go
func reverseString(s []byte) {
   left, right := 0, len(s)-1
   for left < right {
      s[left], s[right] = s[right], s[left]
      left++
      right--
   }
}
```

#### 357. Count Numbers with Unique Digits

go

```go
func countNumbersWithUniqueDigits(n int) int {  
   if n == 0 {  
      return 1  
   }  
   available, curr, res := 9, 9, 10  
   // 0 <= n <= 8  
   for ; n > 1; n-- {  
      curr *= available  
      res += curr  
      available--  
   }  
   return res  
}
```

java

```java
class Solution {  
    public int countNumbersWithUniqueDigits(int n) {  
        if (n == 0) {  
            return 1;  
        }  
        int res = 10, curr = 9, available = 9;  
        // 0 <= n <= 8  
        while (n-- > 1) {  
            curr *= available;  
            res += curr;  
            available--;  
        }  
        return res;  
    }  
}
```

python

```python
class Solution:  
    def countNumbersWithUniqueDigits(self, n: int) -> int:  
        if not n:  
            return 1  
        res, curr, available = 10, 9, 9  
        for _ in range(n, 1, -1):  
            curr *= available  
            res += curr  
            available -= 1  
        return res
```

#### 367. Valid Perfect Square

```go
func isPerfectSquare(num int) bool {
   r := num
   for r*r > num {
      r = (r + num/r) / 2
   }
   return r*r == num
}
```

#### 368. Largest Divisible Subset

go

```go
func largestDivisibleSubset(nums []int) []int {  
   n := len(nums)  
   dp, prev := make([]int, n), make([]int, n)  
   sort.Ints(nums)  
   maxLen, index := 0, -1  
   for i := range nums {  
      dp[i], prev[i] = 1, -1  
      for j := i - 1; j >= 0; j-- {  
         if nums[i]%nums[j] == 0 && dp[j]+1 > dp[i] {  
            dp[i] = dp[j] + 1  
            prev[i] = j  
         }  
      }  
      if dp[i] > maxLen {  
         maxLen = dp[i]  
         index = i  
      }  
   }  
   var res []int  
   for index != -1 {  
      res = append(res, nums[index])  
      index = prev[index]  
   }  
   return res  
}
```

python

```python
class Solution:  
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:  
        n = len(nums)  
        nums.sort()  
        dp, prev, idx, max_len = [1] * n, [-1] * n, -1, 0  
        for i in range(n):  
            for j in range(i - 1, -1, -1):  
                if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:  
                    dp[i] = dp[j] + 1  
                    prev[i] = j  
            if dp[i] > max_len:  
                idx = i  
                max_len = dp[i]  
        res = []  
        while idx != -1:  
            res.append(nums[idx])  
            idx = prev[idx]  
        return res
```

java

```java
class Solution {  
    public List<Integer> largestDivisibleSubset(int[] nums) {  
        int n = nums.length;  
        Arrays.sort(nums);  
        int[] dp = new int[n], prev = new int[n];  
        int index = -1, maxLen = 0;  
        for (int i = 0; i < n; i++) {  
            dp[i] = 1;  
            prev[i] = -1;  
            for (int j = i - 1; j >= 0; j--) {  
                if (nums[i] % nums[j] == 0 && dp[j] + 1 > dp[i]) {  
                    dp[i] = dp[j] + 1;  
                    prev[i] = j;  
                }  
            }  
            if (dp[i] > maxLen) {  
                maxLen = dp[i];  
                index = i;  
            }  
        }  
        ArrayList<Integer> res = new ArrayList<>();  
        while (index != -1) {  
            res.add(nums[index]);  
            index = prev[index];  
        }  
        return res;  
    }  
}
```

#### 375. Guess Number Higher or Lower II

```go
func getMoneyAmount(n int) int {  
   dp := make([][]int, n+1)  
   for i := range dp {  
      dp[i] = make([]int, n+1)  
   }  
   for start := n; start > 0; start-- {  
      for end := start + 1; end < n+1; end++ {  
         res := math.MaxInt  
         for x := start; x < end; x++ {  
            res = min(res, x+max(dp[start][x-1], dp[x+1][end]))  
         }  
         dp[start][end] = res  
      }  
   }  
   return dp[1][n]  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}  
func min(a, b int) int {  
   if a > b {  
      return b  
   }  
   return a  
}
```

```python
class Solution:  
    def getMoneyAmount(self, n: int) -> int:  
        dp = [[0] * (n + 1) for _ in range(n + 1)]  
        for start in range(n, 0, -1):  
            for end in range(start + 1, n + 1):  
                dp[start][end] = min(x + max(dp[start][x - 1], dp[x + 1][end])  
                                     for x in range(start, end))  
        return dp[1][n]
```

#### 376. Wiggle Subsequence

```go
func wiggleMaxLength(nums []int) int {
   curr, prev, res := 0, 0, 1
   for i := 1; i < len(nums); i++ {
      curr = nums[i] - nums[i-1]
      if curr > 0 && prev <= 0 || curr < 0 && prev >= 0 {
         res++
         prev = curr
      }
   }
   return res
}
```



#### 392. Is Subsequence

```go
func isSubsequence(s string, t string) bool {  
   dp := make([][]bool, len(s)+1)  
   for i := range dp {  
      dp[i] = make([]bool, len(t)+1)  
   }  
   for i := range dp[0] {  
      dp[0][i] = true  
   }  
   for i := range s {  
      for j := range t {  
         if s[i] == t[j] {  
            dp[i+1][j+1] = dp[i][j]  
         } else {  
            dp[i+1][j+1] = dp[i+1][j]  
         }  
      }  
   }  
   return dp[len(s)][len(t)]  
}
```

#### 397. Integer Replacement

go

```go
func integerReplacement(n int) int {  
   res := 0  
   for n > 1 {  
      switch {  
      case n&1 == 0:  
         n >>= 1  
      case n == 3 || (n>>1)&1 == 0:  
         n--  
      default:  
         n++  
      }  
      res++  
   }  
   return res  
}
```

python

```python
class Solution:  
    def integerReplacement(self, n: int) -> int:  
        res = 0  
        while n > 1:  
            if n & 1 == 0:  
                n >>= 1  
            elif n == 3 or (n >> 1) & 1 == 0:  
                n -= 1  
            else:  
                n += 1  
            res += 1  
        return res
```


#### 406. Queue Reconstruction by Height

```go
func reconstructQueue(people [][]int) [][]int {
   sort.Slice(people, func(i, j int) bool {
      if people[i][0] == people[j][0] {
         return people[i][1] < people[j][1]
      }
      return people[i][0] > people[j][0]
   })
   var res [][]int
   for _, v := range people {
      res = append(res, v)
      copy(res[v[1]+1:], res[v[1]:])
      res[v[1]] = v
   }
   return res
}
```

#### 409. Longest Palindrome

go

```go
func longestPalindrome(s string) int {  
   counter := make([]int, 'z'-'A'+1)  
   odds := 0  
   for i := range s {  
      counter[s[i]-'A']++  
   }  
   for _, v := range counter {  
      if v&1 == 1 {  
         odds++  
      }  
   }  
   if odds > 0 {  
      return len(s) - odds + 1  
   }  
   return len(s)  
}
```

#### 410. Split Array Largest Sum

go

```go
func splitArray(nums []int, k int) int {  
   sum, maxNum := 0, 0  
   for _, num := range nums {  
      sum += num  
      maxNum = max(maxNum, num)  
   }  
   if k == 1 {  
      return sum  
   }  
   left, right := maxNum, sum  
   for left < right {  
      mid := (left + right) / 2  
      if isValid(nums, k, mid) {  
         right = mid  
      } else {  
         left = mid + 1  
      }  
   }  
   return left  
}  
func max(a, b int) int {  
   return int(math.Max(float64(a), float64(b)))  
}  
func isValid(nums []int, k int, target int) bool {  
   count, sum := 1, 0  
   for _, num := range nums {  
      sum += num  
      if sum > target {  
         count++  
         sum = num  
      }  
      if count > k {  
         return false  
      }  
   }  
   return true  
}
```
#### 413. Arithmetic Slices

java

```java
class Solution {  
    public int numberOfArithmeticSlices(int[] nums) {  
        int curr = 0, res = 0;  
        for (int i = 2; i < nums.length; i++) {  
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {  
                curr++;  
                res += curr;  
            } else {  
                curr = 0;  
            }  
        }  
        return res;  
    }  
}
```

go

```go
func numberOfArithmeticSlices(nums []int) int {  
   curr, res := 0, 0  
   for i := 2; i < len(nums); i++ {  
      if nums[i]-nums[i-1] == nums[i-1]-nums[i-2] {  
         curr++  
         res += curr  
      } else {  
         curr = 0  
      }  
   }  
   return res  
}
```
#### 414. Third Maximum Number

go

```go
func thirdMax(nums []int) int {  
   a, b, c := math.MinInt, math.MinInt, math.MinInt  
   for _, v := range nums {  
      switch {  
      case a < v:  
         a, b, c = v, a, b  
      case v < a && v > b:  
         b, c = v, b  
      case v < b && v > c:  
         c = v  
      }  
   }  
   if c == math.MinInt {  
      return a  
   }  
   return c  
}
```

java

```java
class Solution {  
    public int thirdMax(int[] nums) {  
        long a = Long.MIN_VALUE, b = Long.MIN_VALUE, c = Long.MIN_VALUE;  
        for (int num : nums) {  
            if (a < num) {  
                c = b;  
                b = a;  
                a = num;  
            }  
            if (num < a && num > b) {  
                c = b;  
                b = num;  
            }  
            if (num < b && num > c) {  
                c = num;  
            }  
        }  
        return c == Long.MIN_VALUE ? (int) a : (int) c;  
    }  
}
```


#### 435. Non-overlapping Intervals

```go
func eraseOverlapIntervals(intervals [][]int) int {
   sort.Slice(intervals, func(i, j int) bool {
      return intervals[i][1] < intervals[j][1]
   })
   res := 0
   for i := 0; i < len(intervals)-1; i++ {
      if intervals[i][1] > intervals[i+1][0] {
         intervals[i+1][1] = min(intervals[i+1][1], intervals[i][1])
         res++
      }
   }
   return res
}
func min(a, b int) int {
   if a < b {
      return a
   }
   return b
}
```


#### 441. Arranging Coins

java

```java
class Solution {  
    public int arrangeCoins(int n) {  
        int x = 1;  
        while (n >= x) {  
            n -= x;  
            x++;  
        }  
        return x-1;  
    }  
}
```

go

```go
func arrangeCoins(n int) int {  
   x := 1  
   for n >= x {  
      n -= x  
      x++  
   }  
   return x - 1  
}
```


#### 452. Minimum Number of Arrows to Burst Balloons

```go
func findMinArrowShots(points [][]int) int {
   sort.Slice(points, func(i, j int) bool {
      return points[i][0] < points[j][0]
   })
   res := len(points)
   for i := 0; i < len(points)-1; i++ {
      if points[i+1][0] <= points[i][1] {
         points[i+1][1] = min(points[i+1][1], points[i][1])
         res--
      }
   }
   return res
}
func min(a, b int) int {
   if a < b {
      return a
   }
   return b
}
```

#### 453. Minimum Moves to Equal Array Elements

go

```go
func minMoves(nums []int) int {  
   sum, minValue := 0, nums[0]  
   for _, v := range nums {  
      sum += v  
      if minValue > v {  
         minValue = v  
      }  
   }  
   return sum - minValue*len(nums)  
}
```

java

```java
class Solution {  
    public int minMoves(int[] nums) {  
        int min = nums[0];  
        for (int num : nums) {  
            min = Math.min(min, num);  
        }  
        int res = 0;  
        for (int num : nums) {  
            res += num - min;  
        }  
        return res;  
    }  
}
```

#### 455. Assign Cookies

```go
func findContentChildren(g []int, s []int) int {
   sort.Ints(g)
   sort.Ints(s)
   child := 0
   for pt := 0; pt < len(s) && child < len(g); pt++ {
      if s[pt] >= g[child] {
         child++
      }
   }
   return child
}
```

#### 458. Poor Pigs

https://leetcode.com/problems/poor-pigs/discuss/94266/Another-explanation-and-solution

python

```python
class Solution:
    def poorPigs(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
        res=0 
        while (minutesToTest/minutesToDie+1)**res< buckets :
            res += 1
        return res
```


#### 459. Repeated Substring Pattern

https://leetcode.com/problems/repeated-substring-pattern/discuss/94334/Easy-python-solution-with-explaination

go

```go
func repeatedSubstringPattern(s string) bool {
    ss := (s + s)[1 : 2*len(s)-1]
    return strings.Contains(ss, s)
}
```

#### 461. Hamming Distance

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        h,count = x^y,0
        while h:
            count += 1
            h &= h-1
        return count
```

#### 464. Can I Win

Go

```go
func canIWin(maxChoose int, desired int) bool {  
   if (1+maxChoose)*maxChoose/2 < desired {  
      return false  
   }  
   if desired <= 0 {  
      return true  
   }  
   return helper(desired, maxChoose, 0, map[int]bool{})  
}  
func helper(desired, n int, state int, mp map[int]bool) bool {  
   if _, ok := mp[state]; ok {  
      return mp[state]  
   }  
   for i := 0; i < n; i++ {  
      if state&(1<<i) != 0 {  
         continue  
      }  
      if desired <= i+1 || !helper(desired-(i+1), n, state|(1<<i), mp) {  
         mp[state] = true  
         return true      }  
   }  
   mp[state] = false  
   return false}
```

python

```python
class Solution:  
    def dfs(self, desired, n, mp, state) -> bool:  
        if state in mp:  
            return mp[state]  
        for i in range(n):  
            if state & (1 << i):  
                continue  
            if desired <= 1 + i or not self.dfs(desired - (i + 1), n, mp, state | (1 << i)):  
                mp[state] = True  
                return True        mp[state] = False  
        return False  
    def canIWin(self, maxChoose: int, desired: int) -> bool:  
        if (maxChoose + 1) * maxChoose // 2 < desired:  
            return False  
        if desired <= 0:  
            return True  
        return self.dfs(desired, maxChoose, {}, 0)
```

#### 476. Number Complement

go

```go
func findComplement(num int) int {  
   n := 0  
   for n < num {  
      n = (n << 1) | 1  
   }  
   return n - num  
}
```

java

```java
class Solution {  
    public int findComplement(int num) {  
        int n = 0;  
        while (n < num) {  
            n = (n << 1) | 1;  
        }  
        return n - num;  
    }  
}
```

#### 486. Predict the Winner
 When n is even, we can always consider the array [A B C D]. If A+C>=B+D, the first player will choose A, else D. Thus the first player always wins.

https://leetcode.com/problems/predict-the-winner/discuss/96828/JAVA-9-lines-DP-solution-easy-to-understand-with-improvement-to-O(N)-space-complexity.

```go
func PredictTheWinner(nums []int) bool {  
   n := len(nums)
   if n&1 == 0 {  
      return true  
   }  
   dp := make([][]int, n)  
   for i := range dp {  
      dp[i] = make([]int, n)  
      dp[i][i] = nums[i]  
   }  
   for k := 1; k < n; k++ {  
      for i := 0; i < n-k; i++ {  
         j := i + k  
         dp[i][j] = max(nums[i]-dp[i+1][j], nums[j]-dp[i][j-1])  
      }  
   }  
   return dp[0][n-1] >= 0  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}
```

O(N) space

```go
func PredictTheWinner(nums []int) bool {  
   n := len(nums)  
   if n&1 == 0 {  
      return true  
   }  
   dp := make([]int, n)  
   for i := n - 1; i >= 0; i-- {  
      for j := i; j < n; j++ {  
         if i == j {  
            dp[i] = nums[i]  
         } else {  
            dp[j] = max(nums[i]-dp[j], nums[j]-dp[j-1])  
         }  
      }  
   }  
   return dp[n-1] >= 0  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}
```

python

```python
class Solution:  
    def PredictTheWinner(self, nums: List[int]) -> bool:  
        n = len(nums)  
        dp = [0] * n  
        for i in range(n - 1, -1, -1):  
            for j in range(n):  
                if i == j:  
                    dp[j] = nums[i]  
                else:  
                    dp[j] = max(nums[i] - dp[j], nums[j] - dp[j - 1])  
        return dp[n - 1] >= 0
```



#### 495. Teemo Attacking

go

```go
func findPoisonedDuration(timeSeries []int, duration int) int {
   ret := 0
   for i := 0; i < len(timeSeries)-1; i++ {
      ret += min(duration, timeSeries[i+1]-timeSeries[i])
   }
   ret += duration
   return ret
}
```

python

```python
class Solution:
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        ret = 0
        for k in range(len(timeSeries) - 1):
            ret += min(duration, timeSeries[k + 1] - timeSeries[k])
        ret += duration
        return ret
```

#### 496. Next Greater Element I

```go
func nextGreaterElement(nums1 []int, nums2 []int) []int {  
   res := make([]int, len(nums1))  
   for i := range res {  
      res[i] = -1  
   }  
   mp := make(map[int]int)  
   for k, v := range nums1 {  
      mp[v] = k  
   }  
   var sk []int  
   for i, v := range nums2 {  
      for len(sk) > 0 && v > nums2[sk[len(sk)-1]] {  
         top := sk[len(sk)-1]  
         if idx, ok := mp[nums2[top]]; ok {  
            res[idx] = v  
         }  
         sk = sk[:len(sk)-1]  
      }  
      sk = append(sk, i)  
   }  
   return res  
}
```

#### 541. Reverse String II

go

```go
func reverseStr(s string, k int) string {
   ss := []byte(s)
   for i := 0; i < len(s); i += 2 * k {
      if i+k <= len(s) {
         reverse(ss[i : i+k])
      } else {
         reverse(ss[i:len(s)])
      }
   }
   return string(ss)
}
func reverse(s []byte) {
   left, right := 0, len(s)-1
   for left < right {
      s[left], s[right] = s[right], s[left]
      left++
      right--
   }
}
```


#### 556. Next Greater Element III

```go
func nextGreaterElement(n int) int {  
   b := []byte(strconv.Itoa(n))  
   i := len(b) - 2  
   for ; i >= 0; i-- {  
      if b[i] < b[i+1] {  
         break  
      }  
   }  
   if i == -1 {  
      return -1  
   }  
   j := len(b) - 1  
   for ; j > i; j-- {  
      if b[j] > b[i] {  
         break  
      }  
   }  
   b[i], b[j] = b[j], b[i]  
   reverse(b[i+1:])  
   res, _ := strconv.Atoi(string(b))  
   if res > math.MaxInt32 {  
      return -1  
   }  
   return res  
}  
func reverse(b []byte) {  
   left, right := 0, len(b)-1  
   for left < right {  
      b[left], b[right] = b[right], b[left]  
      left++  
      right--  
   }  
}
```


#### 575. Distribute Candies

python

```python
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(candyType)//2,len(set(candyType)))
```

#### 581. Shortest Unsorted Continuous Subarray

go

```go
func findUnsortedSubarray(nums []int) int {  
   n := len(nums)  
   maxValue, minValue := nums[0], nums[n-1]  
   begin, end := -1, -2  
   for i := 1; i < n; i++ {  
      if nums[i] < maxValue {  
         end = i  
      } else {  
         maxValue = nums[i]  
      }  
      if nums[n-1-i] > minValue {  
         begin = n - 1 - i  
      } else {  
         minValue = nums[n-1-i]  
      }  
   }  
   return end - begin + 1  
}
```

java

```java
class Solution {  
    public int findUnsortedSubarray(int[] nums) {  
        int begin = -1, end = -2, n = nums.length;  
        int max = nums[0], min = nums[n - 1];  
        for (int i = 1; i < n; i++) {  
            if (max > nums[i]) {  
                end = i;  
            } else {  
                max = nums[i];  
            }  
            if (min < nums[n - i - 1]) {  
                begin = n - i - 1;  
            } else {  
                min = nums[n - i - 1];  
            }  
        }  
        return end - begin + 1;  
    }  
}
```


#### 605. Can Place Flowers

https://leetcode.com/problems/can-place-flowers/discuss/103933/simplest-c%2B%2B-code

```go
func canPlaceFlowers(flowerbed []int, n int) bool {
   flowerbed = append([]int{0}, flowerbed...)
   flowerbed = append(flowerbed, 0)
   for i := 1; i < len(flowerbed)-1; i++ {
      if flowerbed[i-1]+flowerbed[i]+flowerbed[i+1] == 0 {
         i++
         n--
      }
   }
   return n <= 0
}
```


#### 676. Implement Magic Dictionary

go

```go
type MagicDictionary struct {  
   vec []string  
}  
  
func Constructor() MagicDictionary {  
   return MagicDictionary{}  
}  
  
func (m *MagicDictionary) BuildDict(dictionary []string) {  
   m.vec = dictionary  
}  
  
func (m *MagicDictionary) Search(word string) bool {  
   for _, w := range m.vec {  
      if len(w) != len(word) {  
         continue  
      }  
      diff := 0  
      for i := range word {  
         if w[i] != word[i] {  
            diff++  
         }  
      }  
      if diff == 1 {  
         return true  
      }  
   }  
   return false  
}
```

rust

```rust
#[derive(Default)]  
struct MagicDictionary {  
    words: Vec<String>,  
}  
  
impl MagicDictionary {  
    fn new() -> Self {  
        Default::default()  
    }  
  
    fn build_dict(&mut self, dictionary: Vec<String>) {  
        self.words = dictionary;  
    }  
  
    fn search(&self, word: String) -> bool {  
        for x in &self.words {  
            if x.len() != word.len() {  
                continue;  
            }  
            let diff = x.chars().into_iter().zip(word.chars().into_iter()).fold(0, |mut acc, (a, b)| {  
                if a != b {  
                    acc += 1;  
                }  
                acc  
            });  
            if diff == 1 {  
                return true;  
            }  
        }  
        false  
    }  
}
```



#### 674. Longest Continuous Increasing Subsequence

```go
func findLengthOfLCIS(nums []int) int {
	dp := make([]int, len(nums))
	res := 0
	for i := range dp {
		dp[i] = 1
		if i > 0 && nums[i] > nums[i-1] {
			dp[i] = dp[i-1] + 1
		}
		res = max(res, dp[i])
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

```python
class Solution:  
    def findLengthOfLCIS(self, nums: List[int]) -> int:  
        n = len(nums)  
        dp = [1] * n  
        res = 1  
        for i in range(1, len(dp)):  
            if nums[i] > nums[i - 1]:  
                dp[i] = dp[i - 1] + 1  
            res = max(res, dp[i])  
        return res
```

#### 678. Valid Parenthesis String

go

```go
func checkValidString(s string) bool {  
   needMin, needMax := 0, 0  
   for i := range s {  
      switch s[i] {  
      case '*':  
         needMax++  
         needMin = max(0, needMin-1)  
      case '(':  
         needMax++  
         needMin++  
      case ')':  
         needMax--  
         needMin = max(0, needMin-1)  
      }  
      if needMax < 0 {  
         return false  
      }  
   }  
   return needMin == 0  
}  
func max(a, b int) int {  
   return int(math.Max(float64(a), float64(b)))  
}
```

java

```java
class Solution {  
    public boolean checkValidString(String s) {  
        int needMin = 0, needMax = 0;  
        for (char c : s.toCharArray()) {  
            switch (c) {  
                case '(' -> {  
                    needMin++;//the number of unbalanced '(' that MUST be paired  
                    needMax++;//maximum number of unbalanced '(' that COULD be paired  
                }  
                case ')' -> {  
                    needMax--;  
                    needMin = Math.max(needMin - 1, 0);  
                }  
                case '*' -> {  
                    needMax++;  
                    needMin = Math.max(needMin - 1, 0);  
                }  
            }  
            if (needMax < 0) {  
                return false;  
            }  
        }  
        return needMin == 0;  
    }  
}
```

#### 724. Find Pivot Index

```go
func pivotIndex(nums []int) int {  
   left, right := 0, 0  
   for _, v := range nums {  
      right += v  
   }  
   for k, v := range nums {  
      right -= v  
      if left == right {  
         return k  
      }  
      left += v  
   }  
   return -1  
}
```

#### 738. Monotone Increasing Digits

```go
func monotoneIncreasingDigits(n int) int {
    num := strconv.Itoa(n)
    b := []byte(num)
    for i := len(b) - 2; i >= 0; i-- {
        if b[i] > b[i+1] {
            for j := i + 1; j < len(b); j++ {
                b[j] = '9'
            }
            b[i]--
        }
    }
    res, _ := strconv.Atoi(string(b))
    return res
}
```

#### 741. Cherry Pickup

In dp definition, no need to consider the position of first leg to reach and and the last leg start from has to be the same.

go

```go
func cherryPickup(grid [][]int) int {  
   n := len(grid)  
   dp := make([][]int, n)  
   for i := range dp {  
      dp[i] = make([]int, n)  
   }  
   dp[0][0] = grid[0][0]  
   for steps := 1; steps < 2*n-1; steps++ {  
      for i := n - 1; i >= 0; i-- {  
         for p := n - 1; p >= 0; p-- {  
            j, q := steps-i, steps-p  
            if out(grid, i, j, p, q) {  
               dp[i][p] = -1  
               continue  
            }  
            if i > 0 {  
               dp[i][p] = max(dp[i][p], dp[i-1][p])  
            }  
            if p > 0 {  
               dp[i][p] = max(dp[i][p], dp[i][p-1])  
            }  
            if i > 0 && p > 0 {  
               dp[i][p] = max(dp[i][p], dp[i-1][p-1])  
            }  
            if dp[i][p] >= 0 {  
               if i != p {  
                  dp[i][p] += grid[p][q]  
               }  
               dp[i][p] += grid[i][j]  
            }  
         }  
      }  
   }  
   return max(dp[n-1][n-1], 0)  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}  
func out(grid [][]int, i, j, p, q int) bool {  
   n := len(grid)  
   if j < 0 || q < 0 || j >= n || q >= n {  
      return true  
   }  
   return grid[i][j] < 0 || grid[p][q] < 0  
}
```

#### 746. Min Cost Climbing Stairs

```go
func minCostClimbingStairs(cost []int) int {
   dp := make([]int, len(cost))
   dp[0], dp[1] = cost[0], cost[1]
   for i := 2; i < len(dp); i++ {
      dp[i] = min(dp[i-1], dp[i-2]) + cost[i]
   }
   return min(dp[len(dp)-1], dp[len(dp)-2])
}
func min(a, b int) int {
   if a < b {
      return a
   }
   return b
}
```

#### 763. Partition Labels

```go
func partitionLabels(s string) []int {
   var res []int
   m := make(map[byte]int)
   for k := range s {
      m[s[k]] = k
   }
   i, last := 0, -1
   for k := range s {
      i = max(i, m[s[k]])
      if k == i {
         res = append(res, i-last)
         last = i
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

#### 787. Cheapest Flights Within K Stops

```go
func min(a, b int) int {
    if a > b {
        return b
    }
    return a
}
func findCheapestPrice(n int, flights [][]int, src int, dst int, k int) int {
    dp := make([][]int, k+2)
    for i := range dp {
        dp[i] = make([]int, n)
        for j := r
        ange dp[i] {
            dp[i][j] = math.MaxInt
        }
    }
    dp[0][src] = 0
    for t := 1; t < k+2; t++ {
        for _, flight := range flights {
            a, b, c := flight[0], flight[1], flight[2]
            if dp[t-1][a] < math.MaxInt {
                dp[t][b] = min(dp[t][b], dp[t-1][a]+c)
            }
        }
    }
    res := math.MaxInt
    for t := 1; t < k+2; t++ {
        res = min(res, dp[t][dst])
    }
    if res == math.MaxInt {
        return -1
    }
    return res
}
```

#### 858. Mirror Reflection

go

```go
func mirrorReflection(p int, q int) int {
   height := q
   for height%p != 0 {
      height += q
   }
   if height/p%2 == 0 {
      return 0
   }
   if height/q%2 == 0 {
      return 2
   }
   return 1
}
```

python

```python
class Solution:
    def mirrorReflection(self, p: int, q: int) -> int:
        h = q
        while h % p != 0:
            h += q
        if h // p % 2 == 0:
            return 0
        if h // q % 2 == 0:
            return 2
        return 1
```


#### 1005. Maximize Sum Of Array After K Negations

```go
func largestSumAfterKNegations(nums []int, k int) int {
   sort.Slice(nums, func(i, j int) bool {
      return math.Abs(float64(nums[i])) < math.Abs(float64(nums[j]))
   })
   for i := len(nums) - 1; k > 0 && i >= 0; i-- {
      if nums[i] < 0 {
         nums[i] = -nums[i]
         k--
      }
   }
   if k%2 == 1 {
      nums[0] = -nums[0]
   }
   sum := 0
   for _, v := range nums {
      sum += v
   }
   return sum
}
```

#### 1021. Remove Outermost Parentheses

go

```go
func removeOuterParentheses(s string) string {
	var res []byte
	opened := 0
	for i := range s {
		switch s[i] {
		case '(':
			if opened > 0 {
				res = append(res, s[i])
			}
			opened++
		case ')':
			if opened > 1 {
				res = append(res, s[i])
			}
			opened--
		}
	}
	return string(res)
}
```

java

```java
class Solution {
    public String removeOuterParentheses(String s) {
        StringBuilder sb = new StringBuilder();
        int opened = 0;
        for (char c : s.toCharArray()) {
            switch (c) {
                case '(' -> {
                    if (opened > 0) {
                        sb.append(c);
                    }
                    opened++;
                }
                case ')' -> {
                    if (opened > 1) {
                        sb.append(c);
                    }
                    opened--;
                }
            }
        }
        return sb.toString();
    }
}
```


#### 1047. Remove All Adjacent Duplicates In String

go

```go
func removeDuplicates(s string) string {
   b := []byte(s)
   slow, fast := 0, 0
   for ; fast < len(s); fast++ {
      b[slow] = b[fast]
      if slow > 0 && b[slow] == b[slow-1] {
         slow--
      } else {
         slow++
      }
   }
   b = b[:slow]
   return string(b)
}
```

go  栈

```go
func removeDuplicates(s string) string {
   t := []byte(s)
   stack := make([]byte, 0)
   for _, v := range t {
      if len(stack) > 0 && stack[len(stack)-1] == v {
         stack = stack[:len(stack)-1]
         continue
      }
      stack = append(stack, v)
   }
   return string(stack)
}
```



#### 1221. Split a String in Balanced Strings

```go
func balancedStringSplit(s string) int {  
   count, res := 0, 0  
   for i := range s {  
      if s[i] == 'L' {  
         count++  
      } else {  
         count--  
      }  
      if count == 0 {  
         res++  
      }  
   }  
   return res  
}
```


#### 1323. Maximum 69 Number

python

```python
class Solution:
    def maximum69Number(self, num: int) -> int:
        if num // 1000 == 6:
            num += 3000
        elif num % 1000 // 100 == 6:
            num += 300
        elif num % 100 // 10 == 6:
            num += 30
        elif num %10 == 6:
            num += 3
        return num
```

go

```go
func maximum69Number(num int) int {
   switch {
   case num/1000 == 6:
      num += 3000
   case num%1000/100 == 6:
      num += 300
   case num%100/10 == 6:
      num += 30
   case num%10 == 6:
      num += 3
   }
   return num
}
```

#### 43. Multiply Strings

go

```go
func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	m, n := len(num1), len(num2)
	arr := make([]int, m+n)
	for i := m - 1; i >= 0; i-- {
		for j := n - 1; j >= 0; j-- {
			sum := int(num1[i]-'0')*int(num2[j]-'0') + arr[i+j+1]
			arr[i+j+1] = sum % 10
			arr[i+j] += sum / 10
		}
	}
	if arr[0] == 0 {
		arr = arr[1:]
	}
	var res strings.Builder
	for _, digit := range arr {
		res.WriteByte(byte(digit + '0'))
	}
	return res.String()
}
```

java

```java
class Solution {
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) return "0";
        int m = num1.length(), n = num2.length();
        int[] arr = new int[n + m];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int sum = (num1.charAt(i) - '0') * (num2.charAt(j) - '0') + arr[i + j + 1];
                arr[i + j + 1] = sum % 10;
                arr[i + j] += sum / 10;
            }
        }
        StringBuilder res = new StringBuilder();
        for (int digit : arr) {
            if (res.isEmpty() && digit == 0) {
                continue;
            }
            res.append(digit);
        }
        return res.toString();
    }
}
```


#### 415. Add Strings

java

```java
class Solution {
    public String addStrings(String num1, String num2) {
        int i = num1.length() - 1, j = num2.length() - 1;
        int carry = 0;
        StringBuilder sb = new StringBuilder();
        while (i >= 0 || j >= 0 || carry > 0) {
            int n1 = 0, n2 = 0;
            if (i >= 0) {
                n1 = num1.charAt(i) - '0';
            }
            if (j >= 0) {
                n2 = num2.charAt(j) - '0';
            }
            carry += n1 + n2;
            sb.append(carry % 10);
            carry /= 10;
            i--;
            j--;
        }
        return sb.reverse().toString();
    }
}
```

go参考https://github.com/ganeshskudva/Leetcode-Golang

```go
func addStrings(num1 string, num2 string) string {
	i, j, carry := len(num1)-1, len(num2)-1, 0
	var sb []byte
	for i >= 0 || j >= 0 || carry > 0 {
		n1, n2 := 0, 0
		if i >= 0 {
			n1 = int(num1[i] - '0')
		}
		if j >= 0 {
			n2 = int(num2[j] - '0')
		}
		carry += n1 + n2
		sb = append(sb, byte(carry%10+'0'))
		carry /= 10
		i--
		j--
	}
	return string(reverse(sb))
}

func reverse(sb []byte) []byte {
	left, right := 0, len(sb)-1
	for left < right {
		sb[left], sb[right] = sb[right], sb[left]
		left++
		right--
	}
	return sb
}
```

python

```python
class Solution:  
    def addStrings(self, num1: str, num2: str) -> str:  
        pt1, pt2 = len(num1) - 1, len(num2) - 1  
        res, carry = '', 0  
        while pt1 >= 0 or pt2 >= 0 or carry:  
            n1 = n2 = 0  
            if pt1 >= 0:  
                n1 = int(num1[pt1])  
            if pt2 >= 0:  
                n2 = int(num2[pt2])  
            carry += n1 + n2  
            res += str(carry % 10)  
            carry //= 10  
            pt1 -= 1  
            pt2 -= 1  
        return res[::-1]
```



#### 950. Reveal Cards In Increasing Order

java

```java
class Solution {
    public int[] deckRevealedIncreasing(int[] deck) {
        Arrays.sort(deck);
        int n = deck.length;
        // the head of queue is the bottom of the deck
        LinkedList<Integer> queue = new LinkedList<>();
        for (int i = n - 1; i >= 0; i--) {
            if (!queue.isEmpty()) {
                queue.offer(queue.remove());
            }
            queue.add(deck[i]);
        }
        Collections.reverse(queue);
        return queue.stream().mapToInt(i -> i).toArray();
    }
}
```

