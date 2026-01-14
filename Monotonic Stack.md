#### [1019. Next Greater Node In Linked List](https://leetcode.com/problems/next-greater-node-in-linked-list/)

java

```java
class Solution {
    public int[] nextLargerNodes(ListNode head) {
        ArrayList<Integer> list = new ArrayList<>();
        for (ListNode curr = head; curr != null; curr = curr.next) {
            list.add(curr.val);
        }
        int n = list.size();
        int[] res = new int[n];
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && list.get(stack.peek()) < list.get(i)) {
                res[stack.pop()] = list.get(i);
            }
            stack.push(i);
        }
        return res;
    }
}
```

#### Interesting Triples

java

```java
public class Solution {
    public boolean getFunTuple(int[] nums) {
        int min1 = Integer.MAX_VALUE;
        int min2 = Integer.MAX_VALUE;
        for (int n : nums) {
            if (n <= min1) {
                min1 = n;
            } else if (n <= min2) {
                min2 = n;
            } else {
                return true;
            }
        }
        return false;
    }
}
```

#### 1475. Final Prices With a Special Discount in a Shop

java

won't ruin the input data

```java
class Solution {
    public int[] finalPrices(int[] prices) {
        int n = prices.length;
        int[] res = prices.clone();
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && prices[stack.peek()] >= prices[i]) {
                Integer prev = stack.pop();
                res[prev] = prices[prev] - prices[i];
            }
            stack.push(i);
        }
        return res;
    }
}
```

#### 901. Online Stock Span

java

```java
class StockSpanner {
    ArrayDeque<int[]> stack;

    public StockSpanner() {
        stack = new ArrayDeque<>();
    }

    public int next(int price) {
        int span = 1;
        while (!stack.isEmpty() && price >= stack.peek()[0]) {
            span += stack.pop()[1];
        }
        stack.push(new int[]{price, span});
        return span;
    }
}
```

#### Convert Expression to Reverse Polish Notation

go

```go
func ConvertToRPN(expression []string) []string {
	res := make([]string, 0)
	n := len(expression)
	stack := make([]string, n)
	top := 0
	priority := map[string]int{
		"*": 2, "/": 2,
		"+": 1, "-": 1,
		"(": 0,
	}
	for _, exp := range expression {
		switch exp {
		case "+", "-", "*", "/":
			for top > 0 && priority[stack[top-1]] >= priority[exp] {
				res = append(res, stack[top-1])
				top--
			}
			stack[top] = exp
			top++
		case "(":
			stack[top] = exp
			top++
		case ")":
			for top > 0 && stack[top-1] != "(" {
				res = append(res, stack[top-1])
				top--
			}
			top--
		default:
			res = append(res, exp)
		}
	}
	for top > 0 {
		res = append(res, stack[top-1])
		top--
	}
	return res
}
```

#### Delete Char

go

```go
func DeleteChar(s string, k int) string {
	var stack []byte
	toDelete := len(s) - k
	for i := 0; i < len(s); i++ {
		digit := s[i]
		for len(stack) > 0 && stack[len(stack)-1] > digit && toDelete > 0 {
			stack = stack[:len(stack)-1]
			toDelete--
		}
		stack = append(stack, digit)
	}
	for len(stack) > 0 && toDelete > 0 {
		stack = stack[:len(stack)-1]
		toDelete--
	}
	return string(stack)
}
```

#### Tall Building

go

```go
func TallBuilding(arr []int) []int {
	n := len(arr)
	dp := make([]int, n)
	for i := range dp {
		dp[i] = 1
	}
	process(dp, false, arr, n)
	process(dp, true, arr, n)
	return dp
}
func process(dp []int, reverse bool, arr []int, n int) {
	var stack []int
	switch reverse {
	case true:
		for i := n - 1; i >= 0; i-- {
			alter(&stack, dp, i, arr)
		}
	case false:
		for i := 0; i < n; i++ {
			alter(&stack, dp, i, arr)
		}
	}
}
func alter(stack *[]int, dp []int, i int, arr []int) {
	dp[i] += len(*stack)
	for len(*stack) > 0 && (*stack)[len(*stack)-1] <= arr[i] {
		*stack = (*stack)[:len(*stack)-1]
	}
	*stack = append(*stack, arr[i])
}
```

#### 42. Trapping Rain Water

必须找到两侧都高于自己的才能计算

java

```java
class Solution {
    public int trap(int[] heights) {
        int res = 0;
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        for (int j = 0; j < heights.length; j++) {
            while (!stack.isEmpty() && heights[j] > heights[stack.peek()]) {
                int pop = stack.pop();
                if (stack.isEmpty()) {
                    break;
                }
                int i = stack.peek();
                int lower = Math.min(heights[i], heights[j]);
                res += (lower - heights[pop]) * (j - i - 1);
            }
            stack.push(j);
        }
        return res;
    }
}
```

Go

```go
func trap(heights []int) int {
	var stack []int
	res := 0
	for i, height := range heights {
		for len(stack) > 0 && heights[stack[len(stack)-1]] < height {
			pop := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				break
			}
			prev := stack[len(stack)-1]
			lower := min(heights[prev], height)
			res += (lower - heights[pop]) * (i - prev - 1)
		}
		stack = append(stack, i)
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

#### 84.  Largest Rectangle in Histogram

经典单调栈 为什么弹栈时用的是>= 因为右侧区域是连通的，最终会算对

维护单调栈，栈内元素递增，每次比较栈顶元素（如果有）

如果栈顶元素更大，就pop掉，找到了栈顶元素右侧的矮子

如果此时依然有栈顶元素，说明栈顶元素更矮，说明找到了当前元素左侧的矮子

java

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        int res = 0, n = heights.length;
        // nearest index (exclusive)
        int[] left = new int[n];
        int[] right = new int[n];
        Arrays.fill(left, -1);
        Arrays.fill(right, n);

        ArrayDeque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && heights[stack.peek()] >= heights[i]) {
                right[stack.pop()] = i;
            }
            if (!stack.isEmpty()) left[i] = stack.peek();
            stack.push(i);
        }
        for (int i = 0; i < n; i++) {
            res = Math.max(res, (right[i] - left[i] - 1) * heights[i]);
        }
        return res;
    }
}
```

go

```go
func largestRectangleArea(heights []int) int {
	n := len(heights)
	left := make([]int, n)
	right := make([]int, n)
	for i := range heights {
		left[i] = -1
		right[i] = n
	}
	stack, top := make([]int, n), 0
	for i, height := range heights {
		for top > 0 && heights[stack[top-1]] >= height {
			right[stack[top-1]] = i
			top--
		}
		// stack[top-1] < height
		if top > 0 {
			left[i] = stack[top-1]
		}
		stack[top] = i
		top++
	}
	res := 0
	for i, height := range heights {
		res = max(res, height*(right[i]-left[i]-1))
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

python

```python
class Solution:  
    def largestRectangleArea(self, heights: List[int]) -> int:  
        n = len(heights)  
        left = [-1 for _ in heights]  
        right = [n for _ in heights]  
        stack, res = deque(), 0  
        for i, height in enumerate(heights):  
            while stack and heights[stack[-1]] >= height:  
                right[stack.pop()] = i  
            if stack:  
                left[i] = stack[-1]  
            stack.append(i)  
        for i, height in enumerate(heights):  
            res = max(res, height * (right[i] - left[i] - 1))  
        return res
```

#### 85. Maximal Rectangle

在只包含0、1的矩阵中找出只包含1的最大矩形

用一维滚动数组逐层计算：只要矩阵为0就更新为0，矩阵为1就累加1

算完一层之后调用84题的函数计算直方图中的最大矩形

go

```go
func maximalRectangle(matrix [][]byte) (res int) {  
   n := len(matrix[0])  
   dp := make([]int, n)  
   for i := range matrix {  
      for j := range dp {  
         if matrix[i][j] == '0' {  
            dp[j] = 0  
         } else {  
            dp[j]++  
         }  
      }  
      res = max(res, largestRectangleArea(dp))  
   }  
   return res  
}  
func largestRectangleArea(heights []int) int {
	n := len(heights)
	left := make([]int, n)
	right := make([]int, n)
	for i := range heights {
		left[i] = -1
		right[i] = n
	}
	stack, top := make([]int, n), 0
	for i, height := range heights {
		for top > 0 && heights[stack[top-1]] >= height {
			right[stack[top-1]] = i
			top--
		}
		// stack[top-1] < height
		if top > 0 {
			left[i] = stack[top-1]
		}
		stack[top] = i
		top++
	}
	res := 0
	for i, height := range heights {
		res = max(res, height*(right[i]-left[i]-1))
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

#### 316. Remove Duplicate Letters

如果栈顶元素的字典序更大，且栈顶元素在后面仍出现过，就pop掉

go

```go
func removeDuplicateLetters(s string) string {  
   last := [26]int{} // last occurred index for each char  
   for i := range s {  
      last[s[i]-'a'] = i  
   }  
   n := len(s)  
   stack := make([]byte, n)  
   top := 0  
   contains := [26]bool{} // chars that saved in the stack  
   for i := range s {  
      if contains[s[i]-'a'] {  
         continue  
      }  
      for top > 0 && stack[top-1] > s[i] && last[stack[top-1]-'a'] > i {  
         contains[stack[top-1]-'a'] = false  
         top--  
      }  
      stack[top] = s[i]  
      top++  
      contains[s[i]-'a'] = true  
   }  
   return string(stack[:top])  
}
```

rust

```rust
use std::collections::HashSet;

impl Solution {
    pub fn remove_duplicate_letters(s: String) -> String {
        let mut stack: Vec<char> = vec![];
        let last = s.as_bytes().iter().enumerate().fold([0; 26], |mut acc, (i, &u)| {
            acc[(u - b'a') as usize] = i;
            acc
        });
        let mut set = HashSet::new();
        for (i, x) in s.chars().enumerate() {
            if set.contains(&x) {
                continue;
            }
            loop {
                match stack.last() {
                    Some(&top) if top as u8 > x as u8 && last[(top as u8 - b'a') as usize] > i => {
                        stack.pop();
                        set.remove(&top);
                    }
                    _ => { break; }
                }
            }
            stack.push(x);
            set.insert(x);
        }
        stack.iter().collect()
    }
}
```

java

```java
class Solution {
    public String removeDuplicateLetters(String s) {
        int[] last = new int[26];
        for (int i = 0; i < s.length(); i++) {
            last[s.charAt(i) - 'a'] = i;
        }
        ArrayDeque<Character> stack = new ArrayDeque<>();
        HashSet<Character> set = new HashSet<>();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (set.contains(ch)) {
                continue;
            }
            while (!stack.isEmpty() && stack.peek() > ch && last[stack.peek() - 'a'] > i) {
                set.remove(stack.pop());
            }
            stack.push(ch);
            set.add(ch);
        }
        StringBuilder sb = new StringBuilder();
        stack.forEach(sb::append);
        return sb.reverse().toString();
    }
}
```

#### 321. Create Maximum Number

https://leetcode.com/problems/create-maximum-number/discuss/77285/Share-my-greedy-solution

```go
func maxNumber(nums1 []int, nums2 []int, k int) []int {
   res := make([]int, k)
   for i := 0; i < k+1; i++ {
      if i <= len(nums1) && k-i <= len(nums2) {
         candidate := merge(prep(nums1, i), prep(nums2, k-i), k)
         if greater(candidate, 0, res, 0) {
            res = candidate
         }
      }
   }
   return res
}
func prep(nums []int, k int) []int {
   drop := len(nums) - k
   stack := make([]int, 0, k)
   for _, v := range nums {
      for drop > 0 && len(stack) > 0 && stack[len(stack)-1] < v {
         stack = stack[:len(stack)-1]
         drop--
      }
      stack = append(stack, v)
   }
   return stack[:k]
}
func merge(nums1, nums2 []int, k int) []int {
   res := make([]int, k)
   for i, j, r := 0, 0, 0; r < k; r++ {
      if greater(nums1, i, nums2, j) {
         res[r] = nums1[i]
         i++
      } else {
         res[r] = nums2[j]
         j++
      }
   }
   return res
}
func greater(nums1 []int, i int, nums2 []int, j int) bool {
   for i < len(nums1) && j < len(nums2) &&
      nums1[i] == nums2[j] {
      i++
      j++
   }
   return j == len(nums2) || (i < len(nums1) && nums1[i] > nums2[j])
}
```

#### 402. Remove K Digits

java

```java
class Solution {
    public String removeKdigits(String num, int k) {
        int n = num.length();
        ArrayDeque<Character> deque = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            char digit = num.charAt(i);
            while (k > 0 && !deque.isEmpty() && deque.peekLast() > digit) {
                deque.pollLast();
                k--;
            }
            if (deque.isEmpty() && digit == '0') {
                continue;
            }
            deque.addLast(digit);
        }
        while (k > 0 && !deque.isEmpty()) {
            deque.pollLast();
            k--;
        }
        if (deque.isEmpty()) {
            return "0";
        }
        StringBuilder sb = new StringBuilder();
        deque.forEach(sb::append);
        return sb.toString();
    }
}
```

python

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        for digit in num:
            while k and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            if not stack and digit == '0':
                continue
            stack.append(digit)
        while k and stack:
            stack.pop()
            k -= 1
        if not stack:
            return '0'
        return "".join(stack)
```

go

```go
func removeKdigits(num string, k int) string {
	var stack []byte
	for i := 0; i < len(num); i++ {
		digit := num[i]
		for k > 0 && len(stack) > 0 && stack[len(stack)-1] > digit {
			stack = stack[:len(stack)-1]
			k--
		}
		if len(stack) == 0 && digit == '0' {
			continue
		}
		stack = append(stack, digit)
	}
	for k > 0 && len(stack) > 0 {
		stack = stack[:len(stack)-1]
		k--
	}
	if len(stack) == 0 {
		return "0"
	}
	return string(stack)
}
```

#### 456. 132 Pattern

go

```go
func find132pattern(nums []int) bool {  
   s3, n := math.MinInt, len(nums)  
   stack := make([]int, n)  
   top := 0  
   for i := n - 1; i >= 0; i-- {  
      if nums[i] < s3 {  
         return true  
      }  
      for top > 0 && nums[i] > stack[top-1] {  
         s3 = stack[top-1]  
         top--  
      }  
      stack[top] = nums[i]  
      top++  
   }  
   return false  
}
```

java

```java
class Solution {  
    public boolean find132pattern(int[] nums) {  
        int n = nums.length;  
        int s3 = Integer.MIN_VALUE;  
        ArrayDeque<Integer> stack = new ArrayDeque<>();  
        for (int i = n - 1; i >= 0; i--) {  
            if (nums[i] < s3) {  
                return true;  
            }  
            while (!stack.isEmpty() && nums[i] > stack.peek()) {  
                s3 = stack.pop();  
            }  
            stack.push(nums[i]);  
        }  
        return false;  
    }  
}
```

python

```python
class Solution:  
    def find132pattern(self, nums: List[int]) -> bool:  
        stack = []  
        s3, n = -10**9-1, len(nums)  
        for i in range(n - 1, -1, -1):  
            if s3 > nums[i]:  
                return True  
            while stack and nums[i] > stack[-1]:  
                s3 = stack.pop()  
            stack.append(nums[i])  
        return False
```

#### 503. Next Greater Element II

go

```go
func nextGreaterElements(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	for i := range res {
		res[i] = -1
	}
	stack := make([]int, 2*n)
	top := 0
	for i := 0; i < 2*n; i++ {
		for top > 0 && nums[stack[top-1]] < nums[i%n] {
			res[stack[top-1]] = nums[i%n]
			top--
		}
		if i < n {
			stack[top] = i % n
			top++
		}
	}
	return res
}
```

java

```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        int[] res = new int[n];
        Arrays.fill(res, -1);
        for (int i = 0; i < 2 * n; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i % n]) {
                res[stack.pop()] = nums[i % n];
            }
            if (i < n) stack.push(i % n);
        }
        return res;
    }
}
```

#### 739. Daily Temperatures

Go

```go
func dailyTemperatures(temperatures []int) []int {
	var stack []int
	n := len(temperatures)
	res := make([]int, n)
	for i, temperature := range temperatures {
		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperature {
			pop := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res[pop] = i - pop
		}
		stack = append(stack, i)
	}
	return res
}
```

java

```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] res = new int[n];
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < temperatures.length; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                Integer prev = stack.pop();
                res[prev] = i - prev;
            }
            stack.push(i);
        }
        return res;
    }
}
```

#### 907. Sum of Subarray Minimums

java

```java
class Solution {
    public int sumSubarrayMins(int[] arr) {
        int n = arr.length;
        int[] left = new int[n], right = new int[n];
        Arrays.fill(left, -1);
        Arrays.fill(right, n);
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && arr[stack.peek()] >= arr[i]) {
                right[stack.pop()] = i;
            }
            if (!stack.isEmpty()) {
                left[i] = stack.peek();
            }
            stack.push(i);
        }
        long res = 0, mod = (long) (1e9 + 7);
        for (int i = 0; i < n; i++) {
            res = (res + (long) arr[i] * (i - left[i]) % mod * (right[i] - i) % mod) % mod;
        }
        return (int) res;
    }
}
```

#### 1504. Count Submatrices With All Ones

在0、1的矩阵中找出只包含1的矩形的数量

主函数用一维滚动数组逐层计算：只要矩阵为0就更新为0，矩阵为1就累加1

把每一行转化为直方图，这就定死了底边

然后对每一列，定为右边界（包含），统计矩形数量，末了加起来就是一行的总量

已经定了两条边，矩形的数量就是某一顶点能够待的位置数量

找出左边的矮子，dp[ i ]在矮子的面积上累加

如果没有左边的矮子，那矩形的数量就是宽 X 高

go

```go
func numSubmat(mat [][]int) (res int) {
	n := len(mat[0])
	dp := make([]int, n)
	for i := range mat {
		for j := range dp {
			if mat[i][j] == 1 {
				dp[j]++
			} else {
				dp[j] = 0
			}
		}
		res += count(dp)
	}
	return res
}
func count(heights []int) int {
	n := len(heights)
	sum := make([]int, n)
	stack := make([]int, n)
	top := 0

	for i, height := range heights {
		for top > 0 && heights[stack[top-1]] >= height {
			top--
		}
		if top > 0 {
			prev := stack[top-1]
			sum[i] = sum[prev] + height*(i-prev)
		} else {
			sum[i] = height * (i + 1)
		}
		stack[top] = i
		top++
	}

	res := 0
	for _, s := range sum {
		res += s
	}
	return res
}
```

rust

```rust
impl Solution {
    pub fn num_submat(mat: Vec<Vec<i32>>) -> i32 {
        let n = mat[0].len();
        let mut heights = vec![0; n];
        mat.iter().fold(0, |acc, row| {
            for (j, &v) in row.iter().enumerate() {
                if v == 1 {
                    heights[j] += 1;
                } else {
                    heights[j] = 0;
                }
            }
            acc + Solution::count(&heights)
        })
    }
    fn count(heights: &Vec<i32>) -> i32 {
        let n = heights.len();
        let mut stack = vec![];
        heights.iter().enumerate().fold(vec![0; n], |mut acc, (i, &height)| {
            loop {
                match stack.last() {
                    Some(&top) if heights[top] >= height => { stack.pop(); }
                    _ => break
                }
            }
            match stack.last() {
                Some(&prev) => acc[i] = acc[prev] + height * (i - prev) as i32,
                _ => acc[i] = height * (i + 1) as i32
            }
            stack.push(i);
            acc
        }).iter().sum()
    }
}
```

python

```python
class Solution:  
    def numSubmat(self, mat: List[List[int]]) -> int:  
        m, n = len(mat), len(mat[0])  
        heights = [0] * n  
        res = 0  
        for i in range(m):  
            for j in range(n):  
                if mat[i][j] == 1:  
                    heights[j] += 1  
                else:  
                    heights[j] = 0  
            res += self.count(heights)  
        return res  
  
    def count(self, heights: [List[int]]) -> int:  
        n = len(heights)  
        stack = deque()  
        res_arr = [0] * n  
        for i, height in enumerate(heights):  
            while stack and heights[stack[-1]] >= height:  
                stack.pop()  
            if stack:  
                prev = stack[-1]  
                res_arr[i] = res_arr[prev] + height * (i - prev)  
            else:  
                # column "i" as the right border.  
                res_arr[i] = height * (i + 1)  
            stack.append(i)  
        return sum(res_arr)
```

