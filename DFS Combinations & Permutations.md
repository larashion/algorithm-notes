含有重复元素的排列组合问题，需要避免构造重复方案，比如只拿第一个后续全部跳过，而不是用hashset过滤，因为没必要构造出重复的答案

debug应当在函数入口打印递归参数的变化

**架构师注解：Go 语言的回溯写法选择** 在 Go 语言中，尽量避免使用 `dfs(append(path, val))` 这种**隐式回溯**写法。

**原因**：当 slice 容量（cap）满时，`append` 会触发扩容并分配新数组。在递归树中，如果父节点的 `path` 容量已满，后续的每一个子分支（Sibling）执行 `append` 时都会**各自**触发一次独立的扩容和内存拷贝。这会导致 $O(N^2)$ 级别的冗余内存分配。

**推荐**：使用**显式回溯**。预先分配足够容量 (`make([]int, 0, n)`)，并在递归前后手动 Push/Pop (`path = append(path, val)` ... `path = path[:len(path)-1]`)。这样可以确保所有递归分支复用同一个底层数组，实现零内存分配（Zero-Allocation）。

#### 17. Letter Combinations of a Phone Number

组合问题，每层递归仅处理一个数字即可

go 

```go
func letterCombinations(digits string) []string {
	n := len(digits)
	if n == 0 {
		return nil
	}
	dict := [10]string{
		"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz",
	}
	var res []string
	// 预分配容量，避免append扩容
	path := make([]byte, 0, n)

	var dfs func(index int)
	dfs = func(index int) {
		if index == n {
			res = append(res, string(path))
			return
		}
		letters := dict[digits[index]-'0']
		for i := 0; i < len(letters); i++ {
			path = append(path, letters[i]) // Push
			dfs(index + 1)
			path = path[:len(path)-1]       // Pop
		}
	}
	dfs(0)
	return res
}
```

rust

```rust
impl Solution {
	pub fn letter_combinations(digits: String) -> Vec<String> {
	    if digits.is_empty() {
	        return vec![];
	    }
	    let board: [&[u8]; 10] = [
	        b"", b"", b"abc", b"def", b"ghi", b"jkl", b"mno", b"pqrs", b"tuv", b"wxyz",
	    ];
	    let mut res: Vec<String> = vec![];
	    let digits = digits.as_bytes();
	    let mut path = Vec::with_capacity(digits.len());
	    Self::dfs(digits, &mut res, board, &mut path);
	    res
	}
	fn dfs(digits: &[u8], res: &mut Vec<String>, board: [&[u8]; 10], path: &mut Vec<u8>) {
	    match digits.first() {
	        Some(&x) => {
	            let num = (x - b'0') as usize;
	            for &c in board[num] {
	                path.push(c);
	                Self::dfs(&digits[1..], res, board, path);
	                path.pop();
	            }
	        }
	        None => {
	            res.push(String::from_utf8(path.to_owned()).unwrap());
	        }
	    }
	}
}
```

java 原地覆写

```java
public List<String> letterCombinations(String digits) {  
    List<String> res = new ArrayList<>();  
    if (digits == null || digits.isEmpty()) {  
        return res;  
    }  
    String[] digitsMap = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};  
    dfs(digits, 0, res, new char[digits.length()], digitsMap);  
    return res;  
}  
  
private void dfs(String digits, int ptr, List<String> res, char[] path, String[] digitsMap) {  
    if (ptr == digits.length()) {  
        res.add(new String(path));  
        return;  
    }  
    String letters = digitsMap[digits.charAt(ptr) - '0'];  
    for (char letter : letters.toCharArray()) {  
        path[ptr] = letter;  
        dfs(digits, ptr + 1, res, path, digitsMap);  
    }  
}
```

python

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits or digits == "":
            return []
        digits_map = [
            "",
            "",
            "abc",
            "def",
            "ghi",
            "jkl",
            "mno",
            "pqrs",
            "tuv",
            "wxyz",
        ]
        res = []
        path = []

        def dfs(ptr: int):
            if ptr == len(digits):
                res.append("".join(path))
                return
            digit = digits[ptr]
            chars = digits_map[int(digit)]
            for char in chars:
                path.append(char)
                dfs(ptr + 1)
                path.pop()

        dfs(0)
        return res
```

#### 22. Generate Parentheses

在系统设计中，我们通常更喜欢 “减法策略” (Count Down)。无状态感：减法策略不需要知道 n 是多少，只需要知道“还有没有剩”。这使得递归函数少传一个参数（不需要传 n，只需要比对 0），函数签名更干净。

go

```go
func generateParenthesis(n int) []string {
	var ans []string
	path := make([]byte, 0, n*2)

	var dfs func(left, right int)
	dfs = func(left, right int) {
		if left == 0 && right == 0 {
			ans = append(ans, string(path))
			return
		}
		if left > 0 {
			path = append(path, '(') // Push
			dfs(left-1, right)
			path = path[:len(path)-1] // Pop
		}
		if right > 0 && right > left {
			path = append(path, ')') // Push
			dfs(left, right-1)
			path = path[:len(path)-1] // Pop
		}
	}

	dfs(n, n)
	return ans
}
```

python

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []

        def dfs(left: int, right: int, path: List[str]):
            if left == 0 == right:
                res.append("".join(path))
                return
            if left > 0:
                path.append('(')
                dfs(left - 1, right, path)
                path.pop()
            if right > 0 and right > left:
                path.append(')')
                dfs(left, right - 1, path)
                path.pop()

        dfs(n, n, [])
        return res
```

java

```java
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        dfs(res, new char[n * 2], 0, n, n);
        return res;
    }

    private void dfs(List<String> res, char[] path, int cur, int left, int right) {
        if (left == 0 && right == 0) {
            res.add(new String(path));
            return;
        }
        if (left > 0) {
            path[cur] = '(';
            dfs(res, path, cur + 1, left - 1, right);
        }
        if (right > 0 && left < right) {
            path[cur] = ')';
            dfs(res, path, cur + 1, left, right - 1);
        }
    }
}
```

rust

```rust
impl Solution {
    pub fn generate_parenthesis(n: i32) -> Vec<String> {
        let mut res = vec![];
        let mut path = Vec::with_capacity((n as usize) << 1);
        Self::dfs(n, n, &mut res, &mut path);
        res
    }
    fn dfs(left: i32, right: i32, res: &mut Vec<String>, path: &mut Vec<u8>) {
        if right == 0 && left == 0 {
            res.push(String::from_utf8(path.to_owned()).unwrap());
        }
        if left > 0 {
            path.push(b'(');
            Self::dfs(left - 1, right, res, path);
            path.pop();
        }
        if right > 0 && right > left {
            path.push(b')');
            Self::dfs(left, right - 1, res, path);
            path.pop();
        }
    }
}
```

#### 39. Combination Sum

https://leetcode.com/problems/combination-sum/solutions/16510/Python-dfs-solution./

“完全背包” (Unbounded Knapsack) 变种问题， “无限制选取的回溯”。

python 意会

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res = []
        n = len(candidates)

        def dfs(candidates: List[int], path: List[int], target):
            if target == 0:
                res.append(path[:])
                return
            for i, v in enumerate(candidates):
                if v > target:
                    break
                dfs(candidates[i:], path + [v], target - v)

        dfs(candidates, [], target)
        return res
```

go

```go
func combinationSum(candidates []int, target int) [][]int {
	sort.Ints(candidates)
	var res [][]int
	// 预估最大深度，例如 target/min(candidates)
	path := make([]int, 0, target/candidates[0]+1) 
	
	var dfs func(start int, target int)
	dfs = func(start int, target int) {
		if target == 0 {
			// 必须拷贝path，因为底层数组会被后续修改
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, temp)
			return
		}

		for i := start; i < len(candidates); i++ {
			v := candidates[i]
			if v > target {
				break
			}
			path = append(path, v) // Push
			dfs(i, target-v)       // index i means we can reuse current candidate
			path = path[:len(path)-1] // Pop
		}
	}
	dfs(0, target)
	return res
}
```

python

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res = []
        path = []

        def dfs(start, target):
            if target == 0:
                res.append(path[:])
                return
            for i in range(start, len(candidates)):
                v = candidates[i]
                if v > target:
                    break
                path.append(v)
                dfs(i, target - v)
                path.pop()

        dfs(0, target)
        return res
```

java

```java
class Solution {
    public List<List<Integer>> combinationSum(int[] nums, int target) {
	    Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        dfs(res, nums, 0, new ArrayList<>(), target);
        return res;
    }

    private void dfs(List<List<Integer>> res, int[] nums, int start, List<Integer> path, int remain) {
        if (remain == 0) {
            res.add(new ArrayList<>(path));
        }
        for (int i = start; i < nums.length; i++) {
            if (nums[i] > remain) {
                break;
            }
            path.add(nums[i]);
            dfs(res, nums, i, path, remain - nums[i]);
            path.removeLast();
        }
    }
}
```

rust

```rust
impl Solution {
    pub fn combination_sum(mut nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let mut res = vec![];
        Self::dfs(&nums, &mut res, &mut vec![], target);
        res
    }
    fn dfs(nums: &[i32], res: &mut Vec<Vec<i32>>, path: &mut Vec<i32>, target: i32) {
        if target == 0 {
            res.push(path.to_owned());
            return;
        }
        for (i, &num) in nums.iter().enumerate() {
            if num > target {
                break;
            }
            path.push(num);
            Self::dfs(&nums[i..], res, path, target - num);
            path.pop();
        }
    }
}
```

#### 40. Combination Sum II

go

```go
func combinationSum2(nums []int, target int) [][]int {
	var res [][]int
	sort.Ints(nums)
	path := make([]int, 0, len(nums))

	var dfs func(start int, target int)
	dfs = func(start int, target int) {
		if target == 0 {
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for i := start; i < len(nums); i++ {
			if i > start && nums[i] == nums[i-1] {
				continue
			}
			if target < nums[i] {
				break
			}
			path = append(path, nums[i]) // Push
			dfs(i+1, target-nums[i])
			path = path[:len(path)-1]    // Pop
		}
	}
	dfs(0, target)
	return res
}
```

java

```java
class Solution {
    public List<List<Integer>> combinationSum2(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        dfs(res, nums, new ArrayList<>(), target, 0);
        return res;
    }

    void dfs(List<List<Integer>> res, int[] nums, ArrayList<Integer> path, int remain, int start) {

        if (remain == 0) {
            res.add(new ArrayList<>(path));
        }
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            if (nums[i] > remain) break;
            path.add(nums[i]);
            dfs(res, nums, path, remain - nums[i], i + 1);
            path.removeLast();
        }
    }
}
```

rust

```rust
impl Solution {
    pub fn combination_sum2(mut nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let mut res = vec![];
        Self::dfs(&nums, &mut res, &mut vec![], target);
        res
    }
    fn dfs(nums: &[i32], res: &mut Vec<Vec<i32>>, path: &mut Vec<i32>, target: i32) {
        if target == 0 {
            res.push(path.to_owned());
            return;
        }
        for (i, &num) in nums.iter().enumerate() {
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }
            if target < nums[i] {
                break;
            }
            path.push(num);
            Self::dfs(&nums[i + 1..], res, path, target - num);
            path.pop();
        }
    }
}
```

#### 46. Permutations

交换法 go

把数组分为两部分：`[已固定的 | 待考虑的]

把 `start` 位置的数，和后面 `start ~ n` 范围内的每一个数 **交换 (Swap)**。

交换后，认为 `start` 位置固定了，递归去处理 `start + 1`。

交换法 Go

```go
func permute(nums []int) [][]int {
	var res [][]int

	// In-place函数
	var dfs func(i int)
	dfs = func(i int) {
		if i == len(nums) {
			res = append(res, append([]int{}, nums...))
			return
		}
		for j := i; j < len(nums); j++ {
			nums[j], nums[i] = nums[i], nums[j]
			dfs(i + 1)
			nums[j], nums[i] = nums[i], nums[j]
		}
	}

	dfs(0)
	return res
}
```

交换法 python

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []

        def dfs(ptr):
            if ptr == len(nums):
                res.append(nums[:])
                return
            for i in range(ptr, len(nums)):
                nums[ptr], nums[i] = nums[i], nums[ptr]
                dfs(ptr + 1)
                nums[ptr], nums[i] = nums[i], nums[ptr]

        dfs(0)
        return res
```

Go

```go
func permute(nums []int) [][]int {
	n := len(nums)
	var res [][]int
	path := make([]int, 0, n)
	used := make([]bool, n)
	
	var dfs func()
	dfs = func() {
		if n == len(path) {
			temp := make([]int, n)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for i, num := range nums {
			if used[i] {
				continue
			}
			used[i] = true
			path = append(path, num) // Push
			dfs()
			path = path[:len(path)-1] // Pop
			used[i] = false
		}
	}

	dfs()
	return res
}
```

java

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        dfs(res, nums, new ArrayList<>(), new boolean[nums.length]);
        return res;
    }

    private void dfs(List<List<Integer>> res, int[] nums, ArrayList<Integer> path, boolean[] used) {
        int n = nums.length;
        if (path.size() == n) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < n; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            int num = nums[i];
            path.add(num);
            dfs(res, nums, path, used);
            path.removeLast();
            used[i] = false;
        }
    }
}
```

rust

```rust
impl Solution {
    pub fn permute(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = vec![];
        let n = nums.len();
        let mut path = Vec::with_capacity(n as usize);
        Self::dfs(&nums, n, &mut res, &mut vec![false; nums.len()], &mut path);
        res
    }
    fn dfs(
        nums: &[i32],
        n: usize,
        res: &mut Vec<Vec<i32>>,
        used: &mut [bool],
        path: &mut Vec<i32>,
    ) {
        if nums.len() == path.len() {
            res.push(path.to_owned());
            return;
        }
        for (i, &num) in nums.iter().enumerate() {
            if used[i] {
                continue;
            }
            used[i] = true;
            path.push(num);
            Self::dfs(nums, n, res, used, path);
            path.pop();
            used[i] = false;
        }
    }
}
```

#### 47. Permutations II

python 意会

```python
class Solution:  
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:  
        nums.sort()  
        res = []  
        self.dfs(nums, [], res)  
        return res  
  
    def dfs(self, nums, path, res):  
        if not nums:  
            res.append(path)  
            return  
        for i in range(len(nums)):  
            if i > 0 and nums[i] == nums[i - 1]:  
                continue  
            self.dfs(nums[:i] + nums[i + 1:], path + [nums[i]], res)
```

go

```go
func permuteUnique(nums []int) [][]int {
	sort.Ints(nums)
	var res [][]int
	n := len(nums)
	used := make([]bool, n)
	path := make([]int, 0, n)

	var dfs func()
	dfs = func() {
		if len(path) == n {
			temp := make([]int, n)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for i := range nums {
			if i > 0 && nums[i-1] == nums[i] && !used[i-1] || used[i] {
				continue
			}
			used[i] = true
			path = append(path, nums[i]) // Push
			dfs()
			path = path[:len(path)-1]    // Pop
			used[i] = false
		}
	}

	dfs()
	return res
}
```

python

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        n = len(nums)
        used = [False] * n
        path = []

        def dfs():
            if len(path) == n:
                res.append(path[:])
                return
            for i in range(n):
                if (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]) or used[i]:
                    continue
                used[i] = True
                path.append(nums[i])
                dfs()
                path.pop()
                used[i] = False

        dfs()
        return res
```

Java

```java
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        dfs(nums, 0, res, new ArrayList<>(), new boolean[nums.length]);
        return res;
    }

    void dfs(int[] nums, int ptr, List<List<Integer>> res, ArrayList<Integer> path, boolean[] used) {
        int n = nums.length;
        if (ptr == n) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < n; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1] || used[i]) {
                continue;
            }
            used[i] = true;
            path.add(nums[i]);
            dfs(nums, ptr + 1, res, path, used);
            path.removeLast();
            used[i] = false;
        }
    }
}
```

rust

```rust
impl Solution {
    pub fn permute_unique(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        nums.sort_unstable();
        let mut res = vec![];
        let n = nums.len();
        Self::dfs(&nums, n, &mut res, &mut vec![], &mut vec![false; n]);
        res
    }
    fn dfs(
        nums: &[i32],
        n: usize,
        res: &mut Vec<Vec<i32>>,
        path: &mut Vec<i32>,
        used: &mut [bool],
    ) {
        if path.len() == n {
            res.push(path.to_owned());
            return;
        }
        for (i, &num) in nums.iter().enumerate() {
            if i > 0 && nums[i] == nums[i - 1] && !used[i - 1] || used[i] {
                continue;
            }
            path.push(num);
            used[i] = true;
            Self::dfs(nums, n, res, path, used);
            used[i] = false;
            path.pop();
        }
    }
}
```

#### 77. Combinations

go

```go
func combine(n, k int) [][]int {
	var res [][]int
	path := make([]int, 0, k)

	var dfs func(start int)
	dfs = func(start int) {
		if len(path) == k {
			temp := make([]int, k)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		need := k - (len(path) + 1)
		for i := start; i < n+1-need; i++ {
			path = append(path, i) // Push
			dfs(i + 1)
			path = path[:len(path)-1] // Pop
		}
	}

	dfs(1)
	return res
}
```

python

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []

        def dfs(start: int, path: List[int]):
            if len(path) == k:
                res.append(path[:])
                return
            need = k - (len(path) + 1)
            for i in range(start, n + 1 - need):
                path.append(i)
                dfs(i + 1, path)
                path.pop()

        dfs(1, [])
        return res

```

rust

```rust
impl Solution {
    pub fn combine(n: i32, k: i32) -> Vec<Vec<i32>> {
        let mut res = vec![];
        let k = k as usize;
        let mut path = Vec::with_capacity(k);
        Self::dfs(1, n, k, &mut res, &mut path);
        res
    }
    fn dfs(start: i32, n: i32, k: usize, res: &mut Vec<Vec<i32>>, path: &mut Vec<i32>) {
        if path.len() == k {
            res.push(path.to_owned());
            return;
        }
        let need = k - (path.len() + 1);
        for i in start..n + 1 - need as i32 {
            path.push(i);
            Self::dfs(i + 1, n, k, res, path);
            path.pop();
        }
    }
}
```

java

```java
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        dfs(1, n, k, res, new ArrayList<>());
        return res;
    }

    void dfs(int start, int n, int k, List<List<Integer>> res, ArrayList<Integer> path) {
        if (path.size() == k) {
            res.add(new ArrayList<>(path));
            return;
        }
        int need = k - (path.size() + 1);
        for (int i = start; i < n + 1 - need; i++) {
            path.add(i);
            dfs(i + 1, n, k, res, path);
            path.removeLast();
        }
    }
}
```

#### 78. Subsets

go

```go
func subsets(nums []int) [][]int {
	var res [][]int
	path := make([]int, 0, len(nums))
	
	var dfs func(start int)
	dfs = func(start int) {
		// Snapshot
		temp := make([]int, len(path))
		copy(temp, path)
		res = append(res, temp)
		
		for i := start; i < len(nums); i++ {
			path = append(path, nums[i]) // Push
			dfs(i + 1)
			path = path[:len(path)-1]    // Pop
		}
	}
	dfs(0)
	return res
}
```

python

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []

        def dfs(start: int, path: List[int]):
            res.append(path[:])
            for i in range(start, len(nums)):
                path.append(nums[i])
                dfs(i + 1, path)
                path.pop()

        dfs(0, [])
        return res
```

java

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        dfs(0, res, nums, new ArrayList<>());
        return res;
    }

    private void dfs(int start, List<List<Integer>> res, int[] nums, ArrayList<Integer> path) {
        res.add(new ArrayList<>(path));
        int n = nums.length;
        for (int i = start; i < n; i++) {
            path.add(nums[i]);
            dfs(i + 1, res, nums, path);
            path.removeLast();
        }
    }
}
```

rust

```rust
impl Solution {  
    pub fn subsets(nums: Vec<i32>) -> Vec<Vec<i32>> {  
        let mut res = vec![];  
        Self::dfs(&nums, &mut res, &mut vec![]);  
        res  
    }  
    fn dfs(nums: &[i32], res: &mut Vec<Vec<i32>>, path: &mut Vec<i32>) {  
        res.push(path.to_owned());  
        for (i, &num) in nums.iter().enumerate() {  
            path.push(num);  
            Self::dfs(&nums[i + 1..], res, path);  
            path.pop();  
        }  
    }  
}
```

#### 90. Subsets II

go

```go
func subsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)
	var res [][]int
	path := make([]int, 0, len(nums))

	var dfs func(start int)
	dfs = func(start int) {
		temp := make([]int, len(path))
		copy(temp, path)
		res = append(res, temp)
		
		for i := start; i < len(nums); i++ {
			if i > start && nums[i] == nums[i-1] {
				continue
			}
			path = append(path, nums[i]) // Push
			dfs(i + 1)
			path = path[:len(path)-1]    // Pop
		}
	}
	dfs(0)
	return res
}
```

python

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []

        def dfs(curr: int, path: List[int]):
            res.append(path[:])
            for i in range(curr, len(nums)):
                if i > curr and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                dfs(i + 1, path)
                path.pop()

        dfs(0, [])
        return res
```

java

```java
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        dfs(res, nums, new ArrayList<>(), 0);
        return res;
    }

    void dfs(List<List<Integer>> res, int[] nums, ArrayList<Integer> path, int start) {
        res.add(new ArrayList<>(path));
        int n = nums.length;
        for (int i = start; i < n; i++) {
            if (i > start && nums[i - 1] == nums[i]) {
                continue;
            }
            path.add(nums[i]);
            dfs(res, nums, path, i + 1);
            path.removeLast();
        }
    }
}
```

rust

```rust
impl Solution {  
    pub fn subsets_with_dup(nums: Vec<i32>) -> Vec<Vec<i32>> {  
        let mut nums = nums;  
        nums.sort_unstable();  
        let mut res = vec![];  
        Self::dfs(&nums, &mut res, &mut vec![]);  
        res  
    }  
    fn dfs(nums: &[i32], res: &mut Vec<Vec<i32>>, path: &mut Vec<i32>) {  
        res.push(path.to_owned());  
        for (i, &num) in nums.iter().enumerate() {  
            if i > 0 && nums[i] == nums[i - 1] {  
                continue;  
            }  
            path.push(num);  
            Self::dfs(&nums[i + 1..], res, path);  
            path.pop();  
        }  
    }  
}
```

#### 216. Combination Sum III

go

```go
func combinationSum3(k int, n int) [][]int {
	var res [][]int
	path := make([]int, 0, k)

	var dfs func(start int, n int)
	dfs = func(start int, n int) {
		if len(path) == k && n == 0 {
			temp := make([]int, k)
			copy(temp, path)
			res = append(res, temp)
			return
		}
		for i := start; i < 10; i++ {
			if n < i {
				break
			}
			path = append(path, i) // Push
			dfs(i+1, n-i)
			path = path[:len(path)-1] // Pop
		}
	}

	dfs(1, n)
	return res
}
```

rust

```rust
impl Solution {
    pub fn combination_sum3(k: i32, n: i32) -> Vec<Vec<i32>> {
        let mut res = vec![];
        let mut path = Vec::with_capacity(k as usize);
        Self::dfs(1, k as usize, n, &mut res, &mut path);
        res
    }
    fn dfs(start: i32, k: usize, n: i32, res: &mut Vec<Vec<i32>>, path: &mut Vec<i32>) {
        if k == path.len() && n == 0 {
            res.push(path.to_owned());
            return;
        }
        for i in start..10 {
            if i > n {
                break;
            }
            path.push(i);
            Self::dfs(i + 1, k, n - i, res, path);
            path.pop();
        }
    }
}
```

java

```java
class Solution {
    List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        dfs(1, k, n, new ArrayList<>(), res);
        return res;
    }

    void dfs(int start, int k, int n, ArrayList<Integer> path, List<List<Integer>> res) {
        if (path.size() == k && n == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i < 10; i++) {
            if (i > n) {
                break;
            }
            path.add(i);
            dfs(i + 1, k, n - i, path, res);
            path.removeLast();
        }
    }
}
```

python

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []

        def dfs(start: int, path: List[int], n: int):
            if len(path) == k and n == 0:
                res.append(path[:])
                return
            for i in range(start, 10):
                if i > n:
                    break
                path.append(i)
                dfs(i + 1, path, n - i)
                path.pop()

        dfs(1, [], n)
        return res
```

#### [491. Non-decreasing Subsequences](https://leetcode.com/problems/non-decreasing-subsequences/)

go

```go
func findSubsequences(nums []int) [][]int {
	var res [][]int
	path := make([]int, 0, len(nums))
	
	var dfs func(start int)
	dfs = func(start int) {
		if len(path) > 1 {
			temp := make([]int, len(path))
			copy(temp, path)
			res = append(res, temp)
		}
		
		visited := make(map[int]bool)
		for i := start; i < len(nums); i++ {
			num := nums[i]
			if len(path) > 0 && num < path[len(path)-1] || visited[num] {
				continue
			}
			visited[num] = true
			path = append(path, num) // Push
			dfs(i + 1)
			path = path[:len(path)-1] // Pop
		}
	}
	dfs(0)
	return res
}
```

rust

```rust
use std::collections::HashSet;

impl Solution {
    pub fn find_subsequences(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = vec![];
        Self::dfs(&nums, &mut res, &mut vec![]);
        res
    }
    fn dfs(nums: &[i32], res: &mut Vec<Vec<i32>>, path: &mut Vec<i32>) {
        let n = path.len();
        if n > 1 {
            res.push(path.to_owned());
        }
        let mut used = HashSet::new();
        for (i, &num) in nums.iter().enumerate() {
            if !path.is_empty() && num < path[n - 1] || used.contains(&num) {
                continue;
            }
            used.insert(num);
            path.push(num);
            Self::dfs(&nums[i + 1..], res, path);
            path.pop();
        }
    }
}
```

java

```java
class Solution {
    List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        dfs(nums, new ArrayList<>(), res, 0);
        return res;
    }

    void dfs(int[] nums, ArrayList<Integer> path, List<List<Integer>> res, int start) {
        if (path.size() > 1) {
            res.add(new ArrayList<>(path));
        }
        HashSet<Integer> set = new HashSet<>();
        for (int i = start; i < nums.length; i++) {
            int v = nums[i];
            if (set.contains(v) || !path.isEmpty() && path.getLast() > v) {
                continue;
            }
            set.add(v);
            path.add(v);
            dfs(nums, path, res, i + 1);
            path.removeLast();
        }
    }
}
```

python

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = []

        def dfs(curr: int, path: List[int]):
            if len(path) > 1:
                res.append(path[:])
            visited = set()
            for i in range(curr, len(nums)):
                v = nums[i]
                if v in visited or path and path[-1] > v:
                    continue
                visited.add(v)
                path.append(v)
                dfs(i + 1, path)
                path.pop()

        dfs(0, [])
        return res
```