#### 93. Restore IP Addresses

go

```go
func restoreIpAddresses(s string) []string {
	var res []string
	var path []string
	var dfs func(int)
	dfs = func(start int) {
		if len(path) == 4 {
			if start == len(s) {
				res = append(res, strings.Join(path, "."))
			}
			return
		}
		// Pruning: remainChars must be in [remainParts, remainParts * 3]
		remainParts := 4 - len(path)
		remainChars := len(s) - start
		if remainChars < remainParts || remainChars > remainParts*3 {
			return
		}

		for i := start; i < start+3 && i < len(s); i++ {
			sub := s[start : i+1]
			if !isValid(sub) {
				continue
			}
			path = append(path, sub)
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}
	dfs(0)
	return res
}

func isValid(s string) bool {
	if len(s) > 1 && s[0] == '0' {
		return false
	}
	v, _ := strconv.Atoi(s)
	return v >= 0 && v <= 255
}
```

java

```java
class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        dfs(s, 0, new ArrayList<>(), res);
        return res;
    }

    private void dfs(String s, int start, List<String> path, List<String> res) {
        if (path.size() == 4) {
            if (start == s.length()) res.add(String.join(".", path));
            return;
        }
        // Pruning
        int remainParts = 4 - path.size();
        int remainChars = s.length() - start;
        if (remainChars < remainParts || remainChars > remainParts * 3) return;

        for (int i = start; i < start + 3 && i < s.length(); i++) {
            String sub = s.substring(start, i + 1);
            if (!isValid(sub)) continue;
            path.add(sub);
            dfs(s, i + 1, path, res);
            path.remove(path.size() - 1);
        }
    }

    private boolean isValid(String s) {
        if (s.length() > 1 && s.charAt(0) == '0') return false;
        int v = Integer.parseInt(s);
        return v >= 0 && v <= 255;
    }
}
```

python

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        res = []
        self.dfs(s, 0, [], res)
        return res

    def dfs(self, s: str, start: int, path: List[str], res: List[str]):
        if len(path) == 4:
            if start == len(s):
                res.append(".".join(path))
            return
        
        # Pruning
        remain_parts = 4 - len(path)
        remain_chars = len(s) - start
        if not (remain_parts <= remain_chars <= remain_parts * 3):
            return

        for i in range(start, min(start + 3, len(s))):
            sub = s[start : i + 1]
            if self.is_valid(sub):
                path.append(sub)
                self.dfs(s, i + 1, path, res)
                path.pop()

    def is_valid(self, s: str) -> bool:
        if len(s) > 1 and s[0] == '0':
            return False
        return 0 <= int(s) <= 255
```

rust

```rust
impl Solution {
    pub fn restore_ip_addresses(s: String) -> Vec<String> {
        let mut res = vec![];
        let mut path = vec![];
        Self::dfs(&s, 0, &mut path, &mut res);
        res
    }

    fn dfs(s: &str, start: usize, path: &mut Vec<String>, res: &mut Vec<String>) {
        if path.len() == 4 {
            if start == s.len() {
                res.push(path.join("."));
            }
            return;
        }
        
        // Pruning
        let remain_parts = 4 - path.len();
        let remain_chars = s.len() - start;
        if remain_chars < remain_parts || remain_chars > remain_parts * 3 {
            return;
        }

        for len in 1..=3 {
            if start + len > s.len() {
                break;
            }
            let sub = &s[start..start + len];
            if Self::is_valid(sub) {
                path.push(sub.to_string());
                Self::dfs(s, start + len, path, res);
                path.pop();
            }
        }
    }

    fn is_valid(s: &str) -> bool {
        if s.len() > 1 && s.starts_with('0') {
            return false;
        }
        s.parse::<u8>().is_ok()
    }
}
```

#### 131. Palindrome Partitioning

go

```go
func partition(s string) [][]string {
	var res [][]string
	dfs(s, 0, []string{}, &res)
	return res
}
func dfs(s string, start int, path []string, res *[][]string) {
	if start == len(s) {
		cp := make([]string, len(path))
		copy(cp, path)
		*res = append(*res, cp)
		return
	}
	for i := start; i < len(s); i++ {
		if !isPalindrome(s, start, i) {
			continue
		}
		path = append(path, s[start:i+1])
		dfs(s, i+1, path, res)
		path = path[:len(path)-1]
	}
}
func isPalindrome(s string, left, right int) bool {
	for left < right {
		if s[left] != s[right] {
			return false
		}
		left++
		right--
	}
	return true
}
```

java

```java
class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        dfs(res, s, new ArrayList<>(), 0);
        return res;
    }

    private void dfs(List<List<String>> res, String s, List<String> path, int start) {
        if (start == s.length()) {
            res.add(new ArrayList<>(path));
        }
        for (int i = start; i < s.length(); i++) {
            if (!isPalindrome(s, start, i)) {
                continue;
            }
            path.add(s.substring(start, i + 1));
            dfs(res, s, path, i+1);
            path.remove(path.size() - 1);
        }
    }

    private boolean isPalindrome(String s, int low, int high) {
        while (low < high) {
            if (s.charAt(low++) != s.charAt(high--)) {
                return false;
            }
        }
        return true;
    }
}
```

python

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        self.dfs(s, 0, [], res)
        return res

    def dfs(self, s: str, start: int, path: List[str], res: List[List[str]]):
        if start == len(s):
            res.append(path[:])
            return
        for i in range(start, len(s)):
            if not self.is_palindrome(s, start, i):
                continue
            path.append(s[start : i + 1])
            self.dfs(s, i + 1, path, res)
            path.pop()

    def is_palindrome(self, s: str, left: int, right: int) -> bool:
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
```

rust

```rust
impl Solution {
    pub fn partition(s: String) -> Vec<Vec<String>> {
        let mut res = vec![];
        let mut path = vec![];
        Self::dfs(&s, 0, &mut path, &mut res);
        res
    }

    fn dfs(s: &str, start: usize, path: &mut Vec<String>, res: &mut Vec<Vec<String>>) {
        if start == s.len() {
            res.push(path.clone());
            return;
        }
        for i in start..s.len() {
            // Slicing bytes view is faster as it avoids redundant UTF-8 boundary checks
            if Self::is_palindrome(&s.as_bytes()[start..i + 1]) {
                path.push(s[start..i + 1].to_string());
                Self::dfs(s, i + 1, path, res);
                path.pop();
            }
        }
    }

    fn is_palindrome(s: &[u8]) -> bool {
        let (mut l, mut r) = (0, s.len() - 1);
        while l < r {
            if s[l] != s[r] {
                return false;
            }
            l += 1;
            r -= 1;
        }
        true
    }
}
```

#### 140. Word Break II

go

```go
func wordBreak(s string, wordDict []string) []string {
	dict := make(map[string]bool)
	for _, w := range wordDict {
		dict[w] = true
	}
	memo := map[int][]string{
		len(s): {""},
	}
	
	var dfs func(int) []string
	dfs = func(i int) []string {
		if res, ok := memo[i]; ok {
			return res
		}
		var res []string
		for j := i + 1; j < len(s)+1; j++ {
			word := s[i:j]
			if !dict[word] {
				continue
			}
			tails := dfs(j)
			for _, tail := range tails {
				if tail == "" {
					res = append(res, word)
				} else {
					res = append(res, word+" "+tail)
				}
			}
		}
		memo[i] = res
		return res
	}
	
	return dfs(0)
}
```

rust

```rust
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

impl Solution {
    pub fn word_break(s: String, word_dict: Vec<String>) -> Vec<String> {
        let set: HashSet<String> = word_dict.into_iter().collect();
        // 优化点 1: Value 类型变为 Rc<Vec<String>>，实现共享所有权
        let mut memo: HashMap<usize, Rc<Vec<String>>> = HashMap::new();
        memo.insert(s.len(), Rc::new(vec!["".to_string()]));
        let result_rc = Self::dfs(&s, 0, &set, &mut memo);
        drop(memo);
        Rc::try_unwrap(result_rc).unwrap_or_else(|rc| (*rc).clone())
    }

    // 返回值改为 Rc<Vec<String>>
    fn dfs(
        s: &str,
        i: usize,
        set: &HashSet<String>,
        memo: &mut HashMap<usize, Rc<Vec<String>>>,
    ) -> Rc<Vec<String>> {
        if let Some(v) = memo.get(&i) {
            // 优化点 2: 这里的 clone 只是增加引用计数，开销极低
            return Rc::clone(v);
        }
        let mut result = vec![];
        for j in i + 1..=s.len() {
            let word = &s[i..j];
            if !set.contains(word) {
                continue;
            }
            let sub_results = Self::dfs(s, j, set, memo);
            // 这里依然需要遍历子结果进行拼接，这是算法逻辑决定的
            for sub_result in sub_results.iter() {
                if sub_result.is_empty() {
                    result.push(word.to_string());
                } else {
                    result.push(format!("{} {}", word, sub_result));
                }
            }
        }
        // 优化点 3: 存入缓存前，包装进 Rc
        let res_rc = Rc::new(result);
        memo.insert(i, Rc::clone(&res_rc));
        res_rc
    }
}
```

java

```java
class Solution {
    List<String> wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<>(wordDict);
        HashMap<Integer, List<String>> memo = new HashMap<>();
        memo.put(s.length(), new ArrayList<>(List.of("")));
        return dfs(s, 0, set, memo);
    }

    private List<String> dfs(String s, int i, HashSet<String> set, HashMap<Integer, List<String>> memo) {
        if (memo.containsKey(i)) return memo.get(i);
        List<String> res = new ArrayList<>();
        for (int j = i + 1; j < s.length() + 1; j++) {
            String word = s.substring(i, j);
            if (!set.contains(word)) {
                continue;
            }
            List<String> subResults = dfs(s, j, set, memo);
            for (String subResult : subResults) {
                if (subResult.isEmpty()) {
                    res.add(word);
                } else {
                    res.add(word + " " + subResult);
                }
            }
        }
        memo.put(i, res);
        return res;
    }
}
```

python

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:

        word_dict = set(wordDict)
        memo = {len(s): [""]}

        def dfs(i: int) -> List[str]:
            if i in memo:
                return memo[i]
            res = []
            for j in range(i + 1, len(s) + 1):
                word = s[i:j]
                if word not in word_dict:
                    continue
                for sub_result in dfs(j):
                    if not sub_result:
                        res.append(word)
                    else:
                        res.append(word + " " + sub_result)
            memo[i] = res
            return res

        return dfs(0)
```

#### 241. Different Ways to Add Parentheses

go

```go
func diffWaysToCompute(expression string) []int {
	memo := make(map[string][]int)
	return dfs(expression, memo)
}
func dfs(expression string, memo map[string][]int) []int {
	if res, ok := memo[expression]; ok {
		return res
	}
	var res []int
	for i := 0; i < len(expression); i++ {
		c := expression[i]
		if c >= '0' && c <= '9' {
			continue
		}
		part1 := dfs(expression[:i], memo)
		part2 := dfs(expression[i+1:], memo)
		for _, p1 := range part1 {
			for _, p2 := range part2 {
				switch c {
				case '+':
					res = append(res, p1+p2)
				case '-':
					res = append(res, p1-p2)
				case '*':
					res = append(res, p1*p2)
				}
			}
		}
	}
	if len(res) == 0 {
		conv, _ := strconv.Atoi(expression)
		res = append(res, conv)
	}
	memo[expression] = res
	return res
}
```

java

```java
class Solution {
    public List<Integer> diffWaysToCompute(String expression) {
        return dfs(expression, new HashMap<>());
    }

    private List<Integer> dfs(String expression, Map<String, List<Integer>> memo) {
        if (memo.containsKey(expression)) {
            return memo.get(expression);
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < expression.length(); i++) {
            char c = expression.charAt(i);
            if (c >= '0' && c <= '9') {
                continue;
            }
            List<Integer> part1 = dfs(expression.substring(0, i), memo);
            List<Integer> part2 = dfs(expression.substring(i + 1), memo);
            for (Integer p1 : part1) {
                for (Integer p2 : part2) {
                    res.add(switch (c) {
                        case '+' -> p1 + p2;
                        case '-' -> p1 - p2;
                        case '*' -> p1 * p2;
                        default -> throw new IllegalArgumentException();
                    });
                }
            }
        }
        if (res.isEmpty()) {
            res.add(Integer.parseInt(expression));
        }
        memo.put(expression, res);
        return res;
    }
}
```

python

```python
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        memo = {}

        def dfs(expr: str) -> List[int]:
            if expr in memo:
                return memo[expr]
            res = []
            for i, char in enumerate(expr):
                if char in "+-*":
                    part1 = dfs(expr[:i])
                    part2 = dfs(expr[i + 1 :])
                    for p1 in part1:
                        for p2 in part2:
                            if char == '+':
                                res.append(p1 + p2)
                            elif char == '-':
                                res.append(p1 - p2)
                            elif char == '*':
                                res.append(p1 * p2)
            if not res:
                res.append(int(expr))
            memo[expr] = res
            return res

        return dfs(expression)
```

rust

```rust
use std::collections::HashMap;
use std::rc::Rc;

impl Solution {
    pub fn diff_ways_to_compute(expression: String) -> Vec<i32> {
        let mut memo = HashMap::new();
        let res = Self::dfs(&expression, &mut memo);
        drop(memo); // Drop memo to ensure the Rc count becomes 1
        Rc::try_unwrap(res).unwrap_or_else(|v| (*v).clone())
    }
    
    fn dfs<'a>(expression: &'a str, memo: &mut HashMap<&'a str, Rc<Vec<i32>>>) -> Rc<Vec<i32>> {
        if let Some(res) = memo.get(expression) {
            return Rc::clone(res);
        }
        let mut res = vec![];
        for (i, c) in expression.chars().enumerate() {
            if c.is_numeric() {
                continue;
            }
            let part1 = Self::dfs(&expression[..i], memo);
            let part2 = Self::dfs(&expression[i + 1..], memo);
            for p1 in part1.iter() {
                for p2 in part2.iter() {
                    match c {
                        '+' => res.push(p1 + p2),
                        '-' => res.push(p1 - p2),
                        '*' => res.push(p1 * p2),
                        _ => panic!(),
                    }
                }
            }
        }
        if res.is_empty() {
            let conv = expression.parse::<i32>().unwrap();
            res.push(conv);
        }
        let res_rc = Rc::new(res);
        memo.insert(expression, Rc::clone(&res_rc));
        res_rc
    }
}
```

#### 282. Expression Add Operators

go

```go
func addOperators(s string, target int) []string {
	var results []string
	dfs(s, target, 0, []byte{}, 0, 0, &results)
	return results
}

func dfs(s string, target int, pos int, path []byte, curr, prev int, results *[]string) {
	if pos == len(s) {
		if curr == target {
			*results = append(*results, string(path))
		}
		return
	}
	for i := pos; i < len(s); i++ {
		if i > pos && s[pos] == '0' {
			break
		}
		numStr := s[pos : i+1]
		num, _ := strconv.Atoi(numStr)
		lenBeforeOp := len(path)
		if pos == 0 {
			path = append(path, numStr...)
			dfs(s, target, i+1, path, num, num, results)
			path = path[:lenBeforeOp]
		} else {
			// +
			path = append(path, '+')
			path = append(path, numStr...)
			dfs(s, target, i+1, path, curr+num, num, results)
			path = path[:lenBeforeOp]

			// -
			path = append(path, '-')
			path = append(path, numStr...)
			dfs(s, target, i+1, path, curr-num, -num, results)
			path = path[:lenBeforeOp]

			// *
			path = append(path, '*')
			path = append(path, numStr...)
			dfs(s, target, i+1, path, curr-prev+prev*num, prev*num, results)
			path = path[:lenBeforeOp]
		}
	}
}
```

python

```python
class Solution:
    def addOperators(self, s: str, target: int) -> List[str]:
        res = []
        self.dfs(s, target, 0, [], 0, 0, res)
        return res

    def dfs(self, s: str, target: int, pos: int, path: List[str], curr: int, prev: int, res: List[str]):
        if pos == len(s):
            if curr == target:
                res.append("".join(path))
            return
        for i in range(pos, len(s)):
            if i > pos and s[pos] == '0':
                break
            val_str = s[pos : i + 1]
            num = int(val_str)
            if pos == 0:
                path.append(val_str)
                self.dfs(s, target, i + 1, path, num, num, res)
                path.pop()
            else:
                path.append('+')
                path.append(val_str)
                self.dfs(s, target, i + 1, path, curr + num, num, res)
                path.pop()
                path.pop()
                
                path.append('-')
                path.append(val_str)
                self.dfs(s, target, i + 1, path, curr - num, -num, res)
                path.pop()
                path.pop()
                
                path.append('*')
                path.append(val_str)
                self.dfs(s, target, i + 1, path, curr - prev + prev * num, prev * num, res)
                path.pop()
                path.pop()
```

rust

```rust
impl Solution {
    pub fn add_operators(s: String, target: i32) -> Vec<String> {
        let mut res = vec![];
        let num_str: Vec<char> = s.chars().collect();
        Self::dfs(&num_str, target as i64, 0, String::new(), 0, 0, &mut res);
        res
    }
    fn dfs(s: &[char], target: i64, pos: usize, path: String, curr: i64, prev: i64, res: &mut Vec<String>) {
        if pos == s.len() {
            if curr == target {
                res.push(path);
            }
            return;
        }
        for i in pos..s.len() {
            if i > pos && s[pos] == '0' { break; }
            let val_str: String = s[pos..=i].iter().collect();
            let val: i64 = val_str.parse().unwrap();
            
            if pos == 0 {
                Self::dfs(s, target, i + 1, val_str, val, val, res);
            } else {
                Self::dfs(s, target, i + 1, format!("{}+{}", path, val), curr + val, val, res);
                Self::dfs(s, target, i + 1, format!("{}-{}", path, val), curr - val, -val, res);
                Self::dfs(s, target, i + 1, format!("{}*{}", path, val), curr - prev + prev * val, prev * val, res);
            }
        }
    }
}
```

#### 301. Remove Invalid Parentheses

go

```go
func removeInvalidParentheses(s string) []string {
	var res []string
	dfs(s, &res, 0, 0, [2]byte{'(', ')'})
	return res
}
func dfs(s string, res *[]string, iStart, jStart int, pair [2]byte) {
	stack := 0
	for i := iStart; i < len(s); i++ {
		if s[i] == pair[0] {
			stack++
		}
		if s[i] == pair[1] {
			stack--
		}
		if stack >= 0 {
			continue
		}
		for j := jStart; j <= i; j++ {
			if s[j] == pair[1] && (j == jStart || s[j-1] != pair[1]) {
				dfs(s[:j]+s[j+1:], res, i, j, pair)
			}
		}
		return
	}
	reversed := reverse(s)
	if pair[0] == '(' {
		dfs(reversed, res, 0, 0, [2]byte{')', '('})
	} else {
		*res = append(*res, reversed)
	}
}
func reverse(s string) string {
	bt := []byte(s)
	left, right := 0, len(bt)-1
	for left < right {
		bt[left], bt[right] = bt[right], bt[left]
		left++
		right--
	}
	return string(bt)
}
```

java

```java
class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        dfs(s, res, 0, 0, new char[]{'(', ')'});
        return res;
    }

    private void dfs(String s, List<String> res, int iStart, int jStart, char[] pair) {
        for (int stack = 0, i = iStart; i < s.length(); i++) {
            if (s.charAt(i) == pair[0]) {
                stack++;
            }
            if (s.charAt(i) == pair[1]) {
                stack--;
            }
            if (stack >= 0) {
                continue;
            }
            for (int j = jStart; j <= i; j++) {
                if (s.charAt(j) == pair[1] && (j == jStart || s.charAt(j - 1) != pair[1])) {
                    dfs(s.substring(0, j) + s.substring(j + 1), res, i, j, pair);
                }
            }
            return;
        }
        String reversed = new StringBuilder(s).reverse().toString();
        if (pair[0] == '(') {
            dfs(reversed, res, 0, 0, new char[]{')', '('});
        } else {
            res.add(reversed);
        }
    }
}
```

python

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        res = []
        self.dfs(s, res, 0, 0, ['(', ')'])
        return res

    def dfs(self, s: str, res: List[str], i_start: int, j_start: int, pair: List[str]):
        stack = 0
        for i in range(i_start, len(s)):
            if s[i] == pair[0]:
                stack += 1
            if s[i] == pair[1]:
                stack -= 1
            if stack >= 0:
                continue
            for j in range(j_start, i + 1):
                if s[j] == pair[1] and (j == j_start or s[j - 1] != pair[1]):
                    self.dfs(s[:j] + s[j + 1:], res, i, j, pair)
            return
        if pair[0] == '(':
            self.dfs(s[::-1], res, 0, 0, pair[::-1])
        else:
            res.append(s[::-1])
```

rust

```rust
impl Solution {
    pub fn remove_invalid_parentheses(s: String) -> Vec<String> {
        let mut res = vec![];
        let chars: Vec<char> = s.chars().collect();
        Self::dfs(&chars, &mut res, 0, 0, &['(', ')']);
        res
    }
    
    fn dfs(s: &[char], res: &mut Vec<String>, i_start: usize, j_start: usize, pair: &[char]) {
        let mut stack = 0;
        for i in i_start..s.len() {
            if s[i] == pair[0] { stack += 1; }
            if s[i] == pair[1] { stack -= 1; }
            if stack >= 0 { continue; }
            
            for j in j_start..=i {
                if s[j] == pair[1] && (j == j_start || s[j-1] != pair[1]) {
                    let mut new_s = s.to_vec();
                    new_s.remove(j);
                    Self::dfs(&new_s, res, i, j, pair);
                }
            }
            return;
        }
        
        let reversed: Vec<char> = s.iter().rev().cloned().collect();
        if pair[0] == '(' {
            Self::dfs(&reversed, res, 0, 0, &[')', '(']);
        } else {
            res.push(reversed.iter().collect());
        }
    }
}
```

#### 341. Flatten Nested List Iterator

java

```java
public class NestedIterator implements Iterator<Integer> {
    private final Queue<Integer> queue;

    public NestedIterator(List<NestedInteger> nestedList) {
        queue = new ArrayDeque<>();
        dfs(nestedList);
    }

    void dfs(List<NestedInteger> nestedList) {
        if (nestedList == null) {
            return;
        }
        for (NestedInteger ni : nestedList) {
            if (ni.isInteger()) {
                queue.offer(ni.getInteger());
            } else {
                dfs(ni.getList());
            }
        }
    }

    public Integer next() {
        if (hasNext()) {
            return queue.poll();
        }
        return null;
    }

    public boolean hasNext() {
        return !queue.isEmpty();
    }
}
```

go

```go
type NestedIterator struct {
	queue []int
}

func Constructor(nestedList []*NestedInteger) *NestedIterator {
	var queue []int
	dfs(nestedList, &queue)
	return &NestedIterator{queue: queue}
}
func dfs(nestedList []*NestedInteger, queue *[]int) {
	for _, nested := range nestedList {
		if nested.IsInteger() {
			*queue = append(*queue, nested.GetInteger())
		} else {
			dfs(nested.GetList(), queue)
		}
	}
}
func (ni *NestedIterator) Next() int {
	if ni.HasNext() {
		res := ni.queue[0]
		ni.queue = ni.queue[1:]
		return res
	}
	return -1
}

func (ni *NestedIterator) HasNext() bool {
	return len(ni.queue) > 0
}
```

rust

```rust
pub struct NestedIterator {
    queue: Vec<i32>,
    cursor: usize,
}

impl NestedIterator {
    pub fn new(nestedList: Vec<NestedInteger>) -> Self {
        let mut queue = vec![];
        Self::dfs(&nestedList, &mut queue);
        Self { queue, cursor: 0 }
    }
    
    fn dfs(nested_list: &[NestedInteger], queue: &mut Vec<i32>) {
        for item in nested_list {
            match item {
                NestedInteger::Int(val) => queue.push(*val),
                NestedInteger::List(list) => Self::dfs(list, queue),
            }
        }
    }
    
    pub fn next(&mut self) -> i32 {
        let res = self.queue[self.cursor];
        self.cursor += 1;
        res
    }
    
    pub fn has_next(&self) -> bool {
        self.cursor < self.queue.len()
    }
}
```

#### 463. Island Perimeter

go

```go
func islandPerimeter(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	res := 0
	dirs := [4][2]int{{1, 0}, {0, 1}, {0, -1}, {-1, 0}}
	for i := range grid {
		for j := range grid[0] {
			if grid[i][j] == 0 {
				continue
			}
			for _, dir := range dirs {
				x := i + dir[0]
				y := j + dir[1]
				if x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == 0 {
					res++
				}
			}
		}
	}
	return res
}
```

rust

```rust
impl Solution {
    pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
        let (m, n) = (grid.len(), grid[0].len());
        let mut res = 0;
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        
        for i in 0..m {
            for j in 0..n {
                if grid[i][j] == 1 {
                    for (dx, dy) in dirs {
                        let x = i as i32 + dx;
                        let y = j as i32 + dy;
                        if x < 0 || x >= m as i32 || y < 0 || y >= n as i32 || grid[x as usize][y as usize] == 0 {
                            res += 1;
                        }
                    }
                }
            }
        }
        res
    }
}
```

#### 647. Palindromic Substrings

go

```go
func countSubstrings(s string) int {
	res := 0
	dfs := func(i, j int, s string) {
		for i >= 0 && j < len(s) && s[i] == s[j] {
			res++
			i--
			j++
		}
	}

	for i := range s {
		dfs(i, i, s)
		dfs(i, i+1, s)
	}
	return res
}
```

rust

```rust
impl Solution {
    pub fn count_substrings(s: String) -> i32 {
        let s_bytes = s.as_bytes();
        let mut res = 0;
        for i in 0..s.len() {
            res += Self::extend(s_bytes, i as i32, i as i32);
            res += Self::extend(s_bytes, i as i32, (i + 1) as i32);
        }
        res
    }
    
    fn extend(s: &[u8], mut i: i32, mut j: i32) -> i32 {
        let mut count = 0;
        while i >= 0 && j < s.len() as i32 && s[i as usize] == s[j as usize] {
            count += 1;
            i -= 1;
            j += 1;
        }
        count
    }
}
```

#### 680. Valid Palindrome II

go

```go
func validPalindrome(s string) bool {
	return check(s, 0, len(s)-1, 1)
}
func check(s string, left int, right int, del int) bool {
	for left < right {
		if s[left] != s[right] {
			return del > 0 && (check(s, left+1, right, del-1) || check(s, left, right-1, del-1))
		}
		left++
		right--
	}
	return true
}
```

rust

```rust
impl Solution {
    pub fn valid_palindrome(s: String) -> bool {
        let chars: Vec<char> = s.chars().collect();
        Self::check(&chars, 0, chars.len() - 1, 1)
    }
    fn check(s: &[char], mut left: usize, mut right: usize, del: i32) -> bool {
        while left < right {
            if s[left] != s[right] {
                if del > 0 {
                    return Self::check(s, left + 1, right, del - 1) 
                        || Self::check(s, left, right - 1, del - 1);
                } else {
                    return false;
                }
            }
            left += 1;
            right -= 1;
        }
        true
    }
}
```

#### 761. Special Binary String

go

```go
func makeLargestSpecial(s string) string {
	count, left := 0, 0
	var res []string
	for right, char := range s {
		if char == '1' {
			count++
		} else {
			count--
		}
		if count == 0 {
			res = append(res, "1"+makeLargestSpecial(s[left+1:right])+"0")
			left = right + 1
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(res)))
	return strings.Join(res, "")
}
```

rust

```rust
impl Solution {
    pub fn make_largest_special(s: String) -> String {
        let mut count = 0;
        let mut i = 0;
        let mut res = vec![];
        let chars: Vec<char> = s.chars().collect();
        
        for (j, &c) in chars.iter().enumerate() {
            if c == '1' { count += 1; } else { count -= 1; }
            if count == 0 {
                let inner = &s[i+1..j];
                res.push(format!("1{}0", Self::make_largest_special(inner.to_string())));
                i = j + 1;
            }
        }
        res.sort_by(|a, b| b.cmp(a));
        res.join("")
    }
}
```

#### 784. Letter Case Permutation

go

```go
func letterCasePermutation(s string) []string {
	var res []string
	dfs([]byte(s), 0, &res)
	return res
}

func dfs(slice []byte, index int, res *[]string) {
	if len(slice) == index {
		*res = append(*res, string(slice))
		return
	}
	if slice[index] >= '0' && slice[index] <= '9' {
		dfs(slice, index+1, res)
		return
	}

	dfs(slice, index+1, res)
    
    if (slice[index] >= 'a' && slice[index] <= 'z') || (slice[index] >= 'A' && slice[index] <= 'Z') {
        slice[index] ^= 32
        dfs(slice, index+1, res)
        slice[index] ^= 32 
    }
}
```

python

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        res = []
        self.dfs(list(s), 0, res)
        return res

    def dfs(self, s: List[str], i: int, res: List[str]):
        if i == len(s):
            res.append("".join(s))
            return
        
        # Branch 1: Keep current char (recurse)
        self.dfs(s, i + 1, res)
        
        # Branch 2: If letter, toggle case and recurse
        if s[i].isalpha():
            s[i] = s[i].swapcase()
            self.dfs(s, i + 1, res)
            s[i] = s[i].swapcase() # backtrack
```

rust

```rust
impl Solution {
    pub fn letter_case_permutation(s: String) -> Vec<String> {
        let mut s: Vec<char> = s.chars().collect();
        let mut res = vec![];
        Self::dfs(&mut s, &mut res, 0);
        res
    }
    fn dfs(s: &mut [char], res: &mut Vec<String>, index: usize) {
        if index == s.len() {
            res.push(s.iter().collect());
            return;
        }
        if s[index].is_numeric() {
            Self::dfs(s, res, index + 1);
            return;
        }
        s[index] = s[index].to_ascii_lowercase();
        Self::dfs(s, res, index + 1);
        
        s[index] = s[index].to_ascii_uppercase();
        Self::dfs(s, res, index + 1);
    }
}
```

#### 834. Sum of Distances in Tree

java

```java
class Solution {
    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++)
            graph.add(new ArrayList<>());

        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        int[] dist = new int[n];
        int[] count = new int[n];
        dfs1(0, -1, graph, count, dist);
        dfs2(0, -1, graph, count, dist, n);
        return dist;
    }

    private void dfs1(int i, int prev, ArrayList<ArrayList<Integer>> graph, int[] count, int[] dist) {
        count[i] = 1;
        dist[i] = 0;
        for (Integer j : graph.get(i)) {
            if (j == prev)
                continue;
            dfs1(j, i, graph, count, dist);
            count[i] += count[j];
            dist[i] += dist[j] + count[j];
        }
    }

    private void dfs2(int i, int prev, ArrayList<ArrayList<Integer>> graph, int[] count, int[] dist, int n) {
        if (prev != -1) {
            dist[i] = (dist[prev] - count[i]) + (n - count[i]);
        }
        for (Integer j : graph.get(i)) {
            if (j == prev)
                continue;
            dfs2(j, i, graph, count, dist, n);
        }
    }
}
```

go

```go
func sumOfDistancesInTree(n int, edges [][]int) []int {
	graph := make([][]int, n)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
	}
	count := make([]int, n)
	res := make([]int, n)

	var dfs1 func(int, int)
	dfs1 = func(u, p int) {
		count[u] = 1
		for _, v := range graph[u] {
			if v == p {
				continue
			}
			dfs1(v, u)
			count[u] += count[v]
			res[u] += res[v] + count[v]
		}
	}

	var dfs2 func(int, int)
	dfs2 = func(u, p int) {
		for _, v := range graph[u] {
			if v == p {
				continue
			}
			res[v] = res[u] - count[v] + (n - count[v])
			dfs2(v, u)
		}
	}

	dfs1(0, -1)
	dfs2(0, -1)
	return res
}
```

rust

```rust
impl Solution {
    pub fn sum_of_distances_in_tree(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        let n = n as usize;
        let mut graph = vec![vec![]; n];
        for edge in edges {
            graph[edge[0] as usize].push(edge[1] as usize);
            graph[edge[1] as usize].push(edge[0] as usize);
        }
        let mut count = vec![1; n];
        let mut res = vec![0; n];
        
        Self::dfs1(0, n, &graph, &mut count, &mut res);
        Self::dfs2(0, n, &graph, &count, &mut res);
        res
    }
    
    fn dfs1(u: usize, p: usize, graph: &Vec<Vec<usize>>, count: &mut Vec<i32>, res: &mut Vec<i32>) {
        for &v in &graph[u] {
            if v == p { continue; }
            Self::dfs1(v, u, graph, count, res);
            count[u] += count[v];
            res[u] += res[v] + count[v];
        }
    }
    
    fn dfs2(u: usize, p: usize, graph: &Vec<Vec<usize>>, count: &Vec<i32>, res: &mut Vec<i32>) {
        for &v in &graph[u] {
            if v == p { continue; }
            res[v] = res[u] - count[v] + (count.len() as i32 - count[v]);
            Self::dfs2(v, u, graph, count, res);
        }
    }
}
```

#### 841. Keys and Rooms

go

```go
func canVisitAllRooms(rooms [][]int) bool {
	visited := make([]bool, len(rooms))
	dfs(rooms, 0, visited)
	for _, v := range visited {
		if !v {
			return false
		}
	}
	return true
}
func dfs(rooms [][]int, u int, visited []bool) {
	visited[u] = true
	for _, v := range rooms[u] {
		if !visited[v] {
			dfs(rooms, v, visited)
		}
	}
}
```

rust

```rust
impl Solution {
    pub fn can_visit_all_rooms(rooms: Vec<Vec<i32>>) -> bool {
        let n = rooms.len();
        let mut visited = vec![false; n];
        Self::dfs(&rooms, 0, &mut visited);
        visited.iter().all(|&x| x)
    }
    fn dfs(rooms: &Vec<Vec<i32>>, curr: usize, visited: &mut Vec<bool>) {
        visited[curr] = true;
        for &next_room in &rooms[curr] {
            if !visited[next_room as usize] {
                Self::dfs(rooms, next_room as usize, visited);
            }
        }
    }
}
```

#### 1286. Iterator for Combination

java

```java
class CombinationIterator {
    List<String> data;
    int cursor;

    public CombinationIterator(String characters, int length) {
        data = new ArrayList<>();
        dfs(characters, length, new StringBuilder());
    }

    private void dfs(String characters, int length, StringBuilder path) {
        if (length == path.length()) {
            data.add(path.toString());
            return;
        }
        int n = characters.length();
        for (int i = 0; i < n; i++) {
            path.append(characters.charAt(i));
            dfs(characters.substring(i + 1), length, path);
            path.deleteCharAt(path.length() - 1);
        }
    }

    public String next() {
        return data.get(cursor++);
    }

    public boolean hasNext() {
        return cursor < data.size();
    }
}
```

rust

```rust
struct CombinationIterator {
    combinations: Vec<String>,
    cursor: usize,
}

impl CombinationIterator {
    fn new(characters: String, combinationLength: i32) -> Self {
        let mut combinations = vec![];
        let mut path = String::new();
        let chars: Vec<char> = characters.chars().collect();
        Self::dfs(&chars, combinationLength as usize, 0, &mut path, &mut combinations);
        Self { combinations, cursor: 0 }
    }
    
    fn dfs(chars: &[char], len: usize, start: usize, path: &mut String, res: &mut Vec<String>) {
        if path.len() == len {
            res.push(path.clone());
            return;
        }
        for i in start..chars.len() {
            path.push(chars[i]);
            Self::dfs(chars, len, i + 1, path, res);
            path.pop();
        }
    }
    
    fn next(&mut self) -> String {
        let res = self.combinations[self.cursor].clone();
        self.cursor += 1;
        res
    }
    
    fn has_next(&self) -> bool {
        self.cursor < self.combinations.len()
    }
}
```

#### 1306. Jump Game III

go

```go
func canReach(arr []int, start int) bool {
	n := len(arr)
	visited := make([]bool, n)
	return dfs(arr, visited, start, n)
}
func dfs(arr []int, visited []bool, start, n int) bool {
	if start >= n || start < 0 || visited[start] {
		return false
	}
	visited[start] = true
	return arr[start] == 0 || dfs(arr, visited, start-arr[start], n) || dfs(arr, visited, start+arr[start], n)
}
```

rust

```rust
impl Solution {
    pub fn can_reach(arr: Vec<i32>, start: i32) -> bool {
        let mut visited = vec![false; arr.len()];
        Self::dfs(&arr, &mut visited, start as usize)
    }
    fn dfs(arr: &Vec<i32>, visited: &mut Vec<bool>, start: usize) -> bool {
        if start >= arr.len() || visited[start] {
            return false;
        }
        if arr[start] == 0 {
            return true;
        }
        visited[start] = true;
        let jump = arr[start] as usize;
        let left_idx = if start >= jump { start - jump } else { usize::MAX };
        let right_idx = start + jump;
        
        Self::dfs(arr, visited, left_idx) || Self::dfs(arr, visited, right_idx)
    }
}
```

rust

```rust
impl Solution {
    pub fn add_operators(s: String, target: i32) -> Vec<String> {
        let mut res = vec![];
        let num_str: Vec<char> = s.chars().collect();
        Self::dfs(&num_str, target as i64, 0, String::new(), 0, 0, &mut res);
        res
    }
    fn dfs(s: &[char], target: i64, pos: usize, path: String, curr: i64, prev: i64, res: &mut Vec<String>) {
        if pos == s.len() {
            if curr == target {
                res.push(path);
            }
            return;
        }
        for i in pos..s.len() {
            if i > pos && s[pos] == '0' { break; }
            let val_str: String = s[pos..=i].iter().collect();
            let val: i64 = val_str.parse().unwrap();
            
            if pos == 0 {
                Self::dfs(s, target, i + 1, val_str, val, val, res);
            } else {
                Self::dfs(s, target, i + 1, format!("{}+{}", path, val), curr + val, val, res);
                Self::dfs(s, target, i + 1, format!("{}-{}", path, val), curr - val, -val, res);
                Self::dfs(s, target, i + 1, format!("{}*{}", path, val), curr - prev + prev * val, prev * val, res);
            }
        }
    }
}
```

#### 301. Remove Invalid Parentheses

go

```go
func removeInvalidParentheses(s string) []string {
	var res []string
	dfs(s, &res, 0, 0, [2]byte{'(', ')'})
	return res
}
func dfs(s string, res *[]string, iStart, jStart int, pair [2]byte) {
	stack := 0
	for i := iStart; i < len(s); i++ {
		if s[i] == pair[0] {
			stack++
		}
		if s[i] == pair[1] {
			stack--
		}
		if stack >= 0 {
			continue
		}
		for j := jStart; j <= i; j++ {
			if s[j] == pair[1] && (j == jStart || s[j-1] != pair[1]) {
				dfs(s[:j]+s[j+1:], res, i, j, pair)
			}
		}
		return
	}
	reversed := reverse(s)
	if pair[0] == '(' {
		dfs(reversed, res, 0, 0, [2]byte{')', '('})
	} else {
		*res = append(*res, reversed)
	}
}
func reverse(s string) string {
	bt := []byte(s)
	left, right := 0, len(bt)-1
	for left < right {
		bt[left], bt[right] = bt[right], bt[left]
		left++
		right--
	}
	return string(bt)
}
```

java

```java
class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        dfs(s, res, 0, 0, new char[]{'(', ')'});
        return res;
    }

    private void dfs(String s, List<String> res, int iStart, int jStart, char[] pair) {
        for (int stack = 0, i = iStart; i < s.length(); i++) {
            if (s.charAt(i) == pair[0]) {
                stack++;
            }
            if (s.charAt(i) == pair[1]) {
                stack--;
            }
            if (stack >= 0) {
                continue;
            }
            for (int j = jStart; j <= i; j++) {
                if (s.charAt(j) == pair[1] && (j == jStart || s.charAt(j - 1) != pair[1])) {
                    dfs(s.substring(0, j) + s.substring(j + 1), res, i, j, pair);
                }
            }
            return;
        }
        String reversed = new StringBuilder(s).reverse().toString();
        if (pair[0] == '(') {
            dfs(reversed, res, 0, 0, new char[]{')', '('});
        } else {
            res.add(reversed);
        }
    }
}
```

python

```python
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        res = []
        self.dfs(s, res, 0, 0, ['(', ')'])
        return res

    def dfs(self, s: str, res: List[str], i_start: int, j_start: int, pair: List[str]):
        stack = 0
        for i in range(i_start, len(s)):
            if s[i] == pair[0]:
                stack += 1
            if s[i] == pair[1]:
                stack -= 1
            if stack >= 0:
                continue
            for j in range(j_start, i + 1):
                if s[j] == pair[1] and (j == j_start or s[j - 1] != pair[1]):
                    self.dfs(s[:j] + s[j + 1:], res, i, j, pair)
            return
        if pair[0] == '(':
            self.dfs(s[::-1], res, 0, 0, pair[::-1])
        else:
            res.append(s[::-1])
```

rust

```rust
impl Solution {
    pub fn remove_invalid_parentheses(s: String) -> Vec<String> {
        let mut res = vec![];
        let chars: Vec<char> = s.chars().collect();
        Self::dfs(&chars, &mut res, 0, 0, &['(', ')']);
        res
    }
    
    fn dfs(s: &[char], res: &mut Vec<String>, i_start: usize, j_start: usize, pair: &[char]) {
        let mut stack = 0;
        for i in i_start..s.len() {
            if s[i] == pair[0] { stack += 1; }
            if s[i] == pair[1] { stack -= 1; }
            if stack >= 0 { continue; }
            
            for j in j_start..=i {
                if s[j] == pair[1] && (j == j_start || s[j-1] != pair[1]) {
                    let mut new_s = s.to_vec();
                    new_s.remove(j);
                    Self::dfs(&new_s, res, i, j, pair);
                }
            }
            return;
        }
        
        let reversed: Vec<char> = s.iter().rev().cloned().collect();
        if pair[0] == '(' {
            Self::dfs(&reversed, res, 0, 0, &[')', '(']);
        } else {
            res.push(reversed.iter().collect());
        }
    }
}
```

#### 341. Flatten Nested List Iterator

java

```java
public class NestedIterator implements Iterator<Integer> {
    private final Queue<Integer> queue;

    public NestedIterator(List<NestedInteger> nestedList) {
        queue = new ArrayDeque<>();
        dfs(nestedList);
    }

    void dfs(List<NestedInteger> nestedList) {
        if (nestedList == null) {
            return;
        }
        for (NestedInteger ni : nestedList) {
            if (ni.isInteger()) {
                queue.offer(ni.getInteger());
            } else {
                dfs(ni.getList());
            }
        }
    }

    public Integer next() {
        if (hasNext()) {
            return queue.poll();
        }
        return null;
    }

    public boolean hasNext() {
        return !queue.isEmpty();
    }
}
```

go

```go
type NestedIterator struct {
	queue []int
}

func Constructor(nestedList []*NestedInteger) *NestedIterator {
	var queue []int
	dfs(nestedList, &queue)
	return &NestedIterator{queue: queue}
}
func dfs(nestedList []*NestedInteger, queue *[]int) {
	for _, nested := range nestedList {
		if nested.IsInteger() {
			*queue = append(*queue, nested.GetInteger())
		} else {
			dfs(nested.GetList(), queue)
		}
	}
}
func (ni *NestedIterator) Next() int {
	if ni.HasNext() {
		res := ni.queue[0]
		ni.queue = ni.queue[1:]
		return res
	}
	return -1
}

func (ni *NestedIterator) HasNext() bool {
	return len(ni.queue) > 0
}
```

rust

```rust
pub struct NestedIterator {
    queue: Vec<i32>,
    cursor: usize,
}

impl NestedIterator {
    pub fn new(nestedList: Vec<NestedInteger>) -> Self {
        let mut queue = vec![];
        Self::dfs(&nestedList, &mut queue);
        Self { queue, cursor: 0 }
    }
    
    fn dfs(nested_list: &[NestedInteger], queue: &mut Vec<i32>) {
        for item in nested_list {
            match item {
                NestedInteger::Int(val) => queue.push(*val),
                NestedInteger::List(list) => Self::dfs(list, queue),
            }
        }
    }
    
    pub fn next(&mut self) -> i32 {
        let res = self.queue[self.cursor];
        self.cursor += 1;
        res
    }
    
    pub fn has_next(&self) -> bool {
        self.cursor < self.queue.len()
    }
}
```

#### 463. Island Perimeter

go

```go
func islandPerimeter(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	res := 0
	dirs := [4][2]int{{1, 0}, {0, 1}, {0, -1}, LAND_DIR_UP := [4][2]int{{1, 0}, {0, 1}, {0, -1}, {-1, 0}}
	for i := range grid {
		for j := range grid[0] {
			if grid[i][j] == 0 {
				continue
			}
			for _, dir := range dirs {
				x := i + dir[0]
				y := j + dir[1]
				if x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == 0 {
					res++
				}
			}
		}
	}
	return res
}
```

rust

```rust
impl Solution {
    pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
        let (m, n) = (grid.len(), grid[0].len());
        let mut res = 0;
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        
        for i in 0..m {
            for j in 0..n {
                if grid[i][j] == 1 {
                    for (dx, dy) in dirs {
                        let x = i as i32 + dx;
                        let y = j as i32 + dy;
                        if x < 0 || x >= m as i32 || y < 0 || y >= n as i32 || grid[x as usize][y as usize] == 0 {
                            res += 1;
                        }
                    }
                }
            }
        }
        res
    }
}
```

#### 647. Palindromic Substrings

go

```go
func countSubstrings(s string) int {
	res := 0
	dfs := func(i, j int, s string) {
		for i >= 0 && j < len(s) && s[i] == s[j] {
			res++
			i--
			j++
		}
	}

	for i := range s {
		dfs(i, i, s)
		dfs(i, i+1, s)
	}
	return res
}
```

rust

```rust
impl Solution {
    pub fn count_substrings(s: String) -> i32 {
        let s_bytes = s.as_bytes();
        let mut res = 0;
        for i in 0..s.len() {
            res += Self::extend(s_bytes, i as i32, i as i32);
            res += Self::extend(s_bytes, i as i32, (i + 1) as i32);
        }
        res
    }
    
    fn extend(s: &[u8], mut i: i32, mut j: i32) -> i32 {
        let mut count = 0;
        while i >= 0 && j < s.len() as i32 && s[i as usize] == s[j as usize] {
            count += 1;
            i -= 1;
            j += 1;
        }
        count
    }
}
```

#### 680. Valid Palindrome II

go

```go
func validPalindrome(s string) bool {
	return check(s, 0, len(s)-1, 1)
}
func check(s string, left int, right int, del int) bool {
	for left < right {
		if s[left] != s[right] {
			return del > 0 && (check(s, left+1, right, del-1) || check(s, left, right-1, del-1))
		}
		left++
		right--
	}
	return true
}
```

rust

```rust
impl Solution {
    pub fn valid_palindrome(s: String) -> bool {
        let chars: Vec<char> = s.chars().collect();
        Self::check(&chars, 0, chars.len() - 1, 1)
    }
    fn check(s: &[char], mut left: usize, mut right: usize, del: i32) -> bool {
        while left < right {
            if s[left] != s[right] {
                if del > 0 {
                    return Self::check(s, left + 1, right, del - 1) 
                        || Self::check(s, left, right - 1, del - 1);
                } else {
                    return false;
                }
            }
            left += 1;
            right -= 1;
        }
        true
    }
}
```

#### 761. Special Binary String

go

```go
func makeLargestSpecial(s string) string {
	count, left := 0, 0
	var res []string
	for right, char := range s {
		if char == '1' {
			count++
		} else {
			count--
		}
		if count == 0 {
			res = append(res, "1"+makeLargestSpecial(s[left+1:right])+"0")
			left = right + 1
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(res)))
	return strings.Join(res, "")
}
```

rust

```rust
impl Solution {
    pub fn make_largest_special(s: String) -> String {
        let mut count = 0;
        let mut i = 0;
        let mut res = vec![];
        let chars: Vec<char> = s.chars().collect();
        
        for (j, &c) in chars.iter().enumerate() {
            if c == '1' { count += 1; } else { count -= 1; }
            if count == 0 {
                let inner = &s[i+1..j];
                res.push(format!("1{}0", Self::make_largest_special(inner.to_string())));
                i = j + 1;
            }
        }
        res.sort_by(|a, b| b.cmp(a));
        res.join("")
    }
}
```

#### 784. Letter Case Permutation

go

```go
func letterCasePermutation(s string) []string {
	var res []string
	dfs([]byte(s), 0, &res)
	return res
}

func dfs(slice []byte, index int, res *[]string) {
	if len(slice) == index {
		*res = append(*res, string(slice))
		return
	}
	if slice[index] >= '0' && slice[index] <= '9' {
		dfs(slice, index+1, res)
		return
	}

	dfs(slice, index+1, res)
    
    if (slice[index] >= 'a' && slice[index] <= 'z') || (slice[index] >= 'A' && slice[index] <= 'Z') {
        slice[index] ^= 32
        dfs(slice, index+1, res)
        slice[index] ^= 32 
    }
}
```

python

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        res = []
        self.dfs(list(s), 0, res)
        return res

    def dfs(self, s: List[str], i: int, res: List[str]):
        if i == len(s):
            res.append("".join(s))
            return
        
        # Branch 1: Keep current char (recurse)
        self.dfs(s, i + 1, res)
        
        # Branch 2: If letter, toggle case and recurse
        if s[i].isalpha():
            s[i] = s[i].swapcase()
            self.dfs(s, i + 1, res)
            s[i] = s[i].swapcase() # backtrack
```

rust

```rust
impl Solution {
    pub fn letter_case_permutation(s: String) -> Vec<String> {
        let mut s: Vec<char> = s.chars().collect();
        let mut res = vec![];
        Self::dfs(&mut s, &mut res, 0);
        res
    }
    fn dfs(s: &mut [char], res: &mut Vec<String>, index: usize) {
        if index == s.len() {
            res.push(s.iter().collect());
            return;
        }
        if s[index].is_numeric() {
            Self::dfs(s, res, index + 1);
            return;
        }
        s[index] = s[index].to_ascii_lowercase();
        Self::dfs(s, res, index + 1);
        
        s[index] = s[index].to_ascii_uppercase();
        Self::dfs(s, res, index + 1);
    }
}
```

#### 834. Sum of Distances in Tree

java

```java
class Solution {
    public int[] sumOfDistancesInTree(int n, int[][] edges) {
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++)
            graph.add(new ArrayList<>());

        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        int[] dist = new int[n];
        int[] count = new int[n];
        dfs1(0, -1, graph, count, dist);
        dfs2(0, -1, graph, count, dist, n);
        return dist;
    }

    private void dfs1(int i, int prev, ArrayList<ArrayList<Integer>> graph, int[] count, int[] dist) {
        count[i] = 1;
        dist[i] = 0;
        for (Integer j : graph.get(i)) {
            if (j == prev)
                continue;
            dfs1(j, i, graph, count, dist);
            count[i] += count[j];
            dist[i] += dist[j] + count[j];
        }
    }

    private void dfs2(int i, int prev, ArrayList<ArrayList<Integer>> graph, int[] count, int[] dist, int n) {
        if (prev != -1) {
            dist[i] = (dist[prev] - count[i]) + (n - count[i]);
        }
        for (Integer j : graph.get(i)) {
            if (j == prev)
                continue;
            dfs2(j, i, graph, count, dist, n);
        }
    }
}
```

go

```go
func sumOfDistancesInTree(n int, edges [][]int) []int {
	graph := make([][]int, n)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
	}
	count := make([]int, n)
	res := make([]int, n)

	var dfs1 func(int, int)
	dfs1 = func(u, p int) {
		count[u] = 1
		for _, v := range graph[u] {
			if v == p {
				continue
			}
			dfs1(v, u)
			count[u] += count[v]
			res[u] += res[v] + count[v]
		}
	}

	var dfs2 func(int, int)
	dfs2 = func(u, p int) {
		for _, v := range graph[u] {
			if v == p {
				continue
			}
			res[v] = res[u] - count[v] + (n - count[v])
			dfs2(v, u)
		}
	}

	dfs1(0, -1)
	dfs2(0, -1)
	return res
}
```

rust

```rust
impl Solution {
    pub fn sum_of_distances_in_tree(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        let n = n as usize;
        let mut graph = vec![vec![]; n];
        for edge in edges {
            graph[edge[0] as usize].push(edge[1] as usize);
            graph[edge[1] as usize].push(edge[0] as usize);
        }
        let mut count = vec![1; n];
        let mut res = vec![0; n];
        
        Self::dfs1(0, n, &graph, &mut count, &mut res);
        Self::dfs2(0, n, &graph, &count, &mut res);
        res
    }
    
    fn dfs1(u: usize, p: usize, graph: &Vec<Vec<usize>>, count: &mut Vec<i32>, res: &mut Vec<i32>) {
        for &v in &graph[u] {
            if v == p { continue; }
            Self::dfs1(v, u, graph, count, res);
            count[u] += count[v];
            res[u] += res[v] + count[v];
        }
    }
    
    fn dfs2(u: usize, p: usize, graph: &Vec<Vec<usize>>, count: &Vec<i32>, res: &mut Vec<i32>) {
        for &v in &graph[u] {
            if v == p { continue; }
            res[v] = res[u] - count[v] + (count.len() as i32 - count[v]);
            Self::dfs2(v, u, graph, count, res);
        }
    }
}
```

#### 841. Keys and Rooms

go

```go
func canVisitAllRooms(rooms [][]int) bool {
	visited := make([]bool, len(rooms))
	dfs(rooms, 0, visited)
	for _, v := range visited {
		if !v {
			return false
		}
	}
	return true
}
func dfs(rooms [][]int, u int, visited []bool) {
	visited[u] = true
	for _, v := range rooms[u] {
		if !visited[v] {
			dfs(rooms, v, visited)
		}
	}
}
```

rust

```rust
impl Solution {
    pub fn can_visit_all_rooms(rooms: Vec<Vec<i32>>) -> bool {
        let n = rooms.len();
        let mut visited = vec![false; n];
        Self::dfs(&rooms, 0, &mut visited);
        visited.iter().all(|&x| x)
    }
    fn dfs(rooms: &Vec<Vec<i32>>, curr: usize, visited: &mut Vec<bool>) {
        visited[curr] = true;
        for &next_room in &rooms[curr] {
            if !visited[next_room as usize] {
                Self::dfs(rooms, next_room as usize, visited);
            }
        }
    }
}
```

#### 1286. Iterator for Combination

java

```java
class CombinationIterator {
    List<String> data;
    int cursor;

    public CombinationIterator(String characters, int length) {
        data = new ArrayList<>();
        dfs(characters, length, new StringBuilder());
    }

    private void dfs(String characters, int length, StringBuilder path) {
        if (length == path.length()) {
            data.add(path.toString());
            return;
        }
        int n = characters.length();
        for (int i = 0; i < n; i++) {
            path.append(characters.charAt(i));
            dfs(characters.substring(i + 1), length, path);
            path.deleteCharAt(path.length() - 1);
        }
    }

    public String next() {
        return data.get(cursor++);
    }

    public boolean hasNext() {
        return cursor < data.size();
    }
}
```

go

```go
type CombinationIterator struct {
	combinations []string
	cursor       int
}

func Constructor(characters string, combinationLength int) CombinationIterator {
	var res []string
	dfs(characters, combinationLength, 0, []byte{}, &res)
	return CombinationIterator{combinations: res}
}

func dfs(chars string, length int, start int, path []byte, res *[]string) {
	if len(path) == length {
		*res = append(*res, string(path))
		return
	}
	for i := start; i < len(chars); i++ {
		dfs(chars, length, i+1, append(path, chars[i]), res)
		path = path[:len(path)] // Backtrack: just slice back
	}
}

func (ci *CombinationIterator) Next() string {
	res := ci.combinations[ci.cursor]
	ci.cursor++
	return res
}

func (ci *CombinationIterator) HasNext() bool {
	return ci.cursor < len(ci.combinations)
}
```

rust

```rust
struct CombinationIterator {
    combinations: Vec<String>,
    cursor: usize,
}

impl CombinationIterator {
    fn new(characters: String, combinationLength: i32) -> Self {
        let mut combinations = vec![];
        let mut path = String::new();
        let chars: Vec<char> = characters.chars().collect();
        Self::dfs(&chars, combinationLength as usize, 0, &mut path, &mut combinations);
        Self { combinations, cursor: 0 }
    }
    
    fn dfs(chars: &[char], len: usize, start: usize, path: &mut String, res: &mut Vec<String>) {
        if path.len() == len {
            res.push(path.clone());
            return;
        }
        for i in start..chars.len() {
            path.push(chars[i]);
            Self::dfs(chars, len, i + 1, path, res);
            path.pop();
        }
    }
    
    fn next(&mut self) -> String {
        let res = self.combinations[self.cursor].clone();
        self.cursor += 1;
        res
    }
    
    fn has_next(&self) -> bool {
        self.cursor < self.combinations.len()
    }
}
```

#### 1306. Jump Game III

go

```go
func canReach(arr []int, start int) bool {
	n := len(arr)
	visited := make([]bool, n)
	return dfs(arr, visited, start, n)
}
func dfs(arr []int, visited []bool, start, n int) bool {
	if start >= n || start < 0 || visited[start] {
		return false
	}
	visited[start] = true
	return arr[start] == 0 || dfs(arr, visited, start-arr[start], n) || dfs(arr, visited, start+arr[start], n)
}
```

rust

```rust
impl Solution {
    pub fn can_reach(arr: Vec<i32>, start: i32) -> bool {
        let mut visited = vec![false; arr.len()];
        Self::dfs(&arr, &mut visited, start as usize)
    }
    fn dfs(arr: &Vec<i32>, visited: &mut Vec<bool>, start: usize) -> bool {
        if start >= arr.len() || visited[start] {
            return false;
        }
        if arr[start] == 0 {
            return true;
        }
        visited[start] = true;
        let jump = arr[start] as usize;
        let left_idx = if start >= jump { start - jump } else { usize::MAX };
        let right_idx = start + jump;
        
        Self::dfs(arr, visited, left_idx) || Self::dfs(arr, visited, right_idx)
    }
}
```