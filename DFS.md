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

#### [1219. Path with Maximum Gold](https://leetcode.com/problems/path-with-maximum-gold/)

java

```java
class Solution {
    public int getMaximumGold(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int total = fullGrid(grid, m, n);
        if (total != -1)
            return total;
        boolean[][] visited = new boolean[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                res = Math.max(res, dfs(grid, m, n, i, j, visited));
            }
        }
        return res;
    }

    private int dfs(int[][] grid, int m, int n, int x, int y, boolean[][] visited) {
        visited[x][y] = true;
        int res = 0;
        int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int[] dir : dirs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] == 0 || visited[nx][ny]) {
                continue;
            }
            res = Math.max(res, dfs(grid, m, n, nx, ny, visited));
        }
        visited[x][y] = false;
        return grid[x][y] + res;
    }

    private int fullGrid(int[][] grid, int m, int n) {
        int total = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) return -1;
                else total += grid[i][j];
            }
        }
        return total;
    }
}
```

#### [1034. Coloring A Border](https://leetcode.com/problems/coloring-a-border/)

```java
class Solution {
    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n], isBoarder = new boolean[m][n];
        dfs(grid, row, col, m, n, visited, isBoarder);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isBoarder[i][j]) grid[i][j] = color;
            }
        }
        return grid;
    }

    private void dfs(int[][] grid, int x, int y, int m, int n, boolean[][] visited, boolean[][] isBoarder) {
        // mark as visited
        visited[x][y] = true;
        // the border is on the boundary of the grid or adjacent to squares of a different color.
        isBoarder[x][y] = x == 0 || x == m - 1 || y == 0 || y == n - 1
                || grid[x][y] != grid[x - 1][y]
                || grid[x][y] != grid[x + 1][y]
                || grid[x][y] != grid[x][y - 1]
                || grid[x][y] != grid[x][y + 1];
        // a recursive call for each of 4 directions
        int[][] dirs = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        for (int[] dir : dirs) {
            int nx = dir[0] + x;
            int ny = dir[1] + y;
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] != grid[x][y] || visited[nx][ny]) {
                continue;
            }
            dfs(grid, nx, ny, m, n, visited, isBoarder);
        }
    }
}
```

#### [797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)

java

```java
class Solution {
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        int n = graph.length;
        dfs(graph, res, new ArrayList<>(), 0, n);
        return res;
    }

    private void dfs(int[][] graph, List<List<Integer>> res, ArrayList<Integer> path, int curr, int n) {
        path.add(curr);
        if (curr == n - 1) {
            res.add(new ArrayList<>(path));
        } else {
            for (int next : graph[curr]) {
                dfs(graph, res, path, next, n);
            }
        }
        path.remove(path.size() - 1);
    }
}
```

#### 934. Shortest Bridge

java

```java
class Solution {
    public int shortestBridge(int[][] grid) {
        int[][] dup = Arrays.stream(grid).map(int[]::clone).toArray(int[][]::new);
        int n = grid.length;
        Queue<int[]> queue = new ArrayDeque<>();
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dup[i][j] == 0) continue;
                dfs(dup, i, j, queue, n, dirs);
                return bfs(dup, queue, n, dirs);
            }
        }
        throw new Error("unreachable");
    }

    private int bfs(int[][] grid, Queue<int[]> queue, int n, int[][] dirs) {
        int step = 0;
        while (!queue.isEmpty()) {
            for (int k = queue.size(); k > 0; k--) {
                int[] poll = queue.poll();
                for (int[] dir : dirs) {
                    int x = poll[0] + dir[0];
                    int y = poll[1] + dir[1];
                    if (x < 0 || y < 0 || x >= n || y >= n || grid[x][y] == 2) continue;
                    if (grid[x][y] == 1) return step;
                    grid[x][y] = 2;
                    queue.offer(new int[]{x, y});
                }
            }
            step++;
        }
        return -1;
    }

    // change connected 1 to 2, collect them to the queue
    private void dfs(int[][] grid, int i, int j, Queue<int[]> queue, int n, int[][] dirs) {
        grid[i][j] = 2;
        queue.offer(new int[]{i, j});
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= n || y >= n) continue;
            if (grid[x][y] == 1) {
                dfs(grid, x, y, queue, n, dirs);
            }
        }
    }
}
```

#### 399. Evaluate Division

java

```java
class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        var graph = graphBuild(equations, values);
        int n = queries.size();
        double[] res = new double[n];
        for (int i = 0; i < n; i++) {
            res[i] = dfs(graph, queries.get(i).get(0), queries.get(i).get(1), new HashSet<>());
        }
        return res;
    }

    private double dfs(HashMap<String, HashMap<String, Double>> graph, String start, String end, HashSet<String> visited) {
        if (!graph.containsKey(start)) {
            return -1;
        }
        if (graph.get(start).containsKey(end)) {
            return graph.get(start).get(end);
        }
        visited.add(start);
        for (String neighbor : graph.get(start).keySet()) {
            if (visited.contains(neighbor)) continue;
            double weight = dfs(graph, neighbor, end, visited);
            if (weight == -1) continue;
            return weight * graph.get(start).get(neighbor);
        }
        return -1;
    }

    private HashMap<String, HashMap<String, Double>> graphBuild(List<List<String>> equations, double[] values) {
        HashMap<String, HashMap<String, Double>> graph = new HashMap<>();
        for (int i = 0; i < equations.size(); i++) {
            String u = equations.get(i).get(0);
            String v = equations.get(i).get(1);
            graph.putIfAbsent(u, new HashMap<>());
            graph.get(u).put(v, values[i]);
            graph.putIfAbsent(v, new HashMap<>());
            graph.get(v).put(u, 1 / values[i]);
        }
        return graph;
    }
}
```

#### 79. Word Search

go

```go
func exist(board [][]byte, word string) bool {
	m, n := len(board), len(board[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	for i := range board {
		for j := range board[i] {
			if dfs(board, word, i, j, m, n, visited) {
				return true
			}
		}
	}
	return false
}

func dfs(board [][]byte, word string, i, j, m, n int, visited [][]bool) bool {
	if len(word) == 1 {
		return word[0] == board[i][j]
	}
	if board[i][j] != word[0] {
		return false
	}
	visited[i][j] = true
	dirs := [4][2]int{{-1, 0}, {0, 1}, {1, 0}, {0, -1}}
	for _, dir := range dirs {
		x := i + dir[0]
		y := j + dir[1]
		if x < 0 || x >= m || y < 0 || y >= n || visited[x][y] {
			continue
		}
		if dfs(board, word[1:], x, y, m, n, visited) {
			return true
		}
	}
	visited[i][j] = false
	return false
}
```

rust

```rust
impl Solution {
    pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
        let m = board.len();
        let n = board[0].len();
        let word: Vec<char> = word.chars().collect();
        let mut visited = vec![vec![false; n]; m];
        for i in 0..m {
            for j in 0..n {
                if Solution::dfs(&board, &word, i, j, m, n, &mut visited) {
                    return true;
                }
            }
        }
        false
    }
    fn dfs(board: &Vec<Vec<char>>, word: &[char], x: usize, y: usize, m: usize, n: usize, visited: &mut Vec<Vec<bool>>) -> bool {
        if word.len() == 1 {
            return board[x][y] == word[0];
        }
        if board[x][y] != word[0] {
            return false;
        }
        visited[x][y] = true;
        let dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]];
        for dir in dirs {
            let x = x as i32 + dir[0];
            let y = y as i32 + dir[1];
            if x < 0 || y < 0 {
                continue;
            }
            let x = x as usize;
            let y = y as usize;
            if x >= m || y >= n || visited[x][y] {
                continue;
            }
            if Solution::dfs(board, &word[1..], x, y, m, n, visited) {
                visited[x][y] = false;
                return true;
            }
        }
        visited[x][y] = false;
        false
    }
}
```

#### 130. Surrounded Regions

go

```go
func solve(board [][]byte) {
	m, n := len(board), len(board[0])
	for i := range board {
		for j := range board[0] {
			if board[i][j] == 'O' && (i == 0 || j == 0 || i == m-1 || j == n-1) {
				dfs(board, i, j, m, n)
			}
		}
	}
	for i := range board {
		for j := range board[0] {
			if board[i][j] == 'O' {
				board[i][j] = 'X'
			}
			if board[i][j] == '#' {
				board[i][j] = 'O'
			}
		}
	}
}

func dfs(board [][]byte, x, y, m, n int) {
	board[x][y] = '#'
	dirs := [4][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	for _, dir := range dirs {
		nx := x + dir[0]
		ny := y + dir[1]
		if nx < 0 || nx >= m || ny < 0 || ny >= n || board[nx][ny] != 'O' {
			continue
		}
		dfs(board, nx, ny, m, n)
	}
}
```

#### 133. Clone Graph

go

```go
func cloneGraph(node *Node) *Node {
	if node == nil {
		return nil
	}
	copies := make(map[*Node]*Node)
	return dfs(node, copies)
}
func dfs(node *Node, copies map[*Node]*Node) *Node {
	if _, ok := copies[node]; !ok {
		copies[node] = &Node{Val: node.Val}
		for _, neighbor := range node.Neighbors {
			copies[node].Neighbors = append(copies[node].Neighbors, dfs(neighbor, copies))
		}
	}
	return copies[node]
}
```

python

```python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        copies = {}
        return self.dfs(node, copies)

    def dfs(self, node: 'Node', copies: dict) -> 'Node':
        if node not in copies:
            copies[node] = Node(node.val, [])
            for neighbor in node.neighbors:
                copies[node].neighbors += self.dfs(neighbor, copies),
        return copies[node]

```

java

```java
class Solution {  
    public Node cloneGraph(Node node) {  
        if (node == null) {  
            return null;  
        }  
        HashMap<Node, Node> copies = new HashMap<>();  
        return dfs(node, copies);  
    }  
  
    Node dfs(Node node, HashMap<Node, Node> copies) {  
        if (!copies.containsKey(node)) {  
            copies.put(node, new Node(node.val));  
            for (Node neighbor : node.neighbors) {  
                copies.get(node).neighbors.add(dfs(neighbor, copies));  
            }  
        }  
        return copies.get(node);  
    }  
}
```

#### 200. Number of Islands

go

```go
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	res := 0
	for i := range grid {
		for j := range grid[0] {
			if grid[i][j] == '0' || visited[i][j] {
				continue
			}
			res++
			dfs(i, j, m, n, grid, visited)
		}
	}
	return res
}
func dfs(i, j, m, n int, grid [][]byte, visited [][]bool) {
	visited[i][j] = true
	dirs := [4][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	for _, dir := range dirs {
		x := i + dir[0]
		y := j + dir[1]
		if x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == '0' || visited[x][y] {
			continue
		}
		dfs(x, y, m, n, grid, visited)
	}
}
```

rust

```rust
impl Solution {
    pub fn num_islands(grid: Vec<Vec<char>>) -> i32 {
        let m = grid.len();
        let n = grid[0].len();
        let mut res = 0;
        let mut visited = vec![vec![false; n]; m];
        for i in 0..m {
            for j in 0..n {
                if grid[i][j] == '0' || visited[i][j] {
                    continue;
                }
                res += 1;
                Solution::dfs(&grid, &mut visited, i, j, m, n)
            }
        }
        res
    }
    fn dfs(grid: &[Vec<char>], visited: &mut [Vec<bool>], i: usize, j: usize, m: usize, n: usize) {
        visited[i][j] = true;
        let dirs = [[-1, 0], [0, -1], [0, 1], [1, 0]];
        for dir in dirs {
            let x = i as i32 + dir[0];
            let y = j as i32 + dir[1];
            if x < 0 || y < 0 {
                continue;
            }
            let x = x as usize;
            let y = y as usize;
            if x >= m || y >= n || grid[x][y] == '0' || visited[x][y] {
                continue;
            }
            Solution::dfs(grid, visited, x, y, m, n)
        }
    }
}
```

#### 207. Course Schedule

有向图中dfs找环

判断DAG

每次当前层将该节点涂成unsafe然后递归查找邻居

java

```java
class Solution {
    private static final int safe = 1;
    private static final int unsafe = 2;

    public boolean canFinish(int n, int[][] prerequisites) {
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++)
            graph.add(new ArrayList<>());
        for (int[] pre : prerequisites)
            graph.get(pre[0]).add(pre[1]);
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            if (cyclic(graph, dp, i)) return false;
        }
        return true;
    }

    boolean cyclic(ArrayList<ArrayList<Integer>> graph, int[] dp, int i) {
        if (dp[i] > 0) return dp[i] == unsafe;
        dp[i] = unsafe;
        for (Integer v : graph.get(i)) {
            if (cyclic(graph, dp, v)) return true;
        }
        dp[i] = safe;
        return false;
    }
}
```

#### 210. Course Schedule II

java

```java
public class Solution {
    private static final int safe = 1;
    private static final int unsafe = 2;

    public int[] findOrder(int n, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++)
            graph.add(new ArrayList<>());
        for (int[] pre : prerequisites) {
            graph.get(pre[0]).add(pre[1]);
        }
        int[] dp = new int[n];

        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++)
            if (cyclic(res, graph, dp, i)) return new int[0];
        return res.stream().mapToInt(i -> i).toArray();
    }

    private boolean cyclic(List<Integer> path, List<List<Integer>> graph, int[] dp, int i) {
        if (dp[i] > 0) return dp[i] == unsafe;
        dp[i] = unsafe;
        for (int j : graph.get(i))
            if (cyclic(path, graph, dp, j)) return true;
        dp[i] = safe;
        path.add(i);
        return false;
    }
}
```

#### 310. Minimum Height Trees

在无向无环图中找到图的中点

java

```java
class Solution {  
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {  
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();  
        for (int i = 0; i < n; i++) {  
            graph.add(new ArrayList<>());  
        }  
        for (int[] edge : edges) {  
            graph.get(edge[0]).add(edge[1]);  
            graph.get(edge[1]).add(edge[0]);  
        }  
        int[] parent = new int[n];  
        int farthest = findFarthest(0, n, graph, parent);  
        int end = findFarthest(farthest, n, graph, parent);  
  
        ArrayList<Integer> path = new ArrayList<>();  
        while (end != -1) {  
            path.add(end);  
            end = parent[end];  
        }  
        Integer a = path.get(path.size() / 2);  
        Integer b = path.get((path.size() - 1) / 2);  
        if (a.equals(b)) {  
            return List.of(a);  
        }  
        return List.of(a, b);  
    }  
  
    int findFarthest(int start, int n, ArrayList<ArrayList<Integer>> graph, int[] parent) {  
        int[] distance = new int[n];  
        Arrays.fill(distance, -1);  
        Arrays.fill(parent, -1);  
        distance[start] = 0;  
        dfs(start, distance, parent, graph);  
        int res = 0, maxDis = 0;  
        for (int i = 0; i < distance.length; i++) {  
            if (distance[i] > maxDis) {  
                maxDis = distance[i];  
                res = i;  
            }  
        }  
        return res;  
    }  
  
    void dfs(int start, int[] distance, int[] parent, ArrayList<ArrayList<Integer>> graph) {  
        for (Integer neighbor : graph.get(start)) {  
            if (distance[neighbor] >= 0) continue;  
            distance[neighbor] = distance[start] + 1;  
            parent[neighbor] = start;  
            dfs(neighbor, distance, parent, graph);  
        }  
    }  
}
```

#### 329. Longest Increasing Path in a Matrix

记忆化搜索，命中缓存就直接返回，长度默认填1，DFS寻找四周的高点累加1

java

```java
class Solution {
    public int longestIncreasingPath(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                res = Math.max(res, dfs(i, j, m, n, dp, matrix));
            }
        }
        return res;
    }

    int dfs(int i, int j, int m, int n, int[][] dp, int[][] matrix) {
        if (dp[i][j] > 0) {
            return dp[i][j];
        }
        int[][] dirs = {{0, 1}, {1, 0}, {-1, 0}, {0, -1}};
        dp[i][j] = 1;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || matrix[x][y] <= matrix[i][j]) {
                continue;
            }
            dp[i][j] = Math.max(dp[i][j], dfs(x, y, m, n, dp, matrix) + 1);
        }
        return dp[i][j];
    }
}
```

go

```go
func longestIncreasingPath(matrix [][]int) int {  
   m, n := len(matrix), len(matrix[0])  
   res := 0  
   dp := make([][]int, m)  
   for i := range dp {  
      dp[i] = make([]int, n)  
   }  
   for i := 0; i < m; i++ {  
      for j := 0; j < n; j++ {  
         res = max(res, dfs(i, j, m, n, matrix, dp))  
      }  
   }  
   return res  
}  
func dfs(i, j, m, n int, matrix, dp [][]int) int {  
   if dp[i][j] > 0 {  
      return dp[i][j]  
   }  
   dirs := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}  
   dp[i][j] = 1  
   for _, dir := range dirs {  
      x := i + dir[0]  
      y := j + dir[1]  
      if x < 0 || y < 0 || x >= m || y >= n || matrix[x][y] <= matrix[i][j] {  
         continue  
      }  
      dp[i][j] = max(dp[i][j], dfs(x, y, m, n, matrix, dp)+1)  
   }  
   return dp[i][j]  
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
use std::cmp::max;  
  
impl Solution {  
    pub fn longest_increasing_path(matrix: Vec<Vec<i32>>) -> i32 {  
        let m = matrix.len();  
        let n = matrix[0].len();  
        let mut dp = vec![vec![0; n]; m];  
        let mut res = 0;  
        for i in 0..m {  
            for j in 0..n {  
                res = max(res, dfs(i, j, &mut dp, &matrix, m, n))  
            }  
        }  
  
        fn dfs(i: usize, j: usize, dp: &mut Vec<Vec<i32>>, matrix: &Vec<Vec<i32>>, m: usize, n: usize) -> i32 {  
            if dp[i][j] > 0 {  
                return dp[i][j];  
            }  
            let dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];  
            dp[i][j] = 1;  
            for dir in dirs {  
                let x = i as i32 + dir[0];  
                let y = j as i32 + dir[1];  
                if x < 0 || y < 0 {  
                    continue;  
                }  
                let x = x as usize;  
                let y = y as usize;  
                if x >= m || y >= n || matrix[x][y] <= matrix[i][j] {  
                    continue;  
                }  
                dp[i][j] = max(dp[i][j], dfs(x, y, dp, matrix, m, n) + 1);  
            }  
            dp[i][j]  
        }  
        res  
    }  
}
```

#### 694.Number of Distinct Islands

testing [https://www.lintcode.com/problem/860/description](https://www.lintcode.com/problem/860/description)

java

```java
public class Solution {
    public int numberofDistinctIslands(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        Set<ArrayList<Integer>> res = new HashSet<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0 || visited[i][j]) continue;
                ArrayList<Integer> path = new ArrayList<>();
                dfs(i, j, grid, path, visited, i, j, m, n);
                res.add(path);
            }
        }
        return res.size();
    }

    void dfs(int i, int j, int[][] grid, ArrayList<Integer> path, boolean[][] visited, int startI, int startJ, int m, int n) {
        visited[i][j] = true;
        path.add(i - startI);
        path.add(j - startJ);
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] dir : dirs) {
            int x = i + dir[0], y = j + dir[1];
            if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y] || grid[x][y] == 0) continue;
            dfs(x, y, grid, path, visited, startI, startJ, m, n);
        }
    }
}
```

#### 695. Max Area of Island

java

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 || visited[i][j]) continue;
                res = Math.max(res, dfs(grid, visited, m, n, i, j));
            }
        }
        return res;
    }

    int dfs(int[][] grid, boolean[][] visited, int m, int n, int i, int j) {
        visited[i][j] = true;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 1;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || visited[x][y] || grid[x][y] == 0) continue;
            res += dfs(grid, visited, m, n, x, y);
        }
        return res;
    }
}
```

#### 721. Accounts Merge

java

```java
class Solution {
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        int n = accounts.size();
        HashMap<String, Integer> mailToIndex = new HashMap<>();
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < accounts.get(i).size(); j++) {
                String curr = accounts.get(i).get(j);
                if (mailToIndex.containsKey(curr)) {
                    Integer prev = mailToIndex.get(curr);
                    graph.get(prev).add(i);
                    graph.get(i).add(prev);
                } else mailToIndex.put(curr, i);
            }
        }
        HashMap<Integer, HashSet<String>> indexToMail = new HashMap<>();
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (visited[i]) continue;
            HashSet<String> path = new HashSet<>();
            DFS(i, graph, path, visited, accounts);
            indexToMail.put(i, path);
        }
        List<List<String>> res = new ArrayList<>();
        for (int i : indexToMail.keySet()) {
            ArrayList<String> path = new ArrayList<>();
            path.add(accounts.get(i).get(0));
            path.addAll(indexToMail.get(i).stream().sorted().toList());
            res.add(path);
        }
        return res;
    }

    void DFS(Integer i, ArrayList<ArrayList<Integer>> graph, HashSet<String> path, boolean[] visited, List<List<String>> accounts) {
        visited[i] = true;
        path.addAll(accounts.get(i).stream().skip(1).toList());
        for (int neighbor : graph.get(i)) {
            if (visited[neighbor]) continue;
            DFS(neighbor, graph, path, visited, accounts);
        }
    }
}
```

#### 785. Is Graph Bipartite?

验证二分图

java

```java
class Solution {
    public boolean isBipartite(int[][] graph) {
        int n = graph.length;
        int[] colors = new int[n];
        for (int i = 0; i < n; i++) {
            if (colors[i] != 0) continue;
            if (!dfs(graph, i, 1, colors)) return false;
        }
        return true;
    }

    boolean dfs(int[][] graph, int i, int color, int[] colors) {
        if (colors[i] != 0) return colors[i] == color;
        colors[i] = color;
        for (int j : graph[i]) {
            if (!dfs(graph, j, -color, colors)) return false;
        }
        return true;
    }
}
```

python

```python
class Solution:  
    def isBipartite(self, graph: List[List[int]]) -> bool:  
        n = len(graph)  
        dp = [0] * n  
        for i in range(n):  
            if not dp[i] and not self.dfs(1, i, dp, graph):  
                return False  
        return True  
    def dfs(self, color: int, i: int, dp: List[int], graph: List[List[int]]) -> bool:  
        if dp[i]:  
            return dp[i] == color  
        dp[i] = color  
        return all(self.dfs(-color, j, dp, graph) for j in graph[i])
```

#### 802. Find Eventual Safe States

标记为unsafe之后递归查找邻居，如果查找到unsafe，就说明成环了，自身也是unsafe，直接return，否则说明自己safe

java

```java
class Solution {
    private static final int safe = 1;
    private static final int unsafe = 2;

    public List<Integer> eventualSafeNodes(int[][] graph) {
        int n = graph.length;
        int[] dp = new int[n];

        for (int i = 0; i < n; i++) {
            dfs(i, dp, graph);
        }
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (dp[i] == safe) res.add(i);
        }
        return res;
    }

    boolean dfs(int i, int[] color, int[][] graph) {
        if (color[i] > 0) return color[i] == safe;
        color[i] = unsafe;
        for (int j : graph[i]) {
            if (!dfs(j, color, graph)) return false;
        }
        color[i] = safe;
        return true;
    }
}
```

go

```go
const safe = 1
const unsafe = 2

func eventualSafeNodes(graph [][]int) []int {
	n := len(graph)
	color := make([]int, n)
	for i := range graph {
		dfs(graph, i, color)
	}
	var res []int
	for i, v := range color {
		if v == safe {
			res = append(res, i)
		}
	}
	return res
}
func dfs(graph [][]int, i int, color []int) bool {
	if color[i] > 0 {
		return color[i] == safe
	}
	color[i] = unsafe
	for _, j := range graph[i] {
		if !dfs(graph, j, color) {
			return false
		}
	}
	color[i] = safe
	return true
}
```

rust

```rust
impl Solution {
    const SAFE: i32 = 1;
    const UNSAFE: i32 = 2;
    pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
        let n = graph.len();
        let mut color = vec![0; n];

        fn dfs(graph: &Vec<Vec<i32>>, color: &mut Vec<i32>, i: usize) -> bool {
            if color[i] > 0 {
                return color[i] == Solution::SAFE;
            }
            color[i] = Solution::UNSAFE;
            for j in &graph[i] {
                if !dfs(graph, color, *j as usize) {
                    return false;
                }
            }
            color[i] = Solution::SAFE;
            true
        }

        for i in 0..n {
            dfs(&graph, &mut color, i);
        }
        color.iter().enumerate().filter(|(_, &v)| v == Solution::SAFE).map(|(i, _)| i as i32).collect::<Vec<i32>>()
    }
}
```

#### 827. Making A Large Island

java

```java
class Solution {
    public int largestIsland(int[][] grid) {
        HashMap<Integer, Integer> area = new HashMap<>();
        int mark = 2, res = 0, n = grid.length;
        int[][] cp = Arrays.copyOf(grid, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (cp[i][j] == 1) {
                    area.put(mark, dfs(cp, i, j, n, mark));
                    res = Math.max(res, area.get(mark++));
                }
            }
        }
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (cp[i][j] == 0) {
                    ArrayList<Integer> seen = new ArrayList<>();
                    int curr = 1;
                    for (int[] dir : dirs) {
                        int x = i + dir[0], y = j + dir[1];
                        if (out(x, y, n) || cp[x][y] == 0 || seen.contains(cp[x][y])) continue;
                        seen.add(cp[x][y]);
                        curr += area.get(cp[x][y]);
                    }
                    res = Math.max(res, curr);
                }
            }
        }
        return res;
    }

    int dfs(int[][] grid, int i, int j, int n, int mark) {
        grid[i][j] = mark;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 1;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (out(x, y, n) || grid[x][y] != 1) continue;
            res += dfs(grid, x, y, n, mark);
        }
        return res;
    }

    boolean out(int x, int y, int n) {
        return 0 > x || x >= n || 0 > y || y >= n;
    }
}
```

#### 886. Possible Bipartition

java

```java
class Solution {

    public boolean possibleBipartition(int n, int[][] dislikes) {
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n + 1; i++) graph.add(new ArrayList<>());
        for (int[] dislike : dislikes) {
            int u = dislike[0], v = dislike[1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
        int[] colors = new int[n + 1];
        for (int i = 1; i < n + 1; i++) {
            if (colors[i] != 0) {
                continue;
            }
            if (!dfs(i, 1, colors, graph)) {
                return false;
            }
        }
        return true;
    }

    boolean dfs(int u, int target, int[] colors, ArrayList<ArrayList<Integer>> graph) {
        if (colors[u] != 0) {
            return colors[u] == target;
        }
        colors[u] = target;
        for (Integer v : graph.get(u)) {
            if (!dfs(v, -target, colors, graph)) {
                return false;
            }
        }
        return true;
    }
}
```

#### 980. Unique Paths III

java

```java
class Solution {
    int empty;

    public int uniquePathsIII(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int startI = 0, startJ = 0;
        empty = 1; // starting point
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    empty++;
                    continue;
                }
                if (grid[i][j] == 1) {
                    startI = i;
                    startJ = j;
                }
            }
        }
        return dfs(grid, startI, startJ, m, n, new boolean[m][n]);
    }

    int dfs(int[][] grid, int i, int j, int m, int n, boolean[][] visited) {
        if (grid[i][j] == 2) {
            return empty == 0 ? 1 : 0;
        }
        visited[i][j] = true;
        empty--;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 0;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == -1 || visited[x][y]) continue;
            res += dfs(grid, x, y, m, n, visited);
        }
        visited[i][j] = false;
        empty++;
        return res;
    }
}
```

compress

```java
class Solution {
    int empty;

    public int uniquePathsIII(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int startI = 0, startJ = 0;
        empty = 1; // starting point
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    empty++;
                    continue;
                }
                if (grid[i][j] == 1) {
                    startI = i;
                    startJ = j;
                }
            }
        }
        return dfs(grid, startI, startJ, m, n, 0);
    }

    int dfs(int[][] grid, int i, int j, int m, int n, int visited) {
        if (grid[i][j] == 2) {
            return empty == 0 ? 1 : 0;
        }
        visited |= 1 << i * n + j;
        empty--;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 0;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == -1 || (visited & 1 << x * n + y) != 0) continue;
            res += dfs(grid, x, y, m, n, visited);
        }
        empty++;
        return res;
    }
}
```

#### 1020. Number of Enclaves

java

```java
class Solution {
    public int numEnclaves(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if ((i == 0 || j == 0 || i == m - 1 || j == n - 1) && grid[i][j] == 1) {
                    dfs(i, j, m, n, visited, grid);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 || visited[i][j]) continue;
                res++;
            }
        }
        return res;
    }

    void dfs(int i, int j, int m, int n, boolean[][] visited, int[][] grid) {
        visited[i][j] = true;
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        for (int[] dir : directions) {
            int x = dir[0] + i;
            int y = dir[1] + j;
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0 || visited[x][y]) continue;
            dfs(x, y, m, n, visited, grid);
            visited[x][y] = true;
        }
    }
}
```

#### 1254. Number of Closed Islands

java

```java
class Solution {
    public int closedIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) continue;
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    dfs(i, j, m, n, visited, grid);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 || visited[i][j]) continue;
                res++;
                dfs(i, j, m, n, visited, grid);
            }
        }
        return res;
    }

    void dfs(int i, int j, int m, int n, boolean[][] visited, int[][] grid) {
        visited[i][j] = true;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 1 || visited[x][y]) continue;
            dfs(x, y, m, n, visited, grid);
        }
    }
}
```

#### 36. Valid Sudoku

```go
func isValidSudoku(board [][]byte) bool {
	var (
		row [9]int
		col [9]int
		box [9]int
	)
	for i := range board {
		for j := range board[0] {
			if board[i][j] == '.' {
				continue
			}
			idx := 1 << (board[i][j] - '0')
			if row[i]&idx > 0 ||
				col[j]&idx > 0 ||
				box[i/3*3+j/3]&idx > 0 {
				return false
			}
			row[i] |= idx
			col[j] |= idx
			box[(i/3)*3+j/3] |= idx
		}
	}
	return true
}
```

#### 37. Sudoku Solver

 / 3  X   3 可以将坐标映射到3 * 3九宫格的左上角

```go
func solveSudoku(board [][]byte) {
	dfs(board)
}
func dfs(board [][]byte) bool {
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] != '.' {
				continue
			}
			for k := '1'; k <= '9'; k++ {
				if valid(i, j, byte(k), board) {
					board[i][j] = byte(k)
					if dfs(board) {
						return true
					}
					board[i][j] = '.'
				}
			}
			return false
		}
	}
	return true
}
func valid(row, col int, k byte, board [][]byte) bool {
	for i := 0; i < 9; i++ {
		if board[row][i] == k {
			return false
		}
	}
	for i := 0; i < 9; i++ {
		if board[i][col] == k {
			return false
		}
	}
	nr := row / 3 * 3
	nc := col / 3 * 3
	for i := nr; i < nr+3; i++ {
		for j := nc; j < nc+3; j++ {
			if board[i][j] == k {
				return false
			}
		}
	}
	return true
}
```

