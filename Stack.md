#### 341. Flatten Nested List Iterator

java

```java
public class NestedIterator implements Iterator<Integer> {
    private final ArrayDeque<NestedInteger> stack;

    public NestedIterator(List<NestedInteger> nestedList) {
        stack = new ArrayDeque<>(nestedList);
    }

    public Integer next() {
        if (hasNext()) {
            return stack.pop().getInteger();
        }
        return null;
    }

    public boolean hasNext() {
        while (!stack.isEmpty() && !stack.peek().isInteger()) {
            List<NestedInteger> list = stack.pop().getList();
            Collections.reverse(list);
            list.forEach(stack::push);
        }
        return !stack.isEmpty();
    }
}
```

#### 20. Valid Parentheses

go

```go
func isValid(s string) bool {
	stack := make([]rune, len(s))
	top := 0
	for _, p := range s {
		switch p {
		case '[':
			stack[top] = ']'
			top++
		case '{':
			stack[top] = '}'
			top++
		case '(':
			stack[top] = ')'
			top++
		default:
			if top == 0 || p != stack[top-1] {
				return false
			}
			top--
		}
	}
	return top == 0
}
```

rust

```rust
impl Solution {  
    pub fn is_valid(s: String) -> bool {  
        let mut stack = vec![];  
        for i in s.chars() {  
            match i {  
                '(' => stack.push(')'),  
                '[' => stack.push(']'),  
                '{' => stack.push('}'),  
                '}' | ']' | ')' if Some(i) != stack.pop() => return false,  
                _ => {}  
            }  
        }  
        stack.is_empty()  
    }  
}
```

java

```java
class Solution {  
    public boolean isValid(String s) {  
        ArrayDeque<Character> stack = new ArrayDeque<>();  
        for (char c : s.toCharArray()) {  
            switch (c) {  
                case '(' -> stack.push(')');  
                case '[' -> stack.push(']');  
                case '{' -> stack.push('}');  
                default -> {  
                    if (stack.isEmpty() || stack.pop() != c) {  
                        return false;  
                    }  
                }  
            }  
        }  
        return stack.isEmpty();  
    }  
}
```

#### Form Minimum Number

https://www.lintcode.com/problem/1890/

java

```java
public class Solution {  
    public String formMinimumNumber(String str) {  
        StringBuilder res = new StringBuilder();  
        int n = str.length();  
        ArrayDeque<Integer> stack = new ArrayDeque<>();  
        for (int i = 0; i < n + 1; i++) {  
            stack.push(i + 1);  
            if (i == n || str.charAt(i) == 'I') {  
                while (!stack.isEmpty()) res.append(stack.pop());  
            }  
        }  
        return res.toString();  
    }  
}
```

#### 32. Longest Valid Parentheses

go stack

```go
func longestValidParentheses(s string) int {  
   stack := make([]int, len(s)+1)  
   stack[0] = -1  
   top, longest := 1, 0  
   for i := range s {  
      if s[i] == ')' && top > 1 && s[stack[top-1]] == '(' {  
         top--  
         longest = max(longest, i-stack[top-1])  
      } else {  
         stack[top] = i  
         top++  
      }  
   }  
   return longest  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}
```

java stack

```java
class Solution {  
    public int longestValidParentheses(String s) {  
        Deque<Integer> stack = new ArrayDeque<>();  
        stack.push(-1);  
        int res = 0;  
        for (int i = 0; i < s.length(); i++) {  
            if (s.charAt(i) == ')' && stack.size() > 1 && s.charAt(stack.peek()) == '(') {  
                stack.pop();  
                res = Math.max(res, i - stack.peek());  
            } else {  
                stack.push(i);  
            }  
        }  
        return res;  
    }  
}
```

#### 71. Simplify Path

go

```go
func simplifyPath(path string) string {  
   s := strings.Split(path, "/")  
   var stack []string  
   for _, dep := range s {  
      switch dep {  
      case "", ".":  
         continue  
      case "..":  
         if len(stack) > 0 {  
            stack = stack[:len(stack)-1]  
         }  
      default:  
         stack = append(stack, dep)  
      }  
   }  
   return "/" + strings.Join(stack, "/")  
}
```

python

```python
class Solution:  
    def simplifyPath(self, path: str) -> str:  
        stack = []  
        for v in str.split(path, '/'):  
            if v == '..' and stack:  
                stack.pop()  
            elif v not in ('', '.', '..'):  
                stack += v,  
        return '/' + '/'.join(stack)
```

rust

```rust
impl Solution {
    pub fn simplify_path(path: String) -> String {
        let path = path
            .split("/")
            .fold(vec![], |mut acc, x| {
                match x {
                    "" | "." => {}
                    ".." => {
                        if !acc.is_empty() {
                            acc.pop();
                        }
                    }
                    _ => acc.push(x)
                }
                acc
            });
        "/".to_string() + &path.join("/")
    }
}
```

#### 150. Evaluate Reverse Polish Notation

go

```go
func evalRPN(tokens []string) int {  
   stack := make([]int, len(tokens))  
   top := 0  
   for _, token := range tokens {  
      v, err := strconv.Atoi(token)  
      if err == nil {  
         stack[top] = v  
         top++  
         continue  
      }  
      num1, num2 := stack[top-2], stack[top-1]  
      top -= 2  
      switch token {  
      case "+":  
         stack[top] = num1 + num2  
      case "-":  
         stack[top] = num1 - num2  
      case "*":  
         stack[top] = num1 * num2  
      case "/":  
         stack[top] = num1 / num2  
      }  
      top++  
   }  
   return stack[0]  
}
```

java

```java
class Solution {
    public int evalRPN(String[] tokens) {
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        for (String token : tokens) {
            switch (token) {
                case "+" -> stack.push(stack.pop() + stack.pop());
                case "-" -> stack.push(-stack.pop() + stack.pop());
                case "*" -> stack.push(stack.pop() * stack.pop());
                case "/" -> {
                    int b = stack.pop();
                    int a = stack.pop();
                    stack.push(a / b);
                }
                default -> stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }
}
```

rust

```rust
impl Solution {
    pub fn eval_rpn(tokens: Vec<String>) -> i32 {
        tokens
            .iter()
            .fold(vec![], |mut acc, x| {
                if let Ok(v) = str::parse::<i32>(x) {
                    acc.push(v);
                } else {
                    let b = acc.pop().unwrap();
                    let a = acc.pop().unwrap();
                    match x.as_str() {
                        "+" => acc.push(a + b),
                        "-" => acc.push(a - b),
                        "*" => acc.push(a * b),
                        "/" => acc.push(a / b),
                        _ => {}
                    }
                }
                acc
            })
            [0]
    }
}
```

#### 155. Min Stack

java

```java
class MinStack {
    ArrayDeque<Integer> stack;
    ArrayDeque<Integer> preMin;

    MinStack() {
        stack = new ArrayDeque<>();
        preMin = new ArrayDeque<>();
    }

    public void push(int val) {
        stack.push(val);
        if (preMin.isEmpty()) preMin.push(val);
        else preMin.push(Math.min(val, preMin.peek()));
    }

    public void pop() {
        stack.pop();
        preMin.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return preMin.peek();
    }
}
```
#### 224.Basic Calculator

go

```go
func calculate(s string) int {
	res, number, sign := 0, 0, 1
	var stack [][2]int
	s += "+"
	for i := range s {
		if s[i] >= '0' && s[i] <= '9' {
			number = 10*number + int(s[i]-'0')
			continue
		}
		res += sign * number
		number = 0
		switch s[i] {
		case '(':
			stack = append(stack, [2]int{res, sign})
			res, sign = 0, 1
		case '+':
			sign = 1
		case '-':
			sign = -1
		case ')':
			prev := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res *= prev[1]
			res += prev[0]
		}
	}
	return res
}
```

java

```java
class Solution {
    public int calculate(String s) {
        ArrayDeque<int[]> stack = new ArrayDeque<>();
        int result = 0, number = 0, sign = 1;
        s += "+";
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                number = 10 * number + (c - '0');
                continue;
            }
            result += sign * number;
            number = 0;
            switch (c) {
                case '+', '-' -> sign = c == '+' ? 1 : -1;
                case '(' -> {
                    stack.push(new int[]{result, sign});
                    result = 0;
                    sign = 1;
                }
                case ')' -> {
                    int[] pop = stack.pop();
                    result *= pop[1];
                    result += pop[0];
                }
            }
        }
        return result;
    }
}
```

rust

```rust
impl Solution {
    pub fn calculate(s: String) -> i32 {
        let mut res = 0;
        let mut number = 0;
        let mut sign = 1;
        let mut stack = vec![];
        let s = s + "+";
        for char in s.chars() {
            if char.is_numeric() {
                number = number * 10 + (char as u8 - b'0') as i32;
                continue;
            }
            res += number * sign;
            number = 0;
            match char {
                '(' => {
                    stack.push([res, sign]);
                    res = 0;
                    sign = 1;
                }
                '+' => sign = 1,
                '-' => sign = -1,
                ')' => {
                    if let Some(v) = stack.pop() {
                        res *= v[1];
                        res += v[0];
                    }
                }
                _ => continue,
            }
        }
        res
    }
}
```

#### 225. Implement Stack using Queues

https://leetcode.com/problems/implement-stack-using-queues/discuss/62516/Concise-1-Queue-Java-C%2B%2B-Python

go

```go
type MyStack struct {
	queue *list.List
}

func Constructor() MyStack {
	return MyStack{list.New()}
}

func (ms *MyStack) Push(x int) {
	ms.queue.PushBack(x)
	for i := 1; i < ms.queue.Len(); i++ {
		ms.queue.PushBack(ms.queue.Remove(ms.queue.Front()))
	}
}

func (ms *MyStack) Pop() int {
	return ms.queue.Remove(ms.queue.Front()).(int)
}

func (ms *MyStack) Top() int {
	return ms.queue.Front().Value.(int)
}

func (ms *MyStack) Empty() bool {
	return ms.queue.Len() == 0
}
```

java

```java
class MyStack {  
    LinkedList<Integer> queue;  
  
    public MyStack() {  
        queue = new LinkedList<>();  
    }  
  
    public void push(int x) {  
        queue.add(x);  
        for (int i = 1; i < queue.size(); i++) {  
            queue.add(queue.remove());  
        }  
    }  
  
    public int pop() {  
        return queue.remove();  
    }  
  
    public int top() {  
        return queue.peek();  
    }  
  
    public boolean empty() {  
        return queue.isEmpty();  
    }  
}
```

rust

```rust
use std::collections::VecDeque;

#[derive(Default)]
struct MyStack {
    queue: VecDeque<i32>,
}

impl MyStack {
    fn new() -> Self {
        Default::default()
    }
    fn push(&mut self, x: i32) {
        self.queue.push_back(x);
        for _ in 1..self.queue.len() {
            if let Some(v) = self.queue.pop_front() {
                self.queue.push_back(v)
            }
        }
    }
    fn pop(&mut self) -> i32 {
        self.queue.pop_front().unwrap()
    }
    fn top(&self) -> i32 {
        self.queue[0]
    }
    fn empty(&self) -> bool {
        self.queue.is_empty()
    }
}
```

####  227. Basic Calculator II

java

```java
class Solution {
    public int calculate(String s) {
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        s += '+';
        char prev = '+';
        int num = 0;
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                num = num * 10 + c - '0';
                continue;
            }
            if (c == ' ') {
                continue;
            }
            switch (prev) {
                case '+' -> stack.push(num);
                case '-' -> stack.push(-num);
                case '*' -> stack.push(stack.pop() * num);
                case '/' -> stack.push(stack.pop() / num);
            }
            prev = c;
            num = 0;
        }
        return stack.stream().mapToInt(i -> i).sum();
    }
}
```

go

```go
func calculate(s string) int {  
   var (  
      num      int  
      operator byte = '+'  
   )  
   stack := make([]int, len(s))  
   top := 0  
   s += "+"  
   for i := range s {  
      if s[i] >= '0' && s[i] <= '9' {  
         num = (num * 10) + int(s[i]-'0')  
         continue  
      }  
      if s[i] == ' ' {  
         continue  
      }  
      switch operator {  
      case '+':  
         stack[top] = num  
      case '-':  
         stack[top] = -num  
      case '*':  
         prev := stack[top-1]  
         top--  
         stack[top] = prev * num  
      case '/':  
         prev := stack[top-1]  
         top--  
         stack[top] = prev / num  
      }  
      top++  
      operator = s[i]  
      num = 0  
   }  
   res := 0  
   for i := 0; i < top; i++ {  
      res += stack[i]  
   }  
   return res  
}
```
#### 232. Implement Queue using Stacks

go

```go
type MyQueue struct {
	in, out []int
}

func Constructor() MyQueue {
	return MyQueue{}
}

func (mq *MyQueue) Push(x int) {
	mq.in = append(mq.in, x)
}

func (mq *MyQueue) Pop() int {
	mq.Peek()
	val := mq.out[len(mq.out)-1]
	mq.out = mq.out[:len(mq.out)-1]
	return val
}

func (mq *MyQueue) Peek() int {
	if len(mq.out) == 0 {
		for len(mq.in) > 0 {
			n := len(mq.in) - 1
			mq.out = append(mq.out, mq.in[n])
			mq.in = mq.in[:n]
		}
	}
	return mq.out[len(mq.out)-1]
}

func (mq *MyQueue) Empty() bool {
	return len(mq.in) == 0 && len(mq.out) == 0
}
```

python

```python
class MyQueue:  
  
    def __init__(self):  
        self.stack_in = deque()  
        self.stack_out = deque()  
  
    def push(self, x: int) -> None:  
        self.stack_in.append(x)  
  
    def pop(self) -> int:  
        self.peek()  
        return self.stack_out.pop()  
  
    def peek(self) -> int:  
        if not self.stack_out:  
            while self.stack_in:  
                self.stack_out.append(self.stack_in.pop())  
        return self.stack_out[-1]  
  
    def empty(self) -> bool:  
        return not self.stack_out and not self.stack_in
```

java

```java
class MyQueue {  
    ArrayDeque<Integer> in;  
    ArrayDeque<Integer> out;  
  
    public MyQueue() {  
        in = new ArrayDeque<>();  
        out = new ArrayDeque<>();  
    }  
  
    public void push(int x) {  
        in.push(x);  
    }  
  
    public int pop() {  
        int res = peek();  
        return out.pop();  
    }  
  
    public int peek() {  
        if (out.isEmpty()) {  
            while (!in.isEmpty()) {  
                out.push(in.pop());  
            }  
        }  
        return out.peek();  
    }  
  
    public boolean empty() {  
        return in.isEmpty() && out.isEmpty();  
    }  
}
```

#### 234. Palindrome Linked List

Go

```go
func isPalindrome(head *ListNode) bool {
    var stack []int
    for curr := head; curr != nil; curr = curr.Next {
        stack = append(stack, curr.Val)
    }
    for curr := head; curr != nil && len(stack) > 0; curr = curr.Next {
        if curr.Val != stack[len(stack)-1] {
            return false
        }
        stack = stack[:len(stack)-1]
    }
    return true
}
```

#### 394. Decode String

go

```go
func decodeString(s string) string {  
   stack := list.New()  
   count, curr := 0, ""  
   for i := range s {  
      if '0' <= s[i] && s[i] <= '9' {  
         count = count*10 + int(s[i]-'0')  
         continue  
      }  
      switch s[i] {  
      case '[':  
         stack.PushBack(count)  
         stack.PushBack(curr)  
         count, curr = 0, ""  
      case ']':  
         prevStr := stack.Remove(stack.Back()).(string)  
         prevCount := stack.Remove(stack.Back()).(int)  
         prevStr += strings.Repeat(curr, prevCount)  
         curr = prevStr  
      default:  
         curr += string(s[i])  
      }  
   }  
   return curr  
}
```

java

```java
class Solution {
    public String decodeString(String s) {
        Deque<Integer> countStack = new ArrayDeque<>();
        Deque<StringBuilder> stringStack = new ArrayDeque<>();
        StringBuilder curr = new StringBuilder();
        int count = 0;
        for (char ch : s.toCharArray()) {
            if (Character.isDigit(ch)) {
                count = count * 10 + ch - '0';
            } else if (ch == '[') {
                countStack.push(count);
                stringStack.push(curr);
                curr = new StringBuilder();
                count = 0;
            } else if (ch == ']') {
                StringBuilder prevStr = stringStack.pop();
                int repeatCount = countStack.pop();
                prevStr.append(curr.toString().repeat(repeatCount));
                curr = prevStr;
            } else {
                curr.append(ch);
            }
        }
        return curr.toString();
    }
}
```

rust

```rust
impl Solution {
    pub fn decode_string(s: String) -> String {
        let mut stack = vec![];
        let mut curr = vec![];
        let mut count = 0;
        for x in s.chars() {
            match x {
                num if num.is_numeric() => count = count * 10 + (num as u8 - b'0') as usize,
                '[' => {
                    stack.push((curr.to_owned(), count));
                    curr = vec![];
                    count = 0
                }
                ']' => {
                    if let Some((mut prev, count)) = stack.pop() {
                        prev.extend(curr.repeat(count));
                        curr = prev
                    }
                }
                _ => curr.push(x)
            }
        }
        curr.iter().collect()
    }
}
```


#### 682. Baseball Game

go

```go
func calPoints(operations []string) int {  
   stack := make([]int, len(operations))  
   top := 0  
   for _, operation := range operations {  
      switch operation {  
      case "C":  
         top--  
      case "D":  
         last := stack[top-1]  
         stack[top] = last * 2  
         top++  
      case "+":  
         last1 := stack[top-1]  
         last2 := stack[top-2]  
         stack[top] = last1 + last2  
         top++  
      default:  
         stack[top], _ = strconv.Atoi(operation)  
         top++  
      }  
   }  
   res := 0  
   for i := 0; i < top; i++ {  
      res += stack[i]  
   }  
   return res  
}
```

java

```java
class Solution {  
    public int calPoints(String[] operations) {  
        int top = 0, n = operations.length;  
        int[] stack = new int[n];  
        for (String op : operations) {  
            switch (op) {  
                case "C" -> top--;  
                case "D" -> {  
                    stack[top] = 2 * stack[top - 1];  
                    top++;  
                }  
                case "+" -> {  
                    stack[top] = stack[top - 2] + stack[top - 1];  
                    top++;  
                }  
                default -> {  
                    stack[top] = Integer.parseInt(op);  
                    top++;  
                }  
            }  
        }  
        int res = 0;  
        for (int i = 0; i < top; i++) {  
            res += stack[i];  
        }  
        return res;  
    }  
}
```

rust

https://leetcode.com/problems/baseball-game/solutions/710195/rust-cheapest-best/

```rust
impl Solution {
    pub fn cal_points(ops: Vec<String>) -> i32 {
        ops.iter()
            .fold(vec![], |mut acc, curr| {
                match curr.as_str() {
                    "+" => acc.push(acc[acc.len() - 1] + acc[acc.len() - 2]),
                    "D" => acc.push(acc[acc.len() - 1] * 2),
                    "C" => { acc.pop(); }
                    x => acc.push(str::parse::<i32>(x).unwrap()),
                }
                acc
            })
            .iter()
            .sum()
    }
}
```
