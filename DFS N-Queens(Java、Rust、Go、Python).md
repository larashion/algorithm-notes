判断攻击时，利用三个整数的位掩码（Bitmask）分别记录列、左对角线、右对角线的限制。通过位运算（左移/右移）更新对角线限制，取反后即可得到当前行所有合法位置。

递归过程中，将皇后位置压入 `path`，进入下一行；回溯时弹出，恢复状态。

#### 51. N-Queens

python

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        full_mask = (1 << n) - 1

        # path 记录每一行皇后所在的列索引
        def backtrack(path: List[int],
                      col_mask: int, diag_left: int, diag_right: int):
            if col_mask == full_mask:
                res.append(['.' * i + 'Q' + '.' * (n - i - 1) for i in path])
                return

            available = full_mask & ~(col_mask | diag_left | diag_right)

            while available:
                low_bit = available & -available
                available ^= low_bit
                col_idx = low_bit.bit_length() - 1

                path.append(col_idx)
                backtrack(path,
                          col_mask | low_bit,
                          (diag_left | low_bit) << 1,
                          (diag_right | low_bit) >> 1)
                path.pop()

        backtrack([], 0, 0, 0)
        return res
```

go

```go
import (
	"math/bits"
	"strings"
)

func solveNQueens(n int) [][]string {
	var res [][]string
	fullMask := (1 << n) - 1
	path := make([]int, 0, n)
	
	var dfs func(colMask, diagLeft, diagRight int)
	dfs = func(colMask, diagLeft, diagRight int) {
		if colMask == fullMask {
			res = append(res, construct(path, n))
			return
		}
		
		available := fullMask & (^(colMask | diagLeft | diagRight))
		for available > 0 {
			lowBit := available & -available
			available ^= lowBit
			colIdx := bits.TrailingZeros(uint(lowBit))
			
			path = append(path, colIdx)
			dfs(colMask|lowBit, (diagLeft|lowBit)<<1, (diagRight|lowBit)>>1)
			path = path[:len(path)-1]
		}
	}
	dfs(0, 0, 0)
	return res
}

func construct(path []int, n int) []string {
	board := make([]string, n)
	for row, col := range path {
		var sb strings.Builder
		sb.Grow(n)
		sb.WriteString(strings.Repeat(".", col))
		sb.WriteByte('Q')
		sb.WriteString(strings.Repeat(".", n-1-col))
		board[row] = sb.String()
	}
	return board
}
```

rust

```rust
struct Solver {
    n: usize,
    full_mask: usize,
    res: Vec<Vec<String>>,
    path: Vec<usize>,
}

impl Solver {
    fn new(n: i32) -> Self {
        let n = n as usize;
        Solver {
            n,
            full_mask: (1 << n) - 1,
            res: vec![],
            path: Vec::with_capacity(n),
        }
    }

    fn dfs(&mut self, col_mask: usize, diag_left: usize, diag_right: usize) {
        if col_mask == self.full_mask {
            self.decode();
            return;
        }

        let mut available = self.full_mask & (!(col_mask | diag_left | diag_right));
        while available > 0 {
            let low_bit = available & available.wrapping_neg();
            let col_idx = low_bit.trailing_zeros() as usize;
            available ^= low_bit;

            self.path.push(col_idx);
            Self::dfs(
                self,
                col_mask | low_bit,
                (diag_left | low_bit) << 1,
                (diag_right | low_bit) >> 1,
            );
            self.path.pop();
        }
    }

    fn decode(&mut self) {
        let mut board = Vec::with_capacity(self.n);
        for &col in &self.path {
            let mut row_str = String::with_capacity(self.n);
            for c in 0..self.n {
                if c == col {
                    row_str.push('Q');
                } else {
                    row_str.push('.');
                }
            }
            board.push(row_str);
        }
        self.res.push(board);
    }
}

impl Solution {
    pub fn solve_n_queens(n: i32) -> Vec<Vec<String>> {
        let mut solver = Solver::new(n);
        Solver::dfs(&mut solver, 0, 0, 0);
        solver.res
    }
}
```

java

```java
class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        int fullMask = (1 << n) - 1;
        dfs(fullMask, n, res, 0, 0, 0, new ArrayList<>());
        return res;
    }

    private void dfs(int fullMask, int n, List<List<String>> res, int colMask, int diagLeft, int diagRight, List<Integer> path) {
        if (colMask == fullMask) {
            res.add(construct(path, n));
            return;
        }

        int available = fullMask & ~(colMask | diagLeft | diagRight);

        while (available > 0) {
            int lowBit = available & -available;
            available ^= lowBit;
            int colIdx = Integer.numberOfTrailingZeros(lowBit);

            path.add(colIdx);
            dfs(fullMask, n, res, colMask | lowBit, (diagLeft | lowBit) << 1, (diagRight | lowBit) >> 1, path);
            path.remove(path.size() - 1);
        }
    }

    private List<String> construct(List<Integer> path, int n) {
        List<String> board = new ArrayList<>(n);
        for (int col : path) {
            char[] row = new char[n];
            Arrays.fill(row, '.');
            row[col] = 'Q';
            board.add(new String(row));
        }
        return board;
    }
}
```

#### 52. N-Queens II

不关心路径的访问顺序，更关心是否访问，可以用bitmap保存当前访问的路径集合

求N皇后的方案总数，而不是具体方案，而且棋盘边长不会超过32，用二进制表示很合适，递归出口为column mask摆满时视为一种可行方案，返回1

rust

```rust
impl Solution {
    pub fn total_n_queens(n: i32) -> i32 {
        Self::dfs((1 << n) - 1, 0, 0, 0)
    }
    
    fn dfs(full_mask: i32, col_mask: i32, diag_left: i32, diag_right: i32) -> i32 {
        if col_mask == full_mask {
            return 1;
        }
        
        let mut available = full_mask & (!(col_mask | diag_left | diag_right));
        let mut count = 0;
        
        while available > 0 {
            let low_bit = available & -available;
            available ^= low_bit;
            
            count += Self::dfs(
                full_mask,
                col_mask | low_bit,
                (diag_left | low_bit) << 1,
                (diag_right | low_bit) >> 1,
            );
        }
        count
    }
}
```

go

```go
func totalNQueens(n int) int {
	fullMask := (1 << n) - 1
	return dfs(fullMask, 0, 0, 0)
}

func dfs(fullMask, colMask, diagLeft, diagRight int) int {
	if colMask == fullMask {
		return 1
	}

	available := fullMask & (^(colMask | diagLeft | diagRight))
	count := 0

	for available > 0 {
		lowBit := available & -available
		available ^= lowBit

		count += dfs(
			fullMask,
			colMask|lowBit,
			(diagLeft|lowBit)<<1,
			(diagRight|lowBit)>>1,
		)
	}
	return count
}
```

java

```java
class Solution {
    public int totalNQueens(int n) {
        return dfs((1 << n) - 1, 0, 0, 0);
    }

    private int dfs(int fullMask, int colMask, int diagLeft, int diagRight) {
        if (colMask == fullMask) {
            return 1;
        }

        int available = fullMask & (~(colMask | diagLeft | diagRight));
        int count = 0;

        while (available > 0) {
            int lowBit = available & -available;
            available ^= lowBit;

            count += dfs(
                fullMask,
                colMask | lowBit,
                (diagLeft | lowBit) << 1,
                (diagRight | lowBit) >> 1
            );
        }
        return count;
    }
}
```

python

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        return self.dfs((1 << n) - 1, 0, 0, 0)

    def dfs(self, full_mask: int, col_mask: int, diag_left: int, diag_right: int) -> int:
        if col_mask == full_mask:
            return 1
        
        available = full_mask & (~(col_mask | diag_left | diag_right))
        count = 0
        
        while available:
            low_bit = available & -available
            available ^= low_bit
            
            count += self.dfs(
                full_mask,
                col_mask | low_bit,
                (diag_left | low_bit) << 1,
                (diag_right | low_bit) >> 1
            )
        return count
```