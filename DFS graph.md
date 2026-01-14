#### [1219. Path with Maximum Gold](https://leetcode.com/problems/path-with-maximum-gold/)

go

```go
func getMaximumGold(grid [][]int) int {
	// 1. Optimization: Check for full grid (Hamiltonian Path)
	if total := checkFullGrid(grid); total != -1 {
		return total
	}
	// 2. General case: DFS from every cell
	return scanGrid(grid)
}

func checkFullGrid(grid [][]int) int {
	total := 0
	for _, row := range grid {
		for _, val := range row {
			if val == 0 {
				return -1
			}
			total += val
		}
	}
	return total
}

func scanGrid(grid [][]int) int {
	maxGold := 0
	m, n := len(grid), len(grid[0])
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] > 0 {
				maxGold = max(maxGold, dfs(grid, i, j))
			}
		}
	}
	return maxGold
}

func dfs(grid [][]int, x, y int) int {
	m, n := len(grid), len(grid[0])
	gold := grid[x][y]
	grid[x][y] = 0 // Mark visited
	
	maxPath := 0
	dirs := [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	
	for _, d := range dirs {
		nx, ny := x+d[0], y+d[1]
		// Pruning: Check validity before recursion
		if nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] > 0 {
			maxPath = max(maxPath, dfs(grid, nx, ny))
		}
	}
	
	grid[x][y] = gold // Backtrack
	return gold + maxPath
}
```

java

```java
class Solution {
    public int getMaximumGold(int[][] grid) {
        // 1. Optimization: Check for full grid (Hamiltonian Path)
        int total = fullGrid(grid);
        if (total != -1) return total;
        
        // 2. General case: DFS from every cell
        return scanGrid(grid);
    }

    private int scanGrid(int[][] grid) {
        int maxGold = 0;
        int m = grid.length, n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != 0) {
                    maxGold = Math.max(maxGold, dfs(grid, i, j));
                }
            }
        }
        return maxGold;
    }

    private int dfs(int[][] grid, int x, int y) {
        int m = grid.length, n = grid[0].length;
        int gold = grid[x][y];
        grid[x][y] = 0; // Mark visited
        int maxPath = 0;

        int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int[] dir : dirs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            // Pruning: Check validity before recursion
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] != 0) {
                maxPath = Math.max(maxPath, dfs(grid, nx, ny));
            }
        }
        grid[x][y] = gold; // Backtrack
        return gold + maxPath;
    }

    private int fullGrid(int[][] grid) {
        int total = 0;
        for (int[] row : grid) {
            for (int val : row) {
                if (val == 0) return -1;
                total += val;
            }
        }
        return total;
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
        let (m, n) = (grid.len() as i32, grid[0].len() as i32);
        let mut res = 0;
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        
        for i in 0..m {
            for j in 0..n {
                if grid[i as usize][j as usize] == 1 {
                    for (dx, dy) in dirs {
                        let x = i + dx;
                        let y = j + dy;
                        if x < 0 || x >= m || y < 0 || y >= n || grid[x as usize][y as usize] == 0 {
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