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