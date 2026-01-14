模板来自左神 https://github.com/algorithmzuo/algorithmbasic2020/blob/a2e2e76a3901889fc7b4747eca120663fecb1028/src/class15/Code01_FriendCircles.java

union find 可以解决：
- 连通块有几个（岛屿数量）
- 最大的连通块有多大（岛屿最大面积）
- 哪些是连通的

find函数用来找到某一坐标的rank，本模板中做了O(1)层面的优化，在i指针不断向上跳的过程中，用栈保留了滑动轨迹，在返回前，将中间节点的boss全部设置为i指针找到的最终boss，把结构拍平了

union函数使两个集团的boss决出胜负，group属性描述小团体数量，所以union成功后就减一

size数组只有当i为boss时，size[ i ] 才有效

合并时小挂大，减少修改次数

可以将递归改为手动压栈，修改find方法即可，需要初始化stack

```java
int find(int i) {  
    int j = 0;  
    for (; i != boss[i]; j++) {  
        stack[j] = i;  
        i = boss[i];  
    }  
    for (j--; j >= 0; j--) {  
        boss[stack[j]] = i;  
    }  
    return i;  
}
```

#### 1579. Remove Max Number of Edges to Keep Graph Fully Traversable

java

```java
class Solution {
    public int maxNumEdgesToRemove(int n, int[][] edges) {
        UnionFind uf = new UnionFind(n + 1);
        int res = 0, e1 = 0, e2 = 0;
        // Alice and Bob
        for (int[] edge : edges) {
            int i = edge[1];
            int j = edge[2];
            if (edge[0] == 3) {
                if (uf.union(i, j)) {
                    e1++;
                    e2++;
                } else {
                    res++;
                }
            }
        }
        // clone is important
        int[] clone = uf.boss.clone();
        // only Alice
        for (int[] edge : edges) {
            int i = edge[1];
            int j = edge[2];
            if (edge[0] == 1) {
                if (uf.union(i, j)) {
                    e1++;
                } else {
                    res++;
                }
            }
        }

        // only Bob
        // reset the boss array 
        uf.boss = clone;
        for (int[] edge : edges) {
            int i = edge[1];
            int j = edge[2];
            if (edge[0] == 2) {
                if (uf.union(i, j)) {
                    e2++;
                } else {
                    res++;
                }
            }
        }
        return e1 == n - 1 && e2 == n - 1 ? res : -1;
    }
}
class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i])
            boss[i] = find(boss[i]);
        return boss[i];
    }

    boolean union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return false;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
        return true;
    }
}
```

Go

```go
func maxNumEdgesToRemove(n int, edges [][]int) int {
	uf := constructor(n + 1)
	e1, e2, res := 0, 0, 0
	for _, edge := range edges {
		if edge[0] == 3 {
			i := edge[1]
			j := edge[2]
			if uf.union(i, j) {
				e1++
				e2++
			} else {
				res++
			}
		}
	}

	dup := make([]int, len(uf.boss))
	copy(dup, uf.boss)

	for _, edge := range edges {
		if edge[0] == 1 {
			i := edge[1]
			j := edge[2]
			if uf.union(i, j) {
				e1++
			} else {
				res++
			}
		}
	}

	uf.boss = dup

	for _, edge := range edges {
		if edge[0] == 2 {
			i := edge[1]
			j := edge[2]
			if uf.union(i, j) {
				e2++
			} else {
				res++
			}
		}
	}

	if e1 == e2 && e1 == n-1 {
		return res
	}
	return -1
}

type UnionFind struct {
	boss []int
	size []int
}

func constructor(n int) *UnionFind {
	boss := make([]int, n)
	for i := range boss {
		boss[i] = i
	}
	size := make([]int, n)
	for i := range size {
		size[i] = 1
	}
	return &UnionFind{boss, size}
}
func (uf UnionFind) find(i int) int {
	if i != uf.boss[i] {
		uf.boss[i] = uf.find(uf.boss[i])
	}
	return uf.boss[i]
}
func (uf UnionFind) union(i, j int) bool {
	f1 := uf.find(i)
	f2 := uf.find(j)
	if f1 == f2 {
		return false
	}
	if uf.size[f1] < uf.size[f2] {
		uf.boss[f1] = f2
		uf.size[f2] += uf.size[f1]
	} else {
		uf.boss[f2] = f1
		uf.size[f1] += uf.size[f2]
	}
	return true
}
```

rust

```rust
impl Solution {
    pub fn max_num_edges_to_remove(n: i32, edges: Vec<Vec<i32>>) -> i32 {
        let mut uf = UnionFind::new(n as usize + 1);
        let mut res = 0;
        let mut e1 = 0;
        let mut e2 = 0;
        for edge in &edges {
            if edge[0] == 3 {
                if uf.union(edge[1] as usize, edge[2] as usize) {
                    e1 += 1;
                    e2 += 1;
                } else {
                    res += 1;
                }
            }
        }
        let clone = uf.boss.clone();
        for edge in &edges {
            if edge[0] == 1 {
                if uf.union(edge[1] as usize, edge[2] as usize) {
                    e1 += 1;
                } else {
                    res += 1;
                }
            }
        }
        uf.boss = clone;
        for edge in &edges {
            if edge[0] == 2 {
                if uf.union(edge[1] as usize, edge[2] as usize) {
                    e2 += 1;
                } else {
                    res += 1;
                }
            }
        }
        if e1 == n - 1 && e2 == n - 1 {
            return res;
        }
        return -1;
    }
}

struct UnionFind {
    boss: Vec<usize>,
    size: Vec<usize>,
}
impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            boss: (0..n).collect(),
            size: vec![1; n],
        }
    }
    fn find(&mut self, i: usize) -> usize {
        if i != self.boss[i] {
            self.boss[i] = self.find(self.boss[i]);
        }
        self.boss[i]
    }
    fn union(&mut self, i: usize, j: usize) -> bool {
        let x = self.find(i);
        let y = self.find(j);
        if x == y {
            return false;
        }
        if self.size[x] < self.size[y] {
            self.boss[x] = y;
            self.size[y] += self.size[x];
        } else {
            self.boss[y] = x;
            self.size[x] += self.size[y];
        }
        true
    }
}
```
#### 399. Evaluate Division

java

```java
class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        UnionFind uf = new UnionFind();
        for (int i = 0; i < equations.size(); i++) {
            String u = equations.get(i).get(0);
            String v = equations.get(i).get(1);
            uf.union(u, v, values[i]);
        }

        double[] res = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            String s1 = queries.get(i).get(0), s2 = queries.get(i).get(1);
            if (uf.contains(s1) && uf.contains(s2) && uf.isConnected(s1, s2)) {
                res[i] = uf.ratio.get(s1) / uf.ratio.get(s2);
            } else {
                res[i] = -1.0;
            }
        }
        return res;
    }

    class UnionFind {
        private final HashMap<String, String> boss;
        final HashMap<String, Double> ratio;

        UnionFind() {
            boss = new HashMap<>();
            ratio = new HashMap<>();
        }

        void add(String s) {
            if (boss.containsKey(s)) {
                return;
            }
            boss.put(s, s);
            ratio.put(s, 1.0);
        }

        public void union(String root, String child, double val) {
            add(root);
            add(child);
            String f1 = find(root);
            String f2 = find(child);
            if (f1.equals(f2)) {
                return;
            }
            boss.put(f1, f2);
            ratio.put(f1, val * ratio.get(child) / ratio.get(root));
        }

        String find(String s) {
            String p = boss.get(s);
            if (!s.equals(p)) {
                boss.put(s, find(p));
                ratio.put(s, ratio.get(s) * ratio.get(p));
            }
            return boss.get(s);
        }

        boolean isConnected(String s1, String s2) {
            return find(s1).equals(find(s2));
        }

        boolean contains(String s) {
            return boss.containsKey(s);
        }
    }
}
```

#### 1361. Validate Binary Tree Nodes

java

```java
class Solution {
    public boolean validateBinaryTreeNodes(int n, int[] leftChild, int[] rightChild) {
        UnionFind uf = new UnionFind(n);
        for (int i = 0; i < n; i++) {
            if (leftChild[i] >= 0 && !uf.union(i, leftChild[i])) {
                return false;
            }
            if (rightChild[i] >= 0 && !uf.union(i, rightChild[i])) {
                return false;
            }
        }
        return uf.components() == 1;
    }
}

class UnionFind {
    private final int[] boss;
    private int components;

    UnionFind(int n) {
        boss = new int[n];
        for (int i = 0; i < n; i++) {
            boss[i] = i;
        }
        components = n;
    }

    public boolean union(int root, int child) {
        int f1 = find(root);
        int f2 = find(child);
        if (f1 == f2 || f2 != child) {
            return false;
        }
        boss[f2] = f1;
        components--;
        return true;
    }

    private int find(int i) {
        if (i != boss[i]) {
            boss[i] = find(boss[i]);
        }
        return boss[i];
    }

    public int components() {
        return components;
    }
}
```

#### 1061. Lexicographically Smallest Equivalent String

java

```java
class Solution {
    public String smallestEquivalentString(String s1, String s2, String baseStr) {
        UnionFind uf = new UnionFind(256);
        for (int i = 0; i < s1.length(); i++) {
            uf.union(s1.charAt(i), s2.charAt(i));
        }
        StringBuilder sb = new StringBuilder();
        baseStr.chars().map(uf::find).forEach(i -> sb.append((char) i));
        return sb.toString();
    }
}

class UnionFind {
    int[] boss;

    UnionFind(int n) {
        boss = new int[n];
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (f1 < f2) {
            boss[f2] = f1;
        } else {
            boss[f1] = f2;
        }
    }
}
```

#### 128. Longest Consecutive Sequence

把元素值和坐标存入哈希表，重复元素只需存一次坐标

连续的数字可以看作连通块，合并之后返回最大的联通块长度

https://leetcode.com/problems/longest-consecutive-sequence/solutions/166544/union-find-thinking-process/

java

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        int n = nums.length;
        UnionFind uf = new UnionFind(n);
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (map.containsKey(num)) continue;
            if (map.containsKey(num - 1)) uf.union(i, map.get(num - 1));
            if (map.containsKey(num + 1)) uf.union(i, map.get(num + 1));
            map.put(num, i);
        }
        return uf.largest();
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i])
            boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;

        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    int largest() {
        return Arrays.stream(size).max().orElse(0);
    }
}
```

#### 130. Surrounded Regions

java

```java
class Solution {
    public void solve(char[][] board) {
        int m = board.length, n = board[0].length;
        UnionFind uf = new UnionFind(m * n + 1);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O')
                    unionAround(board, i, j, m, n, uf);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'X' || uf.isConnected(i * n + j, m * n)) continue;
                board[i][j] = 'X';
            }
        }
    }

    void unionAround(char[][] board, int i, int j, int m, int n, UnionFind uf) {
        if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
            uf.union(i * n + j, m * n);
        }
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || board[x][y] == 'X') continue;
            uf.union(i * n + j, x * n + y);
        }
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;

        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }
}
```

#### 200. Number of Islands

输入01矩阵，1表示陆地，0表示水面，如何求出岛屿数量

go

```go
type UnionFind struct {
	boss []int
	size []int //represent the size of union whose boss is the node
	sets int
}

func construct(n int) *UnionFind {
	boss := make([]int, n)
	size := make([]int, n)
	for i := 0; i < n; i++ {
		boss[i] = i
		size[i] = 1
	}
	return &UnionFind{boss, size, n}
}
func (uf *UnionFind) find(i int) int {
	if i != uf.boss[i] {
		uf.boss[i] = uf.find(uf.boss[i])
	}
	return uf.boss[i]
}
func (uf *UnionFind) union(i, j int) {
	f1 := uf.find(i)
	f2 := uf.find(j)
	if f1 == f2 {
		return
	}
	uf.sets--
	if uf.size[f1] > uf.size[f2] {
		uf.size[f1] += uf.size[f2]
		uf.boss[f2] = f1
	} else {
		uf.size[f2] += uf.size[f1]
		uf.boss[f1] = f2
	}
}
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	uf := construct(m * n)
	count := 0
	for i := range grid {
		for j := range grid[0] {
			if grid[i][j] == '0' {
				count++
				continue
			}
			if i+1 < len(grid) && grid[i+1][j] == '1' {
				uf.union(i*n+j, (i+1)*n+j)
			}
			if j+1 < len(grid[0]) && grid[i][j+1] == '1' {
				uf.union(i*n+j, i*n+j+1)
			}
		}
	}
	return uf.sets - count
}
```

#### 261.Graph Valid Tree

https://www.lintcode.com/problem/178/

java

```java
public class Solution {
    public boolean validTree(int n, int[][] edges) {
        if (n != edges.length + 1) return false;
        UnionFind uf = new UnionFind(n);
        for (int[] edge : edges) {
            uf.union(edge[0], edge[1]);
        }
        return uf.group == 1;
    }
}

class UnionFind {
    int[] boss;
    int[] size;
    int group;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        group = n;
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

#### 305.Number of Islands II

testing https://www.lintcode.com/problem/434/description

java

```java
public class Solution {
    public List<Integer> numIslands2(int m, int n, Point[] operators) {
        int[][] grid = new int[m][n];
        UnionFind uf = new UnionFind(m * n);
        ArrayList<Integer> res = new ArrayList<>();
        for (Point point : operators) {
            int i = point.x, j = point.y;
            grid[i][j] = 1;
            uf.add(i * n + j);
            unionAround(grid, i, j, m, n, uf);
            res.add(uf.group);
        }
        return res;
    }

    void unionAround(int[][] grid, int i, int j, int m, int n, UnionFind uf) {
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] != 1) continue;
            uf.union(i * n + j, x * n + y);
        }
    }
}

class UnionFind {
    int[] boss;
    int[] size;
    int group;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(boss, -1);
    }

    void add(int i) {
        if (boss[i] != -1) return;
        group++;
        boss[i] = i;
        size[i] = 1;
    }

    int find(int i) {
        if (i != boss[i]) {
            boss[i] = find(boss[i]);
        }
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

#### 352. Data Stream as Disjoint Intervals

java

```java
class SummaryRanges {
    UnionFind uf;

    public SummaryRanges() {
        uf = new UnionFind();
    }

    public void addNum(int value) {
        uf.add(value);
        if (uf.contains(value - 1)) uf.union(value, value - 1);
        if (uf.contains(value + 1)) uf.union(value, value + 1);
    }

    public int[][] getIntervals() {
        List<int[]> res = uf.getIntervals();
        return res
                .stream()
                .sorted(Comparator.comparingInt(a -> a[0]))
                .toArray(int[][]::new);
    }
}

class UnionFind {
    HashMap<Integer, Integer> boss;
    HashMap<Integer, int[]> interval;
    HashMap<Integer, Integer> size;

    UnionFind() {
        boss = new HashMap<>();
        size = new HashMap<>();
        interval = new HashMap<>();
    }

    int find(int i) {
        if (i != boss.get(i)) {
            boss.put(i, find(boss.get(i)));
        }
        return boss.get(i);
    }

    void add(int i) {
        if (contains(i)) return;
        boss.put(i, i);
        interval.put(i, new int[]{i, i});
        size.put(i, 1);
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        int left = Math.min(interval.get(f1)[0], interval.get(f2)[0]);
        int right = Math.max(interval.get(f1)[1], interval.get(f2)[1]);
        if (size.get(f1) > size.get(f2)) {
            boss.put(f2, f1);
            size.put(f1, size.get(f1) + size.get(f2));
            interval.put(f1, new int[]{left, right});
        } else {
            boss.put(f1, f2);
            size.put(f2, size.get(f1) + size.get(f2));
            interval.put(f2, new int[]{left, right});
        }
    }

    boolean contains(int i) {
        return boss.containsKey(i);
    }

    List<int[]> getIntervals() {
        return boss.keySet().stream().filter(a -> boss.get(a) == a).map(interval::get).toList();
    }
}
```


#### 547. Number of Provinces

java

```java
class Solution {
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        UnionFind uf = new UnionFind(n);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (isConnected[i][j] == 1) uf.union(i, j);
            }
        }
        return uf.group;
    }
}

class UnionFind {
    int[] stack;
    int[] boss;
    int[] size;
    int group;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        stack = new int[n];
        group = n;
        Arrays.fill(size, 1); 
	    for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        int j = 0;
        for (; i != boss[i]; j++) {
            stack[j] = i;
            i = boss[i];
        }
        for (j--; j >= 0; j--) {
            boss[stack[j]] = i;
        }
        return i;
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

Go

```go
func findCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	uf := construct(n)
	for i := range isConnected {
		for j := i + 1; j < n; j++ {
			if isConnected[i][j] == 1 {
				uf.union(i, j)
			}
		}
	}
	return uf.group
}

type UnionFind struct {
	boss  []int
	size  []int //represent the size of union whose boss is the node
	stack []int
	group int
}

func construct(n int) *UnionFind {
	boss := make([]int, n)
	size := make([]int, n)
	for i := 0; i < n; i++ {
		boss[i] = i
		size[i] = 1
	}
	return &UnionFind{boss, size, make([]int, n), n}
}
func (uf *UnionFind) find(i int) int {
	j := 0
	for ; i != uf.boss[i]; j++ {
		uf.stack[j] = i
		i = uf.boss[i]
	}
	for j--; j >= 0; j-- {
		uf.boss[uf.stack[j]] = i
	}
	return i
}
func (uf *UnionFind) union(i, j int) {
	f1 := uf.find(i)
	f2 := uf.find(j)
	if f1 == f2 {
		return
	}
	uf.group--
	if uf.size[f1] > uf.size[f2] {
		uf.size[f1] += uf.size[f2]
		uf.boss[f2] = f1
	} else {
		uf.size[f2] += uf.size[f1]
		uf.boss[f1] = f2
	}
}
```

#### 684. Redundant Connection

如果发现两个点已经在一个集合里了就return当前边

java

```java
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        UnionFind uf = new UnionFind(n + 1);
        for (int[] edge : edges) {
            int i = edge[0], j = edge[1];
            if (!uf.union(i, j)) return edge;
        }
        throw new Error("Unreachable");
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    boolean union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return false;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
        return true;
    }
}
```

#### 685. Redundant Connection II

一种情况是有环，另一种情况是有一个点出现两条入边

可以将两条入边收集起来，如果收集不到说明有环

java

```java
class Solution {
    public int[] findRedundantDirectedConnection(int[][] edges) {
        int n = edges.length;
        int[] inDegree = new int[n + 1];
        for (int[] edge : edges) inDegree[edge[1]]++;
        ArrayList<Integer> candi = new ArrayList<>();
        for (int i = edges.length - 1; i >= 0; i--) {
            if (inDegree[edges[i][1]] == 2) candi.add(i);
        }
        if (!candi.isEmpty())
            return validAfterRemove(edges, n, candi.get(0)) ? edges[candi.get(0)] : edges[candi.get(1)];
        return cycle(edges, n);
    }

    int[] cycle(int[][] edges, int n) {
        UnionFind uf = new UnionFind(n + 1);
        for (int[] edge : edges) {
            if (uf.union(edge[0], edge[1])) return edge;
        }
        throw new Error("unreachable");
    }

    boolean validAfterRemove(int[][] edges, int n, int delete) {
        UnionFind uf = new UnionFind(n + 1);
        for (int i = 0; i < n; i++) {
            if (i == delete) continue;
            int[] edge = edges[i];
            if (uf.union(edge[0], edge[1])) return false;
        }
        return true;
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    boolean union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return true;
        if (size[f1] > size[f2]) {
            boss[f2] = f1;
            size[f1] += size[f2];
        } else {
            boss[f1] = f2;
            size[f2] += size[f1];
        }
        return false;
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
        HashMap<Integer, ArrayList<Integer>> islands = new HashMap<>();
        UnionFind uf = new UnionFind(m * n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                unionAround(grid, i, j, m, n, uf);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                int boss = uf.find(i * n + j);
                int x = boss / n, y = boss % n;
                islands.putIfAbsent(boss, new ArrayList<>());
                islands.get(boss).add(i - x);
                islands.get(boss).add(j - y);
            }
        }
        return (int) islands.values().stream().distinct().count();
    }

    void unionAround(int[][] grid, int i, int j, int m, int n, UnionFind uf) {
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] dir : dirs) {
            int x = i + dir[0], y = j + dir[1];
            if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == 0) continue;
            uf.union(i * n + j, x * n + y);
        }
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i])
            boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```
#### 695. Max Area of Island

必须以1的数量初始化union find的group

java

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        UnionFind uf = new UnionFind(grid, m, n);
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                for (int[] dir : dirs) {
                    int x = i + dir[0];
                    int y = j + dir[1];
                    if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0) continue;
                    uf.union(i * n + j, x * n + y);
                }
            }
        }
        return uf.largest();
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int[][] grid, int m, int n) {
        size = new int[m * n];
        boss = new int[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                int idx = i * n + j;
                size[idx] = 1;
                boss[idx] = idx;
            }
        }
    }

    int find(int i) {
        if (i != boss[i])
            boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    int largest() {
        return Arrays.stream(size).max().orElse(0);
    }
}
```

#### 721. Accounts Merge

java

```java
class Solution {
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        int n = accounts.size();
        UnionFind uf = new UnionFind(n);
        HashMap<String, Integer> mailToIndex = new HashMap<>();
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < accounts.get(i).size(); j++) {
                String curr = accounts.get(i).get(j);
                if (mailToIndex.containsKey(curr)) uf.union(mailToIndex.get(curr), i);
                else mailToIndex.put(curr, i);
            }
        }
        HashMap<Integer, Set<String>> graph = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int parent = uf.find(i);
            graph.putIfAbsent(parent, new HashSet<>());
            Set<String> set = graph.get(parent);
            set.addAll(accounts.get(i).stream().skip(1).toList());
        }

        List<List<String>> res = new ArrayList<>();
        for (int index : graph.keySet()) {
            ArrayList<String> path = new ArrayList<>();
            path.add(accounts.get(index).get(0));
            path.addAll(graph.get(index).stream().sorted().toList());
            res.add(path);
        }
        return res;
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i])
            boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

#### 765. Couples Holding Hands

https://leetcode.com/problems/couples-holding-hands/solutions/117520/java-union-find-easy-to-understand-5-ms/

Think about each couple as a vertex in the graph. So if there are N couples, there are N vertices. Now if in position _2i_ and _2i +1_ there are person from couple u and couple v sitting there, that means that the permutations are going to involve u and v. So we add an edge to connect u and v.The min number of swaps = N - number of connected components. This follows directly from the theory of permutations. Any permutation can be decomposed into a composition of cyclic permutations. If the cyclic permutation involve k elements, we need k -1 swaps. You can think about each swap as reducing the size of the cyclic permutation by 1. So in the end, if the graph has k connected components, we need N - k swaps to reduce it back to N disjoint vertices.

*We need N nodes or connected components (N couples) in our final result.  
Let's say currently there are k connected components in our graph.  
Components : C1, C2, ... Ck  
Each component is a cyclic permutation. Let's assume there are n1 nodes in C1 component, n2 nodes in C2 component and so on.  
Total number of nodes N = n1 + n2 + ... nk  
To resolve 1 connected component with d nodes will take d-1 swaps.  
So, to resolve components C1, C2, .. Ck we require Total Swaps = (n1 - 1) + (n2 - 1) + ... (nk - 1)  
`Total swaps = (n1 + n2 + .. nk) - (1 + 1 + ... k times) = N - k`

java

```java
class Solution {
    public int minSwapsCouples(int[] row) {
        int n = row.length;
        UnionFind uf = new UnionFind(n / 2);
        for (int i = 0; i < n; i += 2) {
            uf.union(row[i] / 2, row[i + 1] / 2);
        }
        return uf.swap;
    }

    public static void main(String[] args) {
        System.out.println(new Solution().minSwapsCouples(new int[]{3, 2, 0, 1}));
    }
}

class UnionFind {
    int[] stack;
    int[] boss;
    int[] size;
    int swap;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        stack = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        int j = 0;
        for (; i != boss[i]; j++) {
            stack[j] = i;
            i = boss[i];
        }
        for (j--; j >= 0; j--) {
            boss[stack[j]] = i;
        }
        return i;
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        swap++;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
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
        UnionFind uf = new UnionFind(2 * n);
        for (int i = 0; i < n; i++) {
            for (int j : graph[i]) {
                if (uf.isConnected(i, j)) {
                    return false;
                }
                uf.union(i, j + n);
                uf.union(i + n, j);
            }
        }
        return true;
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }
}
```

#### 803. Bricks Falling When Hit

解法来自GraceMeng https://leetcode.com/problems/bricks-falling-when-hit/solutions/195781/union-find-logical-thinking/

hits中的节点如果命中了就涂成2

把图中的1全部union，记录dummy上连通分量的size

如果是grid顶部的节点就连到dummy上

把2还原成1，再次union，记录增加的size

java

```java
class Solution {
    public int[] hitBricks(int[][] grid, int[][] hits) {
        int m = grid.length, n = grid[0].length;
        for (int[] hit : hits) {
            int i = hit[0], j = hit[1];
            if (grid[i][j] == 1) grid[i][j] = 2;
        }
        UnionFind uf = new UnionFind(m * n + 1);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) unionAround(grid, i, j, m, n, uf);
            }
        }
        int prev = uf.size[uf.find(m * n)];
        int[] res = new int[hits.length];
        for (int i = res.length - 1; i >= 0; i--) {
            int x = hits[i][0], y = hits[i][1];
            if (grid[x][y] != 2) continue;
            grid[x][y] = 1;
            unionAround(grid, x, y, m, n, uf);
            int curr = uf.size[uf.find(m * n)];
            res[i] = Math.max(0, curr - prev - 1);
            prev = curr;
        }
        return res;
    }

    void unionAround(int[][] grid, int i, int j, int m, int n, UnionFind uf) {
        int curr = i * n + j;
        if (i == 0) uf.union(curr, m * n);
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] != 1) continue;
            uf.union(curr, x * n + y);
        }
    }
}

class UnionFind {
    int[] stack;
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        stack = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        int j = 0;
        for (; i != boss[i]; j++) {
            stack[j] = i;
            i = boss[i];
        }
        for (j--; j >= 0; j--) {
            boss[stack[j]] = i;
        }
        return i;
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

#### 839. Similar String Groups

java

```java
class Solution {
    public int numSimilarGroups(String[] strs) {
        int n = strs.length;
        UnionFind uf = new UnionFind(n);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (similar(strs[i], strs[j])) uf.union(i, j);
            }
        }
        return uf.group;
    }

    private boolean similar(String s1, String s2) {
        int diff = 0;
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) != s2.charAt(i)) diff++;
            if (diff > 2) return false;
        }
        return diff == 2 || diff == 0;
    }
}

class UnionFind {
    int[] stack;
    int[] boss;
    int[] size;
    int group;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        stack = new int[n];
        group = n;
        Arrays.fill(size, 1); 
	    for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        int j = 0;
        for (; i != boss[i]; j++) {
            stack[j] = i;
            i = boss[i];
        }
        for (j--; j >= 0; j--) {
            boss[stack[j]] = i;
        }
        return i;
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

#### 886. Possible Bipartition

java

```java
class Solution {
    public boolean possibleBipartition(int n, int[][] dislikes) {
        UnionFind uf = new UnionFind(2 * n + 1);
        for (int[] dislike : dislikes) {
            int u = dislike[0], v = dislike[1];
            if (uf.isConnected(u, v)) {
                return false;
            }
            uf.union(u, v + n);
            uf.union(u + n, v);
        }
        return true;
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }
}
```



#### 924. Minimize Malware Spread

java

```java
public class Solution {
    public int minMalwareSpread(int[][] graph, int[] initial) {
        int n = graph.length;
        UnionFind uf = new UnionFind(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (graph[i][j] == 1) uf.union(i, j);
        int[] cnt = uf.countMalware(initial);
        int maxSize = 0, res = Arrays.stream(initial).min().getAsInt();
        for (int i : initial) {
            int j = uf.find(i);
            if (cnt[j] != 1) continue;
            if (uf.size[j] > maxSize) {
                maxSize = uf.size[j];
                res = i;
            } else if (uf.size[j] == maxSize) res = Math.min(res, i);
        }
        return res;
    }
}

class UnionFind {
    private final int[] stack;
    int[] boss;
    int[] size;
    private final int[] malware;// malware in each component

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        stack = new int[n];
        malware = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int[] countMalware(int[] initial) {
        for (int i : initial)
            malware[find(i)]++;
        return malware;
    }

    int find(int i) {
        int j = 0;
        for (; i != boss[i]; j++) {
            stack[j] = i;
            i = boss[i];
        }
        for (j--; j >= 0; j--) {
            boss[stack[j]] = i;
        }
        return i;
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;

        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

#### 947. Most Stones Removed with Same Row or Column

union with the first stone in each row/column

java

```java
class Solution {
    public int removeStones(int[][] stones) {
        int n = stones.length;
        HashMap<Integer, Integer> rowPre = new HashMap<>();
        HashMap<Integer, Integer> colPre = new HashMap<>();
        UnionFind uf = new UnionFind(n);
        for (int i = 0; i < n; i++) {
            int x = stones[i][0];
            int y = stones[i][1];
            rowPre.putIfAbsent(x, i);
            uf.union(i, rowPre.get(x));
            colPre.putIfAbsent(y, i);
            uf.union(i, colPre.get(y));
        }
        return n - uf.group;
    }
}

class UnionFind {
    int[] stack;
    int[] boss;
    int[] size;
    int group;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        stack = new int[n];
        group = n;
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        int j = 0;
        for (; i != boss[i]; j++) {
            stack[j] = i;
            i = boss[i];
        }
        for (j--; j >= 0; j--) {
            boss[stack[j]] = i;
        }
        return i;
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

`0 <= xi, yi <= 10^4`

也可以合并坐标，但坐标的数据量太大了，远超过石子数量，所以用离散的表存储

java

```java
class Solution {
    public int removeStones(int[][] stones) {
        int n = stones.length;
        UnionFind uf = new UnionFind();
        for (int[] stone : stones) uf.union(stone[0], ~stone[1]);
        return n - uf.group;
    }
}

class UnionFind {
    HashMap<Integer, Integer> boss;
    int group;
    HashMap<Integer, Integer> size;

    UnionFind() {
        boss = new HashMap<>();
        size = new HashMap<>();
    }

    int find(int i) {
        if (boss.putIfAbsent(i, i) == null) {
            group++;
            size.put(i, 1);
        }
        if (i != boss.get(i))
            boss.put(i, find(boss.get(i)));
        return boss.get(i);
    }

    void union(int i, int j) {
        int x = find(i);
        int y = find(j);
        if (x == y) return;
        group--;
        if (size.get(x) < size.get(y)) {
            boss.put(x, y);
            size.put(x, size.get(x) + size.get(y));
        } else {
            boss.put(y, x);
            size.put(y, size.get(x) + size.get(y));
        }
    }
}
```

#### 839. Similar String Groups

java

```java
class Solution {
    public boolean equationsPossible(String[] equations) {
        UnionFind uf = new UnionFind(26);
        for (String equation : equations) {
            if (equation.charAt(1) == '=') {
                int u = equation.charAt(0) - 'a';
                int v = equation.charAt(3) - 'a';
                uf.union(u, v);
            }
        }
        for (String equation : equations) {
            if (equation.charAt(1) == '!') {
                int u = equation.charAt(0) - 'a';
                int v = equation.charAt(3) - 'a';
                if (uf.isConnected(u, v)) {
                    return false;
                }
            }
        }
        return true;
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }
}
```

#### 952. Largest Component Size by Common Factor

java

```java
class Solution {
    public int largestComponentSize(int[] nums) {
        int n = nums.length;
        UnionFind uf = new UnionFind(n);
        HashMap<Integer, Integer> lastOccurred = new HashMap<>();
        for (int i = 0; i < n; i++) {
            for (Integer prime : getPrimeFactors(nums[i])) {
                if (lastOccurred.containsKey(prime)) {
                    uf.union(i, lastOccurred.get(prime));
                    continue;
                }
                lastOccurred.put(prime, i);
            }
        }
        return uf.largest();
    }

    HashSet<Integer> getPrimeFactors(int x) {
        HashSet<Integer> res = new HashSet<>();
        for (int i = 2; i * i < x + 1; i++) {
            while (x % i == 0) {
                res.add(i);
                x /= i;
            }
        }
        if (x > 1) {
            res.add(x);
        }
        return res;
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;

        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    int largest() {
        return Arrays.stream(size).max().orElse(0);
    }
}
```

#### 959. Regions Cut By Slashes

java

```java
class Solution {
    public int regionsBySlashes(String[] grid) {
        int n = grid.length;
        UnionFind uf = new UnionFind(n * n * 4);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                unionAround(i, j, n, grid[i].charAt(j), uf);
            }
        }
        return uf.getGroup();
    }

    void unionAround(int i, int j, int n, char square, UnionFind uf) {
        int s0 = compress(i, j, n, 0);
        int s1 = compress(i, j, n, 1);
        int s2 = compress(i, j, n, 2);
        int s3 = compress(i, j, n, 3);
        if (square != '/') {
            uf.union(s0, s1);
            uf.union(s2, s3);
        }
        if (square != '\\') {
            uf.union(s0, s3);
            uf.union(s1, s2);
        }
        if (i > 0) {
            uf.union(s0, compress(i - 1, j, n, 2));
        }
        if (j > 0) {
            uf.union(s3, compress(i, j - 1, n, 1));
        }
    }

    int compress(int i, int j, int n, int k) {
        return (i * n + j) * 4 + k;
    }
}

class UnionFind {
    private final int[] boss;
    private final int[] size;
    private int group;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        group = n;
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    int getGroup() {
        return group;
    }
}
```

#### 1202. Smallest String With Swaps

java

```java
class Solution {
    public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
        int n = s.length();
        UnionFind uf = new UnionFind(n);
        for (List<Integer> pair : pairs) {
            uf.union(pair.get(0), pair.get(1));
        }
        Map<Integer, List<Integer>> graph = IntStream.range(0, n).boxed().collect(Collectors.groupingBy(uf::find));
        char[] res = new char[n];
        for (List<Integer> cc : graph.values()) {
            Iterator<Character> it = cc.stream().map(s::charAt).sorted().iterator();
            for (Integer idx : cc) {
                res[idx] = it.next();
            }
        }
        return new String(res);
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```

#### 1020. Number of Enclaves

java

```java
class Solution {
    public int numEnclaves(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        UnionFind uf = new UnionFind(grid, m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                unionAround(i, j, m, n, uf, grid);
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 || uf.isConnected(i * n + j, m * n)) continue;
                res++;
            }
        }
        return res;
    }

    void unionAround(int i, int j, int m, int n, UnionFind uf, int[][] grid) {
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        if (i == 0 || j == 0 || i == m - 1 || j == n - 1) {
            uf.union(i * n + j, m * n);
        }
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0) continue;
            uf.union(i * n + j, x * n + y);
        }
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int[][] grid, int m, int n) {
        size = new int[m * n + 1];
        boss = new int[m * n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    continue;
                }
                int idx = i * n + j;
                boss[idx] = idx;
                size[idx] = 1;
            }
        }
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }
}
```


#### 1254. Number of Closed Islands

java

```java
class Solution {
    public int closedIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        UnionFind uf = new UnionFind(m * n + 1);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) continue;
                unionAround(grid, i, j, m, n, uf);
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 && uf.boss[i * n + j] == i * n + j)
                    res++;
            }
        }
        return res;
    }

    void unionAround(int[][] grid, int i, int j, int m, int n, UnionFind uf) {
        if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
            uf.union(i * n + j, m * n);
        }
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 1) continue;
            uf.union(i * n + j, x * n + y);
        }
    }
}

class UnionFind {
    int[] boss;
    int[] size;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) {
            boss[i] = i;
        }
    }

    int find(int i) {
        if (i != boss[i])
            boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }
}
```



#### 1319. Number of Operations to Make Network Connected

java

```java
class Solution {
    public int makeConnected(int n, int[][] connections) {
        if (connections.length < n - 1) {
            return -1;
        }
        UnionFind uf = new UnionFind(n);
        for (int[] conn : connections) {
            uf.union(conn[0], conn[1]);
        }
        return uf.getGroup() - 1;
    }
}

class UnionFind {
    int[] boss;
    int[] size;
    int group;

    UnionFind(int n) {
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
        group = n;
    }

    int find(int i) {
        if (i != boss[i]) boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    int getGroup() {
        return group;
    }
}
```

#### 1584. Min Cost to Connect All Points

Kruskal

不用PQ的话会慢5倍

O(E log E) 

java

```java
class Solution {
    public int minCostConnectPoints(int[][] points) {
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[2]));
        int n = points.length;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int dis = Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1]);
                pq.offer(new int[]{i, j, dis});
            }
        }
        UnionFind uf = new UnionFind(n);
        int res = 0;
        while (!pq.isEmpty() && uf.group > 1) {
            int[] poll = pq.poll();
            int x = poll[0], y = poll[1], dis = poll[2];
            if (uf.isConnected(x, y)) continue;
            uf.union(x, y);
            res += dis;
        }
        return res;
    }
}

class UnionFind {
    int[] boss;
    int[] size;
    int group;

    UnionFind(int n) {
        group = n;
        size = new int[n];
        boss = new int[n];
        Arrays.fill(size, 1);
        for (int i = 0; i < n; i++) boss[i] = i;
    }

    int find(int i) {
        if (i != boss[i])
            boss[i] = find(boss[i]);
        return boss[i];
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size[f1] > size[f2]) {
            size[f1] += size[f2];
            boss[f2] = f1;
        } else {
            size[f2] += size[f1];
            boss[f1] = f2;
        }
    }

    boolean isConnected(int i, int j) {
        return find(i) == find(j);
    }

}
```

#### graph valid tree II

https://www.lintcode.com/problem/444/

java

```java
public class Solution {
    UnionFind uf;
    int edge;

    Solution() {
        uf = new UnionFind();
    }

    public void addEdge(int a, int b) {
        edge++;
        uf.union(a, b);
    }

    public boolean isValidTree() {
        return uf.nodes == edge + 1 && uf.group == 1;
    }
}

class UnionFind {

    HashMap<Integer, Integer> boss;
    int group;
    int nodes;
    HashMap<Integer, Integer> size;

    UnionFind() {
        boss = new HashMap<>();
        size = new HashMap<>();
    }

    int find(int i) {
        if (boss.putIfAbsent(i, i) == null) {
            nodes++;
            group++;
            size.put(i, 1);
        }
        if (i != boss.get(i))
            boss.put(i, find(boss.get(i)));
        return boss.get(i);
    }

    void union(int i, int j) {
        int f1 = find(i);
        int f2 = find(j);
        if (f1 == f2) return;
        group--;
        if (size.get(f1) < size.get(f2)) {
            boss.put(f1, f2);
            size.put(f2, size.get(f2) + size.get(f1));
        } else {
            boss.put(f2, f1);
            size.put(f1, size.get(f2) + size.get(f1));
        }
    }
}
```

#### Paper Review

需要会员https://www.lintcode.com/problem/1463/description

java

```java
public class Solution {
    public float getSimilarity(List<String> words1, List<String> words2, List<List<String>> pairs) {
        UnionFind uf = new UnionFind();
        for (List<String> pair : pairs) {
            uf.union(pair.get(0), pair.get(1));
        }
        int m = words1.size(), n = words2.size();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (uf.isConnected(words1.get(i), words2.get(j))) dp[i + 1][j + 1] = dp[i][j] + 1;
                else dp[i + 1][j + 1] = Math.max(dp[i][j + 1], dp[i + 1][j]);
            }
        }
        return (float) dp[m][n] * 2 / (m + n);
    }

    public static void main(String[] args) {
        System.out.println(new Solution().getSimilarity(
                List.of("great", "acting", "skills", "life"),
                List.of("fine", "drama", "talent", "talent"),
                List.of(List.of("great", "good"), List.of("fine", "good"), List.of("acting", "drama"), List.of("skills", "talent"))));
    }
}

class UnionFind {
    ArrayList<String> stack;
    HashMap<String, String> boss;
    int words;
    HashMap<String, Integer> size;

    UnionFind() {
        stack = new ArrayList<>();
        boss = new HashMap<>();
        size = new HashMap<>();
    }

    String find(String i) {
        if (boss.putIfAbsent(i, i) == null) {
            words++;
            size.put(i, 1);
        }
        while (!i.equals(boss.get(i))) {
            stack.add(i);
            i = boss.get(i);
        }
        while (!stack.isEmpty())
            boss.put(stack.remove(0), i);
        return boss.get(i);
    }

    void union(String i, String j) {
        String f1 = find(i);
        String f2 = find(j);
        if (f1.equals(f2)) return;
        if (size.get(f1) < size.get(f2)) {
            boss.put(f1, f2);
            size.put(f2, size.get(f2) + size.get(f1));
        } else {
            boss.put(f2, f1);
            size.put(f1, size.get(f2) + size.get(f1));
        }
    }

    boolean isConnected(String i, String j) {
        return find(i).equals(find(j));
    }
}
```

