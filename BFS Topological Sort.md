
有向无环图中使用。可以避免循环依赖问题，比如编译打包时就用到

#### [1462. Course Schedule IV](https://leetcode.com/problems/course-schedule-iv/)

java

```java
class Solution {
    public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        ArrayList<HashSet<Integer>> pres = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
            pres.add(new HashSet<>());
        }
        int[] indegree = new int[numCourses];
        for (int[] pre : prerequisites) {
            graph.get(pre[0]).add(pre[1]);
            indegree[pre[1]]++;
        }
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.add(i);
            }
        }
        bfs(graph, pres, indegree, queue);
        return Arrays.stream(queries).map(query -> pres.get(query[1]).contains(query[0])).collect(Collectors.toList());
    }

    private void bfs(ArrayList<ArrayList<Integer>> graph, ArrayList<HashSet<Integer>> pres, int[] indegree, ArrayDeque<Integer> queue) {
        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (int v : graph.get(u)) {
                pres.get(v).add(u);
                pres.get(v).addAll(pres.get(u));
                indegree[v]--;
                if (indegree[v] == 0) {
                    queue.offer(v);
                }
            }
        }
    }
}
```


#### 207. Course Schedule

Go

```go
func canFinish(n int, prerequisites [][]int) bool {
	outDegree := make([]int, n)
	graph := make([][]int, n)
	for _, pre := range prerequisites {
		graph[pre[1]] = append(graph[pre[1]], pre[0])
		outDegree[pre[0]]++
	}
	var queue []int
	for i := range outDegree {
		if outDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	var res []int
	for ; len(queue) > 0; queue = queue[1:] {
		i := queue[0]
		res = append(res, i)
		for _, j := range graph[i] {
			outDegree[j]--
			if outDegree[j] == 0 {
				queue = append(queue, j)
			}
		}
	}
	return len(res) == n
}
```

#### 210. Course Schedule II

Go

```go
func findOrder(n int, prerequisites [][]int) []int {
	outDegree := make([]int, n)
	graph := make([][]int, n)
	for _, pre := range prerequisites {
		graph[pre[1]] = append(graph[pre[1]], pre[0])
		outDegree[pre[0]]++
	}
	var queue []int
	for i := range outDegree {
		if outDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	var res []int
	for ; len(queue) > 0; queue = queue[1:] {
		i := queue[0]
		res = append(res, i)
		for _, j := range graph[i] {
			outDegree[j]--
			if outDegree[j] == 0 {
				queue = append(queue, j)
			}
		}
	}
	if len(res) == n {
		return res
	}
	return []int{}
}
```

rust

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn find_order(n: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
        let n = n as usize;
        let mut in_degree = vec![0; n];
        let mut graph = vec![vec![]; n];
        for pre in prerequisites {
            in_degree[pre[0] as usize] += 1;
            graph[pre[1] as usize].push(pre[0] as usize);
        }
        let mut queue = in_degree.iter().enumerate().fold(VecDeque::new(), |mut acc, (i, &v)| {
            if v == 0 {
                acc.push_back(i)
            }
            acc
        });
        let mut res = vec![];
        while let Some(i) = queue.pop_front() {
            res.push(i as i32);
            for &x in &graph[i] {
                in_degree[x] -= 1;
                if in_degree[x] == 0 {
                    queue.push_front(x);
                }
            }
        }
        if res.len() != n { return vec![]; }
        res
    }
}
```

java

```java
class Solution {  
    public int[] findOrder(int n, int[][] prerequisites) {  
        int[] inDegree = new int[n];  
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();  
        for (int i = 0; i < n; i++) {  
            graph.add(new ArrayList<>());  
        }  
        for (int[] pre : prerequisites) {  
            inDegree[pre[0]]++;  
            graph.get(pre[1]).add((pre[0]));  
        }  
        Queue<Integer> queue = new ArrayDeque<>();  
        for (int i = 0; i < n; i++) {  
            if (inDegree[i] == 0) {  
                queue.offer(i);  
            }  
        }  
        int[] res = new int[n];  
        int visited = 0;  
        while (!queue.isEmpty()) {  
            Integer poll = queue.poll();  
            res[visited++] = poll;  
            for (Integer j : graph.get(poll)) {  
                inDegree[j]--;  
                if (inDegree[j] == 0) {  
                    queue.offer(j);  
                }  
            }  
        }  
        return visited == n ? res : new int[0];  
    }  
}
```

#### 310. Minimum Height Trees

java

```java
class Solution {
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) return List.of(0);
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        int[] inDegree = new int[n];
        Queue<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            inDegree[i] = graph.get(i).size();
            if (inDegree[i] == 1) queue.offer(i);
        }
        return bfs(n, queue, inDegree, graph);
    }

    List<Integer> bfs(int n, Queue<Integer> queue, int[] inDegree, ArrayList<ArrayList<Integer>> graph) {
        while (n > 2) {
            int size = queue.size();
            n -= size;
            for (int i = 0; i < size; i++) {
                Integer leaf = queue.poll();
                for (Integer j : graph.get(leaf)) {
                    inDegree[j]--;
                    if (inDegree[j] == 1) queue.offer(j);
                }
            }
        }
        return new ArrayList<>(queue);
    }
}
```

#### 329. Longest Increasing Path in a Matrix

https://leetcode.com/problems/longest-increasing-path-in-a-matrix/solutions/288520/longest-path-in-dag/

go

```go
func longestIncreasingPath(matrix [][]int) int {
	m, n := len(matrix), len(matrix[0])
	inDegree := make([][]int, m)
	for i := range inDegree {
		inDegree[i] = make([]int, n)
	}
	dirs := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	var queue [][2]int
	for i := range matrix {
		for j := range matrix[0] {
			for _, dir := range dirs {
				x := i + dir[0]
				y := j + dir[1]
				if !valid(x, y, m, n) || matrix[x][y] >= matrix[i][j] {
					continue
				}
				inDegree[i][j]++
			}
			if inDegree[i][j] == 0 {
				queue = append(queue, [2]int{i, j})
			}
		}
	}
	res := 0
	for ; len(queue) > 0; res++ {
		for _, node := range queue {
			queue = queue[1:]
			i, j := node[0], node[1]
			for _, dir := range dirs {
				x := i + dir[0]
				y := j + dir[1]
				if !valid(x, y, m, n) || matrix[x][y] <= matrix[i][j] {
					continue
				}
				inDegree[x][y]--
				if inDegree[x][y] == 0 {
					queue = append(queue, [2]int{x, y})
				}
			}
		}
	}
	return res
}
func valid(x, y, m, n int) bool {
	if x < 0 || y < 0 || x >= m || y >= n {
		return false
	}
	return true
}
```

#### 802. Find Eventual Safe States

java

```java
class Solution {
    public List<Integer> eventualSafeNodes(int[][] graph) {
        int n = graph.length;
        int[] outDegree = new int[n];
        Queue<Integer> queue = new ArrayDeque<>();
        ArrayList<ArrayList<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++)
            adj.add(new ArrayList<>());
        for (int i = 0; i < n; i++) {
            for (int j : graph[i])
                adj.get(j).add(i);
            outDegree[i] = graph[i].length;
            if (outDegree[i] == 0)
                queue.offer(i);
        }
        ArrayList<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();
            res.add(poll);
            for (int j : adj.get(poll)) {
//                outDegree[j]--;
                if (--outDegree[j] == 0)
                    queue.offer(j);
            }
        }
        Collections.sort(res);
        return res;
    }

    public static void main(String[] args) {
        System.out.println(new Solution().eventualSafeNodes(new int[][]{
                {1, 2},
                {2, 3},
                {5},
                {0},
                {5},
                {},
                {},
        }));
    }
}
```