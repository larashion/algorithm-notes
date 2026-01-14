#### [1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/)

```java
class Solution {
    public int maxDistance(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        ArrayDeque<int[]> queue = new ArrayDeque<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    queue.offer(new int[]{i, j});
                    visited[i][j] = true;
                }

            }
        }
        if (queue.isEmpty() || queue.size() == m * n) {
            return -1;
        }
        int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int res = 0;
        for (; !queue.isEmpty(); res++) {
            for (int i = queue.size() - 1; i >= 0; i--) {
                int[] poll = queue.poll();
                int x = poll[0];
                int y = poll[1];
                for (int[] dir : dirs) {
                    int nx = x + dir[0];
                    int ny = y + dir[1];
                    if (nx < 0 || ny < 0 || nx >= m || ny >= n || visited[nx][ny]) {
                        continue;
                    }
                    visited[nx][ny] = true;
                    queue.offer(new int[]{nx, ny});
                }
            }
        }
        //when there is no more cell to be added in the queue,
        //we still need to clear the queue, 
        //in the end max++ is pointless,
        //so we need to minus 1 when we return the max or steps
        return res - 1;
    }
}
```

#### [1034. Coloring A Border](https://leetcode.com/problems/coloring-a-border/)

```java
class Solution {
    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        int m = grid.length, n = grid[0].length;
        boolean[][] isBorder = new boolean[m][n];
        bfs(grid, row, col, m, n, isBorder);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isBorder[i][j]) grid[i][j] = color;
            }
        }
        return grid;
    }

    private void bfs(int[][] grid, int row, int col, int m, int n, boolean[][] isBorder) {
        // mark as visited before enqueue
        boolean[][] visited = new boolean[m][n];
        visited[row][col] = true;
        Queue<int[]> queue = new ArrayDeque<>();
        queue.offer(new int[]{row, col});
        int[][] dirs = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        while (!queue.isEmpty()) {
            int[] polled = queue.poll();
            int x = polled[0], y = polled[1];
            // the border is on the boundary of the grid or adjacent to squares of a different color.
            isBorder[x][y] = x == 0 || x == m - 1 || y == 0 || y == n - 1
                    || grid[x][y] != grid[x - 1][y]
                    || grid[x][y] != grid[x + 1][y]
                    || grid[x][y] != grid[x][y - 1]
                    || grid[x][y] != grid[x][y + 1];
            for (int[] dir : dirs) {
                int nx = dir[0] + x;
                int ny = dir[1] + y;
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] != grid[x][y] || visited[nx][ny]) {
                    continue;
                }
                visited[nx][ny] = true;
                queue.offer(new int[]{nx, ny});
            }
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
        ArrayList<Integer> start = new ArrayList<>();
        start.add(0);
        ArrayDeque<ArrayList<Integer>> queue = new ArrayDeque<>();
        queue.add(start);
        while (!queue.isEmpty()) {
            ArrayList<Integer> poll = queue.poll();
            Integer u = poll.get(poll.size() - 1);
            if (u == n - 1) {
                res.add(poll);
                continue;
            }
            for (int v : graph[u]) {
                ArrayList<Integer> clone = new ArrayList<>(poll);
                clone.add(v);
                queue.add(clone);
            }
        }
        return res;
    }
}
```

Go

```go
func allPathsSourceTarget(graph [][]int) [][]int {
	start := []int{0}
	n := len(graph)
	var res [][]int
	for queue := [][]int{start}; len(queue) > 0; queue = queue[1:] {
		poll := queue[0]
		size := len(poll)
		u := poll[size-1]
		if u == n-1 {
			res = append(res, poll)
			continue
		}
		for _, v := range graph[u] {
			dup := make([]int, size)
			copy(dup, poll)
			dup = append(dup, v)
			queue = append(queue, dup)
		}
	}
	return res
}
```

#### [1129. Shortest Path with Alternating Colors](https://leetcode.com/problems/shortest-path-with-alternating-colors/)

```java
class Solution {
    private static final int red = 1;
    private static final int blue = 0;

    public int[] shortestAlternatingPaths(int n, int[][] redEdges, int[][] blueEdges) {
        ArrayList<ArrayList<Integer>> redG = toGraph(n, redEdges);
        ArrayList<ArrayList<Integer>> blueG = toGraph(n, blueEdges);
        int[] res = new int[n];
        Arrays.fill(res, -1);
        bfs(redG, blueG, res, new HashSet<>(), new HashSet<>());
        return res;
    }

    private ArrayList<ArrayList<Integer>> toGraph(int n, int[][] edges) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            res.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            res.get(edge[0]).add(edge[1]);
        }
        return res;
    }


    private void bfs(ArrayList<ArrayList<Integer>> redG, ArrayList<ArrayList<Integer>> blueG, int[] res, HashSet<Integer> visitedR, HashSet<Integer> visitedB) {
        ArrayDeque<int[]> queue = new ArrayDeque<>();
        queue.add(new int[]{0, red, 0});
        queue.add(new int[]{0, blue, 0});
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            int u = poll[0];
            int color = poll[1];
            int step = poll[2];
            res[u] = res[u] == -1 ? step : Math.min(step, res[u]);
            ArrayList<ArrayList<Integer>> edges = color == red ? redG : blueG;
            HashSet<Integer> visited = color == red ? visitedR : visitedB;
            for (int v : edges.get(u)) {
                if (visited.contains(v)) {
                    continue;
                }
                visited.add(v);
                queue.add(new int[]{v, color ^ 1, step + 1});
            }
        }
    }
}
```

#### [1311. Get Watched Videos by Your Friends](https://leetcode.com/problems/get-watched-videos-by-your-friends/)

```java
class Solution {
    public List<String> watchedVideosByFriends(List<List<String>> watchedVideos, int[][] friends, int id, int level) {
        Queue<Integer> queue = BFS(friends, id, level);
        HashMap<String, Integer> freq = new HashMap<>();
        for (Integer f : queue) {
            for (String video : watchedVideos.get(f)) {
                freq.put(video, freq.getOrDefault(video, 0) + 1);
            }
        }
        ArrayList<String> res = new ArrayList<>(freq.keySet());
        res.sort((a, b) -> {
            int fa = freq.get(a);
            int fb = freq.get(b);
            if (fa != fb) {
                return fa - fb;
            } else {
                return a.compareTo(b);
            }
        });
        return res;
    }

    private Queue<Integer> BFS(int[][] friends, int id, int level) {
        int n = friends.length;
        boolean[] visited = new boolean[n];
        visited[id] = true;
        Queue<Integer> queue = new ArrayDeque<>();
        queue.offer(id);
        for (int i = 0; i < level && !queue.isEmpty(); i++) {
            for (int j = queue.size() - 1; j >= 0; j--) {
                int u = queue.poll();
                for (int v : friends[u]) {
                    if (visited[v]) {
                        continue;
                    }
                    visited[v] = true;
                    queue.offer(v);
                }
            }
        }
        return queue;
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
            res[i] = bfs(graph, queries.get(i).get(0), queries.get(i).get(1));
        }
        return res;
    }

    private double bfs(HashMap<String, HashMap<String, Double>> graph, String start, String end) {
        if (!graph.containsKey(start)) {
            return -1;
        }
        record Node(String u, double weight) {}
        ArrayDeque<Node> queue = new ArrayDeque<>();
        queue.add(new Node(start, 1.0));
        HashSet<String> visited = new HashSet<>();
        visited.add(start);
        while (!queue.isEmpty()) {
            Node poll = queue.poll();
            String u = poll.u;
            double weight = poll.weight;
            if (u.equals(end)) {
                return weight;
            }
            for (String v : graph.get(u).keySet()) {
                if (visited.contains(v)) continue;
                visited.add(v);
                queue.offer(new Node(v, graph.get(u).get(v) * weight));
            }
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
            if (!bfs(i, 1, colors, graph)) {
                return false;
            }
        }
        return true;
    }

    boolean bfs(int start, int target, int[] colors, ArrayList<ArrayList<Integer>> graph) {
        colors[start] = target;
        Queue<Integer> queue = new ArrayDeque<>();
        queue.offer(start);
        while (!queue.isEmpty()) {
            Integer u = queue.poll();
            for (Integer v : graph.get(u)) {
                if (colors[v] == colors[u]) {
                    return false;
                }
                if (colors[v] != 0) {
                    continue;
                }
                colors[v] = -colors[u];
                queue.offer(v);
            }
        }
        return true;
    }
}
```

#### 752. Open the Lock

java

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        HashMap<String, Integer> distance = new HashMap<>();
        for (String deadend : deadends) {
            distance.put(deadend, -1);
        }
        if (distance.containsKey("0000")) return -1;
        distance.put("0000", 0);
        ArrayDeque<String> queue = new ArrayDeque<>();
        queue.offer("0000");
        while (!queue.isEmpty()) {
            String poll = queue.poll();
            if (poll.equals(target)) return distance.get(poll);
            for (String s : getNeighbor(poll)) {
                if (distance.containsKey(s)) continue;
                distance.put(s, distance.get(poll) + 1);
                queue.offer(s);
            }
        }
        return -1;
    }

    List<String> getNeighbor(String curr) {
        ArrayList<String> res = new ArrayList<>();
        for (int i = 0; i < curr.length(); i++) {
            StringBuilder sb = new StringBuilder(curr);
            char c = sb.charAt(i);
            if (c == '9')
                sb.setCharAt(i, '0');
            else sb.setCharAt(i, (char) (c + 1));
            res.add(sb.toString());

            if (c == '0')
                sb.setCharAt(i, '9');
            else sb.setCharAt(i, (char) (c - 1));

            res.add(sb.toString());
        }
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
                    bfs(i, j, m, n, visited, grid);
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

    void bfs(int i, int j, int m, int n, boolean[][] visited, int[][] grid) {
        Queue<int[]> queue = new ArrayDeque<>();
        queue.add(new int[]{i, j});
        visited[i][j] = true;
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        while (!queue.isEmpty()) {
            int[] curr = queue.poll();
            for (int[] dir : directions) {
                int x = dir[0] + curr[0];
                int y = dir[1] + curr[1];
                if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0 || visited[x][y]) continue;
                queue.add(new int[]{x, y});
                visited[x][y] = true;
            }
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
            if (!bfs(graph, i, colors)) return false;
        }
        return true;
    }

    boolean bfs(int[][] graph, int start, int[] colors) {
        colors[start] = 1;
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.offer(start);
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();
            for (int neighbor : graph[poll]) {
                if (colors[neighbor] == colors[poll]) return false;
                if (colors[neighbor] != 0) continue;
                colors[neighbor] = -colors[poll];
                queue.offer(neighbor);
            }
        }
        return true;
    }
}
```

Go

```go
func isBipartite(graph [][]int) bool {
	n := len(graph)
	colors := make([]int, n)
	for i := range colors {
		if colors[i] != 0 {
			continue
		}
		if !bfs(graph, colors, i) {
			return false
		}
	}
	return true
}

func bfs(graph [][]int, colors []int, start int) bool {
	colors[start] = 1
	for queue := []int{start}; len(queue) > 0; queue = queue[1:] {
		poll := queue[0]
		for _, neighbor := range graph[poll] {
			if colors[neighbor] == colors[poll] {
				return false
			}
			if colors[neighbor] != 0 {
				continue
			}
			colors[neighbor] = -colors[poll]
			queue = append(queue, neighbor)
		}
	}
	return true
}
```

#### 102. Binary Tree Level Order Traversal

python

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        if not root:
            return []
        dq = collections.deque([root])
        while dq:
            res.append([node.val for node in dq])
            for _ in range(len(dq)):
                node = dq.popleft()
                dq.extend(leaf for leaf in [node.left, node.right] if leaf)
        return res
```

go

```go
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int
	for queue := []*TreeNode{root}; len(queue) > 0; {
		var level []int
		for _, n := range queue {
			level = append(level, n.Val)
			queue = queue[1:]
			if n.Left != nil {
				queue = append(queue, n.Left)
			}
			if n.Right != nil {
				queue = append(queue, n.Right)
			}
		}
		res = append(res, level)
	}
	return res
}
```

java

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            ArrayList<Integer> level = new ArrayList<>();
            for (int i = queue.size() - 1; i >= 0; i--) {
                TreeNode treeNode = queue.poll();
                assert treeNode != null;
                level.add(treeNode.val);
                
                if (treeNode.left != null) {
                    queue.offer(treeNode.left);
                }
                if (treeNode.right != null) {
                    queue.offer(treeNode.right);
                }
            }
            res.add(level);
        }
        return res;
    }
}
```

#### 103. Binary Tree Zigzag Level Order Traversal

go

```go
func zigzagLevelOrder(root *TreeNode) [][]int {
   if root == nil {
      return nil
   }
   var res [][]int
   queue := []*TreeNode{root}
   zigzag := false
   for len(queue) > 0 {
      var level []int
      for _, n := range queue {
         level = append(level, n.Val)
         queue = queue[1:]
         if n.Left != nil {
            queue = append(queue, n.Left)
         }
         if n.Right != nil {
            queue = append(queue, n.Right)
         }
      }
      if zigzag {
         reverse(level)
      }
      res = append(res, level)
      zigzag = !zigzag
   }
   return res
}
func reverse(nodes []int) {
   left, right := 0, len(nodes)-1
   for left < right {
      nodes[left], nodes[right] = nodes[right], nodes[left]
      left++
      right--
   }
}
```

#### 107. Binary Tree Level Order Traversal II

java

```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = levelOrder(root);
        Collections.reverse(res);
        return res;
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            ArrayList<Integer> level = new ArrayList<>();
            for (int i = queue.size() - 1; i >= 0; i--) {
                TreeNode treeNode = queue.poll();
                assert treeNode != null;
                level.add(treeNode.val);

                if (treeNode.left != null) {
                    queue.offer(treeNode.left);
                }
                if (treeNode.right != null) {
                    queue.offer(treeNode.right);
                }
            }
            res.add(level);
        }
        return res;
    }
}
```

go

```go
func levelOrderBottom(root *TreeNode) [][]int {
	res := levelOrder(root)
	return reverse(res)
}
func reverse(s [][]int) [][]int {
	for left, right := 0, len(s)-1; left < right; {
		s[left], s[right] = s[right], s[left]
		left++
		right--
	}
	return s
}
func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int
	for queue := []*TreeNode{root}; len(queue) > 0; {
		var level []int
		for _, n := range queue {
			level = append(level, n.Val)
			queue = queue[1:]
			if n.Left != nil {
				queue = append(queue, n.Left)
			}
			if n.Right != nil {
				queue = append(queue, n.Right)
			}
		}
		res = append(res, level)
	}
	return res
}
```

#### 117. Populating Next Right Pointers in Each Node II

go

```go
func connect(root *Node) *Node {
    if root == nil {
        return nil
    }
    queue := []*Node{root}
    for len(queue) > 0 {
        var tail *Node
        for _, v := range queue {
            if v.Left != nil {
                queue = append(queue, v.Left)
            }
            if v.Right != nil {
                queue = append(queue, v.Right)
            }
            if tail != nil {
                tail.Next = v
            }
            tail = v
            queue = queue[1:]
        }
    }
    return root
}
```

#### 127. Word Ladder

go

```go
type path struct {
	word string
	step int
}

func ladderLength(begin string, end string, wordList []string) int {
	set := make(map[string]bool)
	for _, word := range wordList {
		set[word] = true
	}
	if !set[end] {
		return 0
	}
	for queue := []*path{{begin, 1}}; len(queue) > 0; queue = queue[1:] {
		word, step := queue[0].word, queue[0].step
		if word == end {
			return step
		}
		for i := range word {
			var j byte
			for j = 'a'; j < 'z'+1; j++ {
				modified := word[:i] + string(j) + word[i+1:]
				if set[modified] {
					delete(set, modified)
					queue = append(queue, &path{modified, step + 1})
				}
			}
		}
	}
	return 0
}
```

java

```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        HashSet<String> wordDict = new HashSet<>(wordList);
        if (!wordDict.contains(endWord)) {
            return 0;
        }
        record Node(String word, int step) {}
        Queue<Node> queue = new ArrayDeque<>(List.of(new Node(beginWord, 1)));
        while (!queue.isEmpty()) {
            Node head = queue.poll();
            String curr = head.word;
            int step = head.step;
            if (curr.equals(endWord)) {
                return step;
            }
            for (int i = 0; i < curr.length(); i++) {
                char[] byteArr = curr.toCharArray();
                for (char j = 'a'; j < 'z' + 1; j++) {
                    byteArr[i] = j;
                    String modified = new String(byteArr);
                    if (wordDict.contains(modified)) {
                        wordDict.remove(modified);
                        queue.offer(new Node(modified, step + 1));
                    }
                }
            }
        }
        return 0;
    }
}
```

rust

```rust
use std::collections::{HashSet, VecDeque};
use std::iter::FromIterator;

impl Solution {
    pub fn ladder_length(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
        let mut set: HashSet<String> = HashSet::from_iter(word_list);
        if !set.contains(&end_word) {
            return 0;
        }
        let mut queue = VecDeque::from([(begin_word, 1)]);
        while let Some((curr, step)) = queue.pop_front() {
            if curr == end_word {
                return step;
            }
            for i in 0..curr.len() {
                for j in b'a'..=b'z' {
                    let modified = [&curr[..i], &String::from(j as char), &curr[i + 1..]].concat();
                    if set.contains(&modified) {
                        set.remove(&modified);
                        queue.push_back((modified, step + 1));
                    }
                }
            }
        }
        0
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
				bfs(board, i, j, m, n)
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

func bfs(board [][]byte, i, j, m, n int) {
	dirs := [4][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	board[i][j] = '#'
	for queue := [][2]int{{i, j}}; len(queue) > 0; queue = queue[1:] {
		head := queue[0]
		for _, dir := range dirs {
			x := head[0] + dir[0]
			y := head[1] + dir[1]
			if x < 0 || x >= m || y < 0 || y >= n || board[x][y] != 'O' {
				continue
			}
			board[x][y] = '#'
			queue = append(queue, [2]int{x, y})
		}
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
	res := &Node{Val: node.Val}
	hashmap := map[*Node]*Node{node: res}
	queue := []*Node{node}
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for _, neighbor := range curr.Neighbors {
			if _, ok := hashmap[neighbor]; !ok {
				hashmap[neighbor] = &Node{Val: neighbor.Val}
				queue = append(queue, neighbor)
			}
			hashmap[curr].Neighbors = append(hashmap[curr].Neighbors, hashmap[neighbor])
		}
	}
	return res
}
```

python

```python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        res = Node(node.val, [])
        copies = {node: res}
        q = collections.deque([node])
        while q:
            curr = q.popleft()
            for neighbor in curr.neighbors:
                if neighbor not in copies:
                    copies[neighbor] = Node(neighbor.val, [])
                    q += neighbor,
                copies[curr].neighbors += copies[neighbor],
        return res
```

java

```java
class Solution {
    public Node cloneGraph(Node node) {
        if (node == null) {
            return null;
        }
        Node res = new Node(node.val);
        HashMap<Node, Node> copies = new HashMap<>();
        copies.put(node, res);

        Queue<Node> queue = new ArrayDeque<>();
        queue.add(node);
        while (!queue.isEmpty()) {
            Node curr = queue.poll();
            for (Node neighbor : curr.neighbors) {
                if (!copies.containsKey(neighbor)) {
                    copies.put(neighbor, new Node(neighbor.val));
                    queue.offer(neighbor);
                }
                copies.get(curr).neighbors.add(copies.get(neighbor));
            }
        }
        return res;
    }
}
```

#### 199. Binary Tree Right Side View

Go

```go
func rightSideView(root *TreeNode) []int {
    if root == nil {
        return nil
    }
    queue, res := []*TreeNode{root}, make([]int, 0)
    for len(queue) > 0 {
        rightNode := queue[len(queue)-1]
        res = append(res, rightNode.Val)
        for _, node := range queue {
            queue = queue[1:]
            if node.Left != nil {
                queue = append(queue, node.Left)
            }
            if node.Right != nil {
                queue = append(queue, node.Right)
            }
        }
    }
    return res
}
```

#### 200. Number of Islands

在矩阵上搜索，BFS的空间复杂度更优，是n级别，而不是n^2

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
			visited[i][j] = true
			bfs(i, j, m, n, grid, visited, &res)
		}
	}
	return res
}
func bfs(i, j, m, n int, grid [][]byte, visited [][]bool, res *int) {
	*res++
	dirs := [4][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	for queue := [][2]int{{i, j}}; len(queue) > 0; queue = queue[1:] {
		for _, dir := range dirs {
			x := queue[0][0] + dir[0]
			y := queue[0][1] + dir[1]
			if x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == '0' || visited[x][y] {
				continue
			}
			visited[x][y] = true
			queue = append(queue, [2]int{x, y})
		}
	}
}
```

java

```java
class Solution {
    public int numIslands(char[][] grid) {
        int m = grid.length, n = grid[0].length;
        int res = 0;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '0' || visited[i][j]) {
                    continue;
                }
                res++;
                visited[i][j] = true;
                bfs(i, j, m, n, grid, visited);
            }
        }
        return res;
    }

    void bfs(int i, int j, int m, int n, char[][] grid, boolean[][] visited) {
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Queue<int[]> queue = new ArrayDeque<>(List.of(new int[]{i, j}));
        while (!queue.isEmpty()) {
            int[] head = queue.poll();
            for (int[] dir : dirs) {
                int x = head[0] + dir[0];
                int y = head[1] + dir[1];
                if (x < 0 || y < 0 || x >= m || y >= n || visited[x][y] || grid[x][y] == '0') {
                    continue;
                }
                visited[x][y] = true;
                queue.offer(new int[]{x, y});
            }
        }
    }
}
```

rust

```rust
use std::collections::VecDeque;

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
                visited[i][j] = true;
                Solution::bfs(&grid, &mut visited, i, j, m, n, &mut res)
            }
        }
        res
    }
    fn bfs(grid: &[Vec<char>], visited: &mut [Vec<bool>], i: usize, j: usize, m: usize, n: usize, res: &mut i32) {
        *res += 1;
        let dirs = [[-1, 0], [0, -1], [0, 1], [1, 0]];
        let mut queue = VecDeque::from([[i, j]]);
        while let Some([x, y]) = queue.pop_front() {
            for dir in dirs {
                let x = x as i32 + dir[0];
                let y = y as i32 + dir[1];
                if x < 0 || y < 0 {
                    continue;
                }
                let x = x as usize;
                let y = y as usize;
                if x >= m || y >= n || grid[x][y] == '0' || visited[x][y] {
                    continue;
                }
                visited[x][y] = true;
                queue.push_back([x, y])
            }
        }
    }
}
```

#### 429. N-ary Tree Level Order Traversal

Go

```go
func levelOrder(root *Node) [][]int {
	if root == nil {
		return nil
	}
	var res [][]int
	for queue := []*Node{root}; len(queue) > 0;{
		var level []int
		for _, node := range queue {
			queue = queue[1:]
			level = append(level, node.Val)
			for _, c := range node.Children {
				queue = append(queue, c)
			}
		}
		res = append(res, level)
	}
	return res
}
```

#### 433. Minimum Genetic Mutation

https://leetcode.com/problems/minimum-genetic-mutation/discuss/189662/Python-BFS-(same-as-word-ladder)

go

```go
type path struct {
	gene   string
	mutate int
}

func minMutation(start string, end string, bank []string) int {
	bankSet := make(map[string]bool)
	for _, v := range bank {
		bankSet[v] = true
	}
	if !bankSet[end] {
		return -1
	}
	agct := "AGCT"
	for queue := []*path{{gene: start, mutate: 0}}; len(queue) > 0; {
		curr := queue[0]
		queue = queue[1:]
		if curr.gene == end {
			return curr.mutate
		}
		for i := range curr.gene {
			for j := range agct {
				modified := curr.gene[:i] + string(agct[j]) + curr.gene[i+1:]
				if bankSet[modified] {
					delete(bankSet, modified)
					queue = append(queue, &path{modified, curr.mutate + 1})
				}
			}
		}
	}
	return -1
}
```

python

```python
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        queue = collections.deque([(start, 0)])
        bank_set = set(bank)
        while queue:
            curr, step = queue.popleft()
            if curr == end:
                return step
            for i in range(len(curr)):
                for char in "AGCT":
                    modified = curr[:i] + char + curr[i + 1:]
                    if modified in bank_set:
                        bank_set.remove(modified)
                        queue += (modified, step + 1),
        return -1
```

java

```java
class Solution {
    public int minMutation(String start, String end, String[] bank) {

        HashSet<String> bankSet = new HashSet<>(Arrays.asList(bank));
        if (!bankSet.contains(end)) {
            return -1;
        }

        record Node(String gene, int step) {}
        Queue<Node> queue = new ArrayDeque<>(List.of(new Node(start, 0)));
        char[] AGCT = "AGCT".toCharArray();
        while (!queue.isEmpty()) {
            Node head = queue.poll();
            int step = head.step;
            String curr = head.gene;
            if (curr.equals(end)) {
                return step;
            }
            for (int i = 0; i < curr.length(); i++) {
                char[] chars = curr.toCharArray();
                for (char ch : AGCT) {
                    chars[i] = ch;
                    String modified = new String(chars);
                    if (bankSet.contains(modified)) {
                        bankSet.remove(modified);
                        queue.offer(new Node(modified, step + 1));
                    }
                }
            }
        }
        return -1;
    }
}
```

rust

```rust
use std::collections::{HashSet, VecDeque};
use std::iter::FromIterator;

impl Solution {
    pub fn min_mutation(start_gene: String, end_gene: String, bank: Vec<String>) -> i32 {
        let mut set: HashSet<String> = HashSet::from_iter(bank);
        if !set.contains(&end_gene) {
            return -1;
        }
        let mut queue = VecDeque::from([(start_gene, 0)]);
        while let Some((head, mutated)) = queue.pop_front() {
            if head == end_gene {
                return mutated;
            }
            for i in 0..head.len() {
                for ch in "AGCT".chars() {
                    let modified = [&head[..i], &String::from(ch), &head[i + 1..]].concat();
                    if !set.contains(&modified) {
                        continue;
                    }
                    set.remove(&modified);
                    queue.push_back((modified, mutated + 1))
                }
            }
        }
        -1
    }
}
```

#### 463. Island Perimeter

go

```go
func islandPerimeter(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	res := 0
	var queue [][2]int
	for i := range grid {
		for j := range grid[0] {
			if grid[i][j] == 0 {
				continue
			}
			queue = append(queue, [2]int{i, j})
		}
	}
	dirs := [4][2]int{{1, 0}, {0, 1}, {0, -1}, {-1, 0}}
	for ; len(queue) > 0; queue = queue[1:] {
		head := queue[0]
		for _, dir := range dirs {
			x := head[0] + dir[0]
			y := head[1] + dir[1]
			if x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == 0 {
				res++
			}
		}
	}
	return res
}
```

#### 513. Find Bottom Left Tree Value

```go
func findBottomLeftValue(root *TreeNode) int {
	res := 0
	for queue := []*TreeNode{root}; len(queue) > 0; {
		res = queue[0].Val
		for _, node := range queue {
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
	}
	return res
}
```

#### 515. Find Largest Value in Each Tree Row

go

```go
func largestValues(root *TreeNode) []int {
   res := make([]int, 0)
   if root == nil {
      return res
   }
   queue := []*TreeNode{root}
   for len(queue) > 0 {
      largest := math.MinInt32
      for _, node := range queue {
         queue = queue[1:]
         if node.Val > largest {
            largest = node.Val
         }
         if node.Left != nil {
            queue = append(queue, node.Left)
         }
         if node.Right != nil {
            queue = append(queue, node.Right)
         }
      }
      res = append(res, largest)
   }
   return res
}
```

java

```java
class Solution {
    public List<Integer> largestValues(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        ArrayList<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int largest = Integer.MIN_VALUE;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode poll = queue.poll();
                assert poll != null;
                largest = Math.max(largest, poll.val);
                if (poll.left != null) {
                    queue.offer(poll.left);
                }
                if (poll.right != null) {
                    queue.offer(poll.right);
                }
            }
            res.add(largest);
        }
        return res;
    }
}
```

#### 542. 01 Matrix

go

```go
func updateMatrix(mat [][]int) [][]int {
	m, n := len(mat), len(mat[0])
	dirs := [4][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	res := make([][]int, m)
	copy(res, mat)
	var queue [][2]int
	for i := range res {
		for j := range res[0] {
			if res[i][j] == 0 {
				queue = append(queue, [2]int{i, j})
			} else {
				res[i][j] = -1
			}
		}
	}
	for ; len(queue) > 0; queue = queue[1:] {
		i, j := queue[0][0], queue[0][1]
		for _, dir := range dirs {
			x, y := i+dir[0], j+dir[1]
			if x >= 0 && x < m && y >= 0 && y < n && res[x][y] == -1 {
				res[x][y] = res[i][j] + 1
				queue = append(queue, [2]int{x, y})
			}
		}
	}
	return res
}
```

java

```java
class Solution {
    public int[][] updateMatrix(int[][] mat) {
        int m = mat.length, n = mat[0].length;
        int[][] res = Arrays.copyOf(mat, m);
        Queue<int[]> queue = new ArrayDeque<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (res[i][j] == 0) {
                    queue.offer(new int[]{i, j});
                } else {
                    res[i][j] = -1;
                }
            }
        }
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            int i = poll[0], j = poll[1];
            for (int[] dir : dirs) {
                int x = i + dir[0];
                int y = j + dir[1];
                if (x < 0 || y < 0 || x >= m || y >= n || res[x][y] != -1) {
                    continue;
                }
                res[x][y] = res[i][j] + 1;
                queue.offer(new int[]{x, y});
            }
        }
        return res;
    }
}
```

#### 623. Add One Row to Tree

Go

```go
func addOneRow(root *TreeNode, val int, depth int) *TreeNode {
	if depth == 1 {
		return &TreeNode{val, root, nil}
	}
	queue := []*TreeNode{root}
	for level := 1; len(queue) > 0 && level < depth-1; level++ {
		for _, node := range queue {
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
	}
	for _, node := range queue {
		node.Left = &TreeNode{val, node.Left, nil}
		node.Right = &TreeNode{val, nil, node.Right}
	}
	return root
}

```

java

```java
class Solution {
    public TreeNode addOneRow(TreeNode root, int val, int depth) {
        if (depth == 1) {
            return new TreeNode(val, root, null);
        }
        ArrayDeque<TreeNode> queue = new ArrayDeque<>(List.of(root));
        for (int level = 1; level < depth - 1 && !queue.isEmpty(); level++) {
            for (int i = queue.size() - 1; i >= 0; i--) {
                TreeNode poll = queue.poll();
                assert poll != null;
                if (poll.left != null) {
                    queue.offer(poll.left);
                }
                if (poll.right != null) {
                    queue.offer(poll.right);
                }
            }
        }
        for (TreeNode treeNode : queue) {
            treeNode.left = new TreeNode(val, treeNode.left, null);
            treeNode.right = new TreeNode(val, null, treeNode.right);
        }
        return root;
    }
}
```

#### 637. Average of Levels in Binary Tree

Go

```go
func averageOfLevels(root *TreeNode) []float64 {
	var res []float64
	for queue := []*TreeNode{root}; len(queue) > 0; {
		var level []int
		for _, node := range queue {
			queue = queue[1:]
			level = append(level, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		res = append(res, avg(level))
	}
	return res
}

func avg(nums []int) float64 {
	sum := 0
	for _, num := range nums {
		sum += num
	}
	return float64(sum) / float64(len(nums))
}
```

#### 662. Maximum Width of Binary Tree

go

```go
type tuple struct {
	node *TreeNode
	idx  int
}

func widthOfBinaryTree(root *TreeNode) int {
	res := 0
	for queue := []*tuple{{node: root, idx: 1}}; len(queue) > 0; {
		res = max(res, queue[len(queue)-1].idx-queue[0].idx+1)
		for _, t := range queue {
			item, num := t.node, t.idx
			queue = queue[1:]
			if item.Left != nil {
				queue = append(queue, &tuple{
					node: item.Left,
					idx:  2 * num,
				})
			}
			if item.Right != nil {
				queue = append(queue, &tuple{
					node: item.Right,
					idx:  2*num + 1,
				})
			}
		}
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

java

```java
class Solution {
    public int widthOfBinaryTree(TreeNode root) {
        int res = 0;
        record Node(TreeNode node, int idx) {}
        ArrayDeque<Node> queue = new ArrayDeque<>();
        queue.offer(new Node(root, 1));
        while (!queue.isEmpty()) {
            int right = queue.peekLast().idx;
            int left = queue.peekFirst().idx;
            res = Math.max(res, right - left + 1);
            for (int i = queue.size() - 1; i >= 0; i--) {
                Node poll = queue.poll();
                int idx = poll.idx;
                TreeNode node = poll.node;
                if (node.left != null) {
                    queue.offer(new Node(node.left, idx * 2));
                }
                if (node.right != null) {
                    queue.offer(new Node(node.right, idx * 2 + 1));
                }
            }
        }
        return res;
    }
}
```

#### 694.Number of Distinct Islands

testing https://www.lintcode.com/problem/860/description

java

```java
public class Solution {
    public int numberofDistinctIslands(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        Set<ArrayList<Integer>> islands = new HashSet<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (visited[i][j] || grid[i][j] == 0) continue;
                ArrayList<Integer> path = new ArrayList<>();
                bfs(grid, path, i, j, m, n, visited);
                islands.add(path);
            }
        }
        return islands.size();
    }

    void bfs(int[][] grid, ArrayList<Integer> path, int i, int j, int m, int n, boolean[][] visited) {
        Queue<int[]> queue = new ArrayDeque<>();
        queue.offer(new int[]{i, j});
        visited[i][j] = true;
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            int x = poll[0], y = poll[1];
            path.add(x - i);
            path.add(y - j);
            for (int[] dir : dirs) {
                int nx = x + dir[0], ny = y + dir[1];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || visited[nx][ny] || grid[nx][ny] == 0) continue;
                visited[nx][ny] = true;
                queue.offer(new int[]{nx, ny});
            }
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
                res = Math.max(res, bfs(grid, visited, m, n, i, j));
            }
        }
        return res;
    }

    int bfs(int[][] grid, boolean[][] visited, int m, int n, int i, int j) {
        ArrayDeque<int[]> queue = new ArrayDeque<>(List.of(new int[]{i, j}));
        visited[i][j] = true;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 0;
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            res++;
            for (int[] dir : dirs) {
                int x = poll[0] + dir[0];
                int y = poll[1] + dir[1];
                if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0 || visited[x][y]) continue;
                visited[x][y] = true;
                queue.offer(new int[]{x, y});
            }
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
        HashMap<Integer, Set<String>> indexToMail = new HashMap<>();
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (visited[i]) continue;
            indexToMail.put(i, new HashSet<>());
            BFS(i, graph, visited, indexToMail, accounts);
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

    void BFS(Integer start, ArrayList<ArrayList<Integer>> graph, boolean[] visited, HashMap<Integer, Set<String>> indexToMail, List<List<String>> accounts) {
        Queue<Integer> queue = new ArrayDeque<>();
        queue.offer(start);
        visited[start] = true;
        while (!queue.isEmpty()) {
            int i = queue.poll();
            indexToMail.get(start).addAll(accounts.get(i).stream().skip(1).toList());
            for (int neighbor : graph.get(i)) {
                if (visited[neighbor]) continue;
                visited[neighbor] = true;
                queue.offer(neighbor);
            }
        }
    }
}
```

#### 743. Network Delay Time

用邻接表构图，用堆里取延迟最低的节点访问

D算法，限制了边无负权重

O(V * log(E) )

rust

```rust
use std::cmp::Reverse;
use std::collections::BinaryHeap;

impl Solution {
    pub fn network_delay_time(times: Vec<Vec<i32>>, n: i32, k: i32) -> i32 {
        let mut n = n as usize;
        let graph = times.iter().fold(vec![vec![]; n + 1], |mut acc, time| {
            acc[time[0] as usize].push([time[1], time[2]]);
            acc
        });
        let mut heap = BinaryHeap::from([Reverse((0, k))]);
        let mut visited = vec![false; n + 1];
        while let Some(Reverse((dis, start))) = heap.pop() {
            let start = start as usize;
            if visited[start] { continue; }
            visited[start] = true;
            n -= 1;
            if n == 0 { return dis; }
            for [next, delay] in &graph[start] {
                heap.push(Reverse((dis + delay, *next)));
            }
        }
        -1
    }
}
```

java

```java
class Solution {
    record edge(int pos, int cost) {
    }

    public int networkDelayTime(int[][] times, int n, int k) {
        ArrayList<ArrayList<edge>> graph = new ArrayList<>();
        for (int i = 0; i < n + 1; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] time : times) {
            graph.get(time[0]).add(new edge(time[1], time[2]));
        }
        boolean[] visited = new boolean[n + 1];
        PriorityQueue<edge> heap = new PriorityQueue<>(Comparator.comparingInt(a -> a.cost));
        heap.offer(new edge(k, 0));
        while (!heap.isEmpty()) {
            edge poll = heap.poll();
            int pos = poll.pos, cost = poll.cost;
            if (visited[pos]) continue;
            visited[pos] = true;
            if (--n == 0) return cost;
            for (edge next : graph.get(pos)) {
                heap.offer(new edge(next.pos, cost + next.cost));
            }
        }
        return -1;
    }
}
```

Go

```go
import (
"github.com/emirpasic/gods/trees/binaryheap"
"github.com/emirpasic/gods/utils"
)

func networkDelayTime(times [][]int, n int, k int) int {
	graph := make([][][2]int, n+1)
	for _, time := range times {
		graph[time[0]] = append(graph[time[0]], [2]int{time[1], time[2]})
	}
	visited := make([]bool, n+1)
	heap := binaryheap.NewWith(func(a, b interface{}) int {
		return utils.IntComparator(a.([2]int)[0], b.([2]int)[0])
	})
	heap.Push([2]int{0, k})
	for {
		pop, ok := heap.Pop()
		if !ok {
			return -1
		}
		head := pop.([2]int)
		dis, start := head[0], head[1]
		if visited[start] {
			continue
		}
		visited[start] = true
		n--
		if n == 0 {
			return dis
		}
		for _, v := range graph[start] {
			target, cost := v[0], v[1]
			heap.Push([2]int{cost + dis, target})
		}
	}
}
```

#### 773. Sliding Puzzle

java

```java
class Solution {
    public int slidingPuzzle(int[][] board) {
        List<Integer> target = List.of(1, 2, 3, 4, 5, 0);
        List<Integer> start = Arrays.stream(board).flatMapToInt(Arrays::stream).boxed().toList();
        record Node(List<Integer> curr, int step) {}
        ArrayDeque<Node> queue = new ArrayDeque<>();
        queue.offer(new Node(start, 0));
        HashSet<List<Integer>> visited = new HashSet<>();
        visited.add(start);
        int[][] dirs = {{1, 3}, {0, 2, 4}, {1, 5}, {0, 4}, {1, 3, 5}, {2, 4}};
        while (!queue.isEmpty()) {
            Node poll = queue.poll();
            List<Integer> curr = poll.curr;
            int step = poll.step;
            if (curr.equals(target)) return step;
            int zero = curr.indexOf(0);
            for (int i : dirs[zero]) {
                ArrayList<Integer> next = new ArrayList<>(curr);
                Collections.swap(next, i, zero);
                if (visited.contains(next)) continue;
                visited.add(next);
                queue.add(new Node(next, step + 1));
            }
        }
        return -1;
    }
}
```

rust

```rust
use std::collections::{HashSet, VecDeque};

impl Solution {
    pub fn sliding_puzzle(board: Vec<Vec<i32>>) -> i32 {
        let target = vec![1, 2, 3, 4, 5, 0];
        let start = board.into_iter().flatten().collect::<Vec<i32>>();
        let dirs = vec![vec![1, 3], vec![0, 2, 4], vec![1, 5], vec![0, 4], vec![1, 3, 5], vec![2, 4]];
        let mut queue = VecDeque::from([(start.clone(), 0)]);
        let mut visited = HashSet::from([start]);

        while let Some((curr, step)) = queue.pop_front() {
            if curr == target {
                return step;
            }
            let zero = curr.iter().position(|&x| x == 0).unwrap();
            for &i in &dirs[zero] {
                let mut next = curr.clone();
                next.swap(i, zero);
                if visited.contains(&next) { continue; }
                queue.push_back((next.clone(), step + 1));
                visited.insert(next);
            }
        }
        -1
    }
}
```

#### 784. Letter Case Permutation

[https://leetcode.com/problems/letter-case-permutation/solutions/115485/java-easy-bfs-dfs-solution-with-explanation/](https://leetcode.com/problems/letter-case-permutation/solutions/115485/java-easy-bfs-dfs-solution-with-explanation/)

其实就是二叉树层序遍历，返回最后一层

go

```go
func letterCasePermutation(s string) []string {  
   queue := []string{s}  
   for i := range s {  
      if s[i] >= '0' && s[i] <= '9' {  
         continue  
      }  
      for range queue {  
         ss := []rune(queue[0])  
         queue = queue[1:]  
  
         ss[i] = unicode.ToUpper(ss[i])  
         queue = append(queue, string(ss))  
  
         ss[i] = unicode.ToLower(ss[i])  
         queue = append(queue, string(ss))  
      }  
  
   }  
   return queue  
}
```

rust

```rust
use std::collections::VecDeque;  
  
impl Solution {  
    pub fn letter_case_permutation(s: String) -> Vec<String> {  
        let mut queue = VecDeque::from([s.to_owned()]);  
        for (i, char) in s.chars().enumerate() {  
            if char.is_numeric() {  
                continue;  
            }  
            for _ in 0..queue.len() {  
                if let Some(s) = queue.pop_front() {  
                    let mut s: Vec<char> = s.chars().collect();  
  
                    s[i] = s[i].to_ascii_lowercase();  
                    queue.push_back(s.iter().collect());  
  
                    s[i] = s[i].to_ascii_uppercase();  
                    queue.push_back(s.iter().collect());  
                }  
            }  
        }  
        Vec::from(queue)  
    }  
}
```

java

```java
class Solution {
    public List<String> letterCasePermutation(String s) {
        Queue<String> queue = new ArrayDeque<>();
        queue.offer(s);
        for (int i = 0; i < s.length(); i++) {
            if (Character.isDigit(s.charAt(i))) {
                continue;
            }
            for (int j = queue.size() - 1; j >= 0; j--) {
                String head = queue.poll();
                assert head != null;
                char[] chars = head.toCharArray();

                chars[i] = Character.toLowerCase(chars[i]);
                queue.offer(String.valueOf(chars));

                chars[i] = Character.toUpperCase(chars[i]);
                queue.offer(String.valueOf(chars));
            }
        }
        return queue.stream().toList();
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
                if (cp[i][j] != 1) continue;
                area.put(mark, bfs(cp, i, j, n, mark));
                res = Math.max(res, area.get(mark++));
            }
        }
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (cp[i][j] != 0) continue;
                HashSet<Integer> seen = new HashSet<>();
                int curr = 1;
                for (int[] dir : dirs) {
                    int x = i + dir[0], y = j + dir[1];
                    if (out(x, y, n)) continue;
                    mark = cp[x][y];
                    if (mark <= 1 || seen.contains(mark)) continue;
                    seen.add(mark);
                    curr += area.get(mark);
                }
                res = Math.max(res, curr);
            }
        }
        return res;
    }

    int bfs(int[][] grid, int i, int j, int n, int mark) {
        Queue<int[]> queue = new ArrayDeque<>(List.of(new int[]{i, j}));
        grid[i][j] = mark;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 0;
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            res++;
            for (int[] dir : dirs) {
                int x = poll[0] + dir[0];
                int y = poll[1] + dir[1];
                if (out(x, y, n) || grid[x][y] != 1) continue;
                grid[x][y] = mark;
                queue.offer(new int[]{x, y});
            }
        }
        return res;
    }

    boolean out(int x, int y, int n) {
        return 0 > x || x >= n || 0 > y || y >= n;
    }
}
```

#### 839. Similar String Groups

java

```java
class Solution {
    public int numSimilarGroups(String[] strs) {
        int n = strs.length;
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) graph.add(new ArrayList<>());
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (similar(strs[i], strs[j])) {
                    graph.get(i).add(j);
                    graph.get(j).add(i);
                }
            }
        }
        int res = 0;
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (visited[i]) continue;
            res++;
            bfs(i, visited, graph);
        }
        return res;
    }

    private boolean similar(String s1, String s2) {
        int diff = 0;
        for (int i = 0; i < s1.length(); i++) {
            if (s1.charAt(i) != s2.charAt(i)) diff++;
            if (diff > 2) return false;
        }
        return diff == 2 || diff == 0;
    }

    private void bfs(int i, boolean[] visited, ArrayList<ArrayList<Integer>> graph) {
        ArrayDeque<Integer> queue = new ArrayDeque<>(List.of(i));
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();
            for (Integer j : graph.get(poll)) {
                if (visited[j]) continue;
                visited[j] = true;
                queue.offer(j);
            }
        }
    }
}
```

#### 841. Keys and Rooms

go

```go
func canVisitAllRooms(rooms [][]int) bool {
	seen := map[int]bool{0: true}
	for queue := []int{0}; len(queue) > 0; queue = queue[1:] {
		head := queue[0]
		for _, next := range rooms[head] {
			if seen[next] {
				continue
			}
			queue = append(queue, next)
			seen[next] = true
			if len(rooms) == len(seen) {
				return true
			}
		}
	}
	return len(rooms) == len(seen)
}
```

#### 958. Check Completeness of a Binary Tree

go

```go
func isCompleteTree(root *TreeNode) bool {
	queue := []*TreeNode{root}
	for ; len(queue) > 0 && queue[0] != nil; queue = queue[1:] {
		head := queue[0]
		queue = append(queue, head.Left)
		queue = append(queue, head.Right)
	}
	for len(queue) > 0 && queue[0] == nil {
		queue = queue[1:]
	}
	return len(queue) == 0
}
```

python

```python
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        queue = collections.deque([root])
        while queue and queue[0]:
            head = queue.popleft()
            queue += head.left,
            queue += head.right,
        while queue and not queue[0]:
            queue.popleft()
        return not queue
```

java

```java
class Solution {
    public boolean isCompleteTree(TreeNode root) {
        LinkedList<TreeNode> queue = new LinkedList<>(List.of(root));
        while (queue.peek() != null) {
            TreeNode head = queue.poll();
            queue.offer(head.left);
            queue.offer(head.right);
        }
        return queue.stream().allMatch(Objects::isNull);
    }
}
```

#### 1091. Shortest Path in Binary Matrix

在图上做宽搜需要用visited标记访问，一律在入队列之前标记visited

Go

```go
func shortestPathBinaryMatrix(grid [][]int) int {
	n := len(grid)
	if grid[0][0] == 1 || grid[n-1][n-1] == 1 {
		return -1
	}
	visited := make([][]bool, n)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	var dirs = [8][2]int{{0, 1}, {0, -1}, {1, 1}, {1, 0}, {1, -1}, {-1, -1}, {-1, 0}, {-1, 1}}
	visited[0][0] = true
	for queue := [][3]int{{0, 0, 1}}; len(queue) > 0; queue = queue[1:] {
		i, j, res := queue[0][0], queue[0][1], queue[0][2]
		if i == n-1 && j == n-1 {
			return res
		}
		for _, dir := range dirs {
			x := i + dir[0]
			y := j + dir[1]
			if x < 0 || x >= n || y < 0 || y >= n || grid[x][y] == 1 || visited[x][y] {
				continue
			}
			visited[x][y] = true
			queue = append(queue, [3]int{x, y, res + 1})
		}
	}
	return -1
}
```

java

```java
class Solution {
    public int shortestPathBinaryMatrix(int[][] grid) {
        int n = grid.length;
        if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1) {
            return -1;
        }
        Queue<int[]> queue = new ArrayDeque<>(List.of(new int[]{0, 0, 1}));
        boolean[][] visited = new boolean[n][n];
        visited[0][0] = true;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 1}, {1, 0}, {1, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            int i = poll[0], j = poll[1], res = poll[2];
            if (i == n - 1 && j == n - 1) {
                return res;
            }
            for (int[] dir : dirs) {
                int x = i + dir[0];
                int y = j + dir[1];
                if (x < 0 || y < 0 || x >= n || y >= n || visited[x][y] || grid[x][y] == 1) {
                    continue;
                }
                visited[x][y] = true;
                queue.offer(new int[]{x, y, res + 1});
            }
        }
        return -1;
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
                    bfs(i, j, m, n, visited, grid);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 || visited[i][j]) continue;
                res++;
                bfs(i, j, m, n, visited, grid);
            }
        }
        return res;
    }

    void bfs(int i, int j, int m, int n, boolean[][] visited, int[][] grid) {
        ArrayDeque<int[]> queue = new ArrayDeque<>();
        queue.add(new int[]{i, j});
        visited[i][j] = true;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            int x = poll[0], y = poll[1];
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                if (nx < 0 || ny < 0 || nx >= m || ny >= n || grid[nx][ny] == 1 || visited[nx][ny]) continue;
                visited[nx][ny] = true;
                queue.offer(new int[]{nx, ny});
            }
        }
    }
}
```

#### 1306. Jump Game III

[https://leetcode.com/problems/jump-game-iii/discuss/571683/Python3-3-Lines-DFS.-O(N)-time-and-space.-Recursion](https://leetcode.com/problems/jump-game-iii/discuss/571683/Python3-3-Lines-DFS.-O(N)-time-and-space.-Recursion)

go

```go
func canReach(arr []int, start int) bool {
	n := len(arr)
	visited := make([]bool, n)
	visited[start] = true
	for queue := []int{start}; len(queue) > 0; queue = queue[1:] {
		head := queue[0]
		if arr[head] == 0 {
			return true
		}
		for _, next := range [2]int{head - arr[head], head + arr[head]} {
			if next >= 0 && next < n && !visited[next] {
				visited[next] = true
				queue = append(queue, next)
			}
		}
	}
	return false
}
```

java

```java
class Solution {
    public boolean canReach(int[] arr, int start) {
        int n = arr.length;
        boolean[] visited = new boolean[n];
        visited[start] = true;
        Queue<Integer> queue = new ArrayDeque<>(List.of(start));
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();
            if (arr[poll] == 0) {
                return true;
            }
            for (int next : List.of(poll + arr[poll], poll - arr[poll])) {
                if (next >= 0 && next < n && !visited[next]) {
                    visited[next] = true;
                    queue.offer(next);
                }
            }
        }
        return false;
    }
}
```


