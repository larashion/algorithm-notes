如果是新加入一个元素，就是加到队尾，然后不断上浮

pop之后把队尾元素拿到顶部然后下沉，比较子节点，与更大的子节点交换

#### 23. Merge k Sorted Lists

java

```java
class Solution {
    public static ListNode mergeKLists(ListNode[] lists) {
        if (lists == null) {
            return null;
        }
        PriorityQueue<ListNode> heap = new PriorityQueue<>(Comparator.comparingInt(a -> a.val));
        for (ListNode l : lists)
            if (l != null) heap.offer(l);
        ListNode dummy = new ListNode();
        ListNode tail = dummy;
        while (!heap.isEmpty()) {
            ListNode poll = heap.poll();
            tail.next = poll;
            tail = tail.next;
            if (poll.next != null) {
                heap.add(poll.next);
            }
        }
        return dummy.next;
    }
}
```


#### 215. Kth Largest Element in an Array

仅供娱乐，不符合题意的`O(n)`要求

rust

```rust
use std::cmp::Reverse;
use std::collections::BinaryHeap;

impl Solution {
    pub fn find_kth_largest(nums: Vec<i32>, k: i32) -> i32 {
        let heap = nums.iter().fold(BinaryHeap::new(), |mut acc, x| {
            acc.push(Reverse(x));
            if acc.len() > k as usize {
                acc.pop();
            }
            acc
        });
        let Reverse(&res) = *heap.peek().unwrap();
        res
    }
}
```

java

```java
class Solution {  
    public int findKthLargest(int[] nums, int k) {  
        PriorityQueue<Integer> heap = new PriorityQueue<>();  
        for (int num : nums) {  
            heap.offer(num);  
            if (heap.size() > k) {  
                heap.poll();  
            }  
        }  
        return heap.peek();  
    }  
}
```

#### 218. The Skyline Problem 

https://leetcode.cn/problems/the-skyline-problem/solutions/873332/you-xian-dui-lie-java-by-liweiwei1419-jdb5/

相当于区间修改，求最大值的问题，但是线段树的常数时间太大了

建筑的轮廓可以理解为“左端高度产生”和“右端高度消失”两个事件

收集产生高度差的点即可，初始高度为0
按照横坐标排序，横坐标相同的时候，高度高的在前面
如果高度属于左端点，就加入堆，右端点就删除

find the critical points that change the max height among the buildings on the left

java

```java
public class Solution {
    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<int[]> heights = new ArrayList<>();
        for (int[] b : buildings) {
            heights.add(new int[]{b[0], -b[2]});
            heights.add(new int[]{b[1], b[2]});
        }
        heights.sort((a, b) -> (a[0] != b[0]) ? a[0] - b[0] : a[1] - b[1]);
        HashMap<Integer, Integer> delay = new HashMap<>();
        PriorityQueue<Integer> heap = new PriorityQueue<>((a, b) -> b - a);
        heap.offer(0);
        int prev = 0;
        List<List<Integer>> res = new ArrayList<>();
        for (int[] h : heights) {
            if (h[1] < 0) 
                heap.offer(-h[1]); // meets a new building
            else
	            delay.put(h[1], delay.getOrDefault(h[1], 0) + 1); // Don't add '-'

            while (!heap.isEmpty() && delay.containsKey(heap.peek())) {
                Integer curr = heap.peek();
                if (delay.get(curr) == 1)
                    delay.remove(curr);
                else
                    delay.put(curr, delay.get(curr) - 1);
                heap.poll();
            }
            int curr = heap.peek();
            if (curr != prev) {
                res.add(List.of(h[0], curr));
                prev = curr;
            }
        }
        return res;
    }
}
```

rust

```rust
use std::collections::{BinaryHeap, HashMap};

impl Solution {
    pub fn get_skyline(buildings: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut points = buildings.iter().fold(vec![], |mut acc, b| {
            acc.push([b[0], -b[2]]);
            acc.push([b[1], b[2]]);
            acc
        });
        points.sort();
        let mut delay = HashMap::new();
        let mut max_heap = BinaryHeap::from([0]);
        let mut prev = 0;
        let mut res = vec![];
        for [x, height] in points {
            if height < 0 {
                max_heap.push(-height)
            } else {
                *delay.entry(height).or_insert(0) += 1
            }
            loop {
                match max_heap.peek() {
                    Some(top) if delay.contains_key(top) => {
                        *delay.entry(*top).or_default() -= 1;
                        if delay[top] == 0 {
                            delay.remove(top);
                        }
                        max_heap.pop();
                    }
                    _ => break
                }
            }
            if let Some(&curr) = max_heap.peek() {
                if curr != prev {
                    res.push(vec![x, curr]);
                    prev = curr
                }
            }
        }
        res
    }
}
```

#### 239. Sliding Window Maximum

可以懒删除

java

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] res = new int[n - k + 1];
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> nums[b] - nums[a]);
        for (int i = 0; i < n; i++) {
            pq.offer(i);
            if (i >= k - 1) {
                while (!pq.isEmpty() && pq.peek() < i - k + 1)
                    pq.poll();
                res[i - k + 1] = nums[pq.peek()];
            }
        }
        return res;
    }
}
```

#### 295. Find Median from Data Stream

https://leetcode.com/problems/find-median-from-data-stream/solutions/74047/java-python-two-heap-solution-o-log-n-add-o-1-find/?orderBy=most_votes

rust

```rust
use std::cmp::Reverse;  
use std::collections::BinaryHeap;  
  
#[derive(Default)]  
struct MedianFinder {  
    small: BinaryHeap<i32>,  
    large: BinaryHeap<Reverse<i32>>,  
    length: usize,  
}  
  
impl MedianFinder {  
    fn new() -> Self {  
        Default::default()  
    }  
  
    fn add_num(&mut self, num: i32) {  
        if self.length & 1 == 0 {  
            self.large.push(Reverse(num));  
            let Reverse(x) = self.large.pop().unwrap();  
            self.small.push(x);  
        } else {  
            self.small.push(num);  
            let x = self.small.pop().unwrap();  
            self.large.push(Reverse(x));  
        }  
        self.length += 1;  
    }  
  
    fn find_median(&self) -> f64 {  
        if self.length & 1 == 0 {  
            let x1 = self.small.peek().unwrap();  
            let Reverse(x2) = self.large.peek().unwrap();  
            (x1 + x2) as f64 / 2.0  
        } else {  
            self.small.peek().unwrap().to_owned() as f64  
        }  
    }  
}
```

java

```java
class MedianFinder {  
    PriorityQueue<Integer> large;  
    PriorityQueue<Integer> small;  
    int len;  
  
    public MedianFinder() {  
        large = new PriorityQueue<>();  
        small = new PriorityQueue<>((a, b) -> b - a);  
    }  
  
    public void addNum(int num) {  
        if ((len & 1) == 0) {  
            large.offer(num);  
            small.offer(large.poll());  
        } else {  
            small.offer(num);  
            large.offer(small.poll());  
        }  
        len++;  
    }  
  
    public double findMedian() {  
        if ((len & 1) == 0) {  
            return (large.peek() + small.peek()) / 2.0;  
        } else {  
            return small.peek();  
        }  
    }  
}
```


#### 347. Top K Frequent Elements

java

```go
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> counter = new HashMap<>();
        for (int num : nums) {
            counter.put(num, counter.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Map.Entry<Integer, Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(Map.Entry::getValue));
        for (Map.Entry<Integer, Integer> entry : counter.entrySet()) {
            pq.offer(entry);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        return pq.stream().map(Map.Entry::getKey).mapToInt(i -> i).toArray();
    }
}
```

#### 373. Find K Pairs with Smallest Sums

rust

```rust
use std::cmp::Reverse;
use std::collections::BinaryHeap;

impl Solution {
    pub fn k_smallest_pairs(nums1: Vec<i32>, nums2: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
        fn push(nums1: &[i32], nums2: &[i32], i: usize, j: usize, heap: &mut BinaryHeap<Reverse<(i32, usize, usize)>>) {
            if i < nums1.len() && j < nums2.len() {
                heap.push(Reverse((nums1[i] + nums2[j], i, j)));
            }
        }
        let mut heap: BinaryHeap<Reverse<(i32, usize, usize)>> = BinaryHeap::new();
        push(&nums1, &nums2, 0, 0, &mut heap);

        let mut res = vec![];
        while res.len() < k as usize {
            if let Some(Reverse((_, i, j))) = heap.pop() {
                res.push(vec![nums1[i], nums2[j]]);
                push(&nums1, &nums2, i, j + 1, &mut heap);
                if j == 0 { push(&nums1, &nums2, i + 1, j, &mut heap); }
            } else {
                break;
            }
        }
        res
    }
}
```

java

```java
class Solution {  
  
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {  
        PriorityQueue<int[]> heap = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));  
        pushBatch(nums1, nums2, 0, 0, heap);  
        List<List<Integer>> res = new ArrayList<>();  
        while (!heap.isEmpty() && res.size() < k) {  
            int[] curr = heap.poll();  
            res.add(Arrays.asList(nums1[curr[1]], nums2[curr[2]]));  
            pushBatch(nums1, nums2, curr[1], curr[2] + 1, heap);  
            if (curr[2] == 0) {  
                pushBatch(nums1, nums2, curr[1] + 1, curr[2], heap);  
            }  
        }  
        return res;  
    }  
  
    void pushBatch(int[] nums1, int[] nums2, int i, int j, PriorityQueue<int[]> heap) {  
        if (i < nums1.length && j < nums2.length) {  
            heap.offer(new int[]{nums1[i] + nums2[j], i, j});  
        }  
    }  
}
```

#### 407. Trapping Rain Water II

java

```java
class Solution {
    record Cell(int row, int col, int height) {
    }

    public int trapRainWater(int[][] heights) {
        int m = heights.length, n = heights[0].length;
        boolean[][] visited = new boolean[m][n];
        PriorityQueue<Cell> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.height));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    visited[i][j] = true;
                    pq.offer(new Cell(i, j, heights[i][j]));
                }
            }
        }
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 0;
        while (!pq.isEmpty()) {
            Cell cell = pq.poll();
            for (int[] dir : dirs) {
                int x = cell.row + dir[0];
                int y = cell.col + dir[1];
                if (x < 0 || y < 0 || x >= m || y >= n || visited[x][y]) {
                    continue;
                }
                visited[x][y] = true;
                res += Math.max(0, cell.height - heights[x][y]);
                pq.offer(new Cell(x, y, Math.max(heights[x][y], cell.height)));
            }
        }
        return res;
    }
}
```

rust

```rust
use std::cmp::{max, Ordering};
use std::collections::BinaryHeap;

#[derive(PartialEq, Eq)]
struct Cell {
    row: usize,
    col: usize,
    height: i32,
}

impl PartialOrd<Self> for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        other.height.cmp(&self.height)
    }
}

impl Solution {
    pub fn trap_rain_water(heights: Vec<Vec<i32>>) -> i32 {
        let m = heights.len();
        let n = heights[0].len();
        let mut visited = vec![vec![false; n]; m];
        let mut heap = BinaryHeap::new();
        for i in 0..m {
            for j in 0..n {
                if i == m - 1 || i == 0 || j == n - 1 || j == 0 {
                    heap.push(Cell { row: i, col: j, height: heights[i][j] });
                }
            }
        }
        let dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]];
        let mut res = 0;
        while let Some(cell) = heap.pop() {
            visited[cell.row][cell.col] = true;
            for dir in dirs {
                let x = cell.row as i32 + dir[0];
                let y = cell.col as i32 + dir[1];
                if x < 0 || y < 0 {
                    continue;
                }
                let x = x as usize;
                let y = y as usize;
                if x >= m || y >= n || visited[x][y] {
                    continue;
                }
                visited[x][y] = true;
                res += max(cell.height - heights[x][y], 0);
                heap.push(Cell { row: x, col: y, height: max(heights[x][y], cell.height) });
            }
        }
        res
    }
}
```

#### 451. Sort Characters By Frequency

java

```java
class Solution {
    public String frequencySort(String s) {
        HashMap<String, Integer> map = new HashMap<>();
        Arrays.stream(s.split("")).forEach(c -> map.put(c, map.getOrDefault(c, 0) + 1));
        PriorityQueue<Map.Entry<String, Integer>> heap = new PriorityQueue<>((a, b) -> b.getValue() - a.getValue());
        map.entrySet().forEach(heap::offer);
        StringBuilder sb = new StringBuilder();
        while (!heap.isEmpty()) {
            var head = heap.poll();
            String ss = head.getKey().repeat(head.getValue());
            sb.append(ss);
        }
        return sb.toString();
    }
}
```

rust

```rust
use std::collections::{BinaryHeap, HashMap};

impl Solution {
    pub fn frequency_sort(s: String) -> String {
        let freq = s.chars().into_iter().fold(HashMap::new(), |mut acc, char| {
            *acc.entry(char).or_insert(0) += 1;
            acc
        });
        let mut heap: BinaryHeap<(usize, char)> = freq.iter().map(|(&char, &count)| (count, char)).collect();
        let mut res = Vec::with_capacity(s.len());
        while let Some((count, char)) = heap.pop() {
            res.extend(vec![char].repeat(count))
        }
        res.iter().collect()
    }
}
```

go

```go
type freq struct {
	char rune
	val  int
}

func frequencySort(s string) string {
	hashmap := make(map[rune]int)
	for i := range s {
		hashmap[rune(s[i])]++
	}
	heap := binaryheap.NewWith(func(a, b interface{}) int {
		return -utils.IntComparator(a.(*freq).val, b.(*freq).val)
	})
	for k, v := range hashmap {
		heap.Push(&freq{val: v, char: k})
	}
	res := ""
	for {
		pop, ok := heap.Pop()
		if !ok {
			return res
		}
		head := pop.(*freq)
		res += strings.Repeat(string(head.char), head.val)
	}
}
```

#### 692. Top K Frequent Words

rust

https://leetcode.com/problems/top-k-frequent-words/solutions/2721398/python-rust-c-concise-using-map-heap-with-detailed-comments/

```rust
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

impl Solution {
    pub fn top_k_frequent(words: Vec<String>, k: i32) -> Vec<String> {
        let freq = words.iter().fold(HashMap::new(), |mut acc, word| {
            *acc.entry(word).or_insert(0) += 1;
            acc
        });
        let mut heap: BinaryHeap<(i32, Reverse<&String>)> = freq
            .iter()
            .map(|(&word, &count)| (count, Reverse(word)))
            .collect();
        (0..k)
            .map(|_| {
                match heap.pop() {
                    Some((_, Reverse(x))) => x.to_owned(),
                    None => panic!()
                }
            })
            .collect()
    }
}
```

java

```java
class Solution {
    public static List<String> topKFrequent(String[] words, int k) {
        HashMap<String, Integer> map = new HashMap<>();
        Arrays.stream(words).forEach(a -> map.put(a, map.getOrDefault(a, 0) + 1));
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(
                (a, b) -> {
                    if (a.getValue() == b.getValue()) {
                        return a.getKey().compareTo(b.getKey());
                    }
                    return b.getValue() - a.getValue();
                }
        );
        map.entrySet().forEach(pq::offer);
        ArrayList<String> res = new ArrayList<>();
        while (res.size() < k) {
            res.add(pq.poll().getKey());
        }
        return res;
    }
}
```

go

```go
type freq struct {
	word string
	val  int
}

func topKFrequent(words []string, k int) []string {
	heap := binaryheap.NewWith(func(a, b interface{}) int {
		A := a.(*freq)
		B := b.(*freq)
		if A.val != B.val {
			return -utils.IntComparator(A.val, B.val)
		}
		return utils.StringComparator(A.word, B.word)
	})
	m := make(map[string]int)
	for _, word := range words {
		m[word]++
	}
	for key, value := range m {
		heap.Push(&freq{word: key, val: value})
	}
	var res []string
	for i := 0; i < k; i++ {
		top, _ := heap.Pop()
		res = append(res, top.(*freq).word)
	}
	return res
}
```


#### 778. Swim in Rising Water

java

```java
class Solution {
    record cell(int i, int j, int water) {
    }

    public int swimInWater(int[][] grid) {
        int n = grid.length;
        PriorityQueue<cell> heap = new PriorityQueue<>(Comparator.comparingInt(a -> a.water));
        heap.offer(new cell(0, 0, grid[0][0]));
        int res = 0;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        boolean[][] visited = new boolean[n][n];
        visited[0][0] = true;
        while (!heap.isEmpty()) {
            cell poll = heap.poll();
            int i = poll.i, j = poll.j, water = poll.water;
            res = Math.max(res, water);
            if (i == n - 1 && j == n - 1) return res;
            for (int[] dir : dirs) {
                int x = i + dir[0], y = j + dir[1];
                if (x < 0 || x >= n || y < 0 || y >= n || visited[x][y]) continue;
                visited[x][y] = true;
                heap.offer(new cell(x, y, grid[x][y]));
            }
        }
        throw new Error();
    }
}
```

#### 857. Minimum Cost to Hire K Workers

java

```java
class Solution {
    public double mincostToHireWorkers(int[] quality, int[] wage, int k) {
        int n = quality.length;
        double[][] workers = new double[n][2];
        for (int i = 0; i < n; ++i)
            workers[i] = new double[]{(double) (wage[i]) / quality[i], (double) quality[i]};
        Arrays.sort(workers, Comparator.comparingDouble(a -> a[0]));
        double res = Double.MAX_VALUE, total = 0;
        PriorityQueue<Double> pq = new PriorityQueue<>((a, b) -> (int) (b - a));
        for (double[] worker : workers) {
            total += worker[1];
            pq.add(worker[1]);
            if (pq.size() > k) total -= pq.poll();
            if (pq.size() == k) res = Math.min(res, total * worker[0]);
        }
        return res;
    }
}
```

#### 1046. Last Stone Weight

java

```java
class Solution {
    public int lastStoneWeight(int[] A) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b)-> b - a);
        for (int a : A)
            pq.offer(a);
        while (pq.size() > 1)
            pq.offer(pq.poll() - pq.poll());
        return pq.poll();
    }
}
```

rust

```rust
use std::collections::BinaryHeap;  
  
impl Solution {  
    pub fn last_stone_weight(stones: Vec<i32>) -> i32 {  
        let mut heap: BinaryHeap<i32> = stones.into_iter().collect();  
        while heap.len() > 1 {  
            let x1 = heap.pop().unwrap();  
            let x2 = heap.pop().unwrap();  
            heap.push(x1 - x2);  
        }  
        heap.pop().unwrap()  
    }  
}
```

go

```go
func lastStoneWeight(stones []int) int {
	heap := binaryheap.NewWith(func(a, b interface{}) int {
		return -utils.IntComparator(a, b)
	})
	for _, stone := range stones {
		heap.Push(stone)
	}
	for heap.Size() > 1 {
		a, _ := heap.Pop()
		b, _ := heap.Pop()
		heap.Push(a.(int) - b.(int))
	}
	res, _ := heap.Peek()
	return res.(int)
}
```

#### 1167. Minimum Cost to Connect Sticks

java

```java
public class Solution {  
    public int minimumCost(List<Integer> sticks) {  
        PriorityQueue<Integer> pq = new PriorityQueue<>();  
        sticks.forEach(pq::offer);  
        int res = 0;  
        while (pq.size() > 1) {  
            int A = pq.poll();  
            int B = pq.poll();  
            res += A + B;  
            pq.offer(A + B);  
        }  
        return res;  
    }  
}
```

#### 1851. Minimum Interval to Include Each Query

java

```java
class Solution {
    public int[] minInterval(int[][] intervals, int[] queries) {
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        int n = queries.length, m = intervals.length;
        HashMap<Integer, Integer> res = new HashMap<>();
        int[] Q = queries.clone();
        Arrays.sort(Q);
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1] - a[0]));
        int i = 0;
        for (int query : Q) {
            while (i < m && intervals[i][0] <= query) pq.offer(intervals[i++]);
            while (!pq.isEmpty() && pq.peek()[1] < query) pq.poll();
            res.put(query, pq.isEmpty() ? -1 : pq.peek()[1] - pq.peek()[0] + 1);
        }

        int[] res1 = new int[n];
        for (int j = 0; j < queries.length; j++) {
            int query = queries[j];
            res1[j] = res.get(query);
        }
        return res1;
    }
}
```

rust

```rust
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};


#[derive(PartialEq, Eq)]
struct Interval {
    len: i32,
    left: i32,
    right: i32,
}


impl PartialOrd<Self> for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Interval {
    fn cmp(&self, other: &Self) -> Ordering {
        other.len.cmp(&self.len)
    }
}

impl Solution {
    pub fn min_interval(intervals: Vec<Vec<i32>>, queries: Vec<i32>) -> Vec<i32> {
        let mut intervals = intervals.clone();
        intervals.sort_unstable();
        let mut q = queries.clone();
        q.sort_unstable();
        let m = intervals.len();
        let mut i = 0;
        let mut pq = BinaryHeap::new();
        let res = q.iter().fold(HashMap::new(), |mut acc, &query| {
            while i < m && query >= intervals[i][0] {
                pq.push(Interval {
                    left: intervals[i][0],
                    right: intervals[i][1],
                    len: intervals[i][1] - intervals[i][0] + 1,
                });
                i += 1;
            }
            while !pq.is_empty() && query > pq.peek().unwrap().right {
                pq.pop();
            }
            match pq.peek() {
                None => {
                    acc.insert(query, -1);
                }
                Some(top) => {
                    acc.insert(query, top.len);
                }
            }
            acc
        });
        queries.iter().fold(vec![], |mut acc, query| {
            acc.push(res[query]);
            acc
        })
    }
}
```

#### 1584. Min Cost to Connect All Points

prim

O(E log E)   所有边最多入队一次出队一次 n^2 log n

java

```java
class Solution {
    public int minCostConnectPoints(int[][] points) {
        int n = points.length;
        boolean[] visited = new boolean[n];
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>(Comparator.comparingInt(Pair::getKey));
        int res = 0, i = 0;
        for (int edge = 0; edge < n - 1; edge++) {
            visited[i] = true;
            for (int j = 0; j < n; ++j)
                if (!visited[j]) {
                    int abs = Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1]);
                    pq.offer(new Pair<>(abs, j));
                }
            while (visited[pq.peek().getValue()])
                pq.poll();
            Pair<Integer, Integer> poll = pq.poll();
            res += poll.getKey();
            i = poll.getValue();
        }
        return res;
    }
}
```

prim

O(E * V)    每一次都扫描所有边，持续V次    n^3

java

```java
class Solution {
    public int minCostConnectPoints(int[][] points) {
        int n = points.length, res = 0, i = 0;
        int[] dp = new int[n];
        Arrays.fill(dp, (int) 1e7);
        for (int connected = 0; connected < n - 1; connected++) {
            dp[i] = Integer.MAX_VALUE;
            int min_j = i;
            for (int j = 0; j < n; j++) {
                if (dp[j] == Integer.MAX_VALUE) continue;
                dp[j] = Math.min(
                        dp[j],
                        Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1]));
                if (dp[j] < dp[min_j]) {
                    min_j = j;
                }
            }
            res += dp[min_j];
            i = min_j;
        }
        return res;
    }
}
```

prim

O(E log E) 就是和边的数量有关  O(n^2 * log n)

```java
class Solution {
    public int minCostConnectPoints(int[][] points) {
        int n = points.length;
        int res = 0;
        int[] dp = new int[n];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        boolean[] visited = new boolean[n];
        PriorityQueue<Integer> pq = new PriorityQueue<>(Comparator.comparingInt(a -> dp[a]));
        for (int i = 0; i < n; i++) pq.offer(i);
        List<int[]>[] adj = new List[n];
        for (int i = 0; i < n; i++) {
            adj[i] = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    int dist = Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1]);
                    adj[i].add(new int[]{j, dist});
                }
            }
        }
        while (!pq.isEmpty()) {
            int i = pq.poll();
            if (visited[i]) continue;
            visited[i] = true;
            res += dp[i];
            for (int[] edge : adj[i]) {
                int j = edge[0], w = edge[1];
                if (!visited[j] && w < dp[j]) {
                    dp[j] = w;
                    pq.remove(j);
                    pq.offer(j);
                }
            }
        }
        return res;
    }
}
```