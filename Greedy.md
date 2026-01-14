
#### [1647. Minimum Deletions to Make Character Frequencies Unique](https://leetcode.com/problems/minimum-deletions-to-make-character-frequencies-unique/)
java

```java
class Solution {
    public int minDeletions(String s) {
        int[] counter = new int[26];
        for (char c : s.toCharArray()) {
            counter[c - 'a']++;
        }
        HashSet<Integer> used = new HashSet<>();
        int res = 0;
        for (int i = 0; i < 26; i++) {
            while (counter[i] > 0 && !used.add(counter[i])) {
                counter[i]--;
                res++;
            }
        }
        return res;
    }
}
```

#### 1042. Flower Planting With No Adjacent

java

```java
class Solution {
    private static final int flowerTypes = 4;

    public int[] gardenNoAdj(int n, int[][] paths) {
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] path : paths) {
            int x = path[0], y = path[1];
            graph.get(x - 1).add(y - 1);
            graph.get(y - 1).add(x - 1);
        }
        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            boolean[] colors = new boolean[flowerTypes + 1];
            for (int j : graph.get(i))
                colors[res[j]] = true;
            for (int j : new int[]{1, 2, 3, 4}) {
                if (!colors[j]) res[i] = j;
            }
        }
        return res;
    }
}
```


#### 502. IPO

java

```java
class Solution {
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        for (int i = 0; i < profits.length; i++) {
            minHeap.offer(new int[]{capital[i], profits[i]});
        }
        for (int i = 0; i < k; i++) {
            while (!minHeap.isEmpty() && w >= minHeap.peek()[0]) {
                maxHeap.offer(minHeap.poll());
            }
            if (maxHeap.isEmpty()) {
                break;
            }
            w += maxHeap.poll()[1];
        }
        return w;
    }
}
```

rust

```rust
use std::cmp::Reverse;
use std::collections::BinaryHeap;

impl Solution {
    pub fn find_maximized_capital(k: i32, mut w: i32, profits: Vec<i32>, capital: Vec<i32>) -> i32 {
        let mut min_heap = capital.iter().zip(profits).fold(BinaryHeap::new(), |mut acc, (&cap, prof)| {
            acc.push(Reverse([cap, prof]));
            acc
        });
        let mut max_heap = BinaryHeap::new();
        for _ in 0..k {
            while let Some(Reverse(v)) = min_heap.peek() {
                match v.to_owned() {
                    [cap, prof]if w >= cap => {
                        max_heap.push(prof);
                        min_heap.pop();
                    }
                    _ => break,
                }
            }
            match max_heap.pop() {
                Some(v) => w += v,
                None => break
            }
        }
        w
    }
}
```

#### 621. Task Scheduler

先处理频率最高的任务，分为k组，前k-1足够存下其余元素

每一组的宽度为n+1，因为处理相同任务的冷却时间是n 

加上第k组的，最高频率的任务，需要处理数个单位时间

java

```java
class Solution {
    public int leastInterval(char[] tasks, int n) {
        char[] cnt = new char[26];
        int maxN = 0;
        for (int task : tasks) {
            cnt[task - 'A']++;
            maxN = Math.max(maxN, cnt[task - 'A']);
        }
        int ans = (maxN - 1) * (n + 1);
        for (int i = 0; i < 26; i++)
            if (cnt[i] == maxN) ans++;
        return Math.max(ans, tasks.length);
    }
}
```

rust

```rust
use std::cmp::max;
impl Solution {
    pub fn least_interval(tasks: Vec<char>, n: i32) -> i32 {
        let freq = tasks.iter().fold([0; 26], |mut acc, &task| {
            acc[(task as u8 - b'A') as usize] += 1;
            acc
        });
        let &k = freq.iter().max().unwrap();
        let res = (k - 1) * (n + 1) + freq.iter().filter(|&&x| x == k).count() as i32;
        max(res, tasks.len() as i32)
    }
}
```

#### 630. Course Schedule III

经典贪心，按照结束时间排序。收集课程耗费时间，如果超时就说明之前时间没安排好，所以剔除一个最耗时的

rust

```rust
use std::collections::BinaryHeap;

impl Solution {
    pub fn schedule_course(mut courses: Vec<Vec<i32>>) -> i32 {
        courses.sort_by(|a, b| a[1].cmp(&b[1]));
        let mut time = 0;
        let mut heap = BinaryHeap::new();
        for course in courses {
            time += course[0];
            heap.push(course[0]);
            if time > course[1] {
                time -= heap.pop().unwrap();
            }
        }
        heap.len() as i32
    }
}
```

java

```java
class Solution {
    public int scheduleCourse(int[][] courses) {
        Arrays.sort(courses, Comparator.comparingInt(a -> a[1]));
        int time = 0;
        PriorityQueue<Integer> heap = new PriorityQueue<>((a, b) -> b - a);
        for (int[] course : courses) {
            time += course[0];
            heap.offer(course[0]);
            if (time > course[1]) {
                time -= heap.poll();
            }
        }
        return heap.size();
    }
}
```

go

```go
func scheduleCourse(courses [][]int) int {
	sort.Slice(courses, func(i, j int) bool {
		return courses[i][1] < courses[j][1]
	})
	times := 0
	pq := binaryheap.NewWith(cmp)
	for _, course := range courses {
		times += course[0]
		pq.Push(course[0])
		if times > course[1] {
			top, _ := pq.Pop()
			times -= top.(int)
		}
	}
	return pq.Size()
}
func cmp(a, b interface{}) int {
	return -utils.IntComparator(a.(int), b.(int)) // "-" descending order
}
```

#### 649. Dota2 Senate

https://leetcode.com/problems/dota2-senate/discuss/105858/JavaC%2B%2B-Very-simple-greedy-solution-with-explanation

go

```go
func predictPartyVictory(senate string) string {
   q1, q2 := make([]int, 0), make([]int, 0)
   for i := range senate {
      if senate[i] == 'R' {
         q1 = append(q1, i)
      } else {
         q2 = append(q2, i)
      }
   }
   for len(q1) > 0 && len(q2) > 0 {
      r, d := q1[0], q2[0]
      q1, q2 = q1[1:], q2[1:]
      if r < d {
         q1 = append(q1, r+len(senate))
      } else {
         q2 = append(q2, d+len(senate))
      }
   }
   if len(q1) > len(q2) {
      return "Radiant"
   }
   return "Dire"
}
```

python

```python
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        q1, q2 = deque(), deque()
        for i in range(len(senate)):
            if senate[i] == 'R':
                q1 += i,
            else:
                q2 += i,
        while q1 and q2:
            r, d = q1.popleft(), q2.popleft()
            if r < d:
                q1 += r + len(senate),
            else:
                q2 += d + len(senate),
        return "Radiant" if len(q1) > len(q2) else "Dire"
```

#### 767. Reorganize String

rust

```rust
impl Solution {  
    pub fn reorganize_string(s: String) -> String {  
        let n = s.len();  
        let mut freq = s.as_bytes().iter().fold([0; 26], |mut acc, ch| {  
            acc[(ch - b'a') as usize] += 1;  
            acc  
        });  
        let (&max_count, letter) = freq.iter().enumerate().map(|(a, b)| (b, a)).max().unwrap();  
        if max_count > (n + 1) / 2 {  
            return "".to_string();  
        }  
        let mut res = vec![0; n];  
        let mut idx = 0;  
        while freq[letter] > 0 {  
            res[idx] = letter as u8 + b'a';  
            idx += 2;  
            freq[letter] -= 1;  
        }  
        for (i, count) in freq.iter_mut().enumerate() {  
            while *count > 0 {  
                if idx >= n {  
                    idx = 1;  
                }  
                res[idx] = i as u8 + b'a';  
                idx += 2;  
                *count -= 1;  
            }  
        }  
        res.iter().map(|&x| x as char).collect::<String>()  
    }  
}
```


#### 871. Minimum Number of Refueling Stops

https://leetcode.com/problems/minimum-number-of-refueling-stops/solutions/2451482/elixir-rust-priority-queue-solution/

https://leetcode.com/problems/minimum-number-of-refueling-stops/solutions/2452351/rust-heap-solution/

rust

```rust
use std::collections::BinaryHeap;

impl Solution {
    pub fn min_refuel_stops(target: i32, mut curr: i32, mut stations: Vec<Vec<i32>>) -> i32 {
        let mut heap = BinaryHeap::with_capacity(stations.len());
        stations.push(vec![target, 0]);
        stations.iter().fold(0, |mut acc, station| {
            let position = station[0];
            let fuel = station[1];
            while curr < position {
                match heap.pop() {
                    Some(gas) => {
                        curr += gas;
                        acc += 1
                    }
                    None => return -1
                }
            }
            heap.push(fuel);
            acc
        })
    }
}
```

java

```java
class Solution {
    public int minRefuelStops(int target, int curr, int[][] stations) {
        PriorityQueue<Integer> heap = new PriorityQueue<>((a, b) -> b - a);
        int res = 0, n = stations.length;
        int[][] s = Arrays.copyOf(stations, n + 1);
        s[n] = new int[]{target, 0};
        for (int[] ints : s) {
            int position = ints[0];
            int fuel = ints[1];
            while (curr < position) {
                if (heap.isEmpty()) {
                    return -1;
                }
                curr += heap.poll();
                res++;
            }
            heap.offer(fuel);
        }
        return res;
    }
}
```

go

```go
func minRefuelStops(target int, curr int, stations [][]int) int {
	heap := binaryheap.NewWith(func(a, b interface{}) int {
		return -utils.IntComparator(a, b)
	})
	res := 0
	stations = append(stations, []int{target, 0})
	for _, station := range stations {
		position := station[0]
		fuel := station[1]
		for curr < position {
			val, ok := heap.Pop()
			if !ok {
				return -1
			}
			curr += val.(int)
			res++
		}
		heap.Push(fuel)
	}
	return res
}
```

#### 1007. Minimum Domino Rotations For Equal Row

https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/solutions/252242/java-c-python-different-ideas/?orderBy=most_votes

go

```go
func minDominoRotations(A []int, B []int) int {  
   n := len(A)  
   for i, a, b := 0, 0, 0; i < n && A[i] == A[0] || B[i] == A[0]; i++ {  
      if A[i] != A[0] {  
         a++  
      }  
      if B[i] != A[0] {  
         b++  
      }  
      if i == n-1 {  
         return min(a, b)  
      }  
   }  
   for i, a, b := 0, 0, 0; i < n && A[i] == B[0] || B[i] == B[0]; i++ {  
      if A[i] != B[0] {  
         a++  
      }  
      if B[i] != B[0] {  
         b++  
      }  
      if i == n-1 {  
         return min(a, b)  
      }  
   }  
   return -1  
}  
func min(a, b int) int {  
   if a > b {  
      return b  
   }  
   return a  
}
```

#### 860. Lemonade Change

go

```go
func lemonadeChange(bills []int) bool {
    five, ten := 0, 0
    for _, v := range bills {
        switch v {
        case 5:
            five++
        case 10:
            five--
            ten++
        case 20:
            if ten > 0 {
                ten--
                five--
            } else {
                five -= 3
            }
        }
        if five < 0 || ten < 0 {
            return false
        }
    }
    return true
}
```

#### 881. Boats to Save People

go

```go
func numRescueBoats(people []int, limit int) int {
   sort.Ints(people)
   ret := 0
   light, heavy := 0, len(people)-1
   for light <= heavy {
      if people[light]+people[heavy] <= limit {
         light++
      }
      heavy--
      ret++
   }
   return ret
}
```

python

```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        ret = 0
        people.sort()
        light, heavy = 0, len(people) - 1
        while light <= heavy:
            if people[light] + people[heavy] <= limit:
                light += 1
            heavy -= 1
            ret += 1
        return ret
```

#### Reorder array to construct the minimum number

java

```java
class Solution {
    public String minNumber(int[] nums) {
        int n = nums.length;
        ArrayList<String> strings = new ArrayList<>();
        for (int num : nums) strings.add(String.valueOf(num));
        strings.sort((s1, s2) -> s1.concat(s2).compareTo(s2.concat(s1)));

        String join = String.join("", strings);
        int i = 0;
        while (i < n && join.charAt(i) == '0')
            i++;
        return i == n ? "0" : join.substring(i);
    }
}
```