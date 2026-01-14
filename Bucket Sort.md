#### 347. Top K Frequent Elements

https://leetcode.com/problems/top-k-frequent-elements/solutions/81602/java-o-n-solution-bucket-sort/

go

```go
func topKFrequent(nums []int, k int) []int {  
   hashmap := make(map[int]int)  
   for _, num := range nums {  
      hashmap[num]++  
   }  
  
   n := len(nums)  
   bucket := make([][]int, n+1)  
   for key, value := range hashmap {  
      bucket[value] = append(bucket[value], key)  
   }  
  
   var res []int  
   for i := n; i >= 0 && len(res) < k; i-- {  
      res = append(res, bucket[i]...)  
   }  
   return res  
}
```

rust

```rust
use std::collections::HashMap;

impl Solution {
    pub fn top_k_frequent(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let n = nums.len();
        let map = nums.iter().fold(HashMap::new(), |mut acc, x| {
            *acc.entry(x).or_insert(0) += 1;
            acc
        });
        let mut bucket = map.iter().fold(vec![vec![]; n + 1], |mut acc, (&&key, &value)| {
            acc[value].push(key);
            acc
        });
        let mut res = vec![];
        loop {
            match bucket.pop() {
                Some(x) if res.len() < k as usize => res.extend(x),
                _ => break,
            }
        }
        res
    }
}
```

java

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> counter = new HashMap<>();
        for (int num : nums) {
            counter.put(num, counter.getOrDefault(num, 0) + 1);
        }
        int n = nums.length;
        ArrayList<ArrayList<Integer>> bucket = new ArrayList<>();
        for (int i = 0; i < n + 1; i++) {
            bucket.add(new ArrayList<>());
        }
        for (Integer num : counter.keySet()) {
            bucket.get(counter.get(num)).add(num);
        }
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = bucket.size() - 1; i > 0 && res.size() < k; i--) {
            res.addAll(bucket.get(i));
        }
        return res.stream().mapToInt(i -> i).toArray();
    }
}
```

#### 451. Sort Characters By Frequency

rust

```rust
use std::collections::HashMap;

impl Solution {
    pub fn frequency_sort(s: String) -> String {
        let freq = s.chars().into_iter().fold(HashMap::new(), |mut acc, char| {
            *acc.entry(char).or_insert(0) += 1;
            acc
        });
        let n = s.len();
        let mut bucket = freq.iter().fold(vec![vec![]; n + 1], |mut acc, (&x, &count)| {
            acc[count].push((x, count));
            acc
        });
        let mut res = Vec::with_capacity(n);
        while let Some(v) = bucket.pop() {
            for (x, count) in v {
                res.append(&mut vec![x].repeat(count))
            }
        }
        res.iter().collect()
    }
}
```

go

```go
func frequencySort(s string) string {  
   hashmap := make(map[rune]int)  
   for i := range s {  
      hashmap[rune(s[i])]++  
   }  
   n := len(s)  
   bucket := make([][]rune, n+1)  
   for char, count := range hashmap {  
      bucket[count] = append(bucket[count], char)  
   }  
   res := ""  
   for i := n; i >= 0; i-- {  
      for _, v := range bucket[i] {  
         res += strings.Repeat(string(v), i)  
      }  
   }  
   return res  
}
```

#### 1122. Relative Sort Array

java

```java
class Solution {
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
        int[] bucket = new int[1001];
        for (int i : arr1)
            bucket[i]++;
        int i = 0;
        for (int j : arr2) {
            while (bucket[j]-- > 0) {
                arr1[i++] = j;
            }
        }
        for (int j = 0; j < bucket.length; j++) {
            while (bucket[j]-- > 0) {
                arr1[i++] = j;
            }
        }
        return arr1;
    }
}
```

### 1365. How Many Numbers Are Smaller Than the Current Number

https://leetcode.com/problems/how-many-numbers-are-smaller-than-the-current-number/solutions/575760/Java-simple-to-complex-explained-0-ms-faster-than-100-less-space-than-100-5-lines-of-code/

go

```go
func smallerNumbersThanCurrent(nums []int) []int {
	bucket := make([]int, 102)
	for _, num := range nums {
		bucket[num+1]++
	}
	for i := 1; i < 102; i++ {
		bucket[i] += bucket[i-1]
	}
	res := make([]int, len(nums))
	for i := range nums {
		res[i] = bucket[nums[i]]
	}
	return res
}
```

java

```java
class Solution {
    public int[] smallerNumbersThanCurrent(int[] nums) {
        int[] bucket = new int[102];
        for (int num : nums) {
            bucket[num + 1]++;
        }
        for (int i = 1; i < 102; i++) {
            bucket[i] += bucket[i - 1];
        }
        int n = nums.length;
        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            res[i] = bucket[nums[i]];
        }
        return res;
    }
}
```

rust

```rust
impl Solution {
    pub fn smaller_numbers_than_current(nums: Vec<i32>) -> Vec<i32> {
        let mut bucket = [0; 102];
        for num in &nums {
            bucket[(num + 1) as usize] += 1;
        }
        for i in 1..102 {
            bucket[i] += bucket[i - 1];
        }
        nums.iter().map(|&x| bucket[x as usize]).collect()
    }
}
```