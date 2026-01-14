
#### 1218. Longest Arithmetic Subsequence of Given Difference

Go

```go
func longestSubsequence(arr []int, difference int) int {
	m := make(map[int]int)
	res := 0
	for _, num := range arr {
		if _, ok := m[num-difference]; ok {
			m[num] = m[num-difference] + 1
		} else {
			m[num] = 1
		}
		res = max(res, m[num])
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
    public int longestSubsequence(int[] arr, int difference) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            if (map.containsKey(num - difference)) {
                map.put(num, map.get(num - difference) + 1);
            } else {
                map.put(num, 1);
            }
        }
        return Collections.max(map.values());
    }
}
```
#### 205. Isomorphic Strings

Go

```go
func isIsomorphic(s string, t string) bool {  
   m1, m2 := make([]int, 256), make([]int, 256)  
   for i := range s {  
      if m1[s[i]] != m2[t[i]] {  
         return false  
      }  
      m1[s[i]] = i + 1  
      m2[t[i]] = i + 1  
   }  
   return true  
}
```

java

```java
class Solution {
    public boolean isIsomorphic(String s, String t) {
        int[] m1 = new int[256];
        int[] m2 = new int[256];
        for (int i = 0; i < s.length(); i++) {
            if (m1[s.charAt(i)] != m2[t.charAt(i)]) return false;
            m1[s.charAt(i)] = i + 1;
            m2[t.charAt(i)] = i + 1;
        }
        return true;
    }
}
```


#### 128. Longest Consecutive Sequence

https://leetcode.com/problems/longest-consecutive-sequence/discuss/1252849/go-iterate-over-map-10x-faster-than-over-slice

```go
func longestConsecutive(nums []int) int {
    maxLen := 0
    m := make(map[int]bool)
    for _, v := range nums {
        m[v] = true
    }
    for v := range m {
        if !m[v-1] {
            w := v + 1
            for m[w] {
                w++
            }
            maxLen = max(maxLen, w-v)
        }
    }
    return maxLen
}
```

python

参考https://leetcode.com/problems/longest-consecutive-sequence/discuss/41057/Simple-O(n)-with-Explanation-Just-walk-each-streak

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        max_len = 0
        nums = set(nums)
        for x in nums:
            if x - 1 not in nums:
                y = x + 1
                while y in nums:
                    y += 1
                max_len = max(y - x, max_len)
        return max_len
```

#### 1. Two Sum

python

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        m = {}
        for k, v in enumerate(nums):
            if target - v in m:
                return [k,m[target- v]]
            m[v] = k
```

go

```go
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)
    for k, v := range nums {
        if i, ok := m[target-v]; ok {
            return []int{k, i}
        }
        m[v] = k
    }
    return []int{-1 ,-1}
}
```

java

```java
class Solution {  
  public int[] twoSum(int[] nums, int target) {  
    HashMap<Integer, Integer> hashMap = new HashMap<>();  
    for (int i = 0; i < nums.length; i++) {  
      if (hashMap.containsKey(target - nums[i])) {  
        return new int[] {i, hashMap.get(target - nums[i])};  
      }  
      hashMap.put(nums[i], i);  
    }  
    return new int[] {-1, -1};  
  }  
}
```

rust

```rust
impl Solution {
	fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
	    use std::collections::HashMap;
	    let mut map: HashMap<i32, i32> = HashMap::new();
	    for (i, &v) in nums.iter().enumerate() {
	        let i = i as i32;
	        let prev_value = target - v;
	        match map.get(&prev_value) {
	            Some(&j) => return vec![i, j],
	            None => map.insert(v, i),
	        };
	    }
	    vec![]
	}
}
```


#### 49. Group Anagrams

go

https://leetcode.com/problems/group-anagrams/solutions/1399682/Go-solution/

```go
func groupAnagrams(strings []string) [][]string {  
   var res [][]string  
   hashmap := make(map[[26]rune][]string)  
   for _, s := range strings {  
      var arr [26]rune  
      for i := range s {  
         arr[s[i]-'a']++  
      }  
      hashmap[arr] = append(hashmap[arr], s)  
   }  
   for v := range hashmap {  
      res = append(res, hashmap[v])  
   }  
   return res  
}
```

java

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> map = new HashMap<>();
        Arrays.stream(strs).forEach(str -> {
            char[] counter = new char[26];
            for (char c : str.toCharArray()) {
                counter[c - 'a']++;
            }
            String key = String.valueOf(counter);
            map.putIfAbsent(key, new ArrayList<>());
            map.get(key).add(str);
        });
        return map.values().stream().toList();
    }
}
```

rust

```rust
use std::collections::HashMap;

impl Solution {
    pub fn group_anagrams(strings: Vec<String>) -> Vec<Vec<String>> {
        let mut map: HashMap<[u8; 26], Vec<String>> = HashMap::with_capacity(strings.len());
        for s in strings {
            let key = s.bytes().fold([0u8; 26], |mut key, byte| {
                key[(byte - b'a') as usize] += 1;
                key
            });
            map.entry(key).or_default().push(s);
        }
        map.into_values().collect() // 直接移动出的 vec
    }
}

```

#### 187. Repeated DNA Sequences

bit/sliding window/hashmap

```go
func findRepeatedDnaSequences(s string) []string {  
   const L = 10  
   m := map[byte]int{  
      'A': 0,  
      'C': 1,  
      'G': 2,  
      'T': 3,  
   }  
   var res []string  
   n := len(s)  
   if n <= L {  
      return res  
   }  
   x := 0  
   // use 2bit to store the message of first 9 characters  
   counter := make(map[int]int)  
   for i := range s {  
      x = (x<<2 | m[s[i]]) & (1<<(2*L) - 1)  
      if i+1 >= L {  
         counter[x]++  
         if counter[x] == 2 {  
            res = append(res, s[i+1-L:i+1])  
         }  
      }  
   }  
   return res  
}
```

rust

```rust
use std::collections::HashMap;

impl Solution {
    pub fn find_repeated_dna_sequences(s: String) -> Vec<String> {
        const L: usize = 10;
        let n = s.len();
        if n <= L {
            return vec![];
        }
        let bytes = vec!['A', 'C', 'G', 'T'];
        let nums = vec![0, 1, 2, 3];
        let map: HashMap<_, _> = bytes.iter().zip(nums.iter()).collect();
        let mut x = 0;
        let mut counter = HashMap::new();
        s.chars().enumerate().fold(vec![], |mut acc, (i, char)| {
            let &y = map[&char];
            x = (x << 2 | y) & ((1 << (2 * L)) - 1);
            if i + 1 >= L {
                let count = counter.entry(x).or_insert(0);
                *count += 1;
                if *count == 2 {
                    acc.push(s[i + 1 - L..i + 1].to_string());
                }
            }
            acc
        })
    }
}
```

#### 219. Contains Duplicate II

rust

```rust
use std::collections::HashSet;

impl Solution {
    pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
        let mut set = HashSet::new();
        let k = k as usize;
        for (i, v) in nums.iter().enumerate() {
            if i > k {
                set.remove(&nums[i - k - 1]);
            }
            if !set.insert(v) {
                return true;
            }
        }
        false
    }
}
```

go

```go
func containsNearbyDuplicate(nums []int, k int) bool {  
   st := make(map[int]bool)  
   for i, v := range nums {  
      if i > k {  
         delete(st, nums[i-k-1])  
      }  
      if st[v] {  
         return true  
      }  
      st[v] = true  
   }  
   return false  
}
```

#### 217. Contains Duplicate

python

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))
```

#### 229. Majority Element II

find all elements that appear more than `n/3` times

https://leetcode.com/problems/majority-element-ii/solutions/63502/6-lines-general-case-O(N)-time-and-O(k)-space/

go

```go
func majorityElement(nums []int) []int {  
   counter := make(map[int]int)  
   for _, num := range nums {  
      counter[num]++  
      if len(counter) == 3 {  
         shrink(counter)  
      }  
   }  
   var res []int  
   for num := range counter {  
      if valid(nums, num) {  
         res = append(res, num)  
      }  
   }  
   return res  
}  
func shrink(counter map[int]int) {  
   for key := range counter {  
      counter[key]--  
      if counter[key] == 0 {  
         delete(counter, key)  
      }  
   }  
}  
func valid(nums []int, num int) bool {  
   count := 0  
   for _, v := range nums {  
      if v == num {  
         count++  
      }  
   }  
   return count > len(nums)/3  
}
```

python

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter()
        for num in nums:
            cnt[num] += 1
            if len(cnt) == 3:
                cnt -= collections.Counter(set(cnt))
        return [n for n in cnt if nums.count(n) > len(nums) // 3]
```

rust

```rust
use std::collections::HashMap;

impl Solution {
    pub fn majority_element(nums: Vec<i32>) -> Vec<i32> {
        let counter = nums.iter().fold(HashMap::new(), |mut acc, &num| {
            *acc.entry(num).or_insert(0) += 1;
            if acc.len() == 3 {
                acc = acc
                    .iter()
                    .map(|(&key, &value)| (key, value - 1))
                    .filter(|&(_, value)| value > 0)
                    .collect();
            }
            acc
        });
        counter.into_keys().filter(|x| nums.iter().filter(|&y| x == y).count() > nums.len() / 3).collect()
    }
}
```

Java

```java
class Solution {
    public List<Integer> majorityElement(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
            if (map.size() == 3) {
                map.replaceAll((k, v) -> v - 1);
                map.entrySet().removeIf(entry -> entry.getValue() == 0);
            }
        }
        return map.keySet().stream().filter(a -> Arrays.stream(nums).filter(b -> b == a).count() > nums.length / 3).toList();
    }
}
```

#### 242. Valid Anagram

java

```java
class Solution {  
    public boolean isAnagram(String s, String t) {  
        return Arrays.equals(count(s), count(t));  
    }  
  
    int[] count(String s) {  
        int[] res = new int[26];  
        for (char c : s.toCharArray()) {  
            res[c - 'a']++;  
        }  
        return res;  
    }  
}
```

go

```go
func isAnagram(s string, t string) bool {  
   return count(s) == count(t)  
}  
func count(s string) [26]int {  
   var counter [26]int  
   for i := range s {  
      counter[s[i]-'a']++  
   }  
   return counter  
}
```

rust

```rust
impl Solution {
    pub fn is_anagram(s: String, t: String) -> bool {
        fn count(s: String) -> [i32; 26] {
            s.chars().into_iter().fold([0; 26], |mut acc, x| {
                acc[(x as u8 - b'a') as usize] += 1;
                acc
            })
        }
        count(s) == count(t)
    }
}
```

#### 290. Word Pattern

go

https://leetcode.com/problems/word-pattern/discuss/73409/Short-C%2B%2B-read-words-on-the-fly

```go
func wordPattern(pattern string, s string) bool {  
   words := strings.Split(s, " ")  
   if len(words) != len(pattern) {  
      return false  
   }  
   hashMap := make(map[interface{}]int)  
   for i, word := range words {  
      if hashMap[word] != hashMap[pattern[i]] {  
         return false  
      }  
      hashMap[word] = i + 1  
      hashMap[pattern[i]] = i + 1  
   }  
   return true  
}
```

java

https://leetcode.com/problems/word-pattern/discuss/73402/8-lines-simple-Java

```java
class Solution {  
    public boolean wordPattern(String pattern, String s) {  
        String[] words = s.split(" ");  
        if (pattern.length() != words.length) {  
            return false;  
        }  
        HashMap<Object, Integer> map = new HashMap<>();  
        for (int i = 0; i < words.length; i++) {  
            if (!Objects.equals(map.put(pattern.charAt(i), i), map.put(words[i], i))) {  
                return false;  
            }  
        }  
        return true;  
    }  
}
```

#### 349. Intersection of Two Arrays

go

```go
import "github.com/emirpasic/gods/sets/hashset"

func intersection(nums1 []int, nums2 []int) []int {
	set, res := hashset.New(), make([]int, 0)
	for _, v := range nums1 {
		set.Add(v)
	}
	for _, v := range nums2 {
		if set.Contains(v) {
			res = append(res, v)
			set.Remove(v)
		}
	}
	return res
}
```

python

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        n1, res = set(nums1), []
        for x in nums2:
            if x in n1:
                res += x,
                n1.remove(x)
        return res
```
#### 350. Intersection of Two Arrays II

go

```go
func intersect(nums1 []int, nums2 []int) []int {  
   if len(nums1) > len(nums2) {  
      return intersect(nums2, nums1)  
   }  
   hashmap := make(map[int]int)  
   for _, n1 := range nums1 {  
      hashmap[n1]++  
   }  
   var res []int  
   for _, n2 := range nums2 {  
      if hashmap[n2] > 0 {  
         res = append(res, n2)  
         hashmap[n2]--  
      }  
   }  
   return res  
}
```

#### 380. Insert Delete GetRandom O(1)

https://leetcode.com/problems/insert-delete-getrandom-o1/discuss/85401/Java-solution-using-a-HashMap-and-an-ArrayList-along-with-a-follow-up.-(131-ms)

go

```go
import "math/rand"  
  
type RandomizedSet struct {  
   arr  []int  
   hash map[int]int  
}  
  
func Constructor() RandomizedSet {  
   return RandomizedSet{hash: map[int]int{}}  
}  
  
func (set *RandomizedSet) Insert(val int) bool {  
   if _, ok := set.hash[val]; ok {  
      return false  
   }  
   set.hash[val] = len(set.arr)  
   set.arr = append(set.arr, val)  
   return true  
}  
  
func (set *RandomizedSet) Remove(val int) bool {  
   if index, ok := set.hash[val]; ok {  
      n := len(set.arr)  
      if index < n-1 {  
         set.arr[index] = set.arr[n-1]  
         set.hash[set.arr[n-1]] = index  
      }  
      delete(set.hash, val)  
      set.arr = set.arr[:n-1]  
      return true  
   }  
   return false  
}  
  
func (set *RandomizedSet) GetRandom() int {  
   return set.arr[rand.Intn(len(set.arr))]  
}
```

java

```java
class RandomizedSet {
    ArrayList<Integer> arr;
    HashMap<Integer, Integer> hash;

    public RandomizedSet() {
        arr = new ArrayList<>();
        hash = new HashMap<>();
    }

    public boolean insert(int val) {
        if (hash.containsKey(val)) {
            return false;
        }
        hash.put(val, arr.size());
        arr.add(val);
        return true;
    }

    public boolean remove(int val) {
        if (!hash.containsKey(val)) {
            return false;
        }
        int index = hash.get(val), n = arr.size();
        if (index < n - 1) {
            int lastOne = arr.get(arr.size() - 1);
            arr.set(index, lastOne);
            hash.put(lastOne, index);
        }
        arr.remove(arr.size() - 1);
        hash.remove(val);
        return true;
    }

    public int getRandom() {
        Random random = new Random();
        return arr.get(random.nextInt(arr.size()));
    }
}
```

#### 381. Insert Delete GetRandom O(1) - Duplicates allowed

https://leetcode.com/problems/insert-delete-getrandom-o1/discuss/85401/Java-solution-using-a-HashMap-and-an-ArrayList-along-with-a-follow-up.-(131-ms)

java

```java
class RandomizedCollection {  
    ArrayList<Integer> arr;  
    HashMap<Integer, Set<Integer>> hash;  
  
    public RandomizedCollection() {  
        arr = new ArrayList<>();  
        hash = new HashMap<>();  
    }  
  
    public boolean insert(int val) {  
        boolean contain = hash.containsKey(val);  
        if (!contain) {  
            hash.put(val, new HashSet<>());  
        }  
        hash.get(val).add(arr.size());  
        arr.add(val);  
        return !contain;  
    }  
  
    public boolean remove(int val) {  
        if (!hash.containsKey(val)) {  
            return false;  
        }  
        int index = hash.get(val).iterator().next();  
        hash.get(val).remove(index);  
  
        if (index < arr.size() - 1) {  
            int lastOne = arr.get(arr.size() - 1);  
            arr.set(index, lastOne);  
            hash.get(lastOne).remove(arr.size() - 1);  
            hash.get(lastOne).add(index);  
        }  
        arr.remove(arr.size() - 1);  
        if (hash.get(val).isEmpty()) {  
            hash.remove(val);  
        }  
        return true;  
    }  
  
    public int getRandom() {  
        Random random = new Random();  
        return arr.get(random.nextInt(arr.size()));  
    }  
}
```

go

https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/discuss/1083589/Golang-solution-with-comment

```go
type RandomizedCollection struct {  
   arr  []int  
   hash map[int]map[int]bool  
}  
  
func Constructor() RandomizedCollection {  
   return RandomizedCollection{  
      hash: make(map[int]map[int]bool),  
   }  
}  
  
func (rc *RandomizedCollection) Insert(val int) bool {  
   if len(rc.hash[val]) == 0 {  
      rc.hash[val] = make(map[int]bool)  
   }  
   rc.hash[val][len(rc.arr)] = true  
   rc.arr = append(rc.arr, val)  
   return len(rc.hash[val]) == 1  
}  
  
func (rc *RandomizedCollection) Remove(val int) bool {  
   if _, ok := rc.hash[val]; !ok {  
      return false  
   }  
   for index := range rc.hash[val] {  
      delete(rc.hash[val], index)  
      n := len(rc.arr)  
      if index < n-1 {  
         lastValue := rc.arr[n-1]  
         delete(rc.hash[lastValue], n-1)  
         rc.hash[lastValue][index] = true  
         rc.arr[index] = lastValue  
      }  
      rc.arr = rc.arr[:n-1]  
      break  
   }  
   if len(rc.hash[val]) == 0 {  
      delete(rc.hash, val)  
   }  
   return true  
}  
func (rc *RandomizedCollection) GetRandom() int {  
   return rc.arr[rand.Intn(len(rc.arr))]  
}
```

#### 383. Ransom Note

go

```go
func canConstruct(ransomNote string, magazine string) bool {  
   var counter [26]int  
   for i := range magazine {  
      counter[magazine[i]-'a']++  
   }  
   for i := range ransomNote {  
      counter[ransomNote[i]-'a']--  
   }  
   for _, v := range counter {  
      if v < 0 {  
         return false  
      }  
   }  
   return true  
}
```

rust

```rust
impl Solution {
    pub fn can_construct(ransom_note: String, magazine: String) -> bool {
        let mut count = magazine.as_bytes().iter().fold([0; 26], |mut acc, &ch| {
            acc[(ch - b'a') as usize] += 1;
            acc
        });
        for &x in ransom_note.as_bytes() {
            count[(x - b'a') as usize] -= 1
        }
        count.iter().all(|&x| x >= 0)
    }
}
```

#### 387. First Unique Character in a String

python

```python
class Solution:  
    def firstUniqChar(self, s: str) -> int:  
        counter = Counter(s)
        for (i, char) in enumerate(s):  
            if counter[char] == 1:  
                return i  
        return -1
```

go

```go
func firstUniqChar(s string) int {
   var m [26]int
   for i := range s {
      m[s[i]-'a']++
   }
   for i := range s {
      if m[s[i]-'a'] == 1 {
         return i
      }
   }
   return -1
}
```

java

```java
class Solution {  
    public int firstUniqChar(String s) {  
        int[] freq = new int[26];  
        for (char c : s.toCharArray()) {  
            freq[c - 'a']++;  
        }  
        for (int i = 0; i < s.length(); i++) {  
            if (freq[s.charAt(i) - 'a'] == 1) {  
                return i;  
            }  
        }  
        return -1;  
    }  
}
```


#### 454. 4Sum II

go

```go
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
    m, count := make(map[int]int), 0
    for _, v1 := range nums1 {
        for _, v2 := range nums2 {
            m[v1+v2]++
        }
    }
    for _, v3 := range nums3 {
        for _, v4 := range nums4 {
            count += m[-(v3 + v4)]
        }
    }
    return count
}
```

rust

```rust
use std::collections::HashMap;  
  
impl Solution {  
    pub fn four_sum_count(nums1: Vec<i32>, nums2: Vec<i32>, nums3: Vec<i32>, nums4: Vec<i32>) -> i32 {  
        let mut map = HashMap::new();  
        for i in nums1 {  
            for j in &nums2 {  
                *map.entry(i + j).or_insert(0) += 1;  
            }  
        }  
        let mut res = 0;  
        for i in nums3 {  
            for j in &nums4 {  
                if let Some(count) = map.get(&(-i - j)) {  
                    res += count;  
                }  
            }  
        }  
        res  
    }  
}
```

#### 447. Number of Boomerangs

python

```python
class Solution:  
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:  
        res = 0  
        for (i, x) in enumerate(points):  
            m = collections.defaultdict(int)  
            for (j, y) in enumerate(points):  
                if i == j:  
                    continue  
                a = x[0] - y[0]  
                b = x[1] - y[1]  
                m[a ** 2 + b ** 2] += 1  
            res += sum([x * (x - 1) for x in m.values()])  
        return res
```

go

```go
func numberOfBoomerangs(points [][]int) int {  
   res := 0  
   for i := range points {  
      hashmap := make(map[int]int)  
      for j := range points {  
         if i == j {  
            continue  
         }  
         x := points[i][0] - points[j][0]  
         y := points[i][1] - points[j][1]  
         d := x*x + y*y  
         hashmap[d]++  
      }  
      for _, v := range hashmap {  
         res += v * (v - 1)  
      }  
   }  
   return res  
}
```

rust

```rust
use std::collections::HashMap;

impl Solution {
    pub fn number_of_boomerangs(points: Vec<Vec<i32>>) -> i32 {
        let mut res = 0;
        for (i, x) in points.iter().enumerate() {
            let mut map: HashMap<i32, i32> = HashMap::new();
            for (j, y) in points.iter().enumerate() {
                if i == j {
                    continue;
                }
                *map.entry(dis(x, y)).or_insert(0) += 1;
            }
            res += map.values().map(|v| v * (v - 1)).sum::<i32>();
        }
        fn dis(x: &[i32], y: &[i32]) -> i32 {
            let a = x[0] - y[0];
            let b = x[1] - y[1];
            a.pow(2) + b.pow(2)
        }
        res
    }
}
```


#### 1248. Count Number of Nice Subarrays

java

```java
class Solution {
    public int numberOfSubarrays(int[] nums, int k) {
        int res = 0, acc = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int num : nums) {
            acc += num & 1;
            res += map.getOrDefault(acc - k, 0);
            map.put(acc, map.getOrDefault(acc, 0) + 1);
        }
        return res;
    }
}
```

#### 2215. Find the Difference of Two Arrays

java

```java
class Solution {
    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        return List.of(
                differ(nums1, nums2),
                differ(nums2, nums1));
    }

    List<Integer> differ(int[] nums1, int[] nums2) {
        boolean[] visited = new boolean[2001];
        ArrayList<Integer> res = new ArrayList<>();
        for (int num : nums2) {
            visited[num + 1000] = true;
        }
        for (int num : nums1) {
            if (visited[num + 1000]) {
                continue;
            }
            res.add(num);
            visited[num + 1000] = true;
        }
        return res;
    }
}
```


