#### 460. LFU Cache

java

```java
class LFUCache {
    HashMap<Integer, Integer> data;
    HashMap<Integer, Integer> counter;
    HashMap<Integer, LinkedHashSet<Integer>> map;

    int min;
    int cap;

    public LFUCache(int capacity) {
        data = new HashMap<>();
        counter = new HashMap<>();
        map = new HashMap<>();
        map.put(1, new LinkedHashSet<>());
        min = -1;
        cap = capacity;
    }

    public int get(int key) {
        if (!data.containsKey(key)) {
            return -1;
        }
        makeRecently(key);
        return data.get(key);
    }

    public void put(int key, int value) {
        if (data.containsKey(key)) {
            data.put(key, value);
            makeRecently(key);
            return;
        }
        if (data.size() >= cap) {
            Integer del = map.get(min).iterator().next();
            data.remove(del);
            counter.remove(del);
            map.get(min).remove(del);
        }
        data.put(key, value);
        counter.put(key, 1);
        map.get(1).add(key);
        min = 1;
    }

    void makeRecently(int key) {
        Integer count = counter.get(key);
        counter.put(key, count + 1);

        map.get(count).remove(key);
        map.putIfAbsent(count + 1, new LinkedHashSet<>());
        map.get(count + 1).add(key);

        if (count == min && map.get(count).isEmpty()) {
            min++;
        }
    }
}
```


#### 622. Design Circular Queue

java

```java
class MyCircularQueue {
    int capacity;
    int[] data;
    int head;
    int tail;

    public MyCircularQueue(int k) {
        data = new int[k + 1];
        capacity = k + 1;
    }

    // insert first, then move tail
    public boolean enQueue(int value) {
        if (isFull()) return false;
        data[tail] = value;
        tail = (tail + 1) % capacity;
        return true;
    }

    public boolean deQueue() {
        if (isEmpty()) return false;
        head = (head + 1) % capacity;
        return true;
    }

    public int Front() {
        if (isEmpty()) return -1;
        return data[head];
    }

    public int Rear() {
        if (isEmpty()) return -1;
        return data[(tail - 1 + capacity) % capacity];
    }

    public boolean isEmpty() {
        return head == tail;
    }

    public boolean isFull() {
        return (tail + 1) % capacity == head;
    }
}
```

Go

```go
type MyCircularQueue struct {
	head     int
	tail     int
	capacity int
	data     []int
}

func Constructor(k int) MyCircularQueue {
	return MyCircularQueue{
		data:     make([]int, k+1),
		capacity: k + 1,
	}
}

func (cq *MyCircularQueue) EnQueue(value int) bool {
	if cq.IsFull() {
		return false
	}
	cq.data[cq.tail] = value
	cq.tail = (cq.tail + 1) % cq.capacity
	return true
}

func (cq *MyCircularQueue) DeQueue() bool {
	if cq.IsEmpty() {
		return false
	}
	cq.head = (cq.head + 1) % cq.capacity
	return true
}

func (cq *MyCircularQueue) Front() int {
	if cq.IsEmpty() {
		return -1
	}
	return cq.data[cq.head]
}

func (cq *MyCircularQueue) Rear() int {
	if cq.IsEmpty() {
		return -1
	}
	return cq.data[(cq.tail-1+cq.capacity)%cq.capacity]
}

func (cq *MyCircularQueue) IsEmpty() bool {
	return cq.tail == cq.head
}

func (cq *MyCircularQueue) IsFull() bool {
	return (cq.tail+1)%cq.capacity == cq.head
}
```

#### 641. Design Circular Deque

java

```java
class MyCircularDeque {
    int capacity;
    int[] data;
    int head;
    int tail;


    public MyCircularDeque(int k) {
        data = new int[k + 1];
        capacity = k + 1;
    }

    public boolean insertFront(int value) {
        if (isFull()) return false;
        head = (head - 1 + capacity) % capacity;
        data[head] = value;
        return true;
    }
    // insert first, then move tail
    public boolean insertLast(int value) {
        if (isFull()) return false;
        data[tail] = value;
        tail = (tail + 1) % capacity;
        return true;
    }

    public boolean deleteFront() {
        if (isEmpty()) return false;
        head = (head + 1) % capacity;
        return true;
    }

    public boolean deleteLast() {
        if (isEmpty()) return false;
        tail = (tail - 1 + capacity) % capacity;
        return true;
    }

    public int getFront() {
        if (isEmpty()) return -1;
        return data[head];
    }

    public int getRear() {
        if (isEmpty()) return -1;
        return data[(tail - 1 + capacity) % capacity];
    }

    public boolean isEmpty() {
        return head == tail;
    }

    public boolean isFull() {
        return (tail + 1) % capacity == head;
    }
}
```

#### 2336. Smallest Number in Infinite Set

java

```java
class SmallestInfiniteSet {
    int curr;
    TreeSet<Integer> set;

    public SmallestInfiniteSet() {
        set = new TreeSet<>();
        curr = 1;
    }

    public int popSmallest() {
        if (set.isEmpty()) {
            curr++;
            return curr - 1;
        } else {
            Integer first = set.first();
            set.remove(first);
            return first;
        }
    }

    public void addBack(int num) {
        if (curr > num) {
            set.add(num);
        }
    }
}
```

rust 由于LC的rust版本低，不能提交这个

```rust
use std::collections::BTreeSet;

struct SmallestInfiniteSet {
    curr: i32,
    set: BTreeSet<i32>,
}

impl SmallestInfiniteSet {
    fn new() -> Self {
        return SmallestInfiniteSet { set: BTreeSet::new(), curr: 1 };
    }

    fn pop_smallest(&mut self) -> i32 {
        match self.set.pop_first() {
            None => {
                self.curr += 1;
                self.curr - 1
            }
            Some(v) => v            
        }
    }

    fn add_back(&mut self, num: i32) {
        if self.curr > num {
            self.set.insert(num);
        }
    }
}
```