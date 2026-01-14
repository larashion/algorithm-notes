#### 208. Implement Trie (Prefix Tree)

Go array

```go
type Trie struct {
	root *TrieNode
}
type TrieNode struct {
	pass int
	end  int
	next [26]*TrieNode
}

func Constructor() Trie {
	return Trie{&TrieNode{next: [26]*TrieNode{}}}
}

func (t *Trie) Insert(word string) {
	curr := t.root
	curr.pass++
	for i := range word {
		if curr.next[word[i]-'a'] == nil {
			curr.next[word[i]-'a'] = &TrieNode{next: [26]*TrieNode{}}
		}
		curr = curr.next[word[i]-'a']
		curr.pass++
	}
	curr.end++
}

func (t *Trie) Search(word string) bool {
	curr := t.root
	for i := range word {
		if curr.next[word[i]-'a'] == nil {
			return false
		}
		curr = curr.next[word[i]-'a']
	}
	return curr.end > 0
}

func (t *Trie) StartsWith(prefix string) bool {
	curr := t.root
	if len(prefix) == 0 {
		return false
	}
	for i := range prefix {
		if curr.next[prefix[i]-'a'] == nil {
			return false
		}
		curr = curr.next[prefix[i]-'a']
	}
	return true
}
```

#### 211. Design Add and Search Words Data Structure

Go

```go
type WordDictionary struct {
	root *TrieNode
}
type TrieNode struct {
	next map[byte]*TrieNode
	end  bool
}

func Constructor() WordDictionary {
	return WordDictionary{
		root: &TrieNode{next: map[byte]*TrieNode{}},
	}
}

func (w *WordDictionary) AddWord(word string) {
	curr := w.root
	for i := range word {
		if _, ok := curr.next[word[i]]; !ok {
			curr.next[word[i]] = &TrieNode{next: map[byte]*TrieNode{}}
		}
		curr = curr.next[word[i]]
	}
	curr.end = true
}

func (w *WordDictionary) Search(word string) bool {
	return dfs(word, w.root)
}
func dfs(word string, curr *TrieNode) bool {
	if len(word) == 0 {
		return curr.end
	}
	if word[0] == '.' {
		for _, v := range curr.next {
			if dfs(word[1:], v) {
				return true
			}
		}
	} else {
		child, ok := curr.next[word[0]]
		return ok && dfs(word[1:], child)
	}
	return false
}
```

#### 212. Word Search II

https://leetcode.com/problems/word-search-ii/discuss/59780/Java-15ms-Easiest-Solution-(100.00)

go

```go
func findWords(board [][]byte, words []string) []string {
	trie := &Trie{&TrieNode{next: map[byte]*TrieNode{}}}
	trie.Insert(words)
	m, n := len(board), len(board[0])
	var res []string
	for i := range board {
		for j := range board[0] {
			dfs(&res, board, i, j, m, n, trie.root)
		}
	}
	return res
}

func dfs(res *[]string, board [][]byte, i, j, m, n int, curr *TrieNode) {
	char := board[i][j]
	if curr.next[char] == nil {
		return
	}
	curr = curr.next[char]
	if curr.end != "" {
		*res = append(*res, curr.end)
		curr.end = ""
	}
	board[i][j] = '#'
	dirs := [4][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	for _, dir := range dirs {
		x := i + dir[0]
		y := j + dir[1]
		if x < 0 || x >= m || y < 0 || y >= n || board[x][y] == '#' {
			continue
		}
		dfs(res, board, x, y, m, n, curr)
	}
	board[i][j] = char
}

type Trie struct {
	root *TrieNode
}
type TrieNode struct {
	next map[byte]*TrieNode
	end  string
}

func (t *Trie) Insert(words []string) {
	for _, word := range words {
		curr := t.root
		for i := range word {
			if _, ok := curr.next[word[i]]; !ok {
				curr.next[word[i]] = &TrieNode{next: map[byte]*TrieNode{}}
			}
			curr = curr.next[word[i]]
		}
		curr.end = word
	}
}
```


#### 677. Map Sum Pairs

java

```java
class MapSum {
    Trie trie;

    public MapSum() {
        trie = new Trie();
    }

    public void insert(String key, int val) {
        TrieNode curr = trie.root;
        for (char c : key.toCharArray()) {
            if (curr.next[c - 'a'] == null) {
                curr.next[c - 'a'] = new TrieNode();
            }
            curr = curr.next[c - 'a'];
        }
        curr.val = val;
    }

    public int sum(String prefix) {
        TrieNode curr = trie.root;
        for (char c : prefix.toCharArray()) {
            if (curr.next[c - 'a'] == null)
                return 0;
            curr = curr.next[c - 'a'];
        }
        return dfs(curr);
    }

    int dfs(TrieNode curr) {
        int res = curr.val;
        for (TrieNode node : curr.next) {
            if (node != null) {
                res += dfs(node);
            }
        }
        return res;
    }
}

class Trie {
    TrieNode root;

    Trie() {
        root = new TrieNode();
    }
}

class TrieNode {
    TrieNode[] next;
    int val;

    TrieNode() {
        next = new TrieNode[26];
    }
}
```