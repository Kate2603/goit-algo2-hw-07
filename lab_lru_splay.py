import random
import time
import timeit
from functools import lru_cache
import matplotlib.pyplot as plt

# ======================= ЗАВДАННЯ 1 =======================

# Готовий LRU Cache клас
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity
        self.order = []

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)

# --- Запити ---
def make_queries(n, q, hot_pool=30, p_hot=0.95, p_update=0.03):
    hot = [(random.randint(0, n//2), random.randint(n//2, n-1))
           for _ in range(hot_pool)]
    queries = []
    for _ in range(q):
        if random.random() < p_update:
            idx = random.randint(0, n-1)
            val = random.randint(1, 100)
            queries.append(("Update", idx, val))
        else:
            if random.random() < p_hot:
                left, right = random.choice(hot)
            else:
                left = random.randint(0, n-1)
                right = random.randint(left, n-1)
            queries.append(("Range", left, right))
    return queries

# --- Функції без кешу ---
def range_sum_no_cache(array, left, right):
    return sum(array[left:right+1])

def update_no_cache(array, index, value):
    array[index] = value

# --- Функції з кешем ---
def range_sum_with_cache(array, left, right, cache: LRUCache):
    key = (left, right)
    result = cache.get(key)
    if result == -1:
        result = sum(array[left:right+1])
        cache.put(key, result)
    return result

def update_with_cache(array, index, value, cache: LRUCache):
    array[index] = value
    keys_to_remove = [key for key in cache.cache if key[0] <= index <= key[1]]
    for key in keys_to_remove:
        del cache.cache[key]
        cache.order.remove(key)

# --- Тестування Завдання 1 ---
n = 100_000
q = 50_000
array = [random.randint(1, 100) for _ in range(n)]
queries = make_queries(n, q)

# Без кешу
start = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_no_cache(array, query[1], query[2])
    else:
        update_no_cache(array, query[1], query[2])
no_cache_time = time.time() - start

# З кешем
array2 = array.copy()
cache = LRUCache(capacity=1000)
start = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_with_cache(array2, query[1], query[2], cache)
    else:
        update_with_cache(array2, query[1], query[2], cache)
with_cache_time = time.time() - start

print(f"Без кешу : {no_cache_time:.2f} c")
print(f"LRU-кеш  : {with_cache_time:.2f} c (прискорення ×{no_cache_time / with_cache_time:.1f})")

# ======================= ЗАВДАННЯ 2 =======================

# --- LRU Cache для Фібоначчі ---
@lru_cache(maxsize=None)
def fibonacci_lru(n):
    if n < 2:
        return n
    return fibonacci_lru(n-1) + fibonacci_lru(n-2)

# --- Splay Tree ---
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

class SplayTree:
    def __init__(self):
        self.root = None

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        y.right = x
        return y

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        y.left = x
        return y

    def _splay(self, root, key):
        if root is None or root.key == key:
            return root
        if key < root.key:
            if root.left is None:
                return root
            if key < root.left.key:
                root.left.left = self._splay(root.left.left, key)
                root = self._right_rotate(root)
            elif key > root.left.key:
                root.left.right = self._splay(root.left.right, key)
                if root.left.right:
                    root.left = self._left_rotate(root.left)
            return self._right_rotate(root) if root.left else root
        else:
            if root.right is None:
                return root
            if key > root.right.key:
                root.right.right = self._splay(root.right.right, key)
                root = self._left_rotate(root)
            elif key < root.right.key:
                root.right.left = self._splay(root.right.left, key)
                if root.right.left:
                    root.right = self._right_rotate(root.right)
            return self._left_rotate(root) if root.right else root

    def insert(self, key, value):
        if self.root is None:
            self.root = Node(key, value)
            return
        self.root = self._splay(self.root, key)
        if key == self.root.key:
            self.root.value = value
            return
        new_node = Node(key, value)
        if key < self.root.key:
            new_node.right = self.root
            new_node.left = self.root.left
            self.root.left = None
        else:
            new_node.left = self.root
            new_node.right = self.root.right
            self.root.right = None
        self.root = new_node

    def search(self, key):
        self.root = self._splay(self.root, key)
        if self.root and self.root.key == key:
            return self.root.value
        return None

# --- Фібоначчі через Splay Tree ---
def fibonacci_splay(n, tree):
    val = tree.search(n)
    if val is not None:
        return val
    if n < 2:
        tree.insert(n, n)
        return n
    result = fibonacci_splay(n-1, tree) + fibonacci_splay(n-2, tree)
    tree.insert(n, result)
    return result

# --- Порівняння ---
ns = list(range(0, 1000, 50))
lru_times = []
splay_times = []

for num in ns:
    lru_t = timeit.timeit(lambda: fibonacci_lru(num), number=10) / 10
    tree = SplayTree()
    splay_t = timeit.timeit(lambda: fibonacci_splay(num, tree), number=10) / 10
    lru_times.append(lru_t)
    splay_times.append(splay_t)

# Таблиця
print("\n" + "n".ljust(10) + "LRU Cache Time (s)".ljust(25) + "Splay Tree Time (s)")
print("-" * 55)
for i in range(len(ns)):
    print(f"{ns[i]:<10}{lru_times[i]:<25.8f}{splay_times[i]:.8f}")

# Графік
plt.plot(ns, lru_times, marker='o', label='LRU Cache')
plt.plot(ns, splay_times, marker='x', label='Splay Tree')
plt.title('Порівняння часу виконання для LRU Cache та Splay Tree')
plt.xlabel('Число Фібоначчі (n)')
plt.ylabel('Середній час виконання (секунди)')
plt.legend()
plt.show()
