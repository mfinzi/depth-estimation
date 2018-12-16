import numpy as np
import time


# Generates a random hash function from the family described in section 5
def gaussian_hash(d, r):
    a = np.random.normal(size=d)
    b = np.random.uniform(0, r)
    return lambda v: np.floor(((a @ v) + b) / r)


# Calculates Euclidean distance
def dist(a, b):
    return np.linalg.norm(a-b)


# f: hash function
# k: number of projections per hash
# l: number of hash tables
def make_family(f, k, L):
    G = [[f() for _ in range(k)] for _ in range(L)]
    return G


def make_gaussian_family(d, r, k, L):
    A = np.random.normal(size=(d, k*L))
    b = np.random.uniform(0, r, size=k*L)
    return A, b


def filter_main(src, ref, A, b, r, k, L):
    n, num_channels = src.shape
    mapping = {}
    arr = np.floor(((ref @ A) + b) / r).reshape((n, L, k))
    new_src = np.zeros(src.shape)
    num_seen = 0
    miss = 0
    buckets = [{} for _ in range(L)]
    mapping = {}
    for i in range(n):
        mapping[i] = []
        if i % (n // 10) == 0:
            print(i)
        for j in range(L):
            x = tuple(arr[i, j])
            if x in buckets[j]:
                buckets[j][x].append(i)
            else:
                buckets[j][x] = [i]
            mapping[i].append(buckets[j][x])
    for i in range(n):
        if i % (n // 10) == 0:
            print(i)
        pt_indices = set()
        for bucket in mapping[i]:
            pt_indices = pt_indices.union(bucket)
        if len(pt_indices) > 1:
            pt_indices.discard(i)
            assert i not in pt_indices
        else:
            miss += 1
        pt_indices = np.array(list(pt_indices))
        num_seen += len(pt_indices)
        weights = np.exp(-(np.linalg.norm(ref[i] - ref[pt_indices], axis=-1) ** 2))
        for c in range(num_channels):
            new_src[i,c] = weights @ src[pt_indices, c]
    print(num_seen / n)
    print(miss / n)
    return new_src

def filter(src, ref, r, k, L):
    src = np.array(src)
    ref = np.array(ref)
    n, d = ref.shape
    A, b = make_gaussian_family(d, r, k, L)
    return filter_main(src, ref, A, b, r, k, L)


def preprocess_points(G, pts):
    mapping = {}
    buckets = [{} for _ in range(len(G))]
    j = 0
    for pt_index, p in enumerate(pts):
        mapping[pt_index] = []
        j += 1
        if j % (len(pts) // 10) == 0:
            print(j)
        for i, g in enumerate(G):
            x = tuple([h(p) for h in g])
            if x in buckets[i]:
                buckets[i][x].append(pt_index)
            else:
                buckets[i][x] = [pt_index]
            mapping[pt_index].append(buckets[i][x])

    return mapping, buckets


def approx_nn(mapping, pt_index, pts, k):
    pt_indices = set()
    for bucket in mapping[pt_index]:
        pt_indices = pt_indices.union(bucket)

    pt_indices.discard(pt_index)
    pt_indices = np.array(list(pt_indices))
    weights = np.exp(-(np.linalg.norm(pts[pt_index] - pts[pt_indices], axis=-1) ** 2))
    return pt_indices, weights, len(pt_indices)
    # return min(pts, key = lambda x: dist(q, x), default=None)


def true_nn(points, q, k):
    dists = np.array([dist(q, x) for x in points])
    indices = np.argsort(dists)[1:k+1]
    return indices, dists[indices]
    # return min(points, key = lambda x: dist(q, x))


def time_f(f):
    start = time.time()
    v = f()
    end = time.time()
    return v, end - start


def bad_filter(src, ref):
    src = np.array(src)
    ref = np.array(ref)
    n, d = ref.shape
    channels = src.shape[1]
    r = 0.1
    k = 5
    L = 10
    hash_function = lambda: gaussian_hash(d, r)
    G = make_family(hash_function, k, L)
    print("Preprocessing")
    mapping, buckets = preprocess_points(G, ref)
    print("Preprocessing done")
    new_src = np.zeros(src.shape)
    nearest_k = 1000
    misses = 0
    num_searched = 0
    for i in range(n):
        if i % (n // 10) == 0:
            print(i)
        pt_indices, weights, ns = approx_nn(mapping, i, ref, nearest_k)
        assert i not in pt_indices
        num_searched += ns
        if len(pt_indices) == 0:
            misses += 1
            print("Miss", misses / (i+1))
            pt_indices, dists = true_nn(ref, ref[i], k)
        # weights = weights / np.sum(weights)
        for c in range(channels):
            new_src[i,c] = weights @ src[pt_indices, c]
    print(misses / n)
    print(num_searched / n)
    return new_src

if __name__ == "__main__":
    n = 100000
    d = 20
    r = 2.5
    k = 5
    L = 20
    hash_function = lambda: gaussian_hash(d, r)
    G = make_family(hash_function, k, L)
    cov = np.random.rand(d, d)
    cov = cov @ cov.T
    points = np.random.multivariate_normal(np.arange(d), cov, size=n)

    buckets = preprocess_points(G, points)
    # print("Preprocessing done")
    T1 = []
    T2 = []
    diffs = []
    for i in range(100):
        q = np.random.multivariate_normal(np.arange(d), cov)
        approx, t1 = time_f(lambda: approx_nn(G, buckets, q))
        true, t2 = time_f(lambda: true_nn(points, q))
        T1.append(t1)
        T2.append(t2)
        if not approx is None:
            diffs.append(dist(approx, q) / dist(true, q))
        else:
            print("Could not find", i)
    print("mean,std time LSH:", np.mean(T1), np.std(T1))
    print("mean,std time naive:", np.mean(T2), np.std(T2))
    print("max,mean,std d/d:*", np.max(diffs), np.mean(diffs), np.std(diffs))
    # print(diffs)
