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


def preprocess_points(G, pts):
    buckets = [{} for _ in range(len(G))]
    j = 0
    for p in pts:
        j += 1
        # if j % (len(pts) // 100) == 0:
        #     print(j)
        for i, g in enumerate(G):
            x = tuple([h(p) for h in g])
            if x in buckets[i]:
                buckets[i][x].append(p)
            else:
                buckets[i][x] = [p]

    return buckets


def approx_nn(G, buckets, q):
    pts = []
    for i, g in enumerate(G):
        x = tuple([h(q) for h in g])
        if x in buckets[i]:
            for p in buckets[i][x]:
                pts.append(p)

    return min(pts, key = lambda x: dist(q, x), default=None)


def true_nn(points, q):
    return min(points, key = lambda x: dist(q, x))


def time_f(f):
    start = time.time()
    v = f()
    end = time.time()
    return v, end - start

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
