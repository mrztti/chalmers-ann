import numpy as np
import numpy.random as rd

XOR = np.matrix([[-1., -1., -1], [1., -1., 1.], [-1., 1., 1.], [1., 1., -1.]])
# XOR = np.matrix([[1.,1.,1.]])
M = 4
N = 3
W = rd.normal(loc=0., scale=1 / N, size=(M, N))
thr_v = np.zeros((1, N))
thr_h = np.zeros((M, 1))
P = XOR.shape[0]
k = 1000
lr = 0.1


def p(b):
    return np.vectorize(lambda x: 1. / (1. + np.exp(-2 * x)))(b)
def filter(x):
    return np.vectorize(lambda x: 1. if x < 0 else -1.)(x)


for p0 in XOR:

    bh = np.matmul(W,p0.T) - thr_h
    t_bh0 = np.tanh(bh)
    h = filter(np.random.rand(M, 1) - p(bh))

    v = None
    for t in range(k):
        pbv = p(np.matmul(h.T, W) - thr_v)
        v = filter(np.random.rand(1, N) - pbv)
        pbh = p(np.matmul(W, v.T) - thr_h)
        h = filter(np.random.rand(M, 1) - pbh)
    t_bhk = np.tanh(bh)
    for m in range(M):
        for n in range(N):
            dmn = t_bh0[m, 0] * p0[0, n] - t_bhk[m, 0] * v[0, n]
            W[m][n] += lr * dmn
            print(dmn)
    thr_v -= lr * (p0 - v)
    thr_h -= lr * (t_bh0 - t_bhk)

# DKL = 0.25 * np.log(0.25 / )

si = np.matrix([[-1., -1., -1.]])


for i in range(100):

    pbh = p(np.matmul(W, si.T) - thr_h)
    hi = filter(np.random.rand(M, 1) - pbh)
    pbv = p(np.matmul(hi.T, W) - thr_v)
    si = filter(np.random.rand(1, N) - pbv)
print(si)
















