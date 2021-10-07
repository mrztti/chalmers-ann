import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from progress.bar import FillingSquaresBar

# CONSTANTS
XOR = np.matrix([[-1., -1., -1], [1., -1., 1.], [-1., 1., 1.], [1., 1., -1.]])
Nhs = np.array([1,2,4,8])
n_Nhs = Nhs.shape[0]
Nv = 3
P = XOR.shape[0]
k = 100
eta = 0.1

batch_size = 10
p_val = 40
repetitions = 10
N_outer = 300
N_inner = 1000


def p(b):
    return np.vectorize(lambda x: 1. / (1. + np.exp(-2 * x)))(b)


def filter(x):
    return np.vectorize(lambda x: 1. if x < 0 else -1.)(x)


def update_hidden_neurons(input, weights, thresholds, n):
    b = np.matmul(input.T, weights).T - thresholds
    return filter(np.random.rand(n, 1) - p(b))


def update_visible_neurons(input, weights, thresholds, n):
    b = np.matmul(input.T, weights.T).T - thresholds
    return filter(np.random.rand(n, 1) - p(b))


def KLD(W, thr_v, thr_h):
    P_B = np.zeros((P,1))
    bar = FillingSquaresBar("KLD", max=N_inner * N_outer)
    for i in range(N_outer):
        r = np.sign(rd.randint(0, 2, (1, 3)) - 0.5)
        v = r.T
        h = update_hidden_neurons(v, W, thr_h, Nh)
        for j in range(N_inner):
            v = update_visible_neurons(h, W, thr_v, Nv)
            h = update_hidden_neurons(v, W, thr_h, Nh)
            vi = np.all(XOR == v.T, axis=1)
            if vi.any():
                reached = np.where(vi)
                P_B[reached] += 1
            bar.next()
    bar.finish()
    P_B = P_B / (N_outer * N_inner)
    l = np.log(0.25/P_B[P_B != 0.])
    return 0.25 * np.sum(l)


# Init statistics
means = np.zeros(n_Nhs)
stds = np.zeros(n_Nhs)
index = 0
for Nh in Nhs:
    print("Starting repetitions for Nh = " + str(Nh))
    klds = np.zeros(repetitions)
    for rep in range(repetitions):
        # WEIGHT INIT
        W = rd.normal(loc=0., scale=1, size=(Nv, Nh))
        v = np.zeros((Nv, 1))
        h = np.zeros((Nh, 1))
        thr_v = np.zeros((Nv, 1))
        thr_h = np.zeros((Nh, 1))

        # Init deltas
        d_W = np.zeros((Nv, Nh))
        d_thr_v = np.zeros((Nv, 1))
        d_thr_h = np.zeros((Nh, 1))

        # Iterate over multiple batches
        bar = FillingSquaresBar("Training:", max=p_val*batch_size)
        for epoch in range(p_val):
            for pi in range(batch_size):

                v = XOR[rd.randint(P)].T
                v0 = v.copy()
                h = update_hidden_neurons(v, W, thr_h, Nh)

                # K-loop
                for t in range(k):
                    v = update_visible_neurons(h, W, thr_v, Nv)
                    h = update_hidden_neurons(v, W, thr_h, Nh)

                tan_bh0 = np.tanh(np.matmul(v0.T, W).T - thr_h)
                tan_bh = np.tanh(np.matmul(v.T, W).T - thr_h)

                d_W += eta * (np.matmul(v0, tan_bh0.T) - np.matmul(v, tan_bh.T))
                d_thr_v -= eta * (v0 - v)
                d_thr_h -= eta * (tan_bh0 - tan_bh)
                bar.next()

            W += d_W
            thr_v += d_thr_v
            thr_h += d_thr_h
        bar.finish()

        # Calculate statistics
        kld = KLD(W, thr_v, thr_h)
        klds[rep] = kld
        print("Trial {0} finished with KLD: {1}".format(rep, kld))

    means[index] = np.mean(klds)
    stds[index] = np.std(klds)
    print("Finished Nh-{0} with <KLd>={1}, std={2}".format(Nh, means[index], stds[index]))
    index += 1


fig, ax = plt.subplots()
ax.errorbar(Nhs, means, yerr=stds, fmt='-o', color='red')
ax.set_xlabel('Number of hidden neurons')
ax.set_ylabel('KL-divergence')
ax.set_title('Average and STD of KL-divergence for different amounts of hidden neurons (CD-100, 50 repetitions)')
plt.show()



