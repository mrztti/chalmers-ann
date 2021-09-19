import random
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
N = 200
BETA = 2
P = 45
T_N = 200000
T_REPEAT = 100


# FUNCTIONS
def generate_random_matrix(n):
    r = np.random.randint(2, size=(n, N))
    r[r == 0] = -1
    return r


def perform_one_trial():
    # x1 = np.linspace(0, T_N - 1, T_N)
    # x2 = np.array([0.] * T_N)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # line1, = ax.plot(x1, x2, 'r-')
    # plt.ion()
    # plt.ylim([0., 1.1])
    # plt.xlim([0., int(T_N / 200) - 1])
    # plt.show()

    # Generate weight matrix
    patterns = generate_random_matrix(P)
    W = (1/N) * np.matmul(patterns.T, patterns)
    np.fill_diagonal(W, 0)

    # Choose first pattern
    nu_index = 1
    chosen_pattern = patterns[nu_index]
    current_state = chosen_pattern.copy()

    order_sum = 0

    for iteration in range(int(T_N / 200)):
        for neuron_index in range(N):

            # Feed chosen pattern to network
            selected_weights = np.matrix(W[neuron_index]).T
            b = np.dot(current_state, selected_weights).item()
            # if b == 0:
            #     b = 1

            b1 = np.exp(-2 * BETA * b)

            p_b = 1 / (1 + b1)
            new_neuron_value = -1
            r = random.uniform(0., 1.0)

            if r < p_b:
                new_neuron_value = 1
            current_state[neuron_index] = new_neuron_value

        sum_t = 1/N * np.sum(current_state * chosen_pattern)
        order_sum = (iteration*order_sum + sum_t) / (iteration + 1)
    #     x2[iteration] = order_sum
    #     line1.set_ydata(x2)
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
    #
    # input()
    return order_sum


def perform_n_trials():
    x1 = np.linspace(0, T_REPEAT - 1, T_REPEAT)
    x2 = np.array([0.] * T_REPEAT)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x1, x2, 'r-')
    plt.ion()
    plt.ylim([0., 1.1])
    plt.xlim([0., T_REPEAT - 1])
    plt.show()
    order_sum = 0.

    for i in x1:
        a = perform_one_trial()
        x2[int(i)] = a
        line1.set_ydata(x2)
        fig.canvas.draw()
        fig.canvas.flush_events()
        print(a)
        order_sum += a

    print("<m1> = " + str(order_sum / float(T_REPEAT)))
    input() # Block afterwards

# MAIN CODE
perform_n_trials()

