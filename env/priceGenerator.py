import numpy as np
def make_stock(length=100, num_stocks=2):
    alpha = 0.9
    k = 2
    cov = np.random.normal(0, 5, [num_stocks, num_stocks])
    cov = cov.dot(cov.T)
    A = np.random.multivariate_normal(np.zeros(num_stocks), cov, size=[length])
    B = np.random.multivariate_normal(np.zeros(num_stocks), cov, size=[length])
    bs = [np.zeros(shape=num_stocks)]
    ps = [np.zeros(shape=num_stocks)]

    for a, b in zip(A, B):
        bv = alpha * bs[-1] + b
        bs.append(bv)
        pv = ps[-1] + bs[-2] + k * a
        ps.append(pv)

    #     ps = [0]
    #     for a,b,common in zip(A,BB,commonNoise):
    #         ps.append(ps[-1]+b+k*a+2*common)
    #     P = np.array(ps)
    #     P = np.exp(P/(P.max()-P.min()))
    ps = np.array(ps).T
    R = ps.max(1) - ps.min(1)
    prices = np.exp(ps.T / (R)) *np.random.uniform(10,250,num_stocks)
    return prices