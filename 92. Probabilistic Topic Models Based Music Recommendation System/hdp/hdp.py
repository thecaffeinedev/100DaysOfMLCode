import numpy as np
from scipy.special import digamma
import vocabulary
import pickle

niter = 15
niterq = 10

w = [[]]
wq = [[]]

K0 = 1

V = 0

alpha = 1.0
beta = 0.1
gamma = 1.5
tau = []  # double

kgaps = []
kactive = []
nmk = [[]]
nmkq = [[]]
nkt = [[]]  # the length of the 2nd dimension is the same
nk = []
phi = [[]]  # double
z = [[]]
zq = [[]]

pp = []  # double

ppstep = 10

aalpha = 5.0
balpha = 0.1
abeta = 0.1
bbeta = 0.1
agamma = 5.0
bgamma = 0.1

R = 10
T = 0.0
K = 1
M = 0
Mq = 0
Wq = 0

inited = False
fixedK = False
fixedHyper = False
thetaq = [[]]
theta = [[]]


# initialize parameter

def randGammaInner(rr):
    if rr <= 0.0:
        return 0.0
    elif rr == 1.0:
        return -np.log(np.random.rand())
    elif rr < 1.0:
        cc = 1.0 / rr
        dd = 1.0 / (1.0 - rr)
        while True:
            xx = np.power(np.random.rand(), cc)
            yy = xx + np.power(np.random.rand(), dd)
            if yy <= 1.0:
                return -np.log(np.random.rand()) * xx / yy
    else:
        bb = rr - 1.0
        cc = 3.0 * rr - 0.75;
        while True:
            uu = np.random.rand()
            vv = np.random.rand()
            ww = uu * (1.0 - uu)
            yy = np.sqrt(cc / ww) * (uu - 0.5)
            xx = bb + yy
            if xx >= 0:
                zz = 64.0 * ww * ww * ww * vv * vv
                if (zz <= (1.0 - 2.0 * yy * yy / xx)) or (np.log(zz) <= 2.0 * (bb * np.log(xx / bb) - yy)):
                    return xx


def randGamma(shape, scale):
    return randGammaInner(shape) * scale;


def estimateAlphaMap(nmk, nm, alpha, a, b):
    iter = 200
    M = len(nmk)  # size of topics
    K = len(nmk[0])  # size of key words
    alpha0 = 0.0
    prec = 1e-5

    for i in range(iter):
        summk = 0
        summ = 0
        for m in range(M):
            summ = summ + digamma(K * alpha + nm[m])
            for k in range(K):
                summk = summk + digamma(alpha + nmk[m][k])
        summ = summ - M * digamma(K * alpha)
        summk = summk - M * K * digamma(alpha)
        alpha = (a - 1 + alpha * summk) / (b + K * summ)
        if abs(alpha - alpha0) < prec:
            return alpha
        alpha0 = alpha
    return alpha


# Wq=1
##@return the perplexity of the last query sample
def ppx():
    loglik = 0.0
    global Mq, K, nmkq, alpha, wq, phi, thetaq
    # compute thetaq
    thetaq = [[0 for j in range(K)] for i in range(Mq)]  # double
    for m in range(Mq):
        for k in range(K):
            thetaq[m][k] = (nmkq[m][k] + alpha) / (len(wq[m]) + K * alpha)

    # compute ppx
    for m in range(Mq):
        for n in range(len(wq[m])):
            sum = 0.0
            for k in range(K):
                sum += thetaq[m][k] * phi[k][wq[m][n]]
            loglik += np.log(sum)
    return np.exp(-loglik / Wq)


# end ppx()

# print(ppx())


def randMultDirect(pp):
    for i in range(1, len(pp)):
        pp[i] += pp[i - 1]

    randNum = np.random.rand() * pp[i - 1]
    i = np.searchsorted(pp, randNum)
    return i


# end


def mult(ds, d):
    for i in range(len(ds)):
        ds[i] *= d


def stirling(nn):
    MAXSTIRLING = 20000
    allss = [[] for i in range(MAXSTIRLING)]
    #    lmss = 0.0
    maxnn = 1
    logmaxss = [0.0 for i in range(MAXSTIRLING)]

    if allss[0] == []:
        # print 'null'
        allss[0] = [0.0 for i in range(1)]
        allss[0][0] = 1
        logmaxss[0] = 0

    if nn > maxnn:

        for mm in range(maxnn, nn):
            lenth = len(allss[mm - 1]) + 1
            # print lenth
            allss[mm] = [0.0 for i in range(lenth)]
            # print allss[0][0]
            for xx in range(lenth):
                # allss{mm} = [allss{mm-1}*(mm-1) 0] + ...
                if xx < lenth - 1:
                    allss[mm][xx] += allss[mm - 1][xx] * mm
                if xx != 0:
                    allss[mm][xx] += allss[mm - 1][xx - 1]
            mss = max(allss[mm])
            # print mss

            mult(allss[mm], 1 / mss);
            logmaxss[mm] = logmaxss[mm - 1] + np.log(mss)

        maxnn = nn

    # lmss = logmaxss[nn - 1]
    return allss[nn - 1]


# end



def randAntoniak(alpha, n):
    p = []
    p = stirling(n)
    aa = 1.0
    for m in range(len(p)):
        p[m] *= aa
        aa *= alpha
    return randMultDirect(p) + 1


# end

def randGammaArray(aa):
    gamma = [0.0 for i in range(len(aa))]
    for i in range(len(aa)):
        gamma[i] = randGammaInner(aa[i])
    return gamma


# end


def randDir(aa):
    ww = randGammaArray(aa)

    sum = 0.0
    for i in range(len(ww)):
        sum += ww[i]

    i = 0
    for i in range(len(ww)):
        ww[i] /= sum
    return ww


# end


def updateTau():
    # (40) sample mk
    global K, kactive, nmk, tau, gamma, T
    mk = [0.0 for i in range(K + 1)]
    for kk in range(K):
        k = kactive[kk]
        for m in range(M):
            if nmk[m][k] > 1:
                mk[kk] += randAntoniak(alpha * tau[k], nmk[m][k])
            else:
                mk[kk] += nmk[m][k]
    # numbers of tables
    T = sum(mk)
    mk[K] = gamma
    # (36) sample tau

    tt = randDir(mk)
    kk = 0
    for kk in range(K):
        k = kactive[kk]
        tau[k] = tt[kk]
    tau[K] = tt[K]
    # print "len(tau)",len(tau)


# end


def spawnTopic(m, t):
    global kgaps, kactive, nmk, K, M, V, ppstep, pp, nk
    k = 0
    if len(kgaps) > 0:
        # reuse gap
        # k = kgaps.remove(kgaps.size() - 1);
        k = kgaps[0]
        kgaps.remove(k)
        kactive.append(k)
        nmk[m][k] = 1
        nkt[k][t] = 1
        nk[k] = 1
    else:
        # add element to count arrays
        k = K
        for i in range(M):
            nmk[i].append(0)
        kactive.append(K)
        nmk[m][K] = 1
        nkt.append([0 for i in range(V)])
        # print "nkt",nkt
        nkt[K][t] = 1
        nk.append(1)
        tau.append(0.)
    K = K + 1
    if len(pp) <= K:
        pp = [0.0 for i in range(K + ppstep)]

    return k


# end


def run(niter):
    global M, w, z, nmk, nkt, nk, inited, kactive, M, V, alpha, K
    for iiter in range(niter):
        print iiter
        print 'K=', K, 'alpha=', alpha, 'beta=', beta, 'gamma=', gamma, 'M=', M, 'V=', V
        for m in range(M):
            for n in range(len(w[m])):
                # sampling z
                k = -1
                kold = -1
                t = w[m][n]
                if inited:
                    k = z[m][n]
                    # decrement
                    nmk[m][k] = nmk[m][k] - 1
                    nkt[k][t] = nkt[k][t] - 1
                    nk[k] = nk[k] - 1
                    kold = k
                # compute weight
                psum = 0.0
                for kk in range(K):
                    k = kactive[kk]
                    pp[kk] = (nmk[m][k] + alpha * tau[k]) * (nkt[k][t] + beta) / (nk[k] + V * beta)
                    psum += pp[kk]
                # liklyhood of new component
                if not fixedK:
                    pp[K] = alpha * tau[K] / V
                    psum += pp[K]
                u = np.random.rand()
                u *= psum
                psum = 0
                kk = 0
                for kk in range(K + 1):
                    psum += pp[kk]
                    if u <= psum:
                        break
                # reassign and increment
                if kk < K:
                    k = kactive[kk]
                    z[m][n] = k
                    nmk[m][k] = nmk[m][k] + 1
                    nkt[k][t] = nkt[k][t] + 1
                    nk[k] = nk[k] + 1
                else:
                    assert (not fixedK)
                    z[m][n] = spawnTopic(m, t)
                    updateTau()
                    print 'K=', K
                # empty topic?
                if inited and nk[kold] == 0:
                    kactive.remove(kold)
                    kgaps.append(kold)
                    kgaps.sort()
                    assert (sum(nkt[kold]) == 0 and nk[kold] == 0 and nmk[m][kold] == 0)
                    K = K - 1
                    print 'K=', K

        if not fixedK:
            updateTau()
        if (iiter > 10) and (not fixedHyper):
            updateHyper()


# end

# initialise Markov chain
def init():
    global nmk, nkt, nk, z, M, K, w, kactive, kgaps, tau, V, pp, ppstep, fixedK, inited

    nmk = [[0] * K for i in range(M)]
    nkt = [[0] * V for i in range(K)]
    nk = [0 for i in range(K)]
    z = [[] for i in range(M)]

    for m in range(M):
        z[m] = [0 for i in range(len(w[m]))]

    kactive = range(K)
    tau = [1.0 / K for i in range(K + 1)]

    pp = [0.0 for i in range(K + ppstep)]
    run(1)
    if not fixedK:
        updateTau()
    inited = True


# initialise Markov chain for querying
def initq():
    global K, kgaps, V, kactive, phi, nkt, beta, nk, nmkq, Mq, zq, Wq

    Kg = K + len(kgaps)
    phi = [[0.0] * V for i in range(Kg)]
    for kk in range(K):
        k = kactive[kk]
        for t in range(V):
            phi[k][t] = (nkt[k][t] + beta) / (nk[k] + V * beta)
    nmkq = [[0] * Kg for i in range(Mq)]
    zq = [[] for i in range(Mq)]
    Wq = 0

    for m in range(Mq):
        zq[m] = [0 for i in range(len(wq[m]))]
        for n in range(len(wq[m])):
            k = np.random.randint(K)
            zq[m][n] = k
            nmkq[m][k] = nmkq[m][k] + 1
            Wq = Wq + 1


# query Gibbs sampler. This assumes the standard LDA model as we know the
# dimensionality from the training set, therefore topics need to be pruned.
def runq(niter):
    global nmkq, wq, zq, K, pp, alpha, phi, zq, nmkq

    for qiter in range(niter):
        for m in range(len(nmkq)):
            for n in range(len(wq[m])):
                k = zq[m][n]
                t = wq[m][n]
                nmkq[m][k] = nmkq[m][k] - 1
                psum = 0.0
                for kk in range(K):
                    pp[kk] = (nmkq[m][kk] + alpha) * phi[kk][t]
                    psum = psum + pp[kk]
                u = np.random.rand() * psum
                psum = 0.0
                kk = 0
                for kk in range(K):
                    psum = psum + pp[kk]
                    if u <= psum:
                        break

                zq[m][n] = kk
                nmkq[m][kk] = nmkq[m][kk] + 1


# reorders topics such that no gaps exist in the count arrays and topics
# are ordered with their counts descending. Removes any gap dimensions.
def packTopics():
    global nk, nkt, kgaps, M, nmk, K, kactive, z, w

    knew2k = sorted(range(len(nk)), key=lambda x: nk[x], reverse=True)
    nk = sorted(nk, key=lambda x: knew2k.index(nk.index(x)))
    nkt = sorted(nkt, key=lambda x: knew2k.index(nkt.index(x)))

    for i in range(len(kgaps)):
        del nk[-1]
        del nkt[-1]

    for m in range(M):
        nmk[m] = sorted(nmk[m], key=lambda x: knew2k.index(nmk[m].index(x)))
        for i in range(len(kgaps)):
            del nmk[m][-1]
    kgaps = []
    k2knew = sorted(range(len(knew2k)), key=lambda x: knew2k[x])  # k2knew is a map
    for i in range(K):
        kactive[i] = k2knew[kactive[i]]
    for m in range(M):
        for n in range(len(w[m])):
            z[m][n] = k2knew[z[m][n]]


# update scalar DP hyperparameters alpha, gamma and Dirichlet
# hyperparameter beta. Assumes that T is updated (by updateTau).
def updateHyper():
    global R, gamma, T, agamma, bgamma, K, M, V, alpha, w, aalpha, balpha, nk, nkt, beta, abeta, bbeta

    for r in range(R):
        eta = np.random.dirichlet([gamma + 1, T])[0]
        bloge = bgamma - np.log(eta)
        pie = 1.0 / (1.0 + (T * bloge / (agamma + K - 1)))
        u = np.random.binomial(1, pie)
        gamma = randGamma(agamma + K - 1 + u, 1.0 / bloge)

        qs = 0
        qw = 0
        for m in range(M):
            qs = qs + np.random.binomial(1, len(w[m]) / (len(w[m]) + alpha))
            qw = qw + np.log(np.random.dirichlet([alpha + 1, len(w[m])])[0])
        alpha = randGamma(aalpha + T - qs, 1.0 / (balpha - qw))

    beta = estimateAlphaMap(nkt, nk, beta, abeta, bbeta)


class FileOutput:
    def __init__(self, file):
        self.file = file
        import datetime
        self.file = file + datetime.datetime.now().strftime('_%m%d_%H%M%S.txt')

    def out(self, s):
        with open(self.file, 'a') as f:
            print >> f, s


corpus = vocabulary.load_file('mood.txt')
# corpus = vocabulary.load_corpus('1:50')
voca = vocabulary.Vocabulary(True)
w = [voca.doc_to_ids(doc) for doc in corpus][:86]
wq = [voca.doc_to_ids(doc) for doc in corpus][:10]
M = len(w)
Mq = len(wq)
V = voca.size()

init()
run(niter)
initq()
runq(niterq)
ppx()

theta = [[0 for j in range(K)] for i in range(M)]  # double
for m in range(M):
    for k in range(K):
        theta[m][k] = (nmk[m][k] + alpha) / (len(w[m]) + K * alpha)

output = open('modelData', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(theta, output)

output.close()

output = open('testingData', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(thetaq, output)

output.close()

f = FileOutput("hdp_trainning")

for k in range(K):
    f.out("\n-- topic: %d" % k)
    sortedindex = sorted(range(len(phi[k])), key=lambda x: phi[k][x], reverse=True)
    for itraining in sortedindex[:20]:
        f.out("%s: %f" % (voca[itraining], phi[k][itraining]))

for m in range(len(wq)):
    f.out("\n-- doc: %d" % m)
    sortedindex = sorted(range(len(thetaq[m])), key=lambda x: thetaq[m][x], reverse=True)
    for k in sortedindex[:5]:
        f.out("topic %s: %f" % (k, thetaq[m][k]))













