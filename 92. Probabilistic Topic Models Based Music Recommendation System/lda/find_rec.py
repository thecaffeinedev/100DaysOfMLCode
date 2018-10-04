import pickle
import numpy as np

lines = [','.join(line.rstrip().split(',')[2:]) for line in open('playlist.txt')]

with open('newtheta.pk', 'rb') as a:
    newtheta = pickle.load(a)

with open('theta.pk', 'rb') as a:
    theta = pickle.load(a)

# kllist = np.zeros((len(newtheta), len(theta)))
kllist = [[0.0]*len(theta) for i in range(len(newtheta))]
ThetaIndex = []

# calculate KL divergence
def KL(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return np.sum((np.where(a != 0, a * np.log(a / b), 0)))

def KLpre(thetaq, phi):
    global kllist
    for i in range(len(thetaq)):
        for j in range(len(phi)):
            kllist[i][j] = KL(thetaq[i], phi[j])
        ThetaIndex.append(sorted(range(len(kllist[i])),key = lambda x: kllist[i][x]))
        #print(ThetaIndex[i])

KLpre(newtheta, theta)

#print ThetaIndex

out = open("recommendation.txt", 'w')
for i in range(450, 477):
    out.write('\n----------------------------------\n' + lines[i] +  '\n++++recommed++++\n')
    out.write('\n'.join([lines[j] for j in ThetaIndex[i-450][:50]]))