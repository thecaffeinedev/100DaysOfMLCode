import pickle
import numpy as np

lines = [','.join(line.rstrip().split(',')[2:]) for line in open('playlist.txt')]

with open('testingData', 'rb') as a:
    newtheta = pickle.load(a)

with open('modelData', 'rb') as a:
    theta = pickle.load(a)

#kllist = np.zeros((len(newtheta), len(theta)))
kllist = [[0.0]*len(theta) for i in range(len(newtheta))]
ThetaIndex = []

def KL(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def KLpre(thetaq, phi):
    global kllist
    for i in range(len(thetaq)):
        for j in range(len(phi)):
            kllist[i][j] = KL(thetaq[i], phi[j])

        ThetaIndex.append(sorted(range(len(kllist[i])),key = lambda x: kllist[i][x]))


KLpre(newtheta, theta)

#print ThetaIndex
for i in range(0, 10):
    print lines[i], '\n', '++++recommed++++', '\n', ThetaIndex[i][1:8],'\n'
    # for j in range(5):
    #     print ThetaIndex[i][j]

for i in range(0, 10):
    print lines[i], '\n', '++++recommed++++', '\n', [lines[j] for j in ThetaIndex[i][1:6]],'\n'
    # for j in range(5):
    #     print ThetaIndex[i][j]

#
# def output_topics_doc(p):
#     theta = p
#     for k in np.argsort(-theta[54])[:5]:
#         print ("topic %s: %f" % (k, theta[54][k]))
# #print len(newtheta), len(theta), len(newtheta[1]), len(theta[1]), newtheta[1][1]
# output_topics_doc(theta)