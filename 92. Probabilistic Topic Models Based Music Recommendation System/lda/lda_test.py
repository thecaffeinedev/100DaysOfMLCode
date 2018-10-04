import numpy
import lda


class FileOutput:
    def __init__(self, file):
        self.file = file
        import datetime
        self.file = file + datetime.datetime.now().strftime('_%m%d_%H%M%S.txt')

    def out(self, s):
        with open(self.file, 'a') as f:
            f.write(s+'\n')


def lda_training(f, LDA, voca, iteration=30):
    import time
    t0 = time.time()

    lda = LDA
    pre_pjoint = lda.p_joint()
    f.out("GibbsLDA initial prob=%f" % pre_pjoint)

    count = 0
    for i in range(iteration):
        print("Iteration: "+str(i))
        # if i % 10 == 0:
            # output_topics_word(f, lda, voca)
        lda.sampling()
        pjoint = lda.p_joint()
        f.out("-%d p=%f" % (i + 1, pjoint))
        if pre_pjoint is not None:
            if pre_pjoint < pjoint:
                count += 1
                if count >= 2:
                    # output_topics_word(f, lda, voca)
                    pre_pjoint = None
            else:
                count = 0
                pre_pjoint = pjoint
    output_topics_word(f, lda, voca)
    output_topics_doc(f, lda)

    t1 = time.time()
    f.out("time = %f\n" % (t1 - t0))

def lda_fit(f, LDAnew, phi, voca, iteration=30):
    lda = LDAnew
    pre_pjoint = lda.p_joint()
    f.out("GibbsLDA initial prob=%f" % pre_pjoint)

    count = 0
    for i in range(iteration):
        print("Fit Cycle: "+str(i))
        # if i % 10 == 0:
            # output_topics_doc(f, lda)
        lda.sampling()
        pjoint = lda.p_joint()
        f.out("-%d p=%f" % (i + 1, pjoint))
        if pre_pjoint is not None:
            if pre_pjoint < pjoint:
                count += 1
                if count >= 2:
                    # output_topics_doc(f, lda)
                    pre_pjoint = None
            else:
                count = 0
                pre_pjoint = pjoint
    output_topics_word(f, lda, voca)
    output_topics_doc(f, lda)




def output_topics_word(f, lda, voca):
    phi = lda.phi()
    for k in range(lda.K):
        f.out("\n-- topic: %d" % k)
        for w in numpy.argsort(-phi[k])[:20]:
            f.out("%s: %f" % (voca[w], phi[k, w]))


def output_topics_doc(f, lda):
    theta = lda.theta()
    #print(theta)
    for m, doc in enumerate(lda.docs):
        f.out("\n-- doc: %d" % m)
        for k in numpy.argsort(-theta[m])[:5]:
            f.out("topic %s: %f" % (k, theta[m][k]))


def main():
    import vocabulary
    # from sklearn.decomposition import PCA
    import pickle

    corpus = vocabulary.load_file('mood.txt')
    voca = vocabulary.Vocabulary(True)
    docs = [voca.doc_to_ids(doc) for doc in corpus]
    doctrain = docs[:450]
    doctest = docs[450:]

    # docs = voca.cut_low_freq(docs, 1)
    # SET parameter
    K = 10 # number of topics
    alpha, beta = 0.5, 0.5
    V = voca.size()

    f = FileOutput("lda_trainning")
    f.out("corpus=%d, words=%d, K=%d, alpha=%f, beta=%f" % (len(docs), len(voca.vocas), K, alpha, beta))
    LDA = lda.LDA(K, alpha, beta, docs, V)
    lda_training(f, LDA, voca, iteration=30) # set number of iterations
    theta = LDA.theta()[:450]
    newtheta = LDA.theta()[450:]
    with open("theta.pk", 'wb') as f:
        pickle.dump(theta, f)
    with open("newtheta.pk", 'wb') as f:
        pickle.dump(newtheta, f)



if __name__ == "__main__":
    main()
