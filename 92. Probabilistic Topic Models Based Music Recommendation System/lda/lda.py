import numpy


class LDA:
    # GibbsLDA
    def __init__(self, K, alpha, beta, docs, V):
        self.K = K      # topics
        self.alpha = alpha
        self.beta = beta
        self.docs = docs
        self.V = V              # Vocabulary
        self.z_m_n = []     # topics of words of documents
        self.n_m_k = numpy.zeros((len(self.docs), K)) + alpha     # increament doc-topic count
        self.n_k_t = numpy.zeros((K, V)) + beta     # increament topic-word count
        self.n_k = numpy.zeros(K) + V * beta    # increament topic-word sum

        # initialization
        for m, doc in enumerate(docs):   # document m
            z_n = []
            for t in doc:   # word t
                z = numpy.random.randint(0, K)
                z_n.append(z)
                self.n_m_k[m, z] += 1
                self.n_k_t[z, t] += 1
                self.n_k[z] += 1
            self.z_m_n.append(numpy.array(z_n))     # sample topic index

    def sampling(self):
        # burn-in and sampling
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_k = self.n_m_k[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_k[z] -= 1
                self.n_k_t[z, t] -= 1
                self.n_k[z] -= 1

                # sampling topic new_z for t from p(z|)
                p_z = (self.n_k_t[:, t] * n_m_k) / self.n_k
                new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_k[new_z] += 1
                self.n_k_t[new_z, t] += 1
                self.n_k[new_z] += 1

    def phi(self):
        return self.n_k_t / self.n_k[:, numpy.newaxis]

    def p_joint(self, phi=None):
        # calculate joint probability after sampling
        docs = self.docs
        if phi is None:
            phi = self.phi()
        else:
            phi = phi
        log_pjoint = 0
        l = 0
        kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_k[m] / (len(self.docs[m]) + kalpha)
            for w in doc:
                log_pjoint -= numpy.log(numpy.inner(phi[:, w], theta))
            l += len(doc)
        return numpy.exp(log_pjoint / l)

    def theta(self):
        docs = self.docs
        theta = []
        kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta.append(self.n_m_k[m] / (len(self.docs[m]) + kalpha))
        return theta



