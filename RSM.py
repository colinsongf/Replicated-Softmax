# -*- coding: utf-8 -*-
import numpy as np
import cPickle
from collections import Counter, defaultdict

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

class RSM(object):

    def __init__(self, n_visible, n_hidden, mbtsz, epochs, eta, mrate, np_rng, weightinit=0.001):
        """
        CD-k training of RSM with SGD + Momentum.
        @param n_visible:   num of lexicon
        @param n_hidden:    num of latent topics
        @param epochs:      training epochs
        @param eta:         learning rate
        @param mrate:       momentum rate
        @param mbtsz:       mini-batch size
        @param np_rng:      instances of RandomState
        @param weightinit:  scaling of random weight initialization
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.mbtsz = mbtsz
        self.epochs = epochs
        self.eta = eta
        self.mrate = mrate
        self.np_rng = np_rng
        self.W = weightinit * np_rng.randn(n_visible, n_hidden)
        self.vbias = weightinit * np_rng.randn(n_visible)
        self.hbias = np.zeros((n_hidden))
        # for momentum
        self.mW = np.zeros((n_visible, n_hidden))
        self.mvbias = np.zeros((n_visible))
        self.mhbias = np.zeros((n_hidden))

    def train(self, data):
        for epoch in range(self.epochs):
            self.np_rng.shuffle(data)
            for i in range(0, data.shape[0], self.mbtsz):
                mData = data[i:i + self.mbtsz]
                ph_mean, nv_samples, nh_means = self.cd_k(mData)

                self.mW = self.mW * self.mrate + (np.dot(mData.T, ph_mean) - np.dot(nv_samples.T, nh_means))
                self.mvbias = self.mvbias * self.mrate + np.mean(mData - nv_samples, axis=0)
                self.mhbias = self.mhbias * self.mrate + np.mean(ph_mean - nh_means, axis=0)
                self.W += self.eta * self.mW
                self.vbias += self.eta * self.mvbias
                self.hbias += self.eta * self.mhbias

    def cd_k(self, data, k=1):
        D = data.sum(axis=1)
        ph_mean, ph_sample = self.sample_h(data, D)
        chain_start = ph_sample

        for step in range(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start, D) 
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples, D)
        return ph_mean, nv_samples, nh_means

    def sample_h(self, v0_sample, D):
        h1_mean = sigmoid(np.dot(v0_sample, self.W) + np.outer(D, self.hbias))
        h1_sample = self.np_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return [h1_mean, h1_sample]

    def sample_v(self, h0_sample, D):
        pre_soft = np.exp(np.dot(h0_sample, self.W.T) + self.vbias)
        pre_soft_sum = pre_soft.sum(axis=1).reshape((self.mbtsz, 1))
        v1_mean = pre_soft/pre_soft_sum
        v1_sample = np.zeros((self.mbtsz, v1_mean.shape[1]))
        for i in range(self.mbtsz):
            v1_sample[i] = self.np_rng.multinomial(size=1, n=D[i], pvals=v1_mean[i])
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample, D):
        v1_mean, v1_sample = self.sample_v(h0_sample, D)
        h1_mean, h1_sample = self.sample_h(v1_sample, D)
        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def wordPredict(self, topic, voc):
        vecTopics = np.zeros((topic, topic))
        for i in range(len(vecTopics)):
            vecTopics[i][i] = 1
        for i, vecTopic in enumerate(vecTopics):
            pre_soft = np.exp(np.dot(vecTopic, self.W.T) + self.vbias)
            pre_soft_sum = pre_soft.sum().reshape((1, 1))
            word_distribution = (pre_soft/pre_soft_sum).flatten()
            tmpDict = {}
            for j in range(len(voc)):
                tmpDict[voc[j]] = word_distribution[j]
            print 'topic', str(i), ':', vecTopic
            for word, prob in sorted(tmpDict.items(), key=lambda x:x[1], reverse=True):
                print word, str(prob)
            print '-'

    def saveParams(self, filePath):
        cPickle.dump({'W': self.W,
                      'vbias': self.vbias,
                      'hbias': self.hbias},
                      open(filePath, 'w'))

def inputData(filePath):
    docs = []
    voc = defaultdict(lambda: len(voc))
    file = open(filePath, "r")
    for line in file:
        doc = line.rstrip().split()
        for word in doc:
            voc[word]
        cnt = Counter(doc)
        docs.append(cnt)
    file.close()
    docSize, vocSize = len(docs), len(voc)
    v = np.zeros((docSize, vocSize))
    for i in range(docSize):
        for word, freq in docs[i].most_common():
            wID = voc[word]
            v[i][wID] = freq
    return v, {v:k for k, v in voc.items()}

if __name__ == '__main__':
    docs, voc = inputData('doc_Rsm.txt')
    topic = 3
    rsm = RSM( n_visible=len(docs[0]), 
               n_hidden=topic, 
               mbtsz=5,
               epochs=2000,
               eta=0.01,
               mrate=0.8,
               np_rng=np.random.RandomState(1234))
    rsm.train(docs)
    rsm.wordPredict(topic, voc)
    # rsm.saveParams('param.pcl')