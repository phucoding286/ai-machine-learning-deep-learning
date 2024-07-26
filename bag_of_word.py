import numpy as np

class BagOfWord:
    def __init__(self):
        self.maxlen = 0

    def counter_tranform(self, x):
        sent_enc = []
        for char in list(x):
            appeer_times = 0
            for count in list(x):
                if char == count:
                    appeer_times += 1
            sent_enc.append(appeer_times - 1)
            if len(sent_enc) > self.maxlen:
                self.maxlen = len(sent_enc)
        return sent_enc
    
    def fit_transform(self, x_batch):
        batch = []
        for sentence in x_batch:
            enc_op = self.counter_tranform(sentence)
            batch.append(enc_op)
        for i in range(len(batch)):
            if len(batch[i]) < self.maxlen:
                for _ in range(len(batch[i]), self.maxlen):
                    batch[i].append(0)
            else:
                batch[i] = batch[i][:self.maxlen]
        return np.array(batch)
    
    def counter(self, x):
        sent_enc = []
        for char in list(x):
            appeer_times = 0
            for count in list(x):
                if char == count:
                    appeer_times += 1
            sent_enc.append(appeer_times - 1)
        return sent_enc
    
    def fit(self, x):
        batch = []
        for sentence in x:
            enc_op = self.counter(sentence)
            batch.append(enc_op)
        for i in range(len(batch)):
            if len(batch[i]) < self.maxlen:
                for _ in range(len(batch[i]), self.maxlen):
                    batch[i].append(0)
            else:
                batch[i] = batch[i][:self.maxlen]
        return np.array(batch)
