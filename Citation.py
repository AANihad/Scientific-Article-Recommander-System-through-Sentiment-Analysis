import torch.nn as nn
import numpy as np

class Citation():

    def __init__(self, text, keys, references, sentiments):
        self.text = text
        self.keys = keys
        self.references = references

        self.scoreP = sentiments[0]
        self.scoreO = sentiments[1]
        self.scoreN = sentiments[2]
        self.sentiments = sentiments

    def getText(self):
        return self.text

    def refersTo(self, ref):
        if self.references is not None:
            return(ref in self.references)


    def as_dict(self):
        return {'Citation keys': str(self.keys), 'Citation text': self.text, 'References': self.references}
   
    def mockEvaluate(self):
        self.scoreP = 0.6
        self.scoreO = 0.22
        self.scoreN = 0.18

    def getSentiments(self):
        return self.scoreP, self.scoreO, self.scoreN

    def getDominentSentiment(self):
        return np.argmax(self.sentiments)