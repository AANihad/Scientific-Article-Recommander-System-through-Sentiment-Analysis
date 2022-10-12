import Citation
from statistics import mean

class Reference():
    def __init__(self, text):
        self.text = text
        self.citations=[]

    def addCitation(self, c):
        self.citations.append(c)

    def getText(self):
        return self.text

    def appendText(self, t):
        self.text = self.text.join(' ').join(t)

    def as_dict(self):
        try:
            return {'Reference text': str(self.text), 'Avg Pos': self.avP,'Avg Obj': self.avO,'Avg Neg': self.avN, 'Pos Citation Count': self.sentCounter[0], 'Obj Citation Count': self.sentCounter[1], 'Neg Citation Count': self.sentCounter[2]} #, 'Sentiment': self.sentiment
        except AttributeError:
            return {'Reference text': str(self.text), 'Avg Pos': 0,'Avg Obj': 0,'Avg Neg': 0, 'Pos Citation Count': 0, 'Obj Citation Count': 0, 'Neg Citation Count': 0} 
   
    def calculateSentiments(self):
        self.avP, self.avO, self.avN = 0, 0, 0
        self.sentCounter = [0, 0, 0]
        # print(len(self.citations))
        for c in self.citations:
            p, o, n = c.getSentiments()
            self.avP += p
            self.avO += o
            self.avN += n
            self.sentCounter[c.getDominentSentiment()] +=1
        self.avP /= len(self.getCitations())
        self.avO /= len(self.getCitations())
        self.avN /= len(self.getCitations())

    def getCitations(self):
        return self.citations
