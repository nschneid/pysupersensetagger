'''
Ported from Michael Heilman's LabeledSentence.java
@author: Nathan Schneider (nschneid)
@since: 2012-07-23
'''
from __future__ import print_function, division
import sys, os, re, fileinput, codecs

def wordShape(tkn):
    '''Word shape feature described by Ciaramita & Altun 2006'''
    shape = ''
    prevCharType = -1
    addedStar = False
    for curChar in tkn:
        if curChar>='A' and curChar<='Z':
            charType = 'X'
        elif curChar>='a' and curChar<='z':
            charType = 'x'
        elif curChar>='0' and curChar<='9':
            charType = 'd'
        else:
            charType = curChar
        
        if charType==prevCharType:
            if not addedStar:
                shape = ''.join([shape, '*'])
                addedStar = True
        else:
            addedStar = False
            shape = ''.join([shape, charType])
            
        prevCharType = charType
        
    return shape

class LabeledSentence(list):
    '''
    Stores information about the tokens in a sequence. For each token 
    position is a tuple of the form 
      (token, stem, POS, goldLabel, predictedLabel, wordShape)
    Predicted labels are optionally supplied via setPredictions().
    Properties 'articleId' and 'mostFrequentSenses' may be kept 
    as well.
    '''
    def __init__(self):
        list.__init__(self)
        self._mostFrequentSenses = None
        self._articleId = ''
    
    def addToken(self, token, stem, pos, goldLabel):
        self.append((token, stem, pos, goldLabel, "", wordShape(token)))
        self._mostFrequentSenses = None
        
    def getTokens(self):
        return [x[0] for x in self]
    
    def getTokenAt(self, i):
        return self[i][0]
    
    def getStems(self):
        return [x[1] for x in self]
    
    def getStemAt(self, i):
        return self[i][1]
    
    def getPOS(self):
        return [x[2] for x in self]
    
    def getPOSAt(self, i):
        return self[i][2]
    
    def getLabels(self):
        '''Gold labels'''
        return [x[3] for x in self]
    
    def getLabelAt(self, i):
        '''Gold label'''
        return self[i][3]
    
    def getPredictions(self):
        return [x[4] for x in self]
    
    def getPredictionAt(self, i):
        return self[i][4]
    
    def setPredictions(self, predLabels):
        assert len(predLabels)==len(self)
        for i,itm in enumerate(self):
            tok,stem,pos,gold,_,shape = itm
            self[i] = (tok,stem,pos,gold,predLabels[i],shape)
    
    def setPredictionAt(self, i, label):
        self[i] = self[i][:4]+(label,)+self[i][5:]
    
    def predictionsAreCorrect(self):
        return all(x[3]==x[4] for x in self)
    
    def getWordShapes(self):
        return [x[5] for x in self]
    
    def getWordShapeAt(self, i):
        return self[i][5]
    
    @property
    def articleId(self):
        return self._articleId
    @articleId.setter
    def articleId(self, val):
        self._articleId = val

    @property
    def mostFrequentSenses(self):
        return self._mostFrequentSenses
    @mostFrequentSenses.setter
    def mostFrequentSenses(self, val):
        self._mostFrequentSenses = val

    def taggedString(self, usePredictionsRatherThanGold=True):
        '''
        3-column output
        '''
        return '\n'.join(tok+'\t'+pos+'\t'+(pred if usePredictionsRatherThanGold else gold) for tok,stem,pos,gold,pred,shape in self)

    def __str__(self):
        return self.taggedString()
    