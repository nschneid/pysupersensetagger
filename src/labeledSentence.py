'''
Ported from Michael Heilman's LabeledSentence.java
@author: Nathan Schneider (nschneid)
@since: 2012-07-23
'''
from __future__ import print_function, division
from collections import namedtuple

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

Token = namedtuple('Token', 'token stem pos gold prediction shape')

class LabeledSentence(list):
    '''
    Stores information about the tokens in a sequence. For each token 
    position is a tuple of the form 
      (token, stem, POS, goldLabel, predictedLabel, wordShape)
    Properties 'articleId' and 'mostFrequentSenses' may be kept 
    as well.
    '''
    def __init__(self):
        list.__init__(self)
        self._mostFrequentSenses = None
        self._articleId = ''
    
    def addToken(self, token, stem, pos, goldLabel):
        self.append(Token(token, stem, pos, goldLabel, "", wordShape(token)))
        self._mostFrequentSenses = None
    
    def predictionsAreCorrect(self):
        return all(x.gold == x.prediction for x in self)
    
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

    def __str__(self):
        return '\n'.join('{0.token}\t{0.pos}\t{0.prediction}'.format(tok) for tok in self)
    
