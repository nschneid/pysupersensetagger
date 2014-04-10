#coding=utf-8
'''
Data structure for a sentence and its tokens, including 
metadata such as stems and gold/predicted tags.
Partially ported from Michael Heilman's LabeledSentence.java
@author: Nathan Schneider (nschneid)
@since: 2012-07-23
'''
from __future__ import print_function, division
from collections import namedtuple

I_BAR, I_TILDE, i_BAR, i_TILDE = 'ĪĨīĩ'.decode('utf-8')

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

_Token = namedtuple('Token', 'token stem pos gold prediction shape')
class Token(_Token):
    goldparent = goldstrength = goldlabel = ''
    predparent = predstrength = predlabel = ''

class LabeledSentence(list):
    '''
    Stores information about the tokens in a sequence. For each token 
    position is a tuple of the form 
      (token, stem, POS, goldLabel, predictedLabel, wordShape)
    Properties 'sentId' and 'mostFrequentSenses' may be kept 
    as well.
    '''
    def __init__(self):
        list.__init__(self)
        self._mostFrequentSenses = None
        self._sentId = ''
    
    def addToken(self, token, stem, pos, goldTag, **kwargs):
        tok = Token(token, stem, pos, goldTag, "", wordShape(token))
        for k,v in kwargs.items():
            setattr(tok, k, v)
        self.append(tok)
        self._mostFrequentSenses = None
    
    def predictionsAreCorrect(self):
        return all(x.gold == x.prediction for x in self)
    
    def updatedPredictions(self):
        '''Called once every token has a value for .prediction; 
           sets .predstrength, .predlabel, and .predparent accordingly. 
           Note that .predparent is 0 if no parent and a positive offset otherwise.'''
        curOuterMWE = None
        curInnerMWE = None
        for i,tok in enumerate(self):
            if '-' in tok.prediction:
                predPosition = tok.prediction[:tok.prediction.index('-')]
                predLbl = tok.prediction[tok.prediction.index('-')+1:]
            else:
                predPosition = tok.prediction
                predLbl = ''
            
            tok.predlabel = predLbl
            if predPosition in {'B','I',I_BAR,I_TILDE}:
                if predPosition=='B':
                    assert curInnerMWE is None
                    tok.predparent = 0
                else:
                    assert curOuterMWE is not None
                    tok.predparent = curOuterMWE
                    tok.predstrength = '_' if predPosition==I_BAR else ('~' if predPosition==I_TILDE else '')
                curOuterMWE = i+1
                curInnerMWE = None
            elif predPosition in {'b','i',i_BAR,i_TILDE}:
                assert curOuterMWE is not None
                if predPosition=='b':
                    tok.predparent = 0
                else:
                    assert curInnerMWE is not None
                    tok.predparent = curInnerMWE
                    tok.predstrength = '_' if predPosition==i_BAR else ('~' if predPosition==i_TILDE else '')
                curInnerMWE = i+1
            elif predPosition=='o':
                assert curOuterMWE is not None
                tok.predparent = 0
                curInnerMWE = None
            else:
                assert predPosition=='O'
                assert curInnerMWE is None
                tok.predparent = 0
                curOuterMWE = None
        assert curInnerMWE is None
    
    @property
    def sentId(self):
        return self._sentId
    @sentId.setter
    def sentId(self, val):
        self._sentId = val

    @property
    def mostFrequentSenses(self):
        return self._mostFrequentSenses
    @mostFrequentSenses.setter
    def mostFrequentSenses(self, val):
        self._mostFrequentSenses = val

    def __str__(self):
        '''offset   word   lemma   POS   tag   parent   strength   label   sentID'''
        return '\n'.join(u'{offset}\t{0.token}\t{0.stem}\t{0.pos}\t{0.prediction}\t{0.predparent}\t{0.predstrength}\t{0.predlabel}\t{sentId}'.format(tok,offset=i+1,sentId=self.sentId) 
                         for i,tok in enumerate(self)).encode('utf-8')
    
