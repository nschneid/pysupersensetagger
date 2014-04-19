'''
Utilities for scoring/evaluating data (human annotations or system output).

Plotting functions require the 'pylab' library (includes numpy and matplotlib).

@author: Nathan Schneider (nschneid)
@since: 2010-08-25
'''

# Strive towards Python 3 compatibility
from __future__ import print_function, unicode_literals, division, absolute_import
from future_builtins import map, filter

import re, math
from collections import namedtuple, defaultdict, Counter
from numbers import Number

def harmonicMean(P,R):
    if P+R==0.0: return float('NaN')
    return 2*P*R/(P+R)

_PRFScores = namedtuple('PRFScores', 'numer Pdenom Rdenom P R F')
class PRFScores(_PRFScores):
    '''
    Encodes precision/recall/F1 scores as well as the counts used to calculate them.
    
    >>> scores = PRFScores(numer=5, nGuesses=10, nGold=20)
    >>> scores
    PRFScores(numer=5, Pdenom=10, Rdenom=20, P=0.5, R=0.25, F=0.3333)
    >>> scores.Pdenom
    10
    >>> scores.P > scores.R
    True
    '''
    def __new__(cls, numer, nGuesses, nGold, suppressZeroDenominatorCheck=False):
        if not suppressZeroDenominatorCheck:
            assert nGuesses>0, 'The number of guesses must be > 0'
            assert nGold>0, 'The number of gold items must be > 0'
        P = numer/nGuesses if nGuesses>0 else float('NaN')
        R = numer/nGold if nGold>0 else float('NaN')
        o = _PRFScores.__new__(PRFScores, numer, nGuesses, nGold, P, R, harmonicMean(P,R) if numer>0 else 0.0)
        o.__class__ = PRFScores
        return o
    
    def __repr__(self, decimalPlaces=4):
        s = _PRFScores.__repr__(self)
        # Convert all floating point numbers to the level of precision specified in 'decimalPlaces'
        return re.sub(r'[=](\d+[.]\d+)', lambda m: ('={:.'+str(decimalPlaces)+'}').format(float(m.group(1))), s)

_ConfusionMatrix = namedtuple('ConfusionMatrix', 'Aonly Bonly Both Neither')
class ConfusionMatrix(_ConfusionMatrix):
    '''
    Confusion matrix for 2 annotations of binary-valued items.
    
    For example:
    
        +-----+-----+
        |  0  |  1  | <-- A's judgment
    +---+-----------+
    | 0 |   8     5  
    +---+     
    | 1 |   7    12
    +---+
      ^
      |
      B's judgment
    
    >>> c = ConfusionMatrix(5, 7, 12, 8)
    >>> c.totalItems
    32
    >>> c.Same
    20
    >>> c.pAgreement==20/32.0
    True
    >>> c.pRandomAgreement==(17/32*19/32)+(15/32*13/32)
    True
    >>> print('{:.4} {}'.format(c.CohensKappa, c.agreementCharacterization(c.CohensKappa)))
    0.2411 fair
    '''
        
    @property
    def isBinary(self):
        '''Returns whether this is a binary confusion matrix, in which 
        case the first three entries (Aonly, Bonly, Both) are all numeric values.'''
        for elt in (self.Aonly, self.Bonly, self.Both):
            if not isinstance(elt,Number):
                return False
        return True
    
    @property
    def labels(self):
        if self.isBinary:
            return [False,True]
        return set.union(set(self.Aonly.keys()),
                         set(self.Bonly.keys()),
                         set(lbls[0] for lbls in self.Both.keys()),
                         set(lbls[1] for lbls in self.Both.keys()))
    
    @property
    def totalItems(self):
        if self.isBinary:
            return self.Aonly + self.Bonly + self.Both + self.Neither
        return sum(self.Aonly.values())+sum(self.Bonly.values())+sum(self.Both.values())+self.Neither
    
    @property
    def Same(self):
        if self.isBinary:
            return self.Both + self.Neither
        assert False
    
    @property
    def Different(self):
        if self.isBinary:
            return self.Aonly + self.Bonly
        assert False
    
    @property
    def Atotal(self):
        if self.isBinary:
            return self.Aonly + self.Both
        assert False
    
    @property
    def Btotal(self):
        if self.isBinary:
            return self.Bonly + self.Both
        assert False
        
    @property
    def pA(self):
        return self.Atotal / self.totalItems
    
    @property
    def pB(self):
        return self.Btotal / self.totalItems
    
    @property
    def pAgreement(self):
        return self.Same / self.totalItems
    
    @property
    def pRandomAgreement(self):
        return self.pA * self.pB + (1 - self.pA)*(1 - self.pB)
    
    @property
    def CohensKappa(self):
        return (self.pAgreement - self.pRandomAgreement)/(1 - self.pRandomAgreement)
    
    @property
    def CohensKappaReport(self):
        return '{} ({})'.format(self.CohensKappa, ConfusionMatrix.agreementCharacterization(self.CohensKappa))
    
    @staticmethod
    def agreementCharacterization(cohensKappa):
        '''Heuristic characterization of the Cohen's Kappa value according to Landis & Koch (1977).'''
        if cohensKappa<0:
            return 'no agreement'
        elif cohensKappa==1:
            return 'perfect'
        #       < .2      < .4    < .6        < .8           < 1
        return ('slight', 'fair', 'moderate', 'substantial', 'almost perfect')[int(math.floor(cohensKappa*5))]
    
    def asPRF(self, goldAnnotator='A', **kwargs):
        assert goldAnnotator in ('A','B')
        if goldAnnotator=='A':
            return PRFScores(numer=self.Both, nGuesses=self.Btotal, nGold=self.Atotal, **kwargs)
        return PRFScores(self.Both, nGuesses=self.Atotal, nGold=self.Btotal, **kwargs)
    
    def asTable(self, formatS='{c} ({p:.0%})', type='tab', labels=None, emptyLabel='--', threshold=None, thresholdType='%'):
        assert not self.isBinary,'TODO: implement binary case'
        
        assert type in ('tab','html','csv')
        if labels is None:
            labels = self.labels
        labels = list(labels)
        labels.append(None)
        
        n = self.totalItems
        
        hdr = '<th>/{}</th>'.format(n)
        body = '\n'
        for i,lblA in enumerate(labels):    # rows
            hdr += '<th>{}</th>'.format(lblA if i<len(labels)-1 else emptyLabel)
            body += '\t<tr><th>{}</th>'.format(lblA if i<len(labels)-1 else emptyLabel)
            for j,lblB in enumerate(labels):    # cols
                if i<len(labels)-1: # labelled by A
                    if j<len(labels)-1: # labelled by A & B
                        v = self.Both[(lblA,lblB)]
                    else:   # labeled by A only
                        v = self.Aonly[lblA]
                elif j<len(labels)-1:   # labelled by B only
                    v = self.Bonly[lblB]
                else:   # labeled by neither
                    v = self.Neither
                    
                # render the value
                vS = ''
                if threshold is None or (thresholdType=='%' and v/n>threshold) or (thresholdType=='#' and v>threshold): 
                    vS = formatS.format(c=v,n=n,p=v/n)
                    
                body += '<td>{}</td>'.format(vS)
            body += '</tr>\n'
        
        if type=='html':
            return '<table>\n\t<tr>' + hdr + '</tr>' + body + '</table>'
        
        if type=='tab':
            return ''.join((
                     hdr.replace('</th><th>','\t').replace('<th>','\t').replace('</th>',''), 
                     body.replace('\t','').replace('<tr><th>','').replace('</tr>','').replace('</td><td>','\t').replace('</th><td>','\t').replace('</td>','')
                     ))
            
        if type=='csv':
            return ''.join((
                     hdr.replace('</th><th>','","').replace('<th>',',"').replace('</th>','"'), 
                     body.replace('\t','').replace('<tr><th>','"').replace('</tr>','').replace('</td><td>',',').replace('</th><td>','",').replace('</td>','')
                     ))
    
    def __add__(self, that):
        return ConfusionMatrix(self.Aonly+that.Aonly, self.Bonly+that.Bonly, self.Both+that.Both, self.Neither+that.Neither)
    
    @staticmethod
    def fromSets(setA, setB, others):
        '''Create a confusion matrix for binary-valued data 
        given sets of item identifiers for the two annotators plus a 
        set which may contain additional items.'''
        return ConfusionMatrix(len(setA.difference(setB)), len(setB.difference(setA)), 
                               len(setA.intersection(setB)), len(others.difference(setA, setB)))

    @staticmethod
    def fromDicts(dictA, dictB, others=set()):
        '''Create a confusion matrix for categorical data given 
        dicts mapping item identifiers to labels for the respective 
        annotators. A third set may contain additional items.'''
        labelCounts = [Counter(),Counter(),Counter(),0]
        for item in set.union(set(dictA.keys()),set(dictB.keys()),others):
            signature = (item in dictA, item in dictB)
            if signature==(True,False): # item from annotator A only
                labelCounts[0][dictA[item]] += 1
            elif signature==(False,True):   # item from annotator B only
                labelCounts[1][dictB[item]] += 1
            elif signature==(False,False):  # item from neither annotator
                labelCounts[3] += 1
            else:   # item from both annotators, possibly with different labels
                labelCounts[2][(dictA[item],dictB[item])] += 1
        return ConfusionMatrix(*tuple(labelCounts))
    
    @staticmethod
    def fromIterables(iterA, iterB):
        '''Create a confusion matrix for categorical data given 
        parallel iterables, assumed to be the same length, over corresponding
        labels from the respective annotators for each item. A value of 'None' 
        is taken to mean that no label was provided by that annotator.'''
        from itertools import izip
        labelCounts = [Counter(),Counter(),Counter(),0]
        for lblA,lblB in izip(iterA,iterB):
            signature = (lblA is not None, lblB is not None)
            if signature==(True,False): # item from annotator A only
                labelCounts[0][lblA] += 1
            elif signature==(False,True):   # item from annotator B only
                labelCounts[1][lblB] += 1
            elif signature==(False,False):  # item from neither annotator
                labelCounts[3] += 1
            else:   # item from both annotators, possibly with different labels
                labelCounts[2][(lblA,lblB)] += 1
        return ConfusionMatrix(*tuple(labelCounts))
        

def fcurves():
    from pylab import ogrid, divide, clabel, contour, plot
    X, Y = ogrid[0:1:.001,0:1:.001]    # range of R and P values, respectively. X is a row vector, Y is a column vector.
    F = divide(2*X*Y, X+Y)   # matrix s.t. F[P,R] = 2PR/(P+R)
    plot(X[...,0], X[...,0], color='#cccccc')   # P=R
    clabel(contour(X[...,0], Y[0,...], F, levels=[.5,.7,.9], colors='#aaaaaa', linewidths=2), fmt='%.1f', inline_spacing=1)  # show F score curves at values .5, .7, and .9
    
def prf_plot(prf_scores, *args, **kwargs):
    from pylab import scatter, xlim, ylim, xlabel, ylabel, show
    fcurves()
    recall_vals = [prf.R for prf in prf_scores]
    precision_vals = [prf.P for prf in prf_scores]
    scatter(recall_vals,precision_vals,**kwargs)
    xlim(0,1)
    xlabel('Recall')
    ylim(0,1)
    ylabel('Precision')
    show()

def prf_plot_multiseries(series1, *args):
    '''
    Plots precision vs. recall for multiple data series. Each argument of the form 
    ([<PRFScores1>, <PRFScores2>, ...], <fmt(s)>, <label(s)>)
    provides precision/recall data points in PRFScores instances, and optionally 
    labels and format specifier strings for the matplotlib plot() function.
    '''
    from pylab import plot, xlim, ylim, xlabel, ylabel, text, show
    fcurves()
    for series in (series1,)+args:
        oo = [o for o in series if hasattr(o,'__iter__')]
        osizes = {len(o) for o in oo}
        assert len(osizes)==1,'Unequal sizes of parallel arrays'
        series_size = list(osizes)[0]
        ss = [(o if o in oo else [o]*series_size) for o in series]
        recall_vals = [prf.R for prf in ss[0]]
        precision_vals = [prf.P for prf in ss[0]]
        for r,p,fmt,lbl,msize in zip(recall_vals, precision_vals, 
                               ss[1] if len(ss)>1 else [None]*series_size, 
                               ss[2] if len(ss)>2 else [None]*series_size,
                               ss[3] if len(ss)>3 else [None]*series_size):
            if fmt is not None:
                parts = fmt.split(' ')
                markfmt = parts[0]
                if markfmt!='':
                    if msize is not None:
                        plot(r,p,markfmt,markersize=msize)
                    else:
                        plot(r,p,markfmt)
                
                if lbl is not None:
                    opts = {}
                    if len(parts)>1:
                        if parts[1]!='':
                            opts['color'] = parts[1]
                        if len(parts)>2:
                            if parts[2]!='':
                                size = parts[2]
                                if size not in ('xx-small','x-small','small','medium','large','x-large','xx-large'):
                                    size = float(size)
                                opts['fontsize'] = size
                    text(r+.01,p-.02,lbl,**opts)
            else:
                if msize is not None:
                    plot(r,p,markersize=msize)
                else:
                    plot(r,p)
                
                if lbl is not None:
                    text(r,p,lbl)
            
    xlim(0,1)
    xlabel('Recall')
    ylim(0,1)
    ylabel('Precision')
    show()

def testplot():
    from pylab import rand, legend
    #prf_plot(rand(50),rand(50))
    prf_plot_multiseries(([PRFScores(314.2, 428.7, 555), PRFScores(262.0, 376.2, 555), PRFScores(152.5, 510.0, 555)],
                         ['go','b+','mv'], ['a','b','c'],))
    
    

if __name__=='__main__':
    testplot()
    import doctest
    doctest.testmod()
