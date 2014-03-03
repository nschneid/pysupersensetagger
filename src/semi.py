#coding=utf-8
'''
Semi-Markov dynamic programming for gappy segmentation.

@since: 2014-03-01
@author: Nathan Schneider (nschneid@cs.cmu.edu)
'''
from __future__ import print_function, division
import sys
from pyutil.memoize import memoize

inf = float('inf')


class Mul(object):
    '''Lazy multiplication so more than 2 operands are handled together'''
    def __init__(self, *operands):
        self._operands = list(operands)
    def __repr__(self):
        return 'Mul(' + repr(self._operands) + ')'
    def __mul__(self, that):
        if isinstance(that,Mul):
            self._operands.extend(that._operands)
        else:
            self._operands.append(that)
        return self
    __pow__ = __mul__
    def _times_operands(self, times):
        return times(self._operands)
    
    
class SemiringValue(object):
    def __init__(self, v, plus, times):
        self.v = v
        self.plus = plus
        self.times = times
    def __add__(self, that):
        #print(that._operands, file=sys.stderr)
        you = that._times_operands(self.times) if isinstance(that,Mul) else that
        return self.plus([self.v, you.v])
    def __mul__(self, that):
        #return self.times([self.v, that.v])
        return Mul(self, that)
    def __iter__(self):
        return iter(self.v)

 
class MaxPlus(SemiringValue):
    '''Value in the max-plus semiring pairing a real weight with an index.'''
    zero = (-inf,None,None,None)
    one = (0.0,None,[],[])
    def __init__(self, v=one):
        super(MaxPlus,self).__init__(v, plus=MaxPlus.plus, times=MaxPlus.times)
    def __pow__(self, that): # override. __pow__ has higher operator precedence than __mul__
        if isinstance(that,Mul):
            return Mul(self,*that._operands)
        return Mul(self, that)
    def __mul__(self, that):
        return self.times2(self, that._times_operands(self.times) if isinstance(that,Mul) else that)
    
    @classmethod
    def plus(cls, operands):
        ##'''The plus operator returns the max of the operands, _omitting the first index in the last operand_'''
        #print('+', operands, file=sys.stderr)
        ##weights, indices = zip(*operands)
        ##indices = list(indices)
        ##indices[-1] = indices[-1][1:]
        ##return MaxPlus(max(zip(weights, indices)))
        return MaxPlus(max(operands))
    
    @classmethod
    def times(cls, operands):
        '''The times operator returns the sum of the real weights and the concatenation of the indices'''
        #print('**', operands, file=sys.stderr)
        weights, bps, indices, tags = zip(*operands)
        if len(operands)==4:
            pregap, _, postgap = indices[0:3]
            indices = [indices[0], '~<', indices[1], '>~', indices[2]]
        elif len(operands)==3:
            pregap, _, postgap = indices
            indices = [pregap, '_<', indices[1], '>_', postgap]
        elif len(operands)==2 and False:
            indices = [[indices[0]], indices[1]]
        else:
            indices = sum(indices,[])
        #print(indices, file=sys.stderr)
        return MaxPlus((sum(weights), operands[0], filter(lambda x: x!=(), indices), sum(tags, [])))
    
    @classmethod
    def times2(cls, left, right):
        '''Alternate times operator (always binary): returns the sum of the real weights and the indices from the second operand only.
           (Will be used where the indices on the LHS provide a backpointer but are not incorporated in the new chart item.)'''
        #print('*', list(left), list(right), file=sys.stderr)
        weights, bps, indices, tags = zip(left, right)
        right_indices = indices[1]
        return MaxPlus((sum(weights), left, filter(lambda x: x!=(), right_indices), tags[1]))

def o_singleton_score(i): return MaxPlus((3,None,[(i-1,i,'S')],[(i-1,'O')]))
# word that is not part of any MWE and is not inside a gap

def g_singleton_score(i): return MaxPlus((1,None,[(i-1,i,'s')],[(i-1,'o')]))
# word that is not part of any MWE but is inside a gap

def w_singleton_score(i): return MaxPlus((1,None,[(i-1,i,'W')],[(i-1,'Ĩ')]))
# word that is part of a weak MWE but not a strong MWE

def strong_contig_score(h,i): return MaxPlus((1,None,[(h,i,'_')],[(h,'B')]+[(k,'Ī') for k in range(h+1,i)]))

def weak_contig_score(h,i): return MaxPlus((-0.25,None,[(h,i,'~')],[])) # tags handled by ngpath() callees

def weak_gappy_score(j,k,l,i): return MaxPlus((-0.2,None,[((j,k),'~',(l,i))],[])) # tags handled by ngpath() and gap() callees

def strong_gappy_score(j,k,l,i): return MaxPlus((-1.1,None,[((j,k),'_',(l,i))],[(j,'B')]+[(h,'Ī') for h in range(j+1,k)]+[(h,'Ī') for h in range(l,i)])) # tags not including gap contents

def weak_with_gappy_strong_score(j,k,l,i): return MaxPlus((0.1,None,[((j,k),'',(l,i))],[])) # tags handled by strong_gappy_score()


ZERO = MaxPlus(MaxPlus.zero)
ONE = MaxPlus(MaxPlus.one)
N = 6
path = [MaxPlus((float('nan'), None,None)) for i in range(N+1)] # the chart
def build_chart(N, path):
    # N.B.: 'v' is the best value seen so far for the chart entry being computed
    
    def ngpath(j,i, first_tag='B'):
        # sequence of contiguous strong or singleton expressions
        # forming a contiguous part of a weak MWE
        v = _ngpath(j,i)
        w, bp, indices, tags = v
        if tags[0][1]!=first_tag: # we're post-gap
            tags = [(tags[0][0],first_tag)] + tags[1:]
        assert tags and 'O' not in zip(*tags)[1],(indices,tags)
        return MaxPlus((w, bp, indices, tags))

    @memoize
    def _ngpath(j,i):
        if j==i: return ONE
        assert j<i
        v = ZERO + _ngpath(j,i-1) ** w_singleton_score(i)
        for h in range(j, i-1):
            v += _ngpath(j,h) ** strong_contig_score(h,i)
        return v
    
    @memoize
    def gap(j,i):
        if j==i: return ONE
        assert j<i
        v = ZERO + gap(j,i-1) ** g_singleton_score(i)
        for h in range(j, i-1):
            v += gap(j,h) ** strong_contig_score(h,i)
            v += gap(j,h) ** ngpath(h,i) ** weak_contig_score(h,i)
        
        w, bp, indices, tags = v
        tags = [(k,t.lower().replace('Ī','ī').replace('Ĩ','ĩ')) for k,t in tags]
        v = MaxPlus((w, bp, indices, tags))
        return v
    
    path[0] = MaxPlus()
    for i in range(1, N+1):
        #print(path[i-1].v, sys.stderr)
        v = ZERO + path[i-1] * o_singleton_score(i)
        for j in range(0, i-1):
            v += path[j] * ngpath(j,i) ** weak_contig_score(j,i)
            for k in range(j+1, i-1):
                for l in range(k+1, i):
                    # consider gappy expression going from j to i, where k to l is the gap: j < k < l < i
                    assert j < k < l < i,(i,j,k,l)
                    v += path[j] * ngpath(j,k) ** gap(k,l) ** ngpath(l,i,first_tag='Ĩ') ** weak_gappy_score(j,k,l,i)
                    v += path[j] * gap(k,l) ** strong_gappy_score(j,k,l,i) ** weak_with_gappy_strong_score(j,k,l,i)
        path[i] = v
 
build_chart(N, path)

#print(path[-1].v)

from pprint import pprint

# now decode
segmentation = []
alltags = []
w, bp, indices, tags = path[-1].v
alltags.extend(tags)
print(w)
while True:
    segmentation.insert(0, indices)
    print(bp.v, file=sys.stderr)
    w, bp, indices, tags = bp.v
    alltags.extend(tags)
    if not bp: break
pprint(segmentation)
pprint(sorted(alltags))
assert len(alltags)==N
print('\n'.join(zip(*sorted(alltags))[1]))
