#!/usr/bin/env python2.7
'''
Bidirectional mapping/set of alignments between two sequences. 
The mapping holds token offsets within the sequences.

@author: Nathan Schneider (nschneid)
@since: 2012-07-26
'''
from __future__ import print_function
import os, sys, re
from collections import defaultdict

class Alignment(object):
	'''
	Bidirectional mapping/set of alignments between two sequences. 
	The mapping holds token offsets within the sequences.
	
	>>> source = [0, 1, 2, 3, 4]
	>>> target = [0, 1, 2, 3]
	
	# one-to-one
	
	+ + + + source
	|  /  |
	- - - - target
	
	>>> a = Alignment()
	>>> a.link(0, 2)
	>>> a.link(1, 1)
	>>> a.aligned(3, 0)
	False
	>>> a.link(3, 0)
	>>> a.aligned(3, 0)
	True
	>>> a[0:]
	2
	>>> a[:0]
	3
	>>> print(a[4:])
	None
	>>> a[0:2]
	[(0, 2)]
	>>> a[0:0]
	[]
	>>> a[(0,1):(1,2)]
	[(0, 2), (1, 1)]
	>>> a[(1,2):0]
	[]
	>>> print(a[slice(Ellipsis,0)])
	[(3, 0)]
	>>> a.coversSource(source)
	False
	>>> a.coversTarget(target)
	False
	
	# one-to-many
	
	+ + + +
	|  /| |
	- - - -
	
	>>> b = Alignment('one2many', a[:])
	>>> b
	Alignment('one2many', [(0, 2), (1, 1), (3, 0)])
	>>> b.link(1, 3)
	>>> b[1:]
	set([1, 3])
	>>> b[:3]
	1
	>>> b[:]
	[(0, 2), (1, 1), (1, 3), (3, 0)]
	>>> b.coversSource(source)
	False
	>>> b.coversTarget(target)
	True
	
	# many-to-many
	
	+ + + +
	|\ /| |
	- - - -
	
	>>> c = Alignment('many2many', b[:])
	>>> c.link(4, 2)
	>>> c.link(2, 2)
	>>> c[(0,2,4):(0,1,2)]
	[(0, 2), (2, 2), (4, 2)]
	>>> c.coversSource(source)
	True
	>>> c.coversTarget(target)
	True
	>>> c.adjacencies(source, target)	# doctest:+NORMALIZE_WHITESPACE
	[[0, 0, 1, 0], 
	 [0, 1, 0, 1], 
	 [0, 0, 1, 0], 
	 [1, 0, 0, 0], 
	 [0, 0, 1, 0]]
	
	>>> d = Alignment('many2many', c[:])
	>>> d==b
	False
	>>> d.unlink(4, 2)
	>>> d.unlink(2, 2)
	>>> Alignment('one2many', d[:])==b
	True
	
	# try to add invalid links
	
	>>> a.link(4, 2)	# doctest:+ELLIPSIS
	Traceback (most recent call last):
	...
	ValueError: Illegal alignment: linking 4 to 2 would violate one2one structure
	>>> b.link(4, 2)
	Traceback (most recent call last):
	...
	ValueError: Illegal alignment: linking 4 to 2 would violate one2many structure
	>>> c.link(1, 1)
	Traceback (most recent call last):
	...
	ValueError: Alignment from 1 to 1 already exists
	>>> c.unlink(0, 0)
	Traceback (most recent call last):
	...
	ValueError: No alignment from 0 to 0 exists, so cannot remove it
	'''
	def __init__(self, form='one2one' or 'one2many' or 'many2one' or 'many2many', pairs=None):
		sform, tform = form.split('2')
		assert sform in ['one', 'many']
		assert tform in ['one', 'many']
		self._form = form
		
		# forward mapping (source -> target)
		if tform=='many':
			self._fwd = defaultdict(set)
		else:
			self._fwd = {}
			
		# backward mapping (target -> source)
		if sform=='many':
			self._bwd = defaultdict(set)
		else:
			self._bwd = {}
			
		if pairs is not None:
			for s,t in pairs:
				self.link(s,t)
			
	def link(self, s, t):
		'''
		Add an alignment link from source index s to target index t 
		(will not modify any existing alignments)
		'''
		if self.aligned(s, t):
			raise ValueError('Alignment from {} to {} already exists'.format(s, t))
		
		sform, tform = self._form.split('2')
		if sform=='many':
			self._bwd[t].add(s)
		else:
			if t in self._bwd:
				raise ValueError('Illegal alignment: linking {} to {} would violate {} structure'.format(s, t, self._form))
			self._bwd[t] = s
		
		if tform=='many':
			self._fwd[s].add(t)
		else:
			if s in self._fwd:
				raise ValueError('Illegal alignment: linking {} to {} would violate {} structure'.format(s, t, self._form))
			self._fwd[s] = t
		
		
	def unlink(self, s, t):
		'''
		Removes an existing alignment link from source index s to target index t
		'''
		if not self.aligned(s, t):
			raise ValueError('No alignment from {} to {} exists, so cannot remove it'.format(s, t))
		
		sform, tform = self._form.split('2')
		if sform=='many':
			self._bwd[t].remove(s)
		else:
			del self._bwd[t]
		
		if tform=='many':
			self._fwd[s].remove(t)
		else:
			del self._fwd[s]
	
	def fwd(self, s):
		'''Forward lookup: retrieve target index(es) corresponding to the given source index.
		If one-to-one or many-to-one and the source index is unaligned, returns None.'''
		return self._fwd.get(s, set() if self._form.split('2')[1]=='many' else None)
		
	def bwd(self, t):
		'''Backward lookup: retrieve source index(es) corresponding to the given target index.
		If one-to-one or one-to-many and the target index is unaligned, returns None.'''
		return self._bwd.get(t, set() if self._form.split('2')[0]=='many' else None)
	
	def coversSource(self, sourceIndices):
		return all(self.fwd(s) not in [None,set()] for s in sourceIndices)
		
	def coversTarget(self, targetIndices):
		return all(self.bwd(t) not in [None,set()] for t in targetIndices)
		
	def aligned(self, s, t):
		'''Given a source index and a target index, determines whether an 
		alignment link connects them.'''
		if self._form.split('2')[1]=='many':
			return t in self.fwd(s)
		return self.fwd(s)==t
		
	def adjacencies(self, sourceIndices, targetIndices):
		'''Adjacency matrix.'''
		return [[int(self.aligned(s,t)) for t in targetIndices] for s in sourceIndices]
		
	def __getitem__(self, key):
		assert isinstance(key, slice)
		assert key.step is None
		src, tgt = key.start, key.stop
		if src is not None and tgt is not None:
			# list all existing alignments between the given source index(es) 
			# and the given target index(es)
			if src is Ellipsis:
				src = self._fwd.keys()
			elif not hasattr(src, '__iter__'):
				src = {src}
			
			if tgt is Ellipsis:
				tgt = self._bwd.keys()
			elif not hasattr(tgt, '__iter__'):
				tgt = {tgt}

			return [(s,t) for s in src for t in tgt if self.aligned(s,t)]
		elif (src is None or src is Ellipsis) and (tgt is None or tgt is Ellipsis):
			# list all alignment pairs
			if self._form.split('2')[1]=='many':
				return [(s,t) for s in self._fwd for t in self._fwd.get(s,[])]
			return self._fwd.items()
		elif tgt is None:	# forward lookup
			if hasattr(src, '__iter__'):
				return [self.fwd(s) for s in src]
			else:
				return self.fwd(src)
		else:	# backward lookup
			if hasattr(tgt, '__iter__'):
				return [self.bwd(t) for t in tgt]
			else:
				return self.bwd(tgt)
	
	def __eq__(self, that):
		return self.__dict__==that.__dict__
	
	def __repr__(self):
		return 'Alignment(%s, %s)' % (repr(self._form), repr(self[:]))


class TrackingString(object):
	'''
	A mutable string that tracks the original, i.e. it holds a set of 
	offset mappings that are updated with each modification. Also allows 
	indexing by substring.
	
	>>> s = TrackingString("I'll eat myself if you can find / A smarter hat than me.", minimize_edits=False)
	>>> s[0] = 'i'
	>>> print(s)
	i'll eat myself if you can find / A smarter hat than me.
	>>> print(s._align[:0])
	set([0])
	>>> s["'ll":"'ll"] = ' '
	>>> print(s)
	i 'll eat myself if you can find / A smarter hat than me.
	>>> print(s._align[:1])
	set([])
	>>> print(s._align[:2])
	set([1])
	>>> print(s._align[:3])
	set([2])
	>>> s[-1] = ' .'
	>>> print(s)
	i 'll eat myself if you can find / A smarter hat than me .
	>>> [s._align[:i] for i in range(s.index(' / ')-1,s.index('smarter'))]
	[set([30]), set([31]), set([32]), set([33]), set([34]), set([35])]
	>>> s[' / '] = '\\n'
	>>> print(s)
	i 'll eat myself if you can find
	A smarter hat than me .
	>>> s['A'] = 'a'
	>>> print(s)
	i 'll eat myself if you can find
	a smarter hat than me .
	>>> [s._align[:i] for i in range(5)]==[{0}, set(), {1}, {2}, {3}]
	True
	>>> [s._align[:i] for i in range(s.index('\\n')-1,s.index('smarter'))]==[{30}, {31, 32, 33}, {34}, {35}]
	True
	>>> [s._align[:i] for i in range(-3+len(s),len(s))]
	[set([54]), set([55]), set([55])]
	
	
	>>> s = TrackingString("I'll eat myself if you can find / A smarter hat than me.", minimize_edits='nonword')
	>>> s["'ll":"'ll"] = ' '
	>>> s[-1] = ' .'
	>>> print(s)
	I 'll eat myself if you can find / A smarter hat than me .
	>>> [s._align[:i] for i in range(-3+len(s),len(s))]
	[set([54]), set([]), set([55])]
	>>> s[-2:] = ' !'
	>>> [s._align[:i] for i in range(-3+len(s),len(s))]
	[set([54]), set([]), set([55])]
	>>> s.prepend('** ')
	>>> s.append('!')
	>>> print(s)
	** I 'll eat myself if you can find / A smarter hat than me !!
	>>> del s[0]
	>>> del s[2:-2]
	>>> del s[' !']
	>>> print(s)
	*!
	'''
	def __init__(self, orig, minimize_edits='whitespace' or 'nonword' or 'all' or False, alignment=None):
		self._orig = orig	# original string
		self._s = orig		# current string
		self._align = alignment or Alignment(form='many2many', pairs=zip(range(len(orig)), range(len(orig))))
		self._minimize = minimize_edits
		assert self._minimize in {'whitespace', 'nonword', 'all', False}
		#TODO: option to maintain list of edits?
		
	def __getitem__(self, k):
		if isinstance(k, basestring):
			i = self.index(k)
			k = slice(i, i+len(k))
		elif isinstance(k, slice):
			if isinstance(k.start, basestring):
				k = slice(self.index(k.start), k.stop, k.step)
			if isinstance(k.stop, basestring):
				k = slice(k.start, self.index(k.stop), k.step)
		return self._s[k]
		
	def __setitem__(self, k, v):
		'''
		Modify the string and update alignments to the original string accordingly.
		Edit minimization options from the constructor are honored. 
		Note that if a multi-character string is replaced with a multi-character 
		string, the alignments will be many-to-many, even if the replaced and 
		replacement strings are of the same length.
		'''
		if isinstance(k, basestring):
			i = self.index(k)
			k = slice(i, i+len(k))
		elif isinstance(k, slice):
			if isinstance(k.start, basestring):
				k = slice(self.index(k.start), k.stop, k.step)
			if isinstance(k.stop, basestring):
				k = slice(k.start, self.index(k.stop), k.step)
			assert k.step in [None,1]
		elif isinstance(k, int):
			if k<0: k += len(self)
			k = slice(k, k+1)
		
		# k is a slice; step can be ignored
		k = slice(k.start or 0, k.stop if k.stop is not None else len(self))
		
		# normalize indices to be positive
		if k.start<0:
			k = slice(k.start+len(self), k.stop)
		if k.stop<0:
			k = slice(k.start, k.stop+len(self))
		
		u = self[k]
		assert not len(u)==len(v)==0
		
		# TODO: minimize changes if applicable--i.e. if beginning or end of replacement 
		# matches the substring it replaces, don't consider that part as changing
		if self._minimize:
			REGEXES = {'all': r'.*', 'whitespace': r'\s*', 'nonword': r'\W*'}
			
			# longest common prefix
			lcpLen = 0
			for a,b in zip(u,v):
				if a!=b: break
				lcpLen += 1
			if lcpLen>0:
				lcp = u[:lcpLen]
				m = re.search('^'+REGEXES[self._minimize], lcp, re.U)
				k = slice(k.start+m.end(), k.stop)
				u = self[k]
				v = v[m.end():]
				#print('minimizing prefix',repr(lcp),k,repr(u),repr(v))
			
			# longest common suffix
			lcsLen = 0
			for a,b in zip(u[::-1],v[::-1]):
				if a!=b: break
				lcsLen += 1
			if lcsLen>0:
				lcs = u[-lcsLen:]
				m = re.search(REGEXES[self._minimize]+'$', lcs, re.U)
				k = slice(k.start, k.stop-lcsLen+m.start())
				u = self[k]
				v = v[:len(v)-lcsLen+m.start()]
				#print('minimizing suffix',repr(lcs),m.start(),k,repr(u),repr(v))
		
		lu = len(u)
		lv = len(v)
		if lu==lv==0:
			return
		
		#print(k,repr(u),lu,repr(v),lv)
		
		if lu==0:	# insertion
			edit_type = 'INS'
		elif lv==0:	# deletion
			edit_type = 'DEL'
		else:	# substitution
			edit_type = 'SUB'
		
		uu = tuple(range(k.start, k.stop))
		ww = set(i for ii in self._align[:uu] for i in ii)
		
		if lu!=lv:
			# slide (shift) offsets following the replaced substring to adjust for the 
			# shorter or longer replacement
			slide_delta = lv-lu
			slide_start = k.stop
			
			xx = self._align[slice(Ellipsis, tuple(range(slide_start, len(self))))]
			for i,j in xx:
				self._align.unlink(i,j)
			for i,j in xx:
				self._align.link(i,j+slide_delta)
				
			#print('slide',slide_delta,'from',slide_start,xx)
			#print(self._align[slice(Ellipsis, tuple(range(k.start, len(self))))])
		
		# no internal structure is maintained--i.e. if both replaced and replacement 
		# have multiple characters, there will be a many-to-many alignment, regardless
		# of previous alignments for the replaced substring
		
		for i,j in self._align[tuple(ww):uu]:
			self._align.unlink(i,j)
		for j in range(k.start,k.start+lv):
			for i in ww:
				self._align.link(i,j)
		
		self._s = self._s[:k.start] + v + self._s[k.stop:]
	
	def __delitem__(self, k):
		self[k] = ''
	
	def index(self, t):
		'''Returns the start index of the first occurrence of substring t.'''
		return self._s.index(t)
	
	def __len__(self):
		return len(self._s)
	
	def __str__(self):
		return self._s
	
	def __repr__(self):
		return 'TrackingString<'+repr(self._s)+'>'
	
	def append(self, t):
		self[len(self):] = t
		
	def prepend(self, t):
		self[:0] = t
		
	def clone(self):
		return TrackingString(self._orig, self._minimize, Alignment(self._align._form, self._align[:]))
		
	# TODO: methods for string and regex replacement

if __name__=='__main__':
	import doctest
	doctest.testmod()
