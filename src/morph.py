'''
Created on Jul 24, 2012

@author: Nathan Schneider (nschneid)
'''
from __future__ import print_function
import sys, os, gzip

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

from pyutil.memoize import memoize

morphMap = {} # pos -> {word -> stem}

_options = {'useOldDataFormat': False,
            'useMorphCache': True}

def loadDefaults():
    # TODO: properties allowing for override of _options defaults
    pass

@memoize
def stem(w, p):
    '''
    Given a word and PTB part-of-speech tag, returns the lowercased 
    lemma using WordNet. If not found in WordNet, returns the lowercased 
    word.
    '''
    w = w.lower()
    if p is not None and p.startswith('NNP'): return w
    
    # irregular past tense verbs disambiguated by the fine-grained POS
    if w=='fell' and p=='VBD': return 'fall'
    elif w=='found' and p in {'VBD','VBN'}: return 'find'
    elif w=='lay' and p=='VBD': return 'lie'
    elif w=='saw' and p=='VBD': return 'see'
    elif w=='people' and p=='NNS': return 'person'
    
    if w=='cannot' or "'" in w:
        tt = word_tokenize(w+' .')[:-1] # period ensures no part of the word is interpreted as sentence-final punctuation
    else:
        tt = [w]
    lem = wn.morphy(tt[0], p and {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}.get(p[0]))
    if lem:
        tt[0] = lem
    return tt[0]

"""
def stem(word, pos):
    if _options['useMorphCache']:
        return getStemCache(word, pos)
    else:
        return getStemWN(word, pos)
"""
 
def getStemCache(word, pos):
    if not morphMap:
        if _options['useOldDataFormat']:
            loadMorphDataOriginalFormat()
        else:
            loadMorphDataNewFormat()
    
    return morphMap.get(pos, {word.lower(): word}).get(word.lower(), word.lower())
    # TODO: from the Java implementation it looks like the original casing is retained 
    # if the pos map was not found. is that the intended behavior?

def loadMorphDataOriginalFormat():
    def addMorph(w, pos, stem):
        morphMap.setdefault(pos, {})[w] = stem
        # TODO: intern pos, w, stem?
    
    print('loading morphology information (old format)...', file=sys.stderr)
    assert False
    global morphMap
    morphMap = {}
    with gzip.open(_options.get('morphFile',os.path.dirname(os.path.abspath(__file__))+'/../data/oldgaz/MORPH_CACHE.gz')) as inF:
        for ln in inF:
            pos, w, stemS = ln[:-1].split('\t')
            addMorph(w, pos, stemS)
            addMorph(w, 'UNKNOWN', stemS)
    addMorph('men', 'NNS', 'man')
    

def loadMorphDataNewFormat():
    # TODO: not sure why this is needed...the only different from loadMorphDataOriginalFormat() is the default path...
    def addMorph(w, pos, stem):
        morphMap.setdefault(pos, {})[w] = stem
        # TODO: intern pos, w, stem?
    
    print('loading morphology information (new format)...', file=sys.stderr)
    
    global morphMap
    morphMap = {}
    with gzip.open(_options.get('morphFile',os.path.dirname(os.path.abspath(__file__))+'/../data/morph/MORPH_CACHE.gz')) as inF:
        for ln in inF:
            pos, w, stemS = ln[:-1].split('\t')
            addMorph(w, pos, stemS)
            addMorph(w, 'UNKNOWN', stemS)
    addMorph('men', 'NNS', 'man')
    

def getStemWN(w, pos):
    raise NotImplemented()

# TODO
    
'''
    public String getStemWN(String word, String pos){
        if(!(pos.startsWith("N") || pos.startsWith("V") || pos.startsWith("J") || pos.startsWith("R"))
                || pos.startsWith("NNP"))
        {
            return word.toLowerCase();
        }
                
        String res = word.toLowerCase();
        
        if(res.equals("is") || res.equals("are") || res.equals("were") || res.equals("was")){
            res = "be";
        }else{
            try{
                //Iterator<String> iter = Dictionary.getInstance().getMorphologicalProcessor().lookupAllBaseForms(POS.VERB, res).iterator();
                
                IndexWord iw;
                if(pos.startsWith("V")) iw = Dictionary.getInstance().getMorphologicalProcessor().lookupBaseForm(POS.VERB, res);
                else if(pos.startsWith("N")) iw = Dictionary.getInstance().getMorphologicalProcessor().lookupBaseForm(POS.NOUN, res);
                else if(pos.startsWith("J")) iw = Dictionary.getInstance().getMorphologicalProcessor().lookupBaseForm(POS.ADJECTIVE, res);
                else iw = Dictionary.getInstance().getMorphologicalProcessor().lookupBaseForm(POS.ADVERB, res);
                    
                if(iw == null) return res;
                res = iw.getLemma();
            }catch(NullPointerException e){
                e.printStackTrace();
                System.exit(0);
            }catch(Exception e){
                e.printStackTrace();
            }
        }        
        
        return res;
    }
'''
    
