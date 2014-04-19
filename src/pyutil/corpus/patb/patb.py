#coding=UTF-8
'''
General utilities for working with the Penn Arabic Treebank.

Buckwalter/Unicode conversion map from Andrew Roberts'
http://www.andy-roberts.net/software/buckwalter2unicode/buckwalter2unicode.py

@author: Nathan Schneider (nschneid)
@since: 2010-08-08
'''

# Strive towards Python 3 compatibility
from __future__ import print_function, unicode_literals, division, absolute_import
from future_builtins import map, filter

import re

# Declare a dictionary with Buckwalter's ASCII symbols as the keys, and
# their unicode equivalents as values.
buck2uni = {"'": u"\u0621", # hamza-on-the-line
            "|": u"\u0622", # madda
            ">": u"\u0623", # hamza-on-'alif
            "&": u"\u0624", # hamza-on-waaw
            "<": u"\u0625", # hamza-under-'alif
            "}": u"\u0626", # hamza-on-yaa'
            "A": u"\u0627", # bare 'alif
            "b": u"\u0628", # baa'
            "p": u"\u0629", # taa' marbuuTa
            "t": u"\u062A", # taa'
            "v": u"\u062B", # thaa'
            "j": u"\u062C", # jiim
            "H": u"\u062D", # Haa'
            "x": u"\u062E", # khaa'
            "d": u"\u062F", # daal
            "*": u"\u0630", # dhaal
            "r": u"\u0631", # raa'
            "z": u"\u0632", # zaay
            "s": u"\u0633", # siin
            "$": u"\u0634", # shiin
            "S": u"\u0635", # Saad
            "D": u"\u0636", # Daad
            "T": u"\u0637", # Taa'
            "Z": u"\u0638", # Zaa' (DHaa')
            "E": u"\u0639", # cayn
            "g": u"\u063A", # ghayn
            "_": u"\u0640", # taTwiil
            "f": u"\u0641", # faa'
            "q": u"\u0642", # qaaf
            "k": u"\u0643", # kaaf
            "l": u"\u0644", # laam
            "m": u"\u0645", # miim
            "n": u"\u0646", # nuun
            "h": u"\u0647", # haa'
            "w": u"\u0648", # waaw
            "Y": u"\u0649", # 'alif maqSuura
            "y": u"\u064A", # yaa'
            "F": u"\u064B", # fatHatayn
            "N": u"\u064C", # Dammatayn
            "K": u"\u064D", # kasratayn
            "a": u"\u064E", # fatHa
            "u": u"\u064F", # Damma
            "i": u"\u0650", # kasra
            "~": u"\u0651", # shaddah
            "o": u"\u0652", # sukuun
            "`": u"\u0670", # dagger 'alif
            "{": u"\u0671", # waSla
}

# For a reverse transliteration (Unicode -> Buckwalter)
uni2buck = {v: k for k, v in buck2uni.items()}

def romanize(s):
    '''
    >>> print(romanize('كَاتِبٌ'))
    kaAtibN
    >>> print(romanize('قَابُوس'))
    qaAbuws
    '''
    t = ''
    for c in s:
        if c in uni2buck:
            t += uni2buck[c]
        else:
            t += c
    return t

def arabize(s):
    '''
    >>> print(arabize('kaAtibN'))
    كَاتِبٌ
    >>> print(arabize('qaAbuws'))
    قَابُوس
    '''
    t = ''
    for c in s:
        if c in buck2uni:
            t += buck2uni[c]
        else:
            t += c
    return t
    
def buck2diac(buckAnalysis, keepSegmentation=True):
    '''
    Given a segmented and tagged Arabic word from the Buckwalter analyzer, perform
    orthographic normalization to produce the surface diacritized form 
    (also in the Buckwalter encoding) with morphemes separated by '+' (unless 
    'keepSegmentation' is false, in which case the unsegmented form will be returned).
    
    Normalization involves contracting li+Al+ to li+, adding Shadda (gemination mark) to 
    sun letters after Al+, changing Alef Wasla to Alef, etc.
    '''
    
    assert ' ' not in buckAnalysis,buckAnalysis
    
    # Strip tags from morphemes
    vs = '+'.join(filter(lambda m: m!='(null)', (x.split('/')[0] for x in buckAnalysis.split('+') if x!='')))
    
    # Normalization
    ''' In MADA-3.1/MADA/ALMOR3.pm:
    #update Sept 11 2008: to match up behavior of sun/moon letters in BAMA2
    # code segemnt from BAMA-2
    my $voc_str = $$prefix_value{"diac"}."+".$$stem_value{"diac"}."+".$$suffix_value{"diac"};
    $voc_str =~ s/^((wa|fa)?(bi|ka)?Al)\+([tvd\*rzs\$SDTZln])/$1$4~/; # not moon letters
    $voc_str =~ s/^((wa|fa)?lil)\+([tvd\*rzs\$SDTZln])/$1$3~/; # not moon letters
    $voc_str =~ s/A\+a([pt])/A$1/; # e.g.: Al+HayA+ap
    $voc_str =~ s/\{/A/g; 
    $voc_str =~ s/\+//g; 
    '''
        
    # TODO: this is ad hoc for now--not sure if it's precisely how SAMA/MADA do it.
    vs = re.sub(r'^li[+]Al[+]', 'li+l+', vs)    # contraction
    vs = re.sub('^([+]?)min[+]m(A|an)$', r'\1mi+m~\2', vs)   # contraction, e.g. +min/PREP+mA/REL_PRON+ -> mim~A
    vs = re.sub(r'Y[+]', 'y+', vs)  # Y -> y / _+
    vs = re.sub(r'y[+]F', 'Y+F', vs) # e.g. bi/PREP+maEonY/NOUN+F/CASE_INDEF_GEN -> bimaEonYF
    vs = re.sub(r'y[+]([tn])', r'yo+\1', vs)   # 'o' epenthesis? e.g. wa/CONJ+>aroDay/PV+tu/PVSUFF_SUBJ:1S+hA/PVSUFF_DO:3FS -> wa>aroDayotuhA
    vs = re.sub(r't[+]h', 'to+h', vs)   # 'o' deletion? e.g. +jaEal/PV+at/PVSUFF_SUBJ:3FS+hu/PVSUFF_DO:3MS -> jaEalatohu
    vs = re.sub(r'^Einod[+]a[+]mA$', 'EindamA', vs) # +Einod/NOUN+a/CASE_DEF_ACC+mA/SUB_CONJ+
    vs = re.sub(r'^li[+][*]alika$', 'li*`lika', vs) # +li/PREP+*alika/DEM_PRON_MS+
    # end ad hoc
    
    # add Shadda to sun letters after Al+
    # NOTE: needed to add [+]? in some places after clitics
    vs = re.sub(r'^((wa[+]?|fa[+]?)?(bi[+]?|ka[+]?)?Al)\+([tvd\*rzs\$SDTZln])', r'\1\4~', vs, 1) # not moon letters
    vs = re.sub(r'^((wa[+]?|fa[+]?)?li[+]?l)\+([tvd\*rzs\$SDTZln])', r'\1\3~', vs, 1); # not moon letters
    
    # simplify Aa before p or t, e.g.: Al+HayA+ap
    vs = re.sub(r'A\+a([pt])', r'A\1', vs, 1)
    
    # convert Alef Wasla to plain Alef (http://en.wikipedia.org/wiki/Arabic_diacritics#.CA.BCAlif_wa.E1.B9.A3la)
    vs = vs.replace('{', 'A')
    
    # use the desired segmentation character
    if not keepSegmentation:
        vs = vs.replace('+', '')
    
    return vs

def vowels(romanized=True):
    '''Based on Chris Dyer's implementation in umd.clip.arabic.buck.Analyzer.'''
    BW_VOWELS = {'a','i','o','u','F','N','K','~','_','`'}
    if romanized:
        return BW_VOWELS
    return {buck2uni[v] for v in BW_VOWELS}

def devowel(s, romanized=True, removeAlefHamza=True):
    '''
    Removes the vowels from a Buckwalter-encoded or Arabic string.
    
    >>> print(devowel('maso&uwl+uwna'))
    ms&wl+wn
    >>> print(devowel('مسؤولون'))
    مسؤولون
    >>> print(devowel('مسؤولون', romanized=False))
    مسؤولون
    >>> print(devowel('كَاتِبٌ', romanized=False))
    كاتب
    '''
    t = re.sub('('+'|'.join(re.escape(c) for c in vowels(romanized=romanized))+')', '', s)
    if removeAlefHamza:
        t = re.sub(r'[\{>]', 'A', t) if romanized else re.sub('[‎أ‎ٱ]', '‎ا', t)
    return t

def test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    test()
