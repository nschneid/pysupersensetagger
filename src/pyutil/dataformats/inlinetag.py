#coding=UTF-8
'''
Handles SLAM (SLAsh Markup), an inline format for annotating (single or multi-word)
tokens with tags. E.g.

We/{PRP 1pl} the/DT people/NNS of/IN the/DT {United States}/NNP 

is equivalent to

We        PRP 1pl
the       DT
people    NNS
of        IN
the       DT
United States    NNP

See http://www.ark.cs.cmu.edu/ARKwiki/index.php/Inline_tag_markup_format for the spec.



TODO: make into a command-line utility
slam --detab inputfile1.tab [inputfile2.tab...] > outputfile.slam
  calls tabbed2inline()
slam --entab inputfile1.slam [inputfile2.slam...] > outputfile.tab
  calls inline2tabbed()
and possibly
slam --raw inputfile1.txt [inputfile2.txt...] > outputfile.slam
  which simply escapes tokens of raw text

slam inputfile1.slam [inputfile2.slam...]
  verifies that the files are valid SLAM format, and counts 
  the number of tokens ({singleton, multiword} x {untagged, tagged}), and unique tags

@author: Nathan Schneider
@since: 2011-06-25
'''

import sys, re, fileinput
from collections import Counter

def shorten(s, fromLeft=True):
    if len(s)<=50:
        return s
    elif fromLeft:
        return s[:50]+' ...'
    else:
        return '...' + s[-50:]


def escapeItem(tkn, tag):
    '''
    >>> escapeItem('Angelina Jolie', 'PER')
    '{Angelina Jolie}/PER'
    >>> escapeItem('http://www.google.com/sports', None)
    '{http://www.google.com/sports}/'
    >>> escapeItem('http://www.google.com/', 'URL')
    '{http://www.google.com/}/URL'
    >>> escapeItem('boy/girl', 'girl/boy')
    '{boy/girl}/{girl/boy}'
    >>> escapeItem('anti-missile missile', 'anti-{missile} missile')
    '{anti-missile missile}/{anti-{missile} missile}'
    >>> escapeItem('/', 'PUNC')
    '//PUNC'
    >>> escapeItem('}', 'PUNC')
    '{}}/PUNC'
    >>> escapeItem('', '')
    '{}/{}'
    >>> escapeItem('', None)
    '{}/'
    >>> escapeItem('q', None)
    'q'
    >>> escapeItem('', 'q')
    '{}/q'
    >>> escapeItem('foo', '/')
    'foo//'
    >>> escapeItem('', '/')
    '{}//'
    >>> escapeItem('hi there', '//')
    '{hi there}///'
    >>> escapeItem('}/', 'Bad News Bears')
    Traceback (most recent call last):
      ...
    AssertionError
    '''
    k, g = tkn, tag
    def e(itm):
        return '{'+itm+'}'
    if '{' in k or '}' in k or '/' in k or k=='' or re.search('\s',k):
        if k!='/':
            k = e(k)
    assert '/{' not in k
    assert '}/' not in k
 
    if tag is not None:
        if '{' in g or '}' in g or '/' in g or g=='' or re.search('\s',g):
            if not re.match(r'^/+$', g):
                g = e(g)
        assert '/{' not in g,g
        assert '}/' not in g,g
        return k+'/'+g  # there is a tag
    elif tkn!=k:
        return k+'/'    # token is protected
    else:
        return k    # unprotected token, no tag

def bio2slam(items):
    '''
    >>> sentence = [('Mr.', 'B-PER'), ('Rogers', 'I-PER'), ('is', 'O'), ('from', 'O'), ('Squirrel', 'B-LOC'), ('Hill', 'I-LOC'), ('in', 'O'), ('Pittsburgh', 'B-LOC')]
    >>> bio2slam(sentence)
    '{Mr. Rogers}/PER is from {Squirrel Hill}/LOC in Pittsburgh/LOC'
    '''
    def stripBIO(tags):
        if len(tags)==1:
            if tags[0]=='O': return None
            assert tags[0][:2]=='B-'
            return tags[0][2:]
        assert tags[0][:2]=='B-'
        for i in range(1,len(tags)):
            assert tags[i][:2]=='I-'
            assert tags[i][2:]==tags[i-1][2:]
        return tags[0][2:]
    
    return slam(items, groupWithPreviousX=lambda t: t.startswith('I-'),
                groupTagX=stripBIO)

def slam(items, defaultWhitespace=' ', groupWithPreviousX=None, groupTagX=None):
    '''
    Given a sequence of (token, tag) tuples optionally separated by whitespace
    strings, produces a SLAM representation of the data.
    
    A function which takes a tag string and returns True or False can be provided 
    as groupWithPreviousX to convert multiple consecutive tokens into a multi-word 
    span with a single tag. If groupTagX is present, it will be called on the 
    sequence of tags in the group to obtain the tag for the span. Otherwise, 
    the composite tag will be produced by gluing together the individual tags 
    with 'defaultWhitespace'.
    
    >>> sentence = ['  ', ('It', 'PRP'), '  ', ' ', ('was', 'VBD'), ('a', ''), '  ', ('dark', 'AP-1/JJ'), ('and', 'AP-2/CC'), '   ', ('stormy', 'AP-3/JJ'), '  ', ('night', None), '   ']
    >>> slam(sentence)
    '  It/PRP   was/VBD a/{}  dark/{AP-1/JJ} and/{AP-2/CC}   stormy/{AP-3/JJ}  night   '
    >>> continues = lambda t: t and '-' in t and int(t[t.index('-')+1])>1
    >>> slam(sentence, groupWithPreviousX=continues)
    '  It/PRP   was/VBD a/{}  {dark and   stormy}/{AP-1/JJ AP-2/CC AP-3/JJ}  night   '
    >>> spanCat = lambda tt: tt[0] if len(tt)==1 else tt[0][:tt[0].index('-')]
    >>> slam(sentence, groupWithPreviousX=continues, groupTagX=spanCat)
    '  It/PRP   was/VBD a/{}  {dark and   stormy}/AP  night   '
    >>> sentence2 = map(lambda tkn: (tkn,None), "That 's what she said !".split())
    >>> slam(sentence2, groupWithPreviousX=lambda t: True, groupTagX=lambda tt: None)
    "{That 's what she said !}/"
    >>> slam([(' ', None), ' ', ('', 'hi')])
    '{ }/ {}/hi'
    '''
    
    assert not (groupTagX and not groupWithPreviousX)
    
    # Group tokens and ensure a single whitespace string separates each consecutive pair of tokens
    result = [] # Whitespace strings and tokens grouped into lists
    group = []
    separator = ''  # concatenated consecutive whitespace separating tokens
    for itm in items:
        if isinstance(itm, basestring):
            separator += itm
        else:
            if group and not separator:
                separator = defaultWhitespace
                assert separator
            assert re.match(r'\s*', separator)
            group.append(separator)
            separator = ''
            tkn,tag = itm
            if group and (not groupWithPreviousX or not groupWithPreviousX(tag)):
                result.append(group)
                group = []
            group.append(itm)
    if items:
        if separator:
            group.append(separator)
        result.append(group)
    
    # Generate SLAM string
    resultS = ''
    for g in result:
        followingS = ''
        gtkn = None
        gtags = []
        for i,entry in enumerate(g):
            if isinstance(entry, basestring):
                if i==0:
                    resultS += entry  # preceding whitespace
                elif i==len(g)-1:
                    followingS = entry  # following whitespace
                else:
                    if gtkn is None: gtkn = ''
                    gtkn += entry # whitespace internal to the span/group
            else:
                tkn,tag = entry
                if gtkn is None: gtkn = ''
                gtkn += tkn
                gtags.append(tag)
        if gtkn is not None:
            if groupTagX:
                gtag = groupTagX(gtags)
            elif len(gtags)>1:
                gtags = map(lambda t: '' if t is None else t, gtags)
                gtag = defaultWhitespace.join(gtags)
            else:
                gtag = gtags[0]
            resultS += escapeItem(gtkn, gtag)
        resultS += followingS
    
    return resultS

def parse(s, includeWhitespace=True):
    '''
    Returns a list of (token, tag) tuples representing items. If includeWhitespace is True, these will be separated by whitespace strings.
    
    Currently, word// and word/{/} are treated identically—i.e. as a word tagged with a single slash. Likewise for word/// and word/{//}, etc.
    
    >>> parse("I/PRP 'm/VBP a/DT    thistle/NN -/. sifter/NN")
    [('I', 'PRP'), ' ', ("'m", 'VBP'), ' ', ('a', 'DT'), '    ', ('thistle', 'NN'), ' ', ('-', '.'), ' ', ('sifter', 'NN')]
    >>> parse(" \t \\n \t I have a sieve   of unsifted    thistles   \\n", includeWhitespace=False)
    [('I', None), ('have', None), ('a', None), ('sieve', None), ('of', None), ('unsifted', None), ('thistles', None)]
    >>> parse('{complex phrase}/tag1 word/{complex tag} {Complex phrase}/{Complex tag}')
    [('complex phrase', 'tag1'), ' ', ('word', 'complex tag'), ' ', ('Complex phrase', 'Complex tag')]
    >>> parse('{}/{empty} {null}/{} {}/{} {}/ {}// {}/// {/}/{/} {/}///')
    [('', 'empty'), ' ', ('null', ''), ' ', ('', ''), ' ', ('', None), ' ', ('', '/'), ' ', ('', '//'), ' ', ('/', '/'), ' ', ('/', '//')]
    >>> parse('{antimony arsenic} {aluminum selenium}/X')
    [('antimony arsenic} {aluminum selenium', 'X')]
    >>> parse('{hydrogen oxygen}/X {nitrogen rhenium}/Y')
    [('hydrogen oxygen', 'X'), ' ', ('nitrogen rhenium', 'Y')]
    >>> parse('hi/{W X} there/Y {dude}/Z')
    [('hi', 'W X'), ' ', ('there', 'Y'), ' ', ('dude', 'Z')]
    >>> parse('hi/{W X} {there} dude/ {Z}')
    [('hi', 'W X} {there} dude/ {Z')]
    >>> parse('hi/{W X} {there} dude/{Z}')
    [('hi', 'W X} {there'), ' ', ('dude', 'Z')]
    >>> parse('hi/{W X} {there}// dude/{Z}')
    [('hi', 'W X'), ' ', ('there', '/'), ' ', ('dude', 'Z')]
    >>> parse(r'a/A //PUNC b/B c {d/s{g h}\q}/D eee//')
    [('a', 'A'), ' ', ('/', 'PUNC'), ' ', ('b', 'B'), ' ', ('c', None), ' ', ('d/s{g h}\\\\q', 'D'), ' ', ('eee', '/')]
    >>> parse('{http://www.google.com/sports}//  {http://www.google.com/}/URL')
    [('http://www.google.com/sports', '/'), '  ', ('http://www.google.com/', 'URL')]
    >>> parse('good/')
    [('good', None)]
    >>> parse('good/ boy')
    [('good', None), ' ', ('boy', None)]
    >>> parse('//')
    [('/', None)]
    >>> parse('// ')
    [('/', None), ' ']
    >>> parse('///')
    [('/', '/')]
    >>> parse('//// ')
    [('/', '//'), ' ']
    >>> parse('/BAD')
    Traceback (most recent call last):
      ...
    AssertionError: Tag missing a token: '/BAD'
    >>> parse('/BAD boy')
    Traceback (most recent call last):
      ...
    AssertionError: Tag missing a token: '/BAD boy'
    >>> parse('really /BAD boy')
    Traceback (most recent call last):
      ...
    AssertionError: Tag missing a token: '/BAD boy'
    >>> parse('///BAD')
    Traceback (most recent call last):
      ...
    AssertionError: Invalid unprotected tag starting with "/"
    >>> parse('///BAD/')
    Traceback (most recent call last):
      ...
    AssertionError: Invalid unprotected tag starting with "/"
    >>> parse('///{BAD}')
    Traceback (most recent call last):
      ...
    AssertionError: Invalid unprotected tag starting with "/"
    >>> parse('{/')
    Traceback (most recent call last):
      ...
    AssertionError: Invalid protected token: '{/'
    >>> parse('}/X')
    Traceback (most recent call last):
      ...
    AssertionError: Ill-formed unprotected token: '}/X'
    >>> parse('{hello there} {how are you}')
    Traceback (most recent call last):
      ...
    AssertionError: Invalid protected token: '{hello there} {how are you}'
    >>> parse('hi/{W X} {there}// dude/ {Z}')
    Traceback (most recent call last):
      ...
    AssertionError: Invalid protected token: '{Z}' after 'hi/{W X} {there}// dude/ '
    >>> parse('hi/{ } { } { }/X ')
    Traceback (most recent call last):
      ...
    AssertionError: Ambiguous protected tag/token combination: '/{ } { } { }/'
    >>> parse('w/{ } { }/W hi/{ } { } { } { }/X y/{ } z/Z ')
    Traceback (most recent call last):
      ...
    AssertionError: Ambiguous protected tag/token combination: '/{ } { } { } { }/'
    >>> parse('hi/{} {b} {}/X')
    Traceback (most recent call last):
      ...
    AssertionError: Ambiguous protected tag/token combination: '/{} {b} {}/'
    >>> parse('hi/{{} x/ } } / { {{ {x /x } { // }/X')
    [('hi', '{} x/ } } / { {{ {x /x '), ' ', (' // ', 'X')]
    >>> parse('hi/{{} x/ } / } { {{ {x /x } { // }/X')
    Traceback (most recent call last):
      ...
    AssertionError: Ambiguous protected tag/token combination: '/{{} x/ } / } { {{ {x /x } { // }/'
    '''
    items = []
    tkn = ''
    tag = None
    i = 0
 
    # Exclude ambiguous case. E.g. x/{a} {b} {c}/Y is ambiguous 
    # because the 'b' could belong either to the previous tag or the following token.
    j = 0
    while j<len(s):
        m = re.search(r'/\{.*\}(\s+\{.*\})+\s+\{.*\}/', s[j:])
        if not m:
            break
        if '/{' not in m.group(0)[2:] and '}/' not in m.group(0)[:-2]: 
            assert False, 'Ambiguous protected tag/token combination: {!r}'.format(m.group(0))
        j += m.start()+1
 
    def readPart(tag=False):
        follower = '(\s|$)' if tag else '/'
        if tag and i==len(s):   # tag delimiter slash ends the string
            return (None, i+1)
        assert i<len(s), 'Missing {} at end of string'.format('tag' if tag else 'token')
        if s[i]=='{': # protected token or tag
            assert i+1<len(s), 'Invalid protected {}: {!r}{}'.format('tag' if tag else 'token', shorten(s[i:]), ' after {!r}'.format(shorten(s[:i],False)) if shorten(s[:i],False) else '')
            # get s[i+1:x] for largest x such that the  substring is followed by '}' and delimiter but does not contain '/{' or '}/'
            m = re.match(r'^([^/]|(?<!})/(?!{))*(?=}' + follower + ')', s[i+1:])
            assert m, 'Invalid protected {}: {!r}{}'.format('tag' if tag else 'token', shorten(s[i:]), ' after {!r}'.format(shorten(s[:i],False)) if shorten(s[:i],False) else '')
            t = m.group(0)
            assert '/{' not in t and '}/' not in t, 'Invalid protected {}: {!r}'.format('tag' if tag else 'token', t)
            return (t, i+len(t)+2)
        elif tag and s[i]=='/':
            m = re.match(r'[/]+(?=' + follower + ')', s[i:])
            assert m, 'Invalid unprotected tag starting with "/"'
            t = m.group(0)
            return (t, i+len(t))
        # unprotected token or tag
        m = re.match(r'^[^\s/{}]*' if tag else r'^([^\s/{}]+|/)', s[i:])
        assert m, 'Ill-formed unprotected {}: {!r}'.format('tag' if tag else 'token', shorten(s[i:]))
        t = m.group(0)
        if t=='':
            return (None, i)
        return (t, i+len(t))
 
    while i<len(s):
        # process whitespace
        m = re.match(r'^\s+', s[i:])
        if m:
            if includeWhitespace: items.append(m.group(0))
            i += len(m.group(0))
            continue
 
        i0 = i
        tkn, i = readPart(tag=False)
        tag = None
        if i<len(s):
            if s[i]=='/':   # tag delimiter slash 
                i += 1
                # tag
                tag, i = readPart(tag=True)
            else:
                assert re.match(r'\s', s[i]), 'Tag missing a token: {!r}'.format(shorten(s[i0:]))
                
        items.append((tkn, tag))
        tkn = ''
        tag = None


    if 'testing'!='testing':
        # There will be some mismatches due to unnecessary protection and the slash-only tag conventions
        s2 = ''
        for itm in items:
            if isinstance(itm, basestring):
                s2 += itm
            else:
                s2 += escapeItem(*itm)
        if includeWhitespace:
            assert s==s2,s2
        else:
            assert re.sub(r'\s','',s)==s2,s2
    
    return items

def tabbed2inline(input, **kwargs):
    '''
    Given a file or string where each line is a token followed by a tab and its tag, 
    return the inline format equivalent.
    
    >>> tabbed2inline('hello\\tX\\nthere\\tY\\nMr. Rogers\\tNNP PER\\n\\n1/3\\tCD\\nand\\ndistinguished friends\\n')
    'hello/X there/Y {Mr. Rogers}/{NNP PER}\\n{1/3}/CD and {distinguished friends}/\\n'
    >>> tabbed2inline('She\\tX\\nSells\\t\\n\\nSea\\nShells\\tY\\n\\n\\nBy\\tY\\nThe\\tZ\\nSeashore\\n\\n')
    'She/X Sells/{}\\nSea Shells/Y\\n\\nBy/Y The/Z Seashore\\n\\n'
    >>> tabbed2inline('She\\tX\\nSells\\t\\n\\nSea\\nShells\\tY\\n\\n\\nBy\\tY\\nThe\\tZ\\nSeashore\\n\\n', defaultWhitespace='  ', groupWithPreviousX=lambda t: t=='Y', groupTagX=lambda tt: '_' if len(tt)>1 else tt[0])
    'She/X  Sells/{}\\n{Sea  Shells\\n\\nBy}/_  The/Z  Seashore\\n\\n'
    '''
    if not hasattr(input, 'readline'):
        input = input.splitlines(True)
        
    # Convert to list representation
    result = []
    for ln in input:
        if ln.strip()=='':  # blank line
            result.append('\n')
            continue
        if '\t' in ln:
            tkn, tag = ln[:-1].split('\t')
        else:
            tkn = ln[:-1]
            tag = None
        result.append((tkn, tag))
    result.append('\n')
    
    # Convert to SLAM
    return slam(result, **kwargs)

def inline2tabbed(input):
    '''
    Given a file or string in the inline format, convert to the format 
    where each line is a token followed by a tab and its tag.
    
    >>> inline2tabbed('hello/X there/Y\\n\\n{Mr. Rogers}/{NNP PER}\\n{1/3}/CD and {distinguished friends}//\\n')
    'hello\\tX\\nthere\\tY\\n\\n\\nMr. Rogers\\tNNP PER\\n\\n1/3\\tCD\\nand\\t\\ndistinguished friends\\t/\\n'
    >>> inline2tabbed('She/X Sells/{}\\nSea Shells/Y\\n\\nBy/Y The/Z Seashore\\n\\n')
    'She\\tX\\nSells\\t\\n\\nSea\\t\\nShells\\tY\\n\\n\\nBy\\tY\\nThe\\tZ\\nSeashore\\t\\n\\n'
    '''
    if not hasattr(input, 'readline'):
        input = input.splitlines(True)
    inS = ''
    for ln in input:
        inS += ln
        
    result = parse(inS)
    
    s = ''
    for i,entry in enumerate(result):
        if isinstance(entry, basestring):
            # add a newline before each group of newlines, except at the end
            if i<len(result)-1:
                entry = re.sub(r'(\n+)', r'\n\1', entry)
            
            # convert tabs and spaces to single newlines
            s += entry.replace('\t',' ').replace(' ','\n')
        else:
            tkn, tag = entry
            s += tkn + '\t' # change: tab should be included even if no tag
            if tag is not None:
                s += tag
    
    return s

def describe(items):
    '''
    Summarize the tokens and tags in an iterable over (token, tag) pairs.
    
    >>> print(describe(parse('Hello/X there/Y\\n\\n{Mr. Rogers}/{NNP PER}\\n{1/3}/CD and {}/my 30/{} {distinguished friends}// !\\n', includeWhitespace=False)))
    Tokens                                     9
       Empty                                   1
       With whitespace                         2
       With an alphanumeric character          7
       Unique types                            9
       Top 20: [('and', 1), ('', 1), ('30', 1), ('1/3', 1), ('Mr. Rogers', 1), ('there', 1), ('!', 1), ('Hello', 1), ('distinguished friends', 1)]
    <BLANKLINE>
    Untagged tokens                            2
    Tagged tokens                              7
       Empty tag                               1
       Tag contains only /'s                   1
       Tag with whitespace                     1
       Tag with an alphanumeric character      5
       Unique tags                             7
       Top 20 tags: [('', 1), ('/', 1), ('CD', 1), ('NNP PER', 1), ('Y', 1), ('X', 1), ('my', 1)]
    '''
    tknC, tagC = map(Counter, zip(*items))
    nUntagged = tagC[None]
    del tagC[None]
    
    s = '''
Tokens                                {:6}
   Empty                              {:6}
   With whitespace                    {:6}
   With an alphanumeric character     {:6}
   Unique types                       {:6}
   Top 20: {}
'''.format(sum(tknC.values()), 
           sum(v for k,v in tknC.items() if k==''), 
           sum(v for k,v in tknC.items() if sum(ch.isspace() for ch in k)),
           sum(v for k,v in tknC.items() if sum(ch.isalnum() for ch in k)),
           len(tknC),
           tknC.most_common(20)) + '''
Untagged tokens                       {:6}'''.format(nUntagged) + '''
Tagged tokens                         {:6}
   Empty tag                          {:6}
   Tag contains only /'s              {:6}
   Tag with whitespace                {:6}
   Tag with an alphanumeric character {:6}
   Unique tags                        {:6}
   Top 20 tags: {}
'''.format(sum(tagC.values()), 
           sum(v for g,v in tagC.items() if g==''), 
           sum(v for g,v in tagC.items() if re.match(r'^[/]+$', g)),
           sum(v for g,v in tagC.items() if sum(ch.isspace() for ch in g)),
           sum(v for g,v in tagC.items() if sum(ch.isalnum() for ch in g)),
           len(tagC),
           tagC.most_common(20))
    return s.strip()

def test():
    import doctest
    doctest.testmod()

#test()

if __name__=='__main__':
    args = sys.argv[1:]
    byLine = False
    if '--lines' in args:
        byLine = True   # treat each line as a separate SLAM document (assumes no tokens or tags will contain newlines)
        args.remove('--lines')
    
    if byLine:
        items = []
        for ln in fileinput.input(args):
            items.extend(parse(ln, includeWhitespace=False))
    else:
        data = ' '.join(ln for ln in fileinput.input(args))
        print(len(data),'characters of data')
        items = parse(data, includeWhitespace=False)
        print('parsed')
    print(describe(items))
    
