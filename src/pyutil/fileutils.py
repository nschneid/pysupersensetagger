'''
Utilities for file handling

@author: Nathan Schneider (nschneid)
@since: 2009-01-15
'''

# Strive towards Python 3 compatibility
from __future__ import print_function, unicode_literals, division, absolute_import
from future_builtins import map, filter


import sys, os, re
from types import StringTypes
from itertools import chain

'''
Useful methods in the OS module
os.path.abspath(path) -- absolute version of 'path'
os.chdir(path)
os.getcwd()
os.link(target,alias) -- creates a hard link 'alias' pointing to 'target'
os.listdir(path) -- arbitrarily-ordered ls; excludes . and ..
os.mkdir(path)
os.makedirs(path) -- can create a lineage of directories
os.readlink(path) -- get relative or absolute path pointed to by symbolic link
os.path.join(os.path.dirname(path), os.readlink(path)) -- get absolute path indicated by symbolic link
os.rename(src,dst)
os.walk(topdirectory) -- traverse a directory tree, e.g.: 
  for root, dirs, files in os.walk(topdirectory): 
      # root contains the current directory path, dirs contains subdirectories, files contains files in that directory
os.remove(), os.rmdir(), os.removedirs()
'''


def list_files(dir, regx, **kwds):
    return list_contents(dir, regx, includeFiles=True, includeDirs=False, **kwds)

def list_dirs(dir, regx, **kwds):
    return list_contents(dir, regx, includeFiles=False, includeDirs=True, **kwds)

def list_contents(dir, regx, includeFiles=True, includeDirs=True, recursive=False, absolute=True):
    '''
    Search the contents of the given directory for files and/or directories 
    whose names match the specified regular expression. If the regular 
    expression contains parenthesized subexpressions, the match of the first 
    such expression for each file/directory is returned. Otherwise, returns 
    a list of paths to contained files/directories.
    
    The regular expression must match only the file or directory name, 
    not any other part of its path (even if 'recursive' is True).
    The search does not follow links.
    
    Based on regexplace.make_files_list() by Stefano Spinucci, 2006-02-07 (rev 4), http://code.activestate.com/recipes/473828/
    '''

    assert includeDirs or includeFiles,'Need to search for at least one of (files, directories)'
    
    # if dir is not a directory, exit with error
    assert os.path.isdir(dir),'{0} is not a valid dir to walk'.format(dir)
    

    # compile the search regexp
    cregex=re.compile(regx)

    # initialize the list of contents
    clist = []

    # loop on all files and select files matching 'regx'
    for root, dirs, files in os.walk(dir):
        relevantContents = []
        if includeDirs:
            relevantContents += dirs
        if includeFiles:
            relevantContents += files
        for name in relevantContents:
            m = cregex.search(name)
            if m is not None:
                if len(m.groups())>0:
                    val = m.group(1)    # Note: May be None (if the subexpression is unused in the match)
                else:
                    val = os.path.join(root, name)
                clist.append(val)
        if not recursive:
            break

    # return the file list
    if absolute:    # absolute path
        return clist[:]
    return map(lambda p: os.path.basename(p), clist[:])

def merge_files(sources, target, sourceMode='rb', targetMode='wb', transformX=lambda sF,s,i: s, sourceFileOffset=0):
    '''
    Copies the contents of one or more source files into a target file.
    'sources' may be a file path, a file object, or a sequence of file paths/objects.
    'target' may be a file path or object. If 'sources' ('target') contains file paths, 
    they will be opened with mode 'sourceMode' ('targetMode').
    The file contents are optionally transformed by a specified function (which 
    takes the source file, its contents as a string, and its source file's index in the 
    sequence and outputs the transformed string). 
    By default, the source files are simply concatenated into the target.
    
    >>> with open('/tmp/source1.txt', 'w') as sF1:
    ...   sF1.write('so much depends\\nupon\\n\\na red wheel\\nbarrow\\n')
    >>> with open('/tmp/source2.txt', 'w') as sF2:
    ...   sF2.write('\\nglazed with rain\\nwater\\n\\n')    
    >>> with open('/tmp/source3.txt', 'w') as sF3:
    ...   sF3.write('beside the white\\nchickens.\\n')
    >>> sF1 = open('/tmp/source1.txt', 'r')
    >>> sF2 = open('/tmp/source2.txt', 'r')
    >>> sF3 = open('/tmp/source3.txt', 'r')
    >>> merge_files((sF1, sF2, sF3), '/tmp/target1.txt')
    >>> with open('/tmp/target1.txt', 'r') as tF1:
    ...   t1 = tF1.read()
    >>> sF1.close(); sF2.close(); sF3.close()
    >>> sF2 = open('/tmp/source2.txt', 'r')
    >>> merge_files(('/tmp/source1.txt', sF2, '/tmp/source3.txt'), '/tmp/target2.txt')
    >>> with open('/tmp/target2.txt', 'r') as tF2:
    ...   t2 = tF2.read()
    >>> t = """so much depends
    ... upon
    ... 
    ... a red wheel
    ... barrow
    ... !!!
    ... glazed with rain
    ... water
    ... 
    ... !!!beside the white
    ... chickens.
    ... """
    >>> t1==t2==t.replace('!!!','')
    True
    >>> tX = lambda f,s,i: '!!!'+s if i>0 else s
    >>> merge_files(('/tmp/source1.txt', '/tmp/source2.txt', '/tmp/source3.txt'), '/tmp/target3.txt', transformX=tX)
    >>> with open('/tmp/target3.txt', 'r') as tF3:
    ...   t3 = tF3.read()
    >>> t3==t
    True
    '''
    if isinstance(target,StringTypes):
        with open(target, targetMode) as tgtF:
            return merge_files(sources, tgtF, sourceMode, targetMode, transformX, sourceFileOffset)
    if isinstance(sources,StringTypes):
        return merge_files((sources,), target, sourceMode, targetMode, transformX, sourceFileOffset)
        
    sources = iter(sources)
    i = sourceFileOffset
    for src in sources:
        if src is not None:
            if isinstance(src,StringTypes):
                with open(src, sourceMode) as srcF:
                    return merge_files(chain((srcF,),sources), target, sourceMode, targetMode, transformX, sourceFileOffset=i)
            # src is an open file. Assume it is at the beginning.
            s = src.read()  # Read its contents
            t = transformX(src,s,i)   # Transform its contents
            target.write(t) # Write transformed contents to the target
            i += 1

def this_path(fileVar):
    '''To get the absolute path to the current script or module file:
        os.path.join(os.path.abspath(sys.path[0] or os.curdir), __file__)
    '''
    return os.path.join(os.path.abspath(sys.path[0] or os.curdir), fileVar)

def strip_extension(filename, extsep=os.path.extsep):
    '''Returns the provided filename minus the last . and any characters following it.
    
    >>> print(strip_extension('/apple/banana/my.tasty.file.txt'))
    /apple/banana/my.tasty.file
    >>> print(strip_extension('my.tasty.file$txt.1', extsep='$'))
    my.tasty.file
    '''
    i = filename.rfind(extsep)
    if i>-1:
        return filename[:i]
    return filename

def test():
    import doctest
    doctest.testmod()
    
if __name__=='__main__':
    test()
