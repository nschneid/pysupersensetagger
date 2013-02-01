#!/usr/bin/env python2.7
'''
Allocates source data into batches. Intended for several scenarios, including 
partitioning data into train/dev/test splits, and distributing data for annotation 
or parallel processing.

This involves several steps: 
  selecting items from the input (see --unit, --start, --stop, --max-items), 
  determining their sort order (--pre-sort, -r, -u), 
  assigning items to batches (-n, --max-n, --mode, --max-size, --block-size), 
  sorting items within each batch (--post-sort, -R, -U), 
  numbering the batches (--batch-*), and 
  generating output files for each batch (--prefix, --suffix, --overwrite, --align).

Usage:
$ distribute [opts] inputfile1 [inputfile2 ...]


The simplest way to invoke this command is like
$ distribute inputfile -n 2
which will read lines of 'inputfile' and alternately assign them to two batches. 
Assuming no other batch files are present in the directory, these will be called 
'batch1.txt' and 'batch2.txt', respectively.

$ distribute --pre-sort md5 -n 10 --batch-start 0
will sort lines by their md5 hash, then alternately assign them to one of 10 batches
('batch0.txt' through 'batch9.txt').

$ distribute -n 3 0.8 0.1 0.1 --unit chunk
will read chunks (groups of lines separated by double newlines) from stdin and
partition them into 3 batches: the first 80% of chunks will be assigned to the first 
batch, 10% to the second, and 10% to the third. 

$ distribute --unit token --max-size 10 --pre-sort alpha -u --block-size 10 inputfile1 inputfile2
will alphabetically sort all word types from the two input files and divide them 
into batches of 10, such that the concatenation of successive batches would yield 
the full, alphabetized lexicon of words appearing in the input.

$ distribute --unit token --max-size 10 --post-sort alpha -U inputfile1 inputfile2
will alternately assign tokens of input to the 10-item batches, and then sort each batch 
alphabetically, removing any duplicates within the batch.


So that the number of batches can be computed, at least one of (-n, --max-n, --max-size) 
is required.

Currently, the batch allocations will always be disjoint subsets of the input.

Run with -h for full help information.

@author: Nathan Schneider (nschneid)
@since: 2011-06-26
'''

from __future__ import print_function, division

import sys, os, re, itertools
import argparse
import fileinput, glob
import hashlib, random

if __name__ == "__main__" and __package__ is None:
    import inlinetag
    from ds.set import OrderedSet
else:
    from . import inlinetag
    from ..ds.set import OrderedSet

def uniquify(elts):
    return [elt for elt in OrderedSet(elts)]

md5 = lambda x: hashlib.md5(x).hexdigest()

def requireOneOf(parser, args, aa, pred=lambda x: x is not None):
    for a in aa:
        att = a
        if att.startswith('--'): att = att[2:]
        elif att.startswith('-'): att = att[1:]
        att = att.replace('-','_')
        v = getattr(args, att)
        if pred(v):
            return True
    parser.error("at least one of the following arguments is required: "+', '.join("'"+a+"'" for a in aa))

def intNonneg(s):
    v = int(s)
    assert v>=0
    return v

def intPositive(s):
    v = int(s)
    assert v>0
    return v

def finite(v):
    return v is not None and v!=float('inf') and v!=float('-inf')

def floatNonneg(s):
    v = float(s)
    assert finite(v) and v>=0.0
    return v

def intNonnegOrInfinity(s):
    if s.lower()=='infinity':
        return float('Inf')
    return intNonneg(s)

def intPositiveOrInfinity(s):
    if s.lower()=='infinity':
        return float('Inf')
    return intPositive(s)

def FilePattern(mode):
    return lambda s: [argparse.FileType(mode)(f) for f in glob.glob(s)]

parser = argparse.ArgumentParser(description=__doc__[:__doc__.rindex('\n\n')], epilog=__doc__[__doc__.rindex('\n\n')+2:], formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-n', type=floatNonneg, nargs='+', help='exact number of batches to allocate')
parser.add_argument('--max-n', type=intPositiveOrInfinity, default='infinity', help='maximum number of batches to allocate (default: %(default)s)')
parser.add_argument('--max-items', type=intPositiveOrInfinity, default='infinity', help='maximum number of items to allocate (default: %(default)s)')
parser.add_argument('--max-size', type=intPositiveOrInfinity, default='infinity', help='maximum number of items to allocate to each batch (default: %(default)s)')

parser.add_argument('--unit', choices=('token','slam','field','line','chunk','file'), default='line', help="Units by which the input data should be divided into items (default: '%(default)s'). 'token' breaks on all whitespace; 'field' breaks on tabs and newlines; 'line' breaks on newlines; 'chunk' breaks on double newlines and file boundaries; 'file' breaks on input file boundaries. For units smaller than a line, items will be separated in the output with a single line break.")

parser.add_argument('--start', type=int, default=0, help="Lower offset bound of the items to be selected for allocation (default: %(default)s). Negative offsets are relative to the end of the item list.")
parser.add_argument('--stop', type=int, help="Upper offset bound of the items to be selected for allocation. Negative offsets are relative to the end of the item list. Defaults to min(total number of items in the input, --start + --max-items).")

parser.add_argument('--prefix', default='batch', help="string prefixed to the batch offset in the batch name (default: '%(default)s')")
parser.add_argument('--suffix', default='', help="string suffixed to the batch offset in the batch name (but before the file extension) (default: '%(default)s')")
parser.add_argument('--batch-start', type=intPositive, help='offset of the first batch being allocated. By default, it is determined automatically based on the greatest offset of similar batch files in the current directory.')
parser.add_argument('--batch-width', type=intPositive, default=3, help="minimum number of digits to include in the batch number (padded with leading zeros) (default: %(default)s)")
parser.add_argument('--batch-radix', type=intPositive, default=1, help="value of which the starting batch offset must be a multiple (default: %(default)s)")
parser.add_argument('--overwrite', action='store_true', help="resolve any batch name conflicts by overwriting existing batch files")

parser.add_argument('--mode', choices=('alternate','contiguous'), default='alternate', help="method of populating batches from a presorted list of items: 'alternate' (the default) cycles through batches and assigns one block of items at a time (see --block-size); 'contiguous' assigns each batch a coherent group of --max-size items")

parser.add_argument('--block-size', type=intPositive, default=1, help="number of items in each block when the allocation mode is 'alternate' (default: %(default)s)")

parser.add_argument('--pre-sort', choices=('none','alpha','alphanum','md5','md5item','random'), default='none', help="Sort items prior to batch assignment (default: '%(default)s'). 'alpha' performs plain alphabetic sorting. 'alphanum' numerically interprets digit sequences. 'md5' sorts on the md5 hashes of the item string values. 'md5item' appends the item value to its original offset, so repeated items will have different hashes. 'random' randomly shuffles the items and is thus discouraged if replicability is a concern.")
parser.add_argument('-r', action='store_true', help="pre-sort in reverse order")
parser.add_argument('-u', action='store_true', help="filter out duplicate items prior to batch assignment")

parser.add_argument('--post-sort', choices=('none','alpha','alphanum','md5','md5item','random','orig'), default='none', help="Sort items within each batch (default: '%(default)s'). Same choices as --pre-sort, plus 'orig', which restores the original (relative) ordering of items from the input (they may have been reordered with a pre-sort).")
parser.add_argument('-R', action='store_true', help="post-sort in reverse order")
parser.add_argument('-U', action='store_true', help="filter out duplicate items within each batch")


parser.add_argument('--align', choices=('yes','no'), default='yes', help="generate alignment file with unit offsets in terms of the input (enabled by default)")

parser.add_argument('inputFile', nargs='+', help="input data")




#parser.print_help()


# To accommodate argparser, rearrange to ensure that -n option doesn't immediately precede the positional args
if '-n' in sys.argv:
    i = sys.argv.index('-n')
    j = i+1
    # Consume numeric strings as arguments of -n
    while j<len(sys.argv) and re.match(r'\d+([.](\d+)?)?|[.]\d+', sys.argv[j]):
        j += 1
    # Move -n and its arguments to the end
    aa = sys.argv[1:i] + sys.argv[j:] + sys.argv[i:j]
else:
    aa = sys.argv[1:]



args = parser.parse_args(aa)



# Additional argument checks
requireOneOf(parser, args, ('-n', '--max_n', '--max_size'), pred=finite)
if args.block_size>1:
    assert args.mode=='alternate', '--block-size option is only relevant for --mode alternate'
if args.mode=='contiguous':
    assert finite(args.max_items) or (args.n and len(args.n)>1), '-n proportion arguments, or a finite value for --max-items, are required when using --mode contiguous'
if args.n is not None:
    assert intPositive(args.n[0])==args.n[0], 'the first argument to the -n option must be a positive integer, not {}'.format(args.n[0])
    args.n[0] = int(args.n[0])
    if len(args.n)>1:
        assert len(args.n[1:])==args.n[0], 'the option -n {} is invalid: if proportions are specified they must be given for all {} batches'.format(' '.join(map(str,args.n)), args.n[0])

def items():
    '''Generator over items of input data, as determined by options.'''
    
    # Unless we are processing file units, blank lines are not included in items
    # (though they are relevant as chunk separators).
    s = None
    matchedFiles = []
    if args.inputFile:
        for ptn in args.inputFile:
            pp = glob.glob(ptn)
            if len(pp)==0:
                if '*' in ptn or '?' in ptn or '[' in ptn:
                    pass    # failed pattern contained wildcards, probably OK
                else:
                    file(ptn)   # trigger a "does not exist" error
            matchedFiles.extend(pp)
        assert pp, 'None of the provided patterns match any input file'
    # else, default to stdin
    
    if args.unit=='slam':
        # Slashed markup - requires reading each file in its entirety (multiline spans are permitted)
        for fP in matchedFiles:
            with open(fP) as f:
                s = f.read()
                for itm in inlinetag.parse(s, includeWhitespace=False):
                    yield itm
        return
    
    for ln in fileinput.input(matchedFiles):
        if s is not None:
            if ln.strip()=='' and args.unit=='chunk':
                yield s
                s = None
                continue
            elif fileinput.isfirstline() and args.unit in ('file','chunk'):
                yield s
                s = ''
            
            if args.unit in ('file','chunk'):
                s += ln
        else:
            if args.unit=='file':
                s = ln
            elif ln.strip()!='':
                if args.unit=='chunk':
                    s = ln
                elif args.unit=='line':
                    yield ln
                else:
                    ln = ln.strip()
                    for t in (ln.split() if args.unit=='token' else ln.split('\t')):
                        yield t
                        
    if s is not None:
        yield s
        

itms = list(enumerate(items())) # (index, item) pairs

# Select relevant input items
stop = args.stop
if stop is None:
    stop = min(len(itms), args.start+args.max_items)
    if stop==0:
        stop = len(itms)

itms = itms[args.start:stop]


# pre-uniquify
if args.u:
    itms = uniquify(itms)


def sortItems(xx, method, reverse=False):
    
    def human_key(key):
        '''
        A sorting key for alphanumeric strings.
        ['9', 'aB', '1a2', '11', 'ab', '10', '2', '100ab', 'AB', '10a', '1', '1a', '100', '9.9', '3']
        will be sorted into
        ['1', '1a', '1a2', '2', '3', '9', '9.9', '10', '10a', '11', '100', '100ab', 'ab', 'aB', 'AB']
        
        Source: http://stackoverflow.com/questions/5254021/python-human-sort-of-numbers-with-alpha-numeric-but-in-pyqt-and-a-lt-opera/5254534#5254534
        '''
        parts = re.split('(\d*\.\d+|\d+)', key)
        return tuple((e.swapcase() if i % 2 == 0 else float(e)) for i, e in enumerate(parts))
    
    if method=='random':
        random.shuffle(itms)
        return
    
    ky = None
    if method=='alpha':
        ky = lambda (a,b): b
    elif method=='alphanum':
        ky = lambda (a,b): human_key(b)
    elif method=='md5':
        ky = lambda (a,b): md5(b)
    elif method=='md5item':
        ky = lambda (a,b): md5(str(a)+b)
    elif method=='orig':
        ky = lambda (a,b): a
    
    xx.sort(key=ky, reverse=reverse)


# Pre-sort

if args.pre_sort!='none':
    if args.pre_sort=='random':
        assert not args.r, 'option conflict: cannot reverse (-r) a random sort (--pre-sort random)'
    sortItems(itms, args.pre_sort, reverse=args.r)
    


# Determine the number of batches

nItems = len(itms)
nBatches = None
if args.n is not None:
    nBatches = args.n[0]
if not nBatches:
    a, b = divmod(nItems, args.max_size)
    nBatches = min(a + int(b>0), args.max_n)
    assert nBatches>0
assert nBatches<=args.max_n, 'conflicting options: -n {} but --max-n {}'.format(nBatches, args.max_n[0])
assert not finite(args.max_size) or len(args.n)==1, '--max-size option currently not supported when proportions are specified with -n'

# Batch assignment

batches = [[] for b in range(nBatches)]

capacities = [args.max_size for b in range(nBatches)]

if len(args.n)>1:
    proportions_sum = sum(args.n[1:])
    amounts = [divmod(args.n[b+1]/proportions_sum*nItems, 1.0) for b in range(nBatches)]
    capacities = [int(a) for a,b in amounts]
    while sum(capacities)<nItems:    # deal with fractional items
        i,(a,b) = max(enumerate(amounts), key=lambda x: x[1][1])
        capacities[i] += 1
        amounts[i] = (a+1, 0.0)


itmsI = iter(itms)

def full(b):
    return len(batches[b])==capacities[b]


if args.mode=='alternate':  # alternating mode (will make the batches as evenly-sized as possible)
    try:
        while sum(1 for b in range(nBatches) if not full(b)):  # while there is room in some batch
            for b in range(nBatches):   # iterate through the batches
                for i in range(args.block_size):    # if there is room, assign up to block_size items to the batch
                    if not full(b):
                        batches[b].append(next(itmsI))
                    else: break
    except StopIteration:
        pass
else:   # contiguous mode: assign groups of 'max_size' items until we run out
    try:
        for b in range(nBatches):
            for i in range(capacities[b]):
                batches[b].append(next(itmsI))
    except StopIteration:
        pass


# Uniquify each batch
if args.U:
    for b in range(nBatches):
        batches[b] = uniquify(batches[b])

# Post-sort each batch

if args.post_sort!='none':
    if args.post_sort=='random':
        assert not args.R, 'option conflict: cannot reverse (-R) a random sort (--post-sort random)'
    for batch in batches:
        sortItems(batch, args.post_sort, reverse=args.R)

# Decide how to number the batches

# TODO: If unit=file, we need to do something special with the filenames (suffix them with the batch number or create a symlink?).

batchNamePattern = args.prefix + '{}' + args.suffix
batchStart = args.batch_start

# - Find the maximum batch number already in use
similars = glob.glob(batchNamePattern.format('*')+'.txt')
x = 0
for similar in similars:
    m = re.match(re.escape(args.prefix)+r'(\d+)'+re.escape(args.suffix)+r'[.]txt$', similar)
    if m:
        y = int(m.group(1))
        if y>x: x = y

# - Determine what the next number should be
if not batchStart:
    batchStart = (x//args.batch_radix+1)*args.batch_radix
elif not args.overwrite:
    assert x<batchStart, '--batch-start {} specified, but higher-numbered batches already exist (use --overwrite to replace them)'.format(args.batch_size)
# TODO: relax the above restriction so that it's only an error if there would actually be an overwrite? we know nBatches by now
assert batchStart % args.batch_radix==0, 'conflicting options: --batch-radix {} and --batch-start {} (when both are specified, --batch-start must be a multiple of --batch-radix)'.format(args.batch_radix, args.batch_start)


# Write batches, alignment files to disk

batchNamePattern = batchNamePattern.replace('{}','{:0'+str(args.batch_width)+'}')

batchName = lambda b: batchNamePattern.format(b+batchStart)

separator = ''
if args.unit=='token' or args.unit=='field' or args.unit=='chunk':
    separator = '\n'

for b,batch in enumerate(batches):
    if args.unit!='file':
        batchFP = batchName(b)
        alignFP = batchFP+'.align'
        if not args.overwrite:
            assert not os.path.exists(batchFP),'File already exists: {}'.format(batchFP)
            assert not os.path.exists(alignFP),'File already exists: {}'.format(alignFP)
        with open(batchFP,'w+') as batchF, open(alignFP,'w+') as alignF:
            if len(batch):
                bindices,bitms = list(zip(*batch))
                for itm in bitms:
                    batchF.write(itm+separator)
                alignF.write(repr(bindices))
    else:       # unit = file, so a batch may have multiple files
        assert False,'unit=file not yet implemented'
        for i,itm in batch:
            with open(b+i) as batchF, open(b+i+'.align') as alignF:
                batchF.write(itm)
                alignF.write(itm)
            

for b,bitms in enumerate(batches):
    print(batchName(b) + '\t{} items'.format(len(batches[b])), file=sys.stderr)
print('{} items selected, {} of which were allocated'.format(len(itms), sum(len(batch) for batch in batches)), file=sys.stderr)
if args.pre_sort=='none' and args.post_sort=='none' and not (args.r or args.R or args.u or args.U):
    DELIMS = {'token': ' ', 'field': r'\t', 'line': r'\n', 'chunk': r'\n'}
    if args.mode=='alternate' and args.block_size==1 and args.unit in DELIMS:
        print('Tip: Since mode=alternate and unit={}, results can be recombined with a command like `paste -d"{}" {}*{}`'.format(args.unit, DELIMS[args.unit], args.prefix, args.suffix), file=sys.stderr)
    elif args.mode=='contiguous' and args.unit in DELIMS:
        print('Tip: Since mode=contiguous and unit={}, results can be recombined with a command like `paste -s -d"{}" {}*{}`'.format(args.unit, DELIMS[args.unit], args.prefix, args.suffix), file=sys.stderr)
#print(batches[0])
#print(batches[-1])
