#!/usr/bin/env python2.7
'''
Demonstrates how to use the OntoNotes Python API (provided in the 
OntoNotes DB Tool v. 0.999b) to extract word sense, named entity, 
and proposition information.

Argument: config_path
 Config file path. See src/on/tools/config.example and the DPF 
 documentation for details. To include all of the English 
 newswire data I have used a config file with
    load: english-nw
 (With this setting, the first import of 'on' during each execution 
 of the program will spend about 30 seconds loading the data.)

DETAILS

This is especially useful for proposition information (like PropBank), 
which is in a hard-to-interpret raw format. I'm not a PropBank 
expert, so I may have glossed over some of the details of special 
kinds of arguments. Note that the proposition frame is NOT 
necessarily the same as the OntoNotes word sense, which is NOT 
necessarily the same as the WordNet sense. (Supposedly there are 
mappings between the three lexical resources, but I cannot figure 
out how to access those mappings.) In principle the OntoNotes word 
senses include both verbs and nouns, but few nouns are explicitly 
annotated with a sense. (The OntoNotes 4 documentation reports 70% 
sense coverage of nouns in the WSJ subcorpus, but this is largely due 
to monosemous nouns which are presumably straightfoward but have no 
explicit annotation.)

Not demonstrated here: accessing syntactic information, coreference 
annotations, or the 'speaker' or 'parallel' banks.

The OntoNotes DB Tool can be obtained from Sameer Pradhan's website:
  http://cemantix.org/ontonotes.html
The Python API package needs to be installed, e.g.
  $ python setup.py install
Optionally a MySQL database may be used for the data, but it is 
easier just to have the API access the text files directly.
Refer to the PDF version of the documentation for details (the 
online HTML version may be out of date):
  http://cemantix.org/download/ontonotes/beta/doc/pdf/on.pdf
  
@author: Nathan Schneider (nschneid)
@since: 2012-06-05
'''

from __future__ import print_function
import on, sys

cfgFP = sys.argv[1]
cfg = on.common.util.load_config(cfgFP)
all_on = on.ontonotes(cfg)

def describe_prop(p, indent=''):
   #print(p.corpus_id, p.document_id, p.tree_id, p.id)
   print(indent, p.lemma, p.pb_sense_num, p.quality, p.get_primary_predicate().token_index)
   print(indent, '-'*len(p.lemma)+'---')
   for agroup in p: # argument analogue, i.e. group of coreferent argument fillers
     for a in agroup:
       if isinstance(a, on.corpora.proposition.predicate):
         print(indent, a.enc_self, a.type, a.token_index)
       else:
         print(indent, a.enc_self, a.type)
         for anode in a:
           print(indent, '   ', anode.subtree.get_word_string())
           
MAX_TREES = 10
iTree = 0
for subcorp in all_on:
  for d in subcorp['parse']:
    for t in d:
      print('--------------------------------')
      print(t.get_plain_sentence())
      print(t.id)
      print(t.get_word_string())
      print()
      for node in t.subtrees():
        if node.named_entity:
            print('{', node.named_entity.string, '}', node.named_entity.type, (node.named_entity.start_token_index, node.named_entity.end_token_index), (node.named_entity.start_word_index, node.named_entity.end_word_index))
        if node.is_leaf():
          leaf = node
          if leaf.is_trace() or leaf.is_punct():
            continue
          print(leaf.get_word_string(), leaf.get_token_index(), leaf.get_word_index())
          if leaf.on_sense:
            print('  SENSE: ', leaf.on_sense.lemma, leaf.on_sense.pos, leaf.on_sense.sense)
            definition = on.corpora.sense.on_sense_type.get_name(leaf.on_sense.lemma, leaf.on_sense.pos, leaf.on_sense.sense)
            print('  >>>>>> ', definition)
          if leaf.proposition:
            print('  PROP: ')
            describe_prop(leaf.proposition, indent='  ')
      iTree += 1
      assert iTree<MAX_TREES

'''
Output for an example sentence from the English newswire subcorpus of 
OntoNotes 4.0:

--------------------------------
But this is what George Bush Sr. heard about George Bush Jr. in Abu Dhabi last week.
1@nw/p2.5_a2e/00/p2.5_a2e_0002@all@p2.5_a2e@nw@en@on
But this is what George Bush Sr. heard *T*-1 about George Bush Jr. in Abu Dhabi last week .

But 0 0
this 1 1
is 2 2
  SENSE:  be v 2
  >>>>>>  equivalence between two things, or class inclusion of one thing by another.
  PROP: 
   be 01 gold 2
   -----
   2:0 v 2
   0:0 ARGM-DIS
       But
   1:1 ARG1
       this
   3:2 ARG2
       what George Bush Sr. heard *T*-1 about George Bush Jr. in Abu Dhabi last week
what 3 3
{ George Bush Sr. } PERSON (4, 6) (4, 6)
George 4 4
Bush 5 5
Sr. 6 6
heard 7 7
  SENSE:  hear v 1
  >>>>>>  learn, receive information, take heed of
  PROP: 
   hear 01 gold 7
   -------
   7:0 v 7
   4:1 ARG0
       George Bush Sr.
   8:0 ARG1
       *T*-1
   9:1 ARGM-PRD
       about George Bush Jr.
   13:1 ARGM-LOC
       in Abu Dhabi
   16:1 ARGM-TMP
       last week
about 9 8
{ George Bush Jr. } PERSON (10, 12) (9, 11)
George 10 9
Bush 11 10
Jr. 12 11
in 13 12
{ Abu Dhabi } GPE (14, 15) (13, 14)
Abu 14 13
Dhabi 15 14
last 16 15
week 17 16
'''
