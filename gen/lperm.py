## PYTHON

import sys
from itertools import permutations

def getDimNames(ndims):  
  dim_names = ['i', 'j', 'k', 'l', 'm', 'n']
  return dim_names[0:ndims]
  
def getDimPerms(dim_names):
  return permutations(dim_names)

def getDimPermNames(dim_names):
  PERM = permutations(dim_names)
  l = []
  for p in PERM:
    l.append("".join(p).upper())
  return l
  
def getEnumNames(ndims):
  dim_names = getDimNames(ndims)
  perm_names = getDimPermNames(dim_names)
  enum_names = map( lambda a: "PERM_"+a, perm_names)
  return enum_names
  
def getEnumName(PERM):
  return "PERM_" + "".join(PERM).upper();
 


