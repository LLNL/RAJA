#!/usr/bin/env python

#
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory.
#
# All rights reserved.
#
# For release details and restrictions, please see raja/README-license.txt
#


import sys
from itertools import permutations
from lperm import *


def writeForallPolicy(ndims):

  dim_names = getDimNames(ndims)

  print "// Execute (Termination default)"
  print "struct Forall%d_Execute_Tag {};" % ndims
  print "struct Forall%d_Execute {" % (ndims)
  print "  typedef Forall%d_Execute_Tag PolicyTag;" % ndims
  print "};"
  print ""
  
  print "// Starting (outer) policy for all forall%d policies" % ndims
  polstr = ", ".join(map(lambda a: "typename POL_%s"%(a.upper()), dim_names))
  print "template<%s, typename NEXT=Forall%d_Execute>" % (polstr, ndims)
  print "struct Forall%d_Policy {" % (ndims)
  print "  typedef NEXT NextPolicy;"
  for dim in dim_names:
    print "  typedef POL_%s Policy%s;" % (dim.upper(), dim.upper())
  print "};"
  print ""
  
     

def writeForallExecutor(ndims):

  dim_names = getDimNames(ndims)

  polstr = ", ".join(map(lambda a: "typename POLICY_"+(a.upper()), dim_names))
  setstr = ", ".join(map(lambda a: "typename T"+(a.upper()), dim_names))

  print "template<%s, %s>" % (polstr, setstr)
  print "struct Forall%dExecutor {" % (ndims)
  
  # Create default executor
  args = map(lambda a: "T%s const &is_%s"%(a.upper(), a), dim_names)
  argstr = ", ".join(args)  
  print "  template<typename BODY>"
  print "  inline void operator()(%s, BODY body) const {" % argstr
  print "    RAJA::forall<POLICY_I>(is_i, [=](Index_type i){"
  if ndims == 2:  # 2 dimension termination case:
    print "      RAJA::forall<POLICY_J>(is_j, [=](Index_type j){"
  else: # more than 2 dimensions, we just peel off the outer loop, and call an N-1 executor
    args = map(lambda a: "is_"+a, dim_names[1:])
    setstr = ", ".join(args)
    args = map(lambda a: "Index_type "+a, dim_names[1:])
    idxstr = ", ".join(args)  
    print "      exec(%s, [=](%s){" % (setstr, idxstr)
  
  argstr = ", ".join(dim_names)  
  print "        body(%s);" % argstr
  print "      });"
  print "    });"
  print "  }"
    
  # More than 2 dims: create nested ForallNExecutor  
  if ndims > 2:
    print ""
    args = map(lambda a: "POLICY_"+(a.upper()), dim_names[1:])
    polstr = ", ".join(args)
    args = map(lambda a: "T"+(a.upper()), dim_names[1:])
    argstr = ", ".join(args)
    print "  private:"
    print "    Forall%dExecutor<%s, %s> exec;" % (ndims-1, polstr, argstr)
  print "};"
  print ""    



def writeForall_policyForeward(ndims):
  
  dim_names = getDimNames(ndims)
  
  # Create boiler-plate used for all _policy() fcns
  polstr = ", ".join(map(lambda a: "typename Policy"+a.upper(), dim_names))
  setstr = ", ".join(map(lambda a: "typename T"+a.upper(), dim_names))
  isstr = ", ".join(map(lambda a: "T%s const &is_%s"%(a.upper(), a) , dim_names))
  
  template_string = "template<typename POLICY, %s, %s, typename BODY, typename TAG>" % (polstr, setstr)
  fcnargs_string = "TAG, %s, BODY body" % isstr
  
  
  
  
  print ""
  print template_string
  print "RAJA_INLINE void forall%d_policy(%s);" % (ndims, fcnargs_string)



def writeForall_policy(ndims):
  
  dim_names = getDimNames(ndims)
  
  # Create boiler-plate used for all _policy() fcns
  polstr = ", ".join(map(lambda a: "typename Policy"+a.upper(), dim_names))
  setstr = ", ".join(map(lambda a: "typename T"+a.upper(), dim_names))
  isstr = ", ".join(map(lambda a: "T%s const &is_%s"%(a.upper(), a) , dim_names))

  template_string = "template<typename POLICY, %s, %s, typename BODY>" % (polstr, setstr)
  fcnargs_string = "%s, BODY body" % isstr
    
  polstr  = ", ".join(map(lambda a: "Policy"+a.upper(), dim_names))
  typestr = ", ".join(map(lambda a: "T"+a.upper(), dim_names))
  varstr = ", ".join(map(lambda a: "is_"+a, dim_names))
  
  print ""
  print "/*!"
  print " * \\brief Execute inner loops policy function."
  print " *"
  print " * This is the default termination case."
  print " */"
  print template_string
  print "RAJA_INLINE void forall%d_policy(Forall%d_Execute_Tag, %s){" % (ndims, ndims, fcnargs_string)
  print ""
  print "  // Create executor object to launch loops"
  print "  Forall%dExecutor<%s, %s> exec;" % (ndims, polstr, typestr)
  print ""
  print "  // Launch loop body"
  print "  exec(%s, body);" % varstr
  print "}"
  print ""

  print ""
  print ""
  
  

def writeUserIface(ndims):
  
  dim_names = getDimNames(ndims)
  
  print "/*!"
  print " * \\brief Provides abstraction of a %d-nested loop" % (ndims)
  print " *"
  print " * Provides index typing, and initial nested policy unwrapping"
  print " */"
  
  args = map(lambda a: "typename Idx%s=Index_type"%a.upper(), dim_names)
  idxstr = ", ".join(args)
  args = map(lambda a: "typename T"+a.upper(), dim_names)
  argstr = ", ".join(args)
  print "template<typename POLICY, %s, %s, typename BODY>" % (idxstr, argstr)
  
  args = map(lambda a: "T%s const &is_%s"%(a.upper(), a), dim_names)
  argstr = ", ".join(args)
  print "RAJA_INLINE void forall%d(%s, BODY body){" % (ndims, argstr)
  
  args = map(lambda a: "T"+a.upper(), dim_names)
  argstr = ", ".join(args)
  
  args = map(lambda a: "is_"+a, dim_names)
  argstr2 = ", ".join(args)
  print "  // extract next policy"
  print "  typedef typename POLICY::NextPolicy             NextPolicy;"
  print "  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;"
  print ""
  print "  // extract each loop's execution policy"
  for d in dim_names:
    print "  typedef typename POLICY::Policy%s                Policy%s;" % (d.upper(), d.upper())
  print ""
  print "  // call 'policy' layer with next policy"

  args = map(lambda a: "Policy"+a.upper(), dim_names)
  polstr =  ", ".join(args)
  args = map(lambda a: "is_"+a, dim_names)
  isetstr = ", ".join(args)
  print "  forall%d_policy<NextPolicy, %s>(NextPolicyTag(), %s, " % (ndims, polstr, isetstr)
  
  args = map(lambda a: "Index_type %s"%a, dim_names)
  argstr = ", ".join(args)
  print "    [=](%s){" % argstr
  
  args = map(lambda a: "Idx%s(%s)"%(a.upper(), a), dim_names)
  argstr = ", ".join(args)
  print "      body(%s);" % argstr
  print "    }"
  print "  );"
  print "}"
  print ""



def main(ndims):
  # ACTUAL SCRIPT ENTRY:
  print """//AUTOGENERATED BY gen_forallN_generic.py
/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */
  
#ifndef RAJA_forallN_generic_HXX__
#define RAJA_forallN_generic_HXX__

#include"config.hxx"
#include"int_datatypes.hxx"

namespace RAJA {

""" 
  ndims_list = range(2,ndims+1)
  
  # Create the policy struct so the user can define loop policies
  print ""
  print "/******************************************************************"
  print " *  ForallN generic policies"
  print " ******************************************************************/"
  print ""
  for n in ndims_list:
    writeForallPolicy(n)

  # Forward declarations of forall_policy functions
  print ""
  print "/******************************************************************"
  print " *  forallN_policy() Foreward declarations"
  print " ******************************************************************/"
  print ""
  for n in ndims_list:
    writeForall_policyForeward(n)

  # Create the default executor
  print ""
  print "/******************************************************************"
  print " *  ForallNExecutor(): Default Executors for loops"
  print " ******************************************************************/"
  print ""
  for n in ndims_list:
    writeForallExecutor(n)

  # Create _policy functions
  print ""
  print "/******************************************************************"
  print " *  forallN_policy(), base execution policiess"
  print " ******************************************************************/"
  print ""
  for n in ndims_list:
    writeForall_policy(n)

  # Create user API call
  print ""
  print "/******************************************************************"
  print " *  forallN User API"
  print " ******************************************************************/"
  print ""  
  for n in ndims_list:
    writeUserIface(n)


  print """

} // namespace RAJA
  
#endif
"""

if __name__ == '__main__':
  main(int(sys.argv[1]))
  
