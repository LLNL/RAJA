#!/usr/bin/env python
#
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# All rights reserved.
#
# This source code cannot be distributed without permission and
# further review from Lawrence Livermore National Laboratory.
#


import sys
from itertools import permutations
from lperm import *


def writeForallPolicy(ndims):

  dim_names = getDimNames(ndims)

  print "// Begin OpenMP Parallel Region"
  print "struct Forall%d_OMP_Parallel_Tag {};" % ndims
  print "template<typename NEXT=Forall%d_Execute>" % ndims
  print "struct Forall%d_OMP_Parallel {" % ndims
  print "  typedef Forall%d_OMP_Parallel_Tag PolicyTag;" % ndims
  print "  typedef NEXT NextPolicy;"
  print "};"
  print ""
  
    

def writeForallExecuteOMP(ndims):

  dim_names = getDimNames(ndims)

  for omp_policy in ['omp_parallel_for_exec', 'omp_for_nowait_exec']:
    for depth in range(2,ndims+1):
    
      remainder_ndims = ndims - depth

      polargs = []
      setargs = []
      args =  map(lambda a: "typename POLICY_"+a.upper(), dim_names[depth:])
      args.extend(map(lambda a: "typename T"+a.upper(), dim_names[depth:]))
      argstr = ", ".join(args)   
      print "// OpenMP Executor with collapse(%d) for %s" % (depth, omp_policy)
      print "template<%s>" % argstr
      
      args =  map(lambda a: "RAJA::"+omp_policy, range(0,depth))
      args.extend(map(lambda a: "POLICY_"+a.upper(), dim_names[depth:]))
      args.extend(map(lambda a: "RAJA::RangeSegment", range(0,depth)))
      args.extend(map(lambda a: "T"+a.upper(), dim_names[depth:]))
      argstr = ", ".join(args)   
      print "class Forall%dExecutor<%s> {" % (ndims, argstr)
      print "  public:  "
      
      # Create collapse(depth) executor function
      print "    template<typename BODY>"
      
      args = map(lambda a: "RAJA::RangeSegment const &is_"+ a, dim_names[0:depth])
      args.extend(map(lambda a: "T%s const &is_%s"%(a.upper(),a), dim_names[depth:ndims]))
      argstr = ", ".join(args)  
      print "    inline void operator()(%s, BODY body) const {" % argstr
  #    print "          printf(\"collapse(%d)\\n\");" % depth
      
      # get begin and end indices each collapsed RangeSegment
      for a in dim_names[0:depth]:
        print "      Index_type const %s_start = is_%s.getBegin();" % (a,a)
        print "      Index_type const %s_end   = is_%s.getEnd();" % (a,a)
        print ""
      
      # Generate nested collapsed for loops
      if omp_policy == 'omp_parallel_for_exec':
        print "#pragma omp parallel for schedule(static) collapse(%d)" % depth
      elif omp_policy == 'omp_for_nowait_exec':
        print "#pragma omp for schedule(static) collapse(%d) nowait" % depth
      indent = ""
      for d in dim_names[0:depth]:
        print "      %sfor(Index_type %s = %s_start;%s < %s_end;++ %s){" % (indent, d, d, d, d, d)
        indent += "  "
      
      # No more inner loops, so call the loop body directly
      if remainder_ndims == 0:
        argstr = argstr = ", ".join(dim_names)
        print "      %sbody(%s);" % (indent, argstr)
      
      # Just one inner loop, so issue a RAJA::forall
      elif remainder_ndims == 1:      
        d = dim_names[depth]
        print "      %sRAJA::forall<POLICY_%s>(is_%s, [=](Index_type %s){" % (indent, d.upper(), d, d)
        argstr = argstr = ", ".join(dim_names)
        print "      %s  body(%s);" % (indent, argstr)
        print "      %s});" % (indent)
      
      # More than one inner loop, so call an inner executor
      else:              
        args = map(lambda a: "is_"+a, dim_names[depth:])
        setstr = ", ".join(args)
        args = map(lambda a: "Index_type "+a, dim_names[depth:])
        argstr = ", ".join(args)      
        print "      %sexec(%s, [=](%s){" % (indent, setstr, argstr)
        argstr = argstr = ", ".join(dim_names)
        print "      %s  body(%s);" % (indent, argstr)
        print "      %s});" % (indent)
      
      # Close out collapsed loops
      argstr = "";
      for d in range(0,depth):
        argstr += "} "
      print "      %s" % argstr
      print "    }"
      
        
      # More than 2 dims: create nested ForallNExecutor
      if remainder_ndims >= 2:
        print ""
        args = map(lambda a: "POLICY_"+(a.upper()), dim_names[depth:])
        polstr = ", ".join(args)
        args = map(lambda a: "T"+(a.upper()), dim_names[depth:])
        argstr = ", ".join(args)
        print "  private:"
        print "    Forall%dExecutor<%s, %s> exec;" % (ndims-depth, polstr, argstr)
      print "};"
      print ""    

  print ""



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
  print " * \\brief OpenMP Parallel Region Section policy function."
  print " */"
  print template_string
  print "RAJA_INLINE void forall%d_policy(Forall%d_OMP_Parallel_Tag, %s){" % (ndims, ndims, fcnargs_string)
  print "  typedef typename POLICY::NextPolicy            NextPolicy;"
  print "  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;"
  print ""
  print "  // create OpenMP Parallel Region"
  print "#ifdef _OPENMP"
  print "#pragma omp parallel"
  print "#endif"
  print "  {"
  print "    // execute the next policy"
  print "    forall%d_policy<NextPolicy, %s>(NextPolicyTag(), %s, body);" % (ndims, polstr, varstr)
  print "  }"
  print "}"
  print ""
  
  


def main(ndims):
  # ACTUAL SCRIPT ENTRY:
  print """//AUTOGENERATED BY gen_forallN_generic.py
/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */
  
#ifndef RAJA_forallN_openmp_HXX__
#define RAJA_forallN_openmp_HXX__

#include<RAJA/config.hxx>
#include<RAJA/int_datatypes.hxx>

namespace RAJA {

""" 
  ndims_list = range(2,ndims+1)
  
  # Create the policy struct so the user can define loop policies
  print ""
  print "/******************************************************************"
  print " *  ForallN OpenMP Parallel Region policies"
  print " ******************************************************************/"
  print ""
  for n in ndims_list:
    writeForallPolicy(n)

  # Create _policy functions
  print ""
  print "/******************************************************************"
  print " *  forallN Executor OpenMP auto-collapse rules"
  print " ******************************************************************/"
  print ""
  for n in ndims_list:
    writeForallExecuteOMP(n)


  # Create _policy functions
  print ""
  print "/******************************************************************"
  print " *  forallN_policy(), OpenMP Parallel Region execution"
  print " ******************************************************************/"
  print ""
  for n in ndims_list:
    writeForall_policy(n)


  print """

} // namespace RAJA
  
#endif
"""

if __name__ == '__main__':
  main(int(sys.argv[1]))
  
