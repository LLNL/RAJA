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

  print ""
  print "/******************************************************************"
  print " *  Policy base class, forall%d()" % ndims
  print " ******************************************************************/"
  print ""

  dim_names = getDimNames(ndims)

  print "// Execute (Termination default)"
  print "struct Forall%d_Execute_Tag {};" % ndims
  print "struct Forall%d_Execute {" % (ndims)
  print "  typedef Forall%d_Execute_Tag PolicyTag;" % ndims
  print "};"
  print ""
  
  print "// Starting (outer) policy for all forall%d policies" % ndims
  polstr = ", ".join(map(lambda a: "typename POL_%s=RAJA::seq_exec"%(a.upper()), dim_names))
  print "template<%s, typename NEXT=Forall%d_Execute>" % (polstr, ndims)
  print "struct Forall%d_Policy {" % (ndims)
  print "  typedef NEXT NextPolicy;"
  for dim in dim_names:
    print "  typedef POL_%s Policy%s;" % (dim.upper(), dim.upper())
  print "};"
  print ""
  
  print "// Interchange loop order given permutation"
  print "struct Forall%d_Permute_Tag {};" % ndims
  print "template<typename LOOP_ORDER, typename NEXT=Forall%d_Execute>" % (ndims)
  print "struct Forall%d_Permute {" % (ndims)
  print "  typedef Forall%d_Permute_Tag PolicyTag;" % ndims
  print "  typedef NEXT NextPolicy;"
  print "  typedef LOOP_ORDER LoopOrder;"
  print "};"
  print ""
  
  print "// Begin OpenMP Parallel Region"
  print "struct Forall%d_OMP_Parallel_Tag {};" % ndims
  print "template<typename NEXT=Forall%d_Execute>" % ndims
  print "struct Forall%d_OMP_Parallel {" % ndims
  print "  typedef Forall%d_OMP_Parallel_Tag PolicyTag;" % ndims
  print "  typedef NEXT NextPolicy;"
  print "};"
  print ""
  
  print "// Tiling Policy"
  print "struct Forall%d_Tile_Tag {};" % ndims
  args = map(lambda a: "typename TILE_"+(a.upper()), dim_names)
  argstr = ", ".join(args)
  print "template<%s, typename NEXT=Forall%d_Execute>" % (argstr, ndims)
  print "struct Forall%d_Tile {" % ndims
  print "  typedef Forall%d_Tile_Tag PolicyTag;" % ndims
  print "  typedef NEXT NextPolicy;"
  for dim in dim_names:
    print "  typedef TILE_%s Tile%s;" % (dim.upper(), dim.upper())
  print "};"
  print ""
  

    

def writeForallExecutor(ndims):

  print ""
  print "/******************************************************************"
  print " *  Forall%dExecutor(): Default Executor for loops" % ndims
  print " ******************************************************************/"
  print ""

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
  print "    RAJA::forall<POLICY_I>(is_i, RAJA_LAMBDA(Index_type i){"
  if ndims == 2:  # 2 dimension termination case:
    print "      RAJA::forall<POLICY_J>(is_j, RAJA_LAMBDA(Index_type j){"
  else: # more than 2 dimensions, we just peel off the outer loop, and call an N-1 executor
    args = map(lambda a: "is_"+a, dim_names[1:])
    setstr = ", ".join(args)
    args = map(lambda a: "Index_type "+a, dim_names[1:])
    idxstr = ", ".join(args)  
    print "      exec(%s, RAJA_LAMBDA(%s){" % (setstr, idxstr)
  
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


def writeForallExecuteOMP(ndims):

  dim_names = getDimNames(ndims)
  
  print ""
  print "/******************************************************************"
  print " *  OpenMP Auto-Collapsing Executors for forall%d()" % ndims
  print " ******************************************************************/"
  print ""
  print "#ifdef _OPENMP"
  print ""

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
        print "      %sRAJA::forall<POLICY_%s>(is_%s, RAJA_LAMBDA(Index_type %s){" % (indent, d.upper(), d, d)
        argstr = argstr = ", ".join(dim_names)
        print "      %s  body(%s);" % (indent, argstr)
        print "      %s});" % (indent)
      
      # More than one inner loop, so call an inner executor
      else:              
        args = map(lambda a: "is_"+a, dim_names[depth:])
        setstr = ", ".join(args)
        args = map(lambda a: "Index_type "+a, dim_names[depth:])
        argstr = ", ".join(args)      
        print "      %sexec(%s, RAJA_LAMBDA(%s){" % (indent, setstr, argstr)
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
  print "#endif // _OPENMP"
  print ""


def writeForallPermutations(ndims):
  
  dim_names = getDimNames(ndims)

  # Create boiler-plate used for all _policy() fcns
  polstr = ", ".join(map(lambda a: "typename Policy"+a.upper(), dim_names))
  setstr = ", ".join(map(lambda a: "typename T"+a.upper(), dim_names))
  isstr = ", ".join(map(lambda a: "T%s const &is_%s"%(a.upper(), a) , dim_names))
  
  template_string = "template<typename POLICY, %s, %s, typename BODY>" % (polstr, setstr)
  fcnargs_string = "%s, BODY body" % isstr


  print ""
  print "/******************************************************************"
  print " *  forall%d_permute(): Permutation function overloads" % ndims
  print " ******************************************************************/"
  print ""
  
  # Loop over each permutation specialization
  perms = getDimPerms(dim_names)
  for perm in perms:
    # get enumeration name
    enum_name = getEnumName(perm)
    
    # print function declaration
    print template_string
    print "RAJA_INLINE void forall%d_permute(%s, %s){" % (ndims, enum_name, fcnargs_string)
    print "  typedef typename POLICY::NextPolicy            NextPolicy;"
    print "  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;"
    
    
    # Call next policy with permuted indices and policies
    p_polstr  = ", ".join(map(lambda a: "Policy"+a.upper(), perm))
    p_varstr  = ", ".join(map(lambda a: "is_"+a, perm))
    p_argstr  = ", ".join(map(lambda a: "Index_type "+a, perm))
    argstr    = ", ".join(dim_names)

    print ""
    print "  // Call next policy with permuted indices and policies"
    print "  forall%d_policy<NextPolicy, %s>(NextPolicyTag(), %s," % (ndims, p_polstr, p_varstr)
    print "    RAJA_LAMBDA(%s){" % (p_argstr)
    print "      // Call body with non-permuted indices"
    print "      body(%s);" % (argstr)
    print "    });"
    print "}"
    print ""




def writeForall_policyForeward(ndims):
  
  dim_names = getDimNames(ndims)
  
  # Create boiler-plate used for all _policy() fcns
  polstr = ", ".join(map(lambda a: "typename Policy"+a.upper(), dim_names))
  setstr = ", ".join(map(lambda a: "typename T"+a.upper(), dim_names))
  isstr = ", ".join(map(lambda a: "T%s const &is_%s"%(a.upper(), a) , dim_names))
  
  template_string = "template<typename POLICY, %s, %s, typename BODY>" % (polstr, setstr)
  fcnargs_string = "%s, BODY body" % isstr
  
  
  print ""
  print "/******************************************************************"
  print " *  forall%d_policy() Foreward declarations" % ndims
  print " ******************************************************************/"
  
  print ""
  print template_string
  print "RAJA_INLINE void forall%d_policy(Forall%d_Execute_Tag, %s);" % (ndims, ndims, fcnargs_string)

  print ""
  print template_string
  print "RAJA_INLINE void forall%d_policy(Forall%d_Permute_Tag, %s);" % (ndims, ndims, fcnargs_string)

  print ""
  print template_string
  print "RAJA_INLINE void forall%d_policy(Forall%d_OMP_Parallel_Tag, %s);" % (ndims, ndims, fcnargs_string)

  print ""
  print template_string
  print "RAJA_INLINE void forall%d_policy(Forall%d_Tile_Tag, %s);" % (ndims, ndims, fcnargs_string)

  print ""


def writeForall_policy(ndims):
  
  dim_names = getDimNames(ndims)
  
  # Create boiler-plate used for all _policy() fcns
  polstr = ", ".join(map(lambda a: "typename Policy"+a.upper(), dim_names))
  setstr = ", ".join(map(lambda a: "typename T"+a.upper(), dim_names))
  isstr = ", ".join(map(lambda a: "T%s const &is_%s"%(a.upper(), a) , dim_names))

  template_string = "    template<typename POLICY, %s, %s, typename BODY>" % (polstr, setstr)
  fcnargs_string = "%s, BODY body" % isstr
  
  
  print ""
  print "/******************************************************************"
  print " *  forall%d_policy() Policy Layer, overloads for policy tags" % ndims
  print " ******************************************************************/"
  print ""
  
  polstr  = ", ".join(map(lambda a: "Policy"+a.upper(), dim_names))
  typestr = ", ".join(map(lambda a: "T"+a.upper(), dim_names))
  varstr = ", ".join(map(lambda a: "is_"+a, dim_names))
  
  print ""
  print "/**"
  print " * Execute inner loops policy function."
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
  print "/**"
  print " * Permutation policy function."
  print " * Provides loop interchange."
  print " */"
  print template_string
  print "RAJA_INLINE void forall%d_policy(Forall%d_Permute_Tag, %s){" % (ndims, ndims, fcnargs_string)
  
  print "  // Get the loop permutation"
  print "  typedef typename POLICY::LoopOrder LoopOrder;"
  print ""
  print "  // Call loop interchange overload to re-wrire indicies and policies"
  print "  forall%d_permute<POLICY, %s>(LoopOrder(), %s, body);" % (ndims, polstr, varstr)
  print "}"
  print ""

  
  print ""
  print "/**"
  print " * OpenMP Parallel Region Section policy function."
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
  
  
  print ""
  print "/**"
  print " * Tiling policy function."
  print " */"
  print template_string
  print "RAJA_INLINE void forall%d_policy(Forall%d_Tile_Tag, %s){" % (ndims, ndims, fcnargs_string)
  print "  typedef typename POLICY::NextPolicy            NextPolicy;"
  print "  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;"
  for d in dim_names:
    print "  typedef typename POLICY::Tile%s Tile%s;" % (d.upper(), d.upper())
  print ""
  print "  // execute the next policy"
  
  indent = ""
  close_paren = []
  for d in dim_names:
    print "%s      forall_tile(Tile%s(), is_%s, [=](RAJA::RangeSegment is_%s%s){" % (indent, d.upper(), d, d, d)
    close_paren.append(indent + "      });")
    indent += "  "

  # call body with tiled index sets
  t_varstr = ", ".join(map(lambda a: "is_"+a+a, dim_names))
  print "%s  forall%d_policy<NextPolicy, %s>(NextPolicyTag(), %s, body);" % (indent, ndims, polstr, t_varstr)

  # close forall parenthesis
  close_paren.reverse()
  for c in close_paren:
    print c

  print "}"
  print ""
  print ""
  
  

def writeUserIface(ndims):
  
  dim_names = getDimNames(ndims)
  
  print ""
  print "/******************************************************************"
  print " * forall%d(), User interface" % ndims
  print " * Provides index typing, and initial nested policy unwrapping"
  print " ******************************************************************/"
  print ""
  
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



ndims = int(sys.argv[1])


# ACTUAL SCRIPT ENTRY:
print """//AUTOGENERATED BY genForallN.py
/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */
  
#ifndef RAJA_DOMAIN_FORALL%d_HXX__
#define RAJA_DOMAIN_FORALL%d_HXX__

#include<RAJA/RAJA.hxx>
#include<RAJA/Tile.hxx>

namespace RAJA {

""" % (ndims, ndims)



# Create the policy struct so the user can define loop policies
writeForallPolicy(ndims)




writeForall_policyForeward(ndims)

# Create the default executor
writeForallExecutor(ndims)

# Create the OpenMP collapse() executors
writeForallExecuteOMP(ndims)

# Create all permutation policies
writeForallPermutations(ndims)

# Create _policy functions
writeForall_policy(ndims)

# Create user API call
writeUserIface(ndims)


print """

} // namespace RAJA
  
#endif
"""

