#!/usr/bin/env python

notice= """
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-689114
// 
// All rights reserved.
// 
// This file is part of RAJA. 
// 
// For additional details, please also read raja/README-license.txt.
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
"""



import sys
from itertools import permutations
from lperm import *

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
  print "RAJA_INLINE\nvoid forallN(%s, BODY const &body){" % (argstr)
  
  args = map(lambda a: "T"+a.upper(), dim_names)
  argstr = ", ".join(args)
  
  args = map(lambda a: "is_"+a, dim_names)
  argstr2 = ", ".join(args)
  print "  // extract next policy"
  print "  typedef typename POLICY::NextPolicy             NextPolicy;"
  print "  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;"
  print ""
  print "  // extract each loop's execution policy"
  print "  using ExecPolicies = typename POLICY::ExecPolicies;"
  for i in range(0,ndims):
    d = dim_names[i]
    print "  using Policy%s = typename std::tuple_element<%d, typename ExecPolicies::tuple>::type;" % (d.upper(), i)

    
  args = map(lambda a: "Idx%s"%(a.upper()), dim_names)
  argstr = ", ".join(args)  
  print """  
  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, %s> IDX_CONV;
  IDX_CONV lamb(body);
""" % argstr


  print "  // call policy layer with next policy"

  args = map(lambda a: "Policy"+a.upper(), dim_names)
  polstr =  ", ".join(args)
  args = map(lambda a: "is_"+a, dim_names)
  isetstr = ", ".join(args)
  
  outstr = "  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(), lamb"
  for d in dim_names:
    outstr += ",\n    ForallN_PolicyPair<Policy%s, T%s>(is_%s)" % (d.upper(), d.upper(), d)
  print outstr + ");"
  
  print "}"
  print ""
  

def writeUserIfaceAutoDeduce(ndims):
  
  dim_names = getDimNames(ndims)
  
  
  
  args = map(lambda a: "typename Idx%s"%a.upper(), dim_names)
  idxstr = ", ".join(args)
  args = map(lambda a: "typename T"+a.upper(), dim_names)
  argstr = ", ".join(args)
  print "template<typename POLICY, %s, %s, typename R, typename BODY>" % (idxstr, argstr)
  
  args = map(lambda a: "T%s const &is_%s"%(a.upper(), a), dim_names)
  argstr = ", ".join(args)
  args = map(lambda a: "Idx%s"%(a.upper()), dim_names)  
  idxstr = ", ".join(args)
  print "RAJA_INLINE\nvoid forallN_expanded(%s, BODY const &body, R (BODY::*mf)(%s) const){" % (argstr, idxstr)
  
  args = map(lambda a: "T"+a.upper(), dim_names)
  argstr = ", ".join(args)
  
  args = map(lambda a: "is_"+a, dim_names)
  argstr2 = ", ".join(args)
  print "  // extract next policy"
  print "  typedef typename POLICY::NextPolicy             NextPolicy;"
  print "  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;"
  print ""
  print "  // extract each loop's execution policy"
  print "  using ExecPolicies = typename POLICY::ExecPolicies;"
  for i in range(0,ndims):
    d = dim_names[i]
    print "  using Policy%s = typename std::tuple_element<%d, typename ExecPolicies::tuple>::type;" % (d.upper(), i)

    
  args = map(lambda a: "Idx%s"%(a.upper()), dim_names)
  argstr = ", ".join(args)  
  print """  
  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, %s> IDX_CONV;
  IDX_CONV lamb(body);
""" % argstr


  print "  // call policy layer with next policy"

  args = map(lambda a: "Policy"+a.upper(), dim_names)
  polstr =  ", ".join(args)
  args = map(lambda a: "is_"+a, dim_names)
  isetstr = ", ".join(args)
  
  outstr = "  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(), lamb"
  for d in dim_names:
    outstr += ",\n    ForallN_PolicyPair<Policy%s, T%s>(is_%s)" % (d.upper(), d.upper(), d)
  print outstr + ");"
  
  print "}"
  print ""
  
  print "/*!"
  print " * \\brief Provides abstraction of a %d-nested loop" % (ndims)
  print " *"
  print " * Provides index typing, and initial nested policy unwrapping"
  print " */"
  args = map(lambda a: "typename T"+a.upper(), dim_names)
  templatestr = ", ".join(args)
  args = map(lambda a: "T%s const &is_%s"%(a.upper(), a), dim_names)
  paramstr = ", ".join(args)
  args = map(lambda a: "is_"+a, dim_names)
  argstr = ", ".join(args)
  print """template<typename POLICY, %s, typename BODY>
RAJA_INLINE 
void forallN(%s, BODY body){
  forallN_expanded<POLICY>(%s, body, &BODY::operator());
}""" % (templatestr, paramstr, argstr) 
  print ""



def main(ndims):
  # ACTUAL SCRIPT ENTRY:
  print """//AUTOGENERATED BY gen_forallN_generic.py
%s
  
#ifndef RAJA_forallN_generic_HXX__
#define RAJA_forallN_generic_HXX__

#include "forallN_generic_lf.hxx"

namespace RAJA {

""" % notice
  ndims_list = range(1,ndims+1)
  
  # Create user API call
  for n in ndims_list:
    writeUserIface(n)


  print """

} // namespace RAJA
  
#endif
"""

if __name__ == '__main__':
  main(int(sys.argv[1]))
  
