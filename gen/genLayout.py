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

def writeEnumDecl(ndims_list):

  print """
  
/******************************************************************
 *  Permutation tags
 ******************************************************************/
 
"""

  for ndims in ndims_list:
    # Get names of each permutation    
    enum_names = getEnumNames(ndims)
    
    # Write an enum for each permutation
    for enum in enum_names:
      print "struct %s {};" % enum          
    continue
    
  print ""

       
  
def writeLayoutImpl(ndims_list):

  for ndims in ndims_list:
    dim_names = getDimNames(ndims)
    
    print ""
    print "/******************************************************************"
    print " *  Implementation for Layout%dD" % ndims
    print " ******************************************************************/"
    print ""
                
    # Loop over each permutation specialization
    perms = getDimPerms(dim_names)
    for perm in perms:
      # get enumeration name
      enum = getEnumName(perm)
      
      # Start the partial specialization
      args = map(lambda a: "typename Idx%s"%a.upper(), dim_names)
      argstr = ", ".join(args)
      print "template<typename IdxLin, %s>" % argstr
      
      args = map(lambda a: "Idx%s"%a.upper(), dim_names)
      argstr = ", ".join(args)
      print "struct Layout<IdxLin, %s, %s> {" % (enum, argstr)
    
      # Create typedefs to capture the template parameters
      print "  typedef %s Permutation;" % enum
      print "  typedef IdxLin IndexLinear;"
      for a in dim_names:
        print "  typedef Idx%s Index%s;" % (a.upper(), a.upper())
      print ""


      # Add local variables
      args = map(lambda a: "Index_type const size_"+a, dim_names)
      for arg in args:
        print "  %s;" % arg
        
      # Add stride variables
      print ""
      args = map(lambda a: "Index_type const stride_"+a, dim_names)
      for arg in args:
        print "  %s;" % arg
      print ""
    
      # Define constructor
      args = map(lambda a: "Index_type n"+a, dim_names)
      argstr = ", ".join(args)    
      print "  RAJA_INLINE RAJA_HOST_DEVICE constexpr Layout(%s):" % (argstr)    
      
      # initialize size of each dim
      args = map(lambda a: "size_%s(n%s)"%(a,a), dim_names)
      
      # initialize stride of each dim      
      for i in range(0,ndims):
        remain = perm[i+1:]
        if len(remain) > 0:
          remain = map(lambda a: "n"+a, remain)
          stride = "stride_%s(%s)" % ( perm[i],  "*".join(remain) )          
        else:
          stride = "stride_%s(1)" % ( perm[i] )
        args.append(stride)
      args.sort()
          
      # output all initializers
      argstr = ", ".join(args)
      print "    %s" % argstr                    
      print "  {}"
      print ""
      
      

      # Define () Operator, the indices -> linear function
      args = map(lambda a: "Idx%s %s"%(a.upper(), a) , dim_names)
      argstr = ", ".join(args)   
      idxparts = []
      for i in range(0,ndims):
        remain = perm[i+1:]        
        if len(remain) > 0:
          idxparts.append("convertIndex<Index_type>(%s)*stride_%s" % (perm[i], perm[i]))
        else:
          idxparts.append("convertIndex<Index_type>(%s)" % perm[i])
      idx = " + ".join(idxparts)  

      print "  RAJA_INLINE RAJA_HOST_DEVICE constexpr IdxLin operator()(%s) const {" % (argstr)
      print "    return convertIndex<IdxLin>(" + idx + ");"
      print "  }"
      print ""
               
               
                 
      # Define the linear->indices functions      
      args = map(lambda a: "Idx%s &%s"%(a.upper(), a), dim_names)
      argstr = ", ".join(args)
      print "  RAJA_INLINE RAJA_HOST_DEVICE void toIndices(IdxLin lin, %s) const {" % (argstr)
      print "    constexpr Index_type linear = convertIndex<Index_type>(lin);"
      for i in range(0, ndims):
        idx = perm[i]
        prod = "*".join(map(lambda a: "size_%s"%a, perm[i+1 : ndims]))
        if prod != '':
          print "    Index_type _%s = linear / (%s);" % (idx, prod)
          print "    %s = Idx%s(_%s);" % (idx, idx.upper(), idx)
          print "    linear -= _%s*(%s);" % (idx, prod)
      print "    %s = Idx%s(linear);" % (perm[ndims-1], perm[ndims-1].upper()) 
      print "  }"
      
      # Close out class
      print "};"

      print ""    
      print ""          



def main(ndims):
  print """//AUTOGENERATED BY genLayout.py
%s
  
#ifndef RAJA_LAYOUT_HXX__
#define RAJA_LAYOUT_HXX__

#include "RAJA/IndexValue.hxx"

namespace RAJA {

/******************************************************************
 *  Generic prototype for all Layouts
 ******************************************************************/

template<typename IdxLin, typename Perm, typename ... IdxList>
struct Layout {};

""" % notice

  ndims_list = range(1,ndims+1)

  # Dump all declarations (with documentation, etc)
  writeEnumDecl(ndims_list)

  # Dump all implementations and specializations
  writeLayoutImpl(ndims_list)

  print """

} // namespace RAJA

#endif
  """

if __name__ == '__main__':
  main(int(sys.argv[1]))
  
