/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining loop permutations for forallN templates.
 *
 ******************************************************************************
 */

#ifndef RAJA_FORALLN_PERMUTATIONS_HPP
#define RAJA_FORALLN_PERMUTATIONS_HPP

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
// For additional details, please also read RAJA/LICENSE.
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


#include "RAJA/internal/LegacyCompatibility.hpp"

namespace RAJA
{

using PERM_I = VarOps::index_sequence<0>;
using PERM_IJ = VarOps::index_sequence<0, 1>;
using PERM_JI = VarOps::index_sequence<1, 0>;
using PERM_IJK = VarOps::index_sequence<0, 1, 2>;
using PERM_IKJ = VarOps::index_sequence<0, 2, 1>;
using PERM_JIK = VarOps::index_sequence<1, 0, 2>;
using PERM_JKI = VarOps::index_sequence<1, 2, 0>;
using PERM_KIJ = VarOps::index_sequence<2, 0, 1>;
using PERM_KJI = VarOps::index_sequence<2, 1, 0>;
using PERM_IJKL = VarOps::index_sequence<0, 1, 2, 3>;
using PERM_IJLK = VarOps::index_sequence<0, 1, 3, 2>;
using PERM_IKJL = VarOps::index_sequence<0, 2, 1, 3>;
using PERM_IKLJ = VarOps::index_sequence<0, 2, 3, 1>;
using PERM_ILJK = VarOps::index_sequence<0, 3, 1, 2>;
using PERM_ILKJ = VarOps::index_sequence<0, 3, 2, 1>;
using PERM_JIKL = VarOps::index_sequence<1, 0, 2, 3>;
using PERM_JILK = VarOps::index_sequence<1, 0, 3, 2>;
using PERM_JKIL = VarOps::index_sequence<1, 2, 0, 3>;
using PERM_JKLI = VarOps::index_sequence<1, 2, 3, 0>;
using PERM_JLIK = VarOps::index_sequence<1, 3, 0, 2>;
using PERM_JLKI = VarOps::index_sequence<1, 3, 2, 0>;
using PERM_KIJL = VarOps::index_sequence<2, 0, 1, 3>;
using PERM_KILJ = VarOps::index_sequence<2, 0, 3, 1>;
using PERM_KJIL = VarOps::index_sequence<2, 1, 0, 3>;
using PERM_KJLI = VarOps::index_sequence<2, 1, 3, 0>;
using PERM_KLIJ = VarOps::index_sequence<2, 3, 0, 1>;
using PERM_KLJI = VarOps::index_sequence<2, 3, 1, 0>;
using PERM_LIJK = VarOps::index_sequence<3, 0, 1, 2>;
using PERM_LIKJ = VarOps::index_sequence<3, 0, 2, 1>;
using PERM_LJIK = VarOps::index_sequence<3, 1, 0, 2>;
using PERM_LJKI = VarOps::index_sequence<3, 1, 2, 0>;
using PERM_LKIJ = VarOps::index_sequence<3, 2, 0, 1>;
using PERM_LKJI = VarOps::index_sequence<3, 2, 1, 0>;
using PERM_IJKLM = VarOps::index_sequence<0, 1, 2, 3, 4>;
using PERM_IJKML = VarOps::index_sequence<0, 1, 2, 4, 3>;
using PERM_IJLKM = VarOps::index_sequence<0, 1, 3, 2, 4>;
using PERM_IJLMK = VarOps::index_sequence<0, 1, 3, 4, 2>;
using PERM_IJMKL = VarOps::index_sequence<0, 1, 4, 2, 3>;
using PERM_IJMLK = VarOps::index_sequence<0, 1, 4, 3, 2>;
using PERM_IKJLM = VarOps::index_sequence<0, 2, 1, 3, 4>;
using PERM_IKJML = VarOps::index_sequence<0, 2, 1, 4, 3>;
using PERM_IKLJM = VarOps::index_sequence<0, 2, 3, 1, 4>;
using PERM_IKLMJ = VarOps::index_sequence<0, 2, 3, 4, 1>;
using PERM_IKMJL = VarOps::index_sequence<0, 2, 4, 1, 3>;
using PERM_IKMLJ = VarOps::index_sequence<0, 2, 4, 3, 1>;
using PERM_ILJKM = VarOps::index_sequence<0, 3, 1, 2, 4>;
using PERM_ILJMK = VarOps::index_sequence<0, 3, 1, 4, 2>;
using PERM_ILKJM = VarOps::index_sequence<0, 3, 2, 1, 4>;
using PERM_ILKMJ = VarOps::index_sequence<0, 3, 2, 4, 1>;
using PERM_ILMJK = VarOps::index_sequence<0, 3, 4, 1, 2>;
using PERM_ILMKJ = VarOps::index_sequence<0, 3, 4, 2, 1>;
using PERM_IMJKL = VarOps::index_sequence<0, 4, 1, 2, 3>;
using PERM_IMJLK = VarOps::index_sequence<0, 4, 1, 3, 2>;
using PERM_IMKJL = VarOps::index_sequence<0, 4, 2, 1, 3>;
using PERM_IMKLJ = VarOps::index_sequence<0, 4, 2, 3, 1>;
using PERM_IMLJK = VarOps::index_sequence<0, 4, 3, 1, 2>;
using PERM_IMLKJ = VarOps::index_sequence<0, 4, 3, 2, 1>;
using PERM_JIKLM = VarOps::index_sequence<1, 0, 2, 3, 4>;
using PERM_JIKML = VarOps::index_sequence<1, 0, 2, 4, 3>;
using PERM_JILKM = VarOps::index_sequence<1, 0, 3, 2, 4>;
using PERM_JILMK = VarOps::index_sequence<1, 0, 3, 4, 2>;
using PERM_JIMKL = VarOps::index_sequence<1, 0, 4, 2, 3>;
using PERM_JIMLK = VarOps::index_sequence<1, 0, 4, 3, 2>;
using PERM_JKILM = VarOps::index_sequence<1, 2, 0, 3, 4>;
using PERM_JKIML = VarOps::index_sequence<1, 2, 0, 4, 3>;
using PERM_JKLIM = VarOps::index_sequence<1, 2, 3, 0, 4>;
using PERM_JKLMI = VarOps::index_sequence<1, 2, 3, 4, 0>;
using PERM_JKMIL = VarOps::index_sequence<1, 2, 4, 0, 3>;
using PERM_JKMLI = VarOps::index_sequence<1, 2, 4, 3, 0>;
using PERM_JLIKM = VarOps::index_sequence<1, 3, 0, 2, 4>;
using PERM_JLIMK = VarOps::index_sequence<1, 3, 0, 4, 2>;
using PERM_JLKIM = VarOps::index_sequence<1, 3, 2, 0, 4>;
using PERM_JLKMI = VarOps::index_sequence<1, 3, 2, 4, 0>;
using PERM_JLMIK = VarOps::index_sequence<1, 3, 4, 0, 2>;
using PERM_JLMKI = VarOps::index_sequence<1, 3, 4, 2, 0>;
using PERM_JMIKL = VarOps::index_sequence<1, 4, 0, 2, 3>;
using PERM_JMILK = VarOps::index_sequence<1, 4, 0, 3, 2>;
using PERM_JMKIL = VarOps::index_sequence<1, 4, 2, 0, 3>;
using PERM_JMKLI = VarOps::index_sequence<1, 4, 2, 3, 0>;
using PERM_JMLIK = VarOps::index_sequence<1, 4, 3, 0, 2>;
using PERM_JMLKI = VarOps::index_sequence<1, 4, 3, 2, 0>;
using PERM_KIJLM = VarOps::index_sequence<2, 0, 1, 3, 4>;
using PERM_KIJML = VarOps::index_sequence<2, 0, 1, 4, 3>;
using PERM_KILJM = VarOps::index_sequence<2, 0, 3, 1, 4>;
using PERM_KILMJ = VarOps::index_sequence<2, 0, 3, 4, 1>;
using PERM_KIMJL = VarOps::index_sequence<2, 0, 4, 1, 3>;
using PERM_KIMLJ = VarOps::index_sequence<2, 0, 4, 3, 1>;
using PERM_KJILM = VarOps::index_sequence<2, 1, 0, 3, 4>;
using PERM_KJIML = VarOps::index_sequence<2, 1, 0, 4, 3>;
using PERM_KJLIM = VarOps::index_sequence<2, 1, 3, 0, 4>;
using PERM_KJLMI = VarOps::index_sequence<2, 1, 3, 4, 0>;
using PERM_KJMIL = VarOps::index_sequence<2, 1, 4, 0, 3>;
using PERM_KJMLI = VarOps::index_sequence<2, 1, 4, 3, 0>;
using PERM_KLIJM = VarOps::index_sequence<2, 3, 0, 1, 4>;
using PERM_KLIMJ = VarOps::index_sequence<2, 3, 0, 4, 1>;
using PERM_KLJIM = VarOps::index_sequence<2, 3, 1, 0, 4>;
using PERM_KLJMI = VarOps::index_sequence<2, 3, 1, 4, 0>;
using PERM_KLMIJ = VarOps::index_sequence<2, 3, 4, 0, 1>;
using PERM_KLMJI = VarOps::index_sequence<2, 3, 4, 1, 0>;
using PERM_KMIJL = VarOps::index_sequence<2, 4, 0, 1, 3>;
using PERM_KMILJ = VarOps::index_sequence<2, 4, 0, 3, 1>;
using PERM_KMJIL = VarOps::index_sequence<2, 4, 1, 0, 3>;
using PERM_KMJLI = VarOps::index_sequence<2, 4, 1, 3, 0>;
using PERM_KMLIJ = VarOps::index_sequence<2, 4, 3, 0, 1>;
using PERM_KMLJI = VarOps::index_sequence<2, 4, 3, 1, 0>;
using PERM_LIJKM = VarOps::index_sequence<3, 0, 1, 2, 4>;
using PERM_LIJMK = VarOps::index_sequence<3, 0, 1, 4, 2>;
using PERM_LIKJM = VarOps::index_sequence<3, 0, 2, 1, 4>;
using PERM_LIKMJ = VarOps::index_sequence<3, 0, 2, 4, 1>;
using PERM_LIMJK = VarOps::index_sequence<3, 0, 4, 1, 2>;
using PERM_LIMKJ = VarOps::index_sequence<3, 0, 4, 2, 1>;
using PERM_LJIKM = VarOps::index_sequence<3, 1, 0, 2, 4>;
using PERM_LJIMK = VarOps::index_sequence<3, 1, 0, 4, 2>;
using PERM_LJKIM = VarOps::index_sequence<3, 1, 2, 0, 4>;
using PERM_LJKMI = VarOps::index_sequence<3, 1, 2, 4, 0>;
using PERM_LJMIK = VarOps::index_sequence<3, 1, 4, 0, 2>;
using PERM_LJMKI = VarOps::index_sequence<3, 1, 4, 2, 0>;
using PERM_LKIJM = VarOps::index_sequence<3, 2, 0, 1, 4>;
using PERM_LKIMJ = VarOps::index_sequence<3, 2, 0, 4, 1>;
using PERM_LKJIM = VarOps::index_sequence<3, 2, 1, 0, 4>;
using PERM_LKJMI = VarOps::index_sequence<3, 2, 1, 4, 0>;
using PERM_LKMIJ = VarOps::index_sequence<3, 2, 4, 0, 1>;
using PERM_LKMJI = VarOps::index_sequence<3, 2, 4, 1, 0>;
using PERM_LMIJK = VarOps::index_sequence<3, 4, 0, 1, 2>;
using PERM_LMIKJ = VarOps::index_sequence<3, 4, 0, 2, 1>;
using PERM_LMJIK = VarOps::index_sequence<3, 4, 1, 0, 2>;
using PERM_LMJKI = VarOps::index_sequence<3, 4, 1, 2, 0>;
using PERM_LMKIJ = VarOps::index_sequence<3, 4, 2, 0, 1>;
using PERM_LMKJI = VarOps::index_sequence<3, 4, 2, 1, 0>;
using PERM_MIJKL = VarOps::index_sequence<4, 0, 1, 2, 3>;
using PERM_MIJLK = VarOps::index_sequence<4, 0, 1, 3, 2>;
using PERM_MIKJL = VarOps::index_sequence<4, 0, 2, 1, 3>;
using PERM_MIKLJ = VarOps::index_sequence<4, 0, 2, 3, 1>;
using PERM_MILJK = VarOps::index_sequence<4, 0, 3, 1, 2>;
using PERM_MILKJ = VarOps::index_sequence<4, 0, 3, 2, 1>;
using PERM_MJIKL = VarOps::index_sequence<4, 1, 0, 2, 3>;
using PERM_MJILK = VarOps::index_sequence<4, 1, 0, 3, 2>;
using PERM_MJKIL = VarOps::index_sequence<4, 1, 2, 0, 3>;
using PERM_MJKLI = VarOps::index_sequence<4, 1, 2, 3, 0>;
using PERM_MJLIK = VarOps::index_sequence<4, 1, 3, 0, 2>;
using PERM_MJLKI = VarOps::index_sequence<4, 1, 3, 2, 0>;
using PERM_MKIJL = VarOps::index_sequence<4, 2, 0, 1, 3>;
using PERM_MKILJ = VarOps::index_sequence<4, 2, 0, 3, 1>;
using PERM_MKJIL = VarOps::index_sequence<4, 2, 1, 0, 3>;
using PERM_MKJLI = VarOps::index_sequence<4, 2, 1, 3, 0>;
using PERM_MKLIJ = VarOps::index_sequence<4, 2, 3, 0, 1>;
using PERM_MKLJI = VarOps::index_sequence<4, 2, 3, 1, 0>;
using PERM_MLIJK = VarOps::index_sequence<4, 3, 0, 1, 2>;
using PERM_MLIKJ = VarOps::index_sequence<4, 3, 0, 2, 1>;
using PERM_MLJIK = VarOps::index_sequence<4, 3, 1, 0, 2>;
using PERM_MLJKI = VarOps::index_sequence<4, 3, 1, 2, 0>;
using PERM_MLKIJ = VarOps::index_sequence<4, 3, 2, 0, 1>;
using PERM_MLKJI = VarOps::index_sequence<4, 3, 2, 1, 0>;
}

#endif /* RAJA_FORALLN_PERMUTATIONS_HPP */
