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


#include "camp/camp.hpp"

#include <array>

namespace RAJA
{


template <camp::idx_t... Indices>
constexpr auto perm_as_array(camp::idx_seq<Indices...>)
    -> std::array<Index_type, sizeof...(Indices)>
{
  return std::array<Index_type, sizeof...(Indices)>{Indices...};
}

using PERM_I = camp::idx_seq<0>;
using PERM_IJ = camp::idx_seq<0, 1>;
using PERM_JI = camp::idx_seq<1, 0>;
using PERM_IJK = camp::idx_seq<0, 1, 2>;
using PERM_IKJ = camp::idx_seq<0, 2, 1>;
using PERM_JIK = camp::idx_seq<1, 0, 2>;
using PERM_JKI = camp::idx_seq<1, 2, 0>;
using PERM_KIJ = camp::idx_seq<2, 0, 1>;
using PERM_KJI = camp::idx_seq<2, 1, 0>;
using PERM_IJKL = camp::idx_seq<0, 1, 2, 3>;
using PERM_IJLK = camp::idx_seq<0, 1, 3, 2>;
using PERM_IKJL = camp::idx_seq<0, 2, 1, 3>;
using PERM_IKLJ = camp::idx_seq<0, 2, 3, 1>;
using PERM_ILJK = camp::idx_seq<0, 3, 1, 2>;
using PERM_ILKJ = camp::idx_seq<0, 3, 2, 1>;
using PERM_JIKL = camp::idx_seq<1, 0, 2, 3>;
using PERM_JILK = camp::idx_seq<1, 0, 3, 2>;
using PERM_JKIL = camp::idx_seq<1, 2, 0, 3>;
using PERM_JKLI = camp::idx_seq<1, 2, 3, 0>;
using PERM_JLIK = camp::idx_seq<1, 3, 0, 2>;
using PERM_JLKI = camp::idx_seq<1, 3, 2, 0>;
using PERM_KIJL = camp::idx_seq<2, 0, 1, 3>;
using PERM_KILJ = camp::idx_seq<2, 0, 3, 1>;
using PERM_KJIL = camp::idx_seq<2, 1, 0, 3>;
using PERM_KJLI = camp::idx_seq<2, 1, 3, 0>;
using PERM_KLIJ = camp::idx_seq<2, 3, 0, 1>;
using PERM_KLJI = camp::idx_seq<2, 3, 1, 0>;
using PERM_LIJK = camp::idx_seq<3, 0, 1, 2>;
using PERM_LIKJ = camp::idx_seq<3, 0, 2, 1>;
using PERM_LJIK = camp::idx_seq<3, 1, 0, 2>;
using PERM_LJKI = camp::idx_seq<3, 1, 2, 0>;
using PERM_LKIJ = camp::idx_seq<3, 2, 0, 1>;
using PERM_LKJI = camp::idx_seq<3, 2, 1, 0>;
using PERM_IJKLM = camp::idx_seq<0, 1, 2, 3, 4>;
using PERM_IJKML = camp::idx_seq<0, 1, 2, 4, 3>;
using PERM_IJLKM = camp::idx_seq<0, 1, 3, 2, 4>;
using PERM_IJLMK = camp::idx_seq<0, 1, 3, 4, 2>;
using PERM_IJMKL = camp::idx_seq<0, 1, 4, 2, 3>;
using PERM_IJMLK = camp::idx_seq<0, 1, 4, 3, 2>;
using PERM_IKJLM = camp::idx_seq<0, 2, 1, 3, 4>;
using PERM_IKJML = camp::idx_seq<0, 2, 1, 4, 3>;
using PERM_IKLJM = camp::idx_seq<0, 2, 3, 1, 4>;
using PERM_IKLMJ = camp::idx_seq<0, 2, 3, 4, 1>;
using PERM_IKMJL = camp::idx_seq<0, 2, 4, 1, 3>;
using PERM_IKMLJ = camp::idx_seq<0, 2, 4, 3, 1>;
using PERM_ILJKM = camp::idx_seq<0, 3, 1, 2, 4>;
using PERM_ILJMK = camp::idx_seq<0, 3, 1, 4, 2>;
using PERM_ILKJM = camp::idx_seq<0, 3, 2, 1, 4>;
using PERM_ILKMJ = camp::idx_seq<0, 3, 2, 4, 1>;
using PERM_ILMJK = camp::idx_seq<0, 3, 4, 1, 2>;
using PERM_ILMKJ = camp::idx_seq<0, 3, 4, 2, 1>;
using PERM_IMJKL = camp::idx_seq<0, 4, 1, 2, 3>;
using PERM_IMJLK = camp::idx_seq<0, 4, 1, 3, 2>;
using PERM_IMKJL = camp::idx_seq<0, 4, 2, 1, 3>;
using PERM_IMKLJ = camp::idx_seq<0, 4, 2, 3, 1>;
using PERM_IMLJK = camp::idx_seq<0, 4, 3, 1, 2>;
using PERM_IMLKJ = camp::idx_seq<0, 4, 3, 2, 1>;
using PERM_JIKLM = camp::idx_seq<1, 0, 2, 3, 4>;
using PERM_JIKML = camp::idx_seq<1, 0, 2, 4, 3>;
using PERM_JILKM = camp::idx_seq<1, 0, 3, 2, 4>;
using PERM_JILMK = camp::idx_seq<1, 0, 3, 4, 2>;
using PERM_JIMKL = camp::idx_seq<1, 0, 4, 2, 3>;
using PERM_JIMLK = camp::idx_seq<1, 0, 4, 3, 2>;
using PERM_JKILM = camp::idx_seq<1, 2, 0, 3, 4>;
using PERM_JKIML = camp::idx_seq<1, 2, 0, 4, 3>;
using PERM_JKLIM = camp::idx_seq<1, 2, 3, 0, 4>;
using PERM_JKLMI = camp::idx_seq<1, 2, 3, 4, 0>;
using PERM_JKMIL = camp::idx_seq<1, 2, 4, 0, 3>;
using PERM_JKMLI = camp::idx_seq<1, 2, 4, 3, 0>;
using PERM_JLIKM = camp::idx_seq<1, 3, 0, 2, 4>;
using PERM_JLIMK = camp::idx_seq<1, 3, 0, 4, 2>;
using PERM_JLKIM = camp::idx_seq<1, 3, 2, 0, 4>;
using PERM_JLKMI = camp::idx_seq<1, 3, 2, 4, 0>;
using PERM_JLMIK = camp::idx_seq<1, 3, 4, 0, 2>;
using PERM_JLMKI = camp::idx_seq<1, 3, 4, 2, 0>;
using PERM_JMIKL = camp::idx_seq<1, 4, 0, 2, 3>;
using PERM_JMILK = camp::idx_seq<1, 4, 0, 3, 2>;
using PERM_JMKIL = camp::idx_seq<1, 4, 2, 0, 3>;
using PERM_JMKLI = camp::idx_seq<1, 4, 2, 3, 0>;
using PERM_JMLIK = camp::idx_seq<1, 4, 3, 0, 2>;
using PERM_JMLKI = camp::idx_seq<1, 4, 3, 2, 0>;
using PERM_KIJLM = camp::idx_seq<2, 0, 1, 3, 4>;
using PERM_KIJML = camp::idx_seq<2, 0, 1, 4, 3>;
using PERM_KILJM = camp::idx_seq<2, 0, 3, 1, 4>;
using PERM_KILMJ = camp::idx_seq<2, 0, 3, 4, 1>;
using PERM_KIMJL = camp::idx_seq<2, 0, 4, 1, 3>;
using PERM_KIMLJ = camp::idx_seq<2, 0, 4, 3, 1>;
using PERM_KJILM = camp::idx_seq<2, 1, 0, 3, 4>;
using PERM_KJIML = camp::idx_seq<2, 1, 0, 4, 3>;
using PERM_KJLIM = camp::idx_seq<2, 1, 3, 0, 4>;
using PERM_KJLMI = camp::idx_seq<2, 1, 3, 4, 0>;
using PERM_KJMIL = camp::idx_seq<2, 1, 4, 0, 3>;
using PERM_KJMLI = camp::idx_seq<2, 1, 4, 3, 0>;
using PERM_KLIJM = camp::idx_seq<2, 3, 0, 1, 4>;
using PERM_KLIMJ = camp::idx_seq<2, 3, 0, 4, 1>;
using PERM_KLJIM = camp::idx_seq<2, 3, 1, 0, 4>;
using PERM_KLJMI = camp::idx_seq<2, 3, 1, 4, 0>;
using PERM_KLMIJ = camp::idx_seq<2, 3, 4, 0, 1>;
using PERM_KLMJI = camp::idx_seq<2, 3, 4, 1, 0>;
using PERM_KMIJL = camp::idx_seq<2, 4, 0, 1, 3>;
using PERM_KMILJ = camp::idx_seq<2, 4, 0, 3, 1>;
using PERM_KMJIL = camp::idx_seq<2, 4, 1, 0, 3>;
using PERM_KMJLI = camp::idx_seq<2, 4, 1, 3, 0>;
using PERM_KMLIJ = camp::idx_seq<2, 4, 3, 0, 1>;
using PERM_KMLJI = camp::idx_seq<2, 4, 3, 1, 0>;
using PERM_LIJKM = camp::idx_seq<3, 0, 1, 2, 4>;
using PERM_LIJMK = camp::idx_seq<3, 0, 1, 4, 2>;
using PERM_LIKJM = camp::idx_seq<3, 0, 2, 1, 4>;
using PERM_LIKMJ = camp::idx_seq<3, 0, 2, 4, 1>;
using PERM_LIMJK = camp::idx_seq<3, 0, 4, 1, 2>;
using PERM_LIMKJ = camp::idx_seq<3, 0, 4, 2, 1>;
using PERM_LJIKM = camp::idx_seq<3, 1, 0, 2, 4>;
using PERM_LJIMK = camp::idx_seq<3, 1, 0, 4, 2>;
using PERM_LJKIM = camp::idx_seq<3, 1, 2, 0, 4>;
using PERM_LJKMI = camp::idx_seq<3, 1, 2, 4, 0>;
using PERM_LJMIK = camp::idx_seq<3, 1, 4, 0, 2>;
using PERM_LJMKI = camp::idx_seq<3, 1, 4, 2, 0>;
using PERM_LKIJM = camp::idx_seq<3, 2, 0, 1, 4>;
using PERM_LKIMJ = camp::idx_seq<3, 2, 0, 4, 1>;
using PERM_LKJIM = camp::idx_seq<3, 2, 1, 0, 4>;
using PERM_LKJMI = camp::idx_seq<3, 2, 1, 4, 0>;
using PERM_LKMIJ = camp::idx_seq<3, 2, 4, 0, 1>;
using PERM_LKMJI = camp::idx_seq<3, 2, 4, 1, 0>;
using PERM_LMIJK = camp::idx_seq<3, 4, 0, 1, 2>;
using PERM_LMIKJ = camp::idx_seq<3, 4, 0, 2, 1>;
using PERM_LMJIK = camp::idx_seq<3, 4, 1, 0, 2>;
using PERM_LMJKI = camp::idx_seq<3, 4, 1, 2, 0>;
using PERM_LMKIJ = camp::idx_seq<3, 4, 2, 0, 1>;
using PERM_LMKJI = camp::idx_seq<3, 4, 2, 1, 0>;
using PERM_MIJKL = camp::idx_seq<4, 0, 1, 2, 3>;
using PERM_MIJLK = camp::idx_seq<4, 0, 1, 3, 2>;
using PERM_MIKJL = camp::idx_seq<4, 0, 2, 1, 3>;
using PERM_MIKLJ = camp::idx_seq<4, 0, 2, 3, 1>;
using PERM_MILJK = camp::idx_seq<4, 0, 3, 1, 2>;
using PERM_MILKJ = camp::idx_seq<4, 0, 3, 2, 1>;
using PERM_MJIKL = camp::idx_seq<4, 1, 0, 2, 3>;
using PERM_MJILK = camp::idx_seq<4, 1, 0, 3, 2>;
using PERM_MJKIL = camp::idx_seq<4, 1, 2, 0, 3>;
using PERM_MJKLI = camp::idx_seq<4, 1, 2, 3, 0>;
using PERM_MJLIK = camp::idx_seq<4, 1, 3, 0, 2>;
using PERM_MJLKI = camp::idx_seq<4, 1, 3, 2, 0>;
using PERM_MKIJL = camp::idx_seq<4, 2, 0, 1, 3>;
using PERM_MKILJ = camp::idx_seq<4, 2, 0, 3, 1>;
using PERM_MKJIL = camp::idx_seq<4, 2, 1, 0, 3>;
using PERM_MKJLI = camp::idx_seq<4, 2, 1, 3, 0>;
using PERM_MKLIJ = camp::idx_seq<4, 2, 3, 0, 1>;
using PERM_MKLJI = camp::idx_seq<4, 2, 3, 1, 0>;
using PERM_MLIJK = camp::idx_seq<4, 3, 0, 1, 2>;
using PERM_MLIKJ = camp::idx_seq<4, 3, 0, 2, 1>;
using PERM_MLJIK = camp::idx_seq<4, 3, 1, 0, 2>;
using PERM_MLJKI = camp::idx_seq<4, 3, 1, 2, 0>;
using PERM_MLKIJ = camp::idx_seq<4, 3, 2, 0, 1>;
using PERM_MLKJI = camp::idx_seq<4, 3, 2, 1, 0>;

static constexpr auto PERM_I_v = perm_as_array(camp::idx_seq<0>{});
static constexpr auto PERM_IJ_v = perm_as_array(camp::idx_seq<0, 1>{});
static constexpr auto PERM_JI_v = perm_as_array(camp::idx_seq<1, 0>{});
static constexpr auto PERM_IJK_v = perm_as_array(camp::idx_seq<0, 1, 2>{});
static constexpr auto PERM_IKJ_v = perm_as_array(camp::idx_seq<0, 2, 1>{});
static constexpr auto PERM_JIK_v = perm_as_array(camp::idx_seq<1, 0, 2>{});
static constexpr auto PERM_JKI_v = perm_as_array(camp::idx_seq<1, 2, 0>{});
static constexpr auto PERM_KIJ_v = perm_as_array(camp::idx_seq<2, 0, 1>{});
static constexpr auto PERM_KJI_v = perm_as_array(camp::idx_seq<2, 1, 0>{});
static constexpr auto PERM_IJKL_v = perm_as_array(camp::idx_seq<0, 1, 2, 3>{});
static constexpr auto PERM_IJLK_v = perm_as_array(camp::idx_seq<0, 1, 3, 2>{});
static constexpr auto PERM_IKJL_v = perm_as_array(camp::idx_seq<0, 2, 1, 3>{});
static constexpr auto PERM_IKLJ_v = perm_as_array(camp::idx_seq<0, 2, 3, 1>{});
static constexpr auto PERM_ILJK_v = perm_as_array(camp::idx_seq<0, 3, 1, 2>{});
static constexpr auto PERM_ILKJ_v = perm_as_array(camp::idx_seq<0, 3, 2, 1>{});
static constexpr auto PERM_JIKL_v = perm_as_array(camp::idx_seq<1, 0, 2, 3>{});
static constexpr auto PERM_JILK_v = perm_as_array(camp::idx_seq<1, 0, 3, 2>{});
static constexpr auto PERM_JKIL_v = perm_as_array(camp::idx_seq<1, 2, 0, 3>{});
static constexpr auto PERM_JKLI_v = perm_as_array(camp::idx_seq<1, 2, 3, 0>{});
static constexpr auto PERM_JLIK_v = perm_as_array(camp::idx_seq<1, 3, 0, 2>{});
static constexpr auto PERM_JLKI_v = perm_as_array(camp::idx_seq<1, 3, 2, 0>{});
static constexpr auto PERM_KIJL_v = perm_as_array(camp::idx_seq<2, 0, 1, 3>{});
static constexpr auto PERM_KILJ_v = perm_as_array(camp::idx_seq<2, 0, 3, 1>{});
static constexpr auto PERM_KJIL_v = perm_as_array(camp::idx_seq<2, 1, 0, 3>{});
static constexpr auto PERM_KJLI_v = perm_as_array(camp::idx_seq<2, 1, 3, 0>{});
static constexpr auto PERM_KLIJ_v = perm_as_array(camp::idx_seq<2, 3, 0, 1>{});
static constexpr auto PERM_KLJI_v = perm_as_array(camp::idx_seq<2, 3, 1, 0>{});
static constexpr auto PERM_LIJK_v = perm_as_array(camp::idx_seq<3, 0, 1, 2>{});
static constexpr auto PERM_LIKJ_v = perm_as_array(camp::idx_seq<3, 0, 2, 1>{});
static constexpr auto PERM_LJIK_v = perm_as_array(camp::idx_seq<3, 1, 0, 2>{});
static constexpr auto PERM_LJKI_v = perm_as_array(camp::idx_seq<3, 1, 2, 0>{});
static constexpr auto PERM_LKIJ_v = perm_as_array(camp::idx_seq<3, 2, 0, 1>{});
static constexpr auto PERM_LKJI_v = perm_as_array(camp::idx_seq<3, 2, 1, 0>{});
static constexpr auto PERM_IJKLM_v = perm_as_array(camp::idx_seq<0, 1, 2, 3, 4>{});
static constexpr auto PERM_IJKML_v = perm_as_array(camp::idx_seq<0, 1, 2, 4, 3>{});
static constexpr auto PERM_IJLKM_v = perm_as_array(camp::idx_seq<0, 1, 3, 2, 4>{});
static constexpr auto PERM_IJLMK_v = perm_as_array(camp::idx_seq<0, 1, 3, 4, 2>{});
static constexpr auto PERM_IJMKL_v = perm_as_array(camp::idx_seq<0, 1, 4, 2, 3>{});
static constexpr auto PERM_IJMLK_v = perm_as_array(camp::idx_seq<0, 1, 4, 3, 2>{});
static constexpr auto PERM_IKJLM_v = perm_as_array(camp::idx_seq<0, 2, 1, 3, 4>{});
static constexpr auto PERM_IKJML_v = perm_as_array(camp::idx_seq<0, 2, 1, 4, 3>{});
static constexpr auto PERM_IKLJM_v = perm_as_array(camp::idx_seq<0, 2, 3, 1, 4>{});
static constexpr auto PERM_IKLMJ_v = perm_as_array(camp::idx_seq<0, 2, 3, 4, 1>{});
static constexpr auto PERM_IKMJL_v = perm_as_array(camp::idx_seq<0, 2, 4, 1, 3>{});
static constexpr auto PERM_IKMLJ_v = perm_as_array(camp::idx_seq<0, 2, 4, 3, 1>{});
static constexpr auto PERM_ILJKM_v = perm_as_array(camp::idx_seq<0, 3, 1, 2, 4>{});
static constexpr auto PERM_ILJMK_v = perm_as_array(camp::idx_seq<0, 3, 1, 4, 2>{});
static constexpr auto PERM_ILKJM_v = perm_as_array(camp::idx_seq<0, 3, 2, 1, 4>{});
static constexpr auto PERM_ILKMJ_v = perm_as_array(camp::idx_seq<0, 3, 2, 4, 1>{});
static constexpr auto PERM_ILMJK_v = perm_as_array(camp::idx_seq<0, 3, 4, 1, 2>{});
static constexpr auto PERM_ILMKJ_v = perm_as_array(camp::idx_seq<0, 3, 4, 2, 1>{});
static constexpr auto PERM_IMJKL_v = perm_as_array(camp::idx_seq<0, 4, 1, 2, 3>{});
static constexpr auto PERM_IMJLK_v = perm_as_array(camp::idx_seq<0, 4, 1, 3, 2>{});
static constexpr auto PERM_IMKJL_v = perm_as_array(camp::idx_seq<0, 4, 2, 1, 3>{});
static constexpr auto PERM_IMKLJ_v = perm_as_array(camp::idx_seq<0, 4, 2, 3, 1>{});
static constexpr auto PERM_IMLJK_v = perm_as_array(camp::idx_seq<0, 4, 3, 1, 2>{});
static constexpr auto PERM_IMLKJ_v = perm_as_array(camp::idx_seq<0, 4, 3, 2, 1>{});
static constexpr auto PERM_JIKLM_v = perm_as_array(camp::idx_seq<1, 0, 2, 3, 4>{});
static constexpr auto PERM_JIKML_v = perm_as_array(camp::idx_seq<1, 0, 2, 4, 3>{});
static constexpr auto PERM_JILKM_v = perm_as_array(camp::idx_seq<1, 0, 3, 2, 4>{});
static constexpr auto PERM_JILMK_v = perm_as_array(camp::idx_seq<1, 0, 3, 4, 2>{});
static constexpr auto PERM_JIMKL_v = perm_as_array(camp::idx_seq<1, 0, 4, 2, 3>{});
static constexpr auto PERM_JIMLK_v = perm_as_array(camp::idx_seq<1, 0, 4, 3, 2>{});
static constexpr auto PERM_JKILM_v = perm_as_array(camp::idx_seq<1, 2, 0, 3, 4>{});
static constexpr auto PERM_JKIML_v = perm_as_array(camp::idx_seq<1, 2, 0, 4, 3>{});
static constexpr auto PERM_JKLIM_v = perm_as_array(camp::idx_seq<1, 2, 3, 0, 4>{});
static constexpr auto PERM_JKLMI_v = perm_as_array(camp::idx_seq<1, 2, 3, 4, 0>{});
static constexpr auto PERM_JKMIL_v = perm_as_array(camp::idx_seq<1, 2, 4, 0, 3>{});
static constexpr auto PERM_JKMLI_v = perm_as_array(camp::idx_seq<1, 2, 4, 3, 0>{});
static constexpr auto PERM_JLIKM_v = perm_as_array(camp::idx_seq<1, 3, 0, 2, 4>{});
static constexpr auto PERM_JLIMK_v = perm_as_array(camp::idx_seq<1, 3, 0, 4, 2>{});
static constexpr auto PERM_JLKIM_v = perm_as_array(camp::idx_seq<1, 3, 2, 0, 4>{});
static constexpr auto PERM_JLKMI_v = perm_as_array(camp::idx_seq<1, 3, 2, 4, 0>{});
static constexpr auto PERM_JLMIK_v = perm_as_array(camp::idx_seq<1, 3, 4, 0, 2>{});
static constexpr auto PERM_JLMKI_v = perm_as_array(camp::idx_seq<1, 3, 4, 2, 0>{});
static constexpr auto PERM_JMIKL_v = perm_as_array(camp::idx_seq<1, 4, 0, 2, 3>{});
static constexpr auto PERM_JMILK_v = perm_as_array(camp::idx_seq<1, 4, 0, 3, 2>{});
static constexpr auto PERM_JMKIL_v = perm_as_array(camp::idx_seq<1, 4, 2, 0, 3>{});
static constexpr auto PERM_JMKLI_v = perm_as_array(camp::idx_seq<1, 4, 2, 3, 0>{});
static constexpr auto PERM_JMLIK_v = perm_as_array(camp::idx_seq<1, 4, 3, 0, 2>{});
static constexpr auto PERM_JMLKI_v = perm_as_array(camp::idx_seq<1, 4, 3, 2, 0>{});
static constexpr auto PERM_KIJLM_v = perm_as_array(camp::idx_seq<2, 0, 1, 3, 4>{});
static constexpr auto PERM_KIJML_v = perm_as_array(camp::idx_seq<2, 0, 1, 4, 3>{});
static constexpr auto PERM_KILJM_v = perm_as_array(camp::idx_seq<2, 0, 3, 1, 4>{});
static constexpr auto PERM_KILMJ_v = perm_as_array(camp::idx_seq<2, 0, 3, 4, 1>{});
static constexpr auto PERM_KIMJL_v = perm_as_array(camp::idx_seq<2, 0, 4, 1, 3>{});
static constexpr auto PERM_KIMLJ_v = perm_as_array(camp::idx_seq<2, 0, 4, 3, 1>{});
static constexpr auto PERM_KJILM_v = perm_as_array(camp::idx_seq<2, 1, 0, 3, 4>{});
static constexpr auto PERM_KJIML_v = perm_as_array(camp::idx_seq<2, 1, 0, 4, 3>{});
static constexpr auto PERM_KJLIM_v = perm_as_array(camp::idx_seq<2, 1, 3, 0, 4>{});
static constexpr auto PERM_KJLMI_v = perm_as_array(camp::idx_seq<2, 1, 3, 4, 0>{});
static constexpr auto PERM_KJMIL_v = perm_as_array(camp::idx_seq<2, 1, 4, 0, 3>{});
static constexpr auto PERM_KJMLI_v = perm_as_array(camp::idx_seq<2, 1, 4, 3, 0>{});
static constexpr auto PERM_KLIJM_v = perm_as_array(camp::idx_seq<2, 3, 0, 1, 4>{});
static constexpr auto PERM_KLIMJ_v = perm_as_array(camp::idx_seq<2, 3, 0, 4, 1>{});
static constexpr auto PERM_KLJIM_v = perm_as_array(camp::idx_seq<2, 3, 1, 0, 4>{});
static constexpr auto PERM_KLJMI_v = perm_as_array(camp::idx_seq<2, 3, 1, 4, 0>{});
static constexpr auto PERM_KLMIJ_v = perm_as_array(camp::idx_seq<2, 3, 4, 0, 1>{});
static constexpr auto PERM_KLMJI_v = perm_as_array(camp::idx_seq<2, 3, 4, 1, 0>{});
static constexpr auto PERM_KMIJL_v = perm_as_array(camp::idx_seq<2, 4, 0, 1, 3>{});
static constexpr auto PERM_KMILJ_v = perm_as_array(camp::idx_seq<2, 4, 0, 3, 1>{});
static constexpr auto PERM_KMJIL_v = perm_as_array(camp::idx_seq<2, 4, 1, 0, 3>{});
static constexpr auto PERM_KMJLI_v = perm_as_array(camp::idx_seq<2, 4, 1, 3, 0>{});
static constexpr auto PERM_KMLIJ_v = perm_as_array(camp::idx_seq<2, 4, 3, 0, 1>{});
static constexpr auto PERM_KMLJI_v = perm_as_array(camp::idx_seq<2, 4, 3, 1, 0>{});
static constexpr auto PERM_LIJKM_v = perm_as_array(camp::idx_seq<3, 0, 1, 2, 4>{});
static constexpr auto PERM_LIJMK_v = perm_as_array(camp::idx_seq<3, 0, 1, 4, 2>{});
static constexpr auto PERM_LIKJM_v = perm_as_array(camp::idx_seq<3, 0, 2, 1, 4>{});
static constexpr auto PERM_LIKMJ_v = perm_as_array(camp::idx_seq<3, 0, 2, 4, 1>{});
static constexpr auto PERM_LIMJK_v = perm_as_array(camp::idx_seq<3, 0, 4, 1, 2>{});
static constexpr auto PERM_LIMKJ_v = perm_as_array(camp::idx_seq<3, 0, 4, 2, 1>{});
static constexpr auto PERM_LJIKM_v = perm_as_array(camp::idx_seq<3, 1, 0, 2, 4>{});
static constexpr auto PERM_LJIMK_v = perm_as_array(camp::idx_seq<3, 1, 0, 4, 2>{});
static constexpr auto PERM_LJKIM_v = perm_as_array(camp::idx_seq<3, 1, 2, 0, 4>{});
static constexpr auto PERM_LJKMI_v = perm_as_array(camp::idx_seq<3, 1, 2, 4, 0>{});
static constexpr auto PERM_LJMIK_v = perm_as_array(camp::idx_seq<3, 1, 4, 0, 2>{});
static constexpr auto PERM_LJMKI_v = perm_as_array(camp::idx_seq<3, 1, 4, 2, 0>{});
static constexpr auto PERM_LKIJM_v = perm_as_array(camp::idx_seq<3, 2, 0, 1, 4>{});
static constexpr auto PERM_LKIMJ_v = perm_as_array(camp::idx_seq<3, 2, 0, 4, 1>{});
static constexpr auto PERM_LKJIM_v = perm_as_array(camp::idx_seq<3, 2, 1, 0, 4>{});
static constexpr auto PERM_LKJMI_v = perm_as_array(camp::idx_seq<3, 2, 1, 4, 0>{});
static constexpr auto PERM_LKMIJ_v = perm_as_array(camp::idx_seq<3, 2, 4, 0, 1>{});
static constexpr auto PERM_LKMJI_v = perm_as_array(camp::idx_seq<3, 2, 4, 1, 0>{});
static constexpr auto PERM_LMIJK_v = perm_as_array(camp::idx_seq<3, 4, 0, 1, 2>{});
static constexpr auto PERM_LMIKJ_v = perm_as_array(camp::idx_seq<3, 4, 0, 2, 1>{});
static constexpr auto PERM_LMJIK_v = perm_as_array(camp::idx_seq<3, 4, 1, 0, 2>{});
static constexpr auto PERM_LMJKI_v = perm_as_array(camp::idx_seq<3, 4, 1, 2, 0>{});
static constexpr auto PERM_LMKIJ_v = perm_as_array(camp::idx_seq<3, 4, 2, 0, 1>{});
static constexpr auto PERM_LMKJI_v = perm_as_array(camp::idx_seq<3, 4, 2, 1, 0>{});
static constexpr auto PERM_MIJKL_v = perm_as_array(camp::idx_seq<4, 0, 1, 2, 3>{});
static constexpr auto PERM_MIJLK_v = perm_as_array(camp::idx_seq<4, 0, 1, 3, 2>{});
static constexpr auto PERM_MIKJL_v = perm_as_array(camp::idx_seq<4, 0, 2, 1, 3>{});
static constexpr auto PERM_MIKLJ_v = perm_as_array(camp::idx_seq<4, 0, 2, 3, 1>{});
static constexpr auto PERM_MILJK_v = perm_as_array(camp::idx_seq<4, 0, 3, 1, 2>{});
static constexpr auto PERM_MILKJ_v = perm_as_array(camp::idx_seq<4, 0, 3, 2, 1>{});
static constexpr auto PERM_MJIKL_v = perm_as_array(camp::idx_seq<4, 1, 0, 2, 3>{});
static constexpr auto PERM_MJILK_v = perm_as_array(camp::idx_seq<4, 1, 0, 3, 2>{});
static constexpr auto PERM_MJKIL_v = perm_as_array(camp::idx_seq<4, 1, 2, 0, 3>{});
static constexpr auto PERM_MJKLI_v = perm_as_array(camp::idx_seq<4, 1, 2, 3, 0>{});
static constexpr auto PERM_MJLIK_v = perm_as_array(camp::idx_seq<4, 1, 3, 0, 2>{});
static constexpr auto PERM_MJLKI_v = perm_as_array(camp::idx_seq<4, 1, 3, 2, 0>{});
static constexpr auto PERM_MKIJL_v = perm_as_array(camp::idx_seq<4, 2, 0, 1, 3>{});
static constexpr auto PERM_MKILJ_v = perm_as_array(camp::idx_seq<4, 2, 0, 3, 1>{});
static constexpr auto PERM_MKJIL_v = perm_as_array(camp::idx_seq<4, 2, 1, 0, 3>{});
static constexpr auto PERM_MKJLI_v = perm_as_array(camp::idx_seq<4, 2, 1, 3, 0>{});
static constexpr auto PERM_MKLIJ_v = perm_as_array(camp::idx_seq<4, 2, 3, 0, 1>{});
static constexpr auto PERM_MKLJI_v = perm_as_array(camp::idx_seq<4, 2, 3, 1, 0>{});
static constexpr auto PERM_MLIJK_v = perm_as_array(camp::idx_seq<4, 3, 0, 1, 2>{});
static constexpr auto PERM_MLIKJ_v = perm_as_array(camp::idx_seq<4, 3, 0, 2, 1>{});
static constexpr auto PERM_MLJIK_v = perm_as_array(camp::idx_seq<4, 3, 1, 0, 2>{});
static constexpr auto PERM_MLJKI_v = perm_as_array(camp::idx_seq<4, 3, 1, 2, 0>{});
static constexpr auto PERM_MLKIJ_v = perm_as_array(camp::idx_seq<4, 3, 2, 0, 1>{});
static constexpr auto PERM_MLKJI_v = perm_as_array(camp::idx_seq<4, 3, 2, 1, 0>{});
}

#endif /* RAJA_FORALLN_PERMUTATIONS_HPP */
