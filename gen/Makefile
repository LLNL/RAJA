## MAKEFILE ##

#
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory.
#
# All rights reserved.
#
# For release details and restrictions, please see raja/README-license.txt
#


MAXDIM=5
PREFIX=../include/RAJA
PYTHON=/usr/bin/env python

all:
	${PYTHON} ./gen_forallN_generic.py ${MAXDIM} > ${PREFIX}/forallN_generic.hxx
	${PYTHON} ./gen_forallN_permute.py ${MAXDIM} > ${PREFIX}/forallN_permute.hxx
	${PYTHON} ./gen_forallN_tile.py    ${MAXDIM} > ${PREFIX}/forallN_tile.hxx
	${PYTHON} ./gen_forallN_openmp.py    ${MAXDIM} > ${PREFIX}/exec-openmp/forallN_openmp.hxx
	${PYTHON} ./genLayout.py ${MAXDIM} > ${PREFIX}/Layout.hxx
	${PYTHON} ./genView.py ${MAXDIM} > ${PREFIX}/View.hxx


