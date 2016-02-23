#!/bin/bash
#
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# All rights reserved.
#
# This source code cannot be distributed without permission and
# further review from Lawrence Livermore National Laboratory.
#



#
#
# This script will regenerate the nested-loop files
#
#


PREFIX=../include/RAJA
PYTHON="/usr/bin/env python"

$PYTHON ./genForallN.py 2 > $PREFIX/forall2.hxx
$PYTHON ./genForallN.py 3 > $PREFIX/forall3.hxx
$PYTHON ./genForallN.py 4 > $PREFIX/forall4.hxx
$PYTHON ./genForallN.py 5 > $PREFIX/forall5.hxx

$PYTHON ./genLayout.py 5 > $PREFIX/Layout.hxx

$PYTHON ./genView.py 5 > $PREFIX/View.hxx


