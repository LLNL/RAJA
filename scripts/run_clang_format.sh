#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

module load clang

find . -type f -iname '*.hpp' -o -iname '*.cpp' | grep -v -e 'blt' -e 'tpl' -e 'examples/' -e 'exercises/' -e 'test/' | xargs clang-format -i 
