#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

module load clang

find . -type f -iname '*.hpp' -o -iname '*.cpp' | grep -v -e 'blt' -e 'tpl' -e 'examples/' -e 'exercises/' -e 'test/' | xargs clang-format -i 
