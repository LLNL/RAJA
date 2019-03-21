#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

find . -type f -iname '*.hpp' -o -iname '*.cpp' | grep -v -e 'blt' -e 'tpl' -e 'RAJA/examples' | xargs ./scripts/clang-format-linux -i 
