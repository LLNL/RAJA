#!/usr/bin/env python

import os
import sys
import subprocess
import commands
import re

#print sys.argv[0]
#print sys.argv[1]

output = subprocess.check_output(sys.argv[1],shell=True)

minTime=[]
rowCounter=0
for row in output.splitlines():
    index = 0
    if( rowCounter > 0 ):
        for col in row.split():
            length = len(minTime)
            if length <= index:
                minTime.append(float(col))
            else:
                minTime[index] = min(minTime[index],col)
            index += 1
    rowCounter += 1

print minTime