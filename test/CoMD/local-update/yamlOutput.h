/// \file
/// Write simulation information in YAML format.

#ifndef __YAML_OUTPUT_H
#define __YAML_OUTPUT_H

#include <stdio.h>

/// Provide access to the YAML file in other compliation units.
extern FILE* yamlFile;

void yamlBegin(void);
void yamlEnd(void);

void yamlAppInfo(FILE* file);

void printSeparator(FILE* file);

#endif
