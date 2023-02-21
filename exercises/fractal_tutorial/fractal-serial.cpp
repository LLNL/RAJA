//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// Code courtesy of Dr. Martin Burtscher and the Efficient Computing Lab
// at Texas State University. More can be found at 
// https://userweb.cs.txstate.edu/~burtscher/research.html
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include <sys/time.h>

#define xMin 0.74395
#define xMax 0.74973
#define yMin 0.11321
#define yMax 0.11899

static void WriteBMP(int x, int y, unsigned char *bmp, const char * name)
{
  const unsigned char bmphdr[54] = {66, 77, 255, 255, 255, 255, 0, 0, 0, 0, 54, 4, 0, 0, 40, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0, 8, 0, 0, 0, 0, 0, 255, 255, 255, 255, 196, 14, 0, 0, 196, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  unsigned char hdr[1078];
  int i, j, c, xcorr, diff;
  FILE *f;

  xcorr = (x+3) >> 2 << 2;  // BMPs have to be a multiple of 4 pixels wide.
  diff = xcorr - x;
  for (i = 0; i < 54; i++) hdr[i] = bmphdr[i];
    *((int*)(&hdr[18])) = xcorr;
    *((int*)(&hdr[22])) = y;
    *((int*)(&hdr[34])) = xcorr*y;
    *((int*)(&hdr[2])) = xcorr*y + 1078;
    for (i = 0; i < 256; i++) {
      j = i*4 + 54;
      hdr[j+0] = i;  // blue
      hdr[j+1] = i;  // green
      hdr[j+2] = i;  // red
      hdr[j+3] = 0;  // dummy
    }

    f = fopen(name, "wb");
    assert(f != NULL);
    c = fwrite(hdr, 1, 1078, f);
    assert(c == 1078);
    if (diff == 0) {
      c = fwrite(bmp, 1, x*y, f);
      assert(c == x*y);
    } else {
      *((int*)(&hdr[0])) = 0;  // need up to three zero bytes
      for (j = 0; j < y; j++) {
        c = fwrite(&bmp[j * x], 1, x, f);
	assert(c == x);
	c = fwrite(hdr, 1, diff, f);
	assert(c == diff);
      }
    }
  fclose(f);
}

int main(int argc, char *argv[])
{
  double dx, dy;
  int width, maxdepth;
  struct timeval start, end;

  /* check command line */
  if(argc != 3) {fprintf(stderr, "usage: exe <width> <depth>\n"); exit(-1);}
  width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "edge_length must be at least 10\n"); exit(-1);}
  maxdepth = atoi(argv[2]);
  if (maxdepth < 10) {fprintf(stderr, "max_depth must be at least 10\n"); exit(-1);}

  dx = (xMax - xMin) / width;
  dy = (yMax - yMin) / width;

  printf("computing %d by %d fractal with a maximum depth of %d\n", width, width, maxdepth);

  unsigned char *cnt = (unsigned char*)malloc(width * width * sizeof(unsigned char));
  double x2, y2, x, y, cx, cy;
  int depth;

  gettimeofday(&start, NULL);

  /* compute fractal */
  for(int r = 0; r < width; r++){
    cy = yMin + r * dy; //compute row #
    for(int c = 0; c < width; c++) {
      cx = xMin + c * dx; //compute column #
      x = -cx;
      y = -cy;
      depth = maxdepth;
      do {
        x2 = x * x;
        y2 = y * y;
        y = 2 * x * y - cy;
        x = x2 - y2 - cx;
        depth--;
      } while ((depth > 0) && ((x2 + y2) <= 5.0));
      cnt[r * width + c] = depth & 255;
    }
  }
						
  /* end time */
  gettimeofday(&end, NULL);
  printf("compute time: %.8f s\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);

  /* verify result by writing it to a file */
  if (width <= 1024) {
    WriteBMP(width, width, cnt, "fractal.bmp");
  }

  free(cnt);
  return 0;
}
