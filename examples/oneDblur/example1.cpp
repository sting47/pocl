/* example1 - Simple example from OpenCL specification.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <sys/time.h>

#define N 1024 //262144 //16384 //128

timeval t1, t2;
#define begin_timing gettimeofday(&t1, NULL); for (int xyz = 0; xyz < 1; xyz++) {
#define end_timing } gettimeofday(&t2, NULL);

#ifdef __cplusplus
#  define CALLAPI "C"
#else 
#  define CALLAPI
#endif

extern CALLAPI int exec_dot_product_kernel (const char *program_source, 
            int n, void *srcA, void *srcB, void *dst);

int
main (void)
{
  FILE *source_file;
  char *source;
  int source_size;
  cl_int *srcA, *srcB;
  cl_int *dst;
  int ierr;
  int i;

  source_file = fopen("example1.cl", "r");
  if (source_file == NULL) 
    source_file = fopen (SRCDIR "/example1.cl", "r");

  assert(source_file != NULL && "example1.cl not found!");

  fseek (source_file, 0, SEEK_END);
  source_size = ftell (source_file);
  fseek (source_file, 0, SEEK_SET);

  source = (char *) malloc (source_size +1 );
  assert (source != NULL);

  fread (source, source_size, 1, source_file);
  source[source_size] = '\0';

  fclose (source_file);

  srcA = (cl_int *) malloc (N * sizeof (cl_int));
  srcB = (cl_int *) malloc (N * sizeof (cl_int));
  dst = (cl_int *) malloc (N * sizeof (cl_int));

  for (i = 0; i < N; ++i)
    {
      srcA[i] = (cl_int)i;
    }

  // for whole OpenCL execution time counting
  // timeval t1, t2;
  // gettimeofday(&t1, NULL);
  // begin_timing;
  ierr = exec_dot_product_kernel (source, N, srcA, srcB, dst);
  // end_timing;
  // gettimeofday(&t2, NULL);

  // float putong_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;
  // printf("Execution time: %f \n", putong_time);
  // if (ierr) printf ("ERROR\n");

#if 0
  for (i = 0; i < N; i=i+8)
    {
      printf ("%d, %d, %d, %d, %d, %d, %d, %d\n",
        dst[i], dst[i+1], dst[i+2], dst[i+3],
        dst[i+4], dst[i+5], dst[i+6], dst[i+7]);
    //   if (srcA[i].s[0] * srcB[i].s[0] +
    //       srcA[i].s[1] * srcB[i].s[1] +
    //       srcA[i].s[2] * srcB[i].s[2] +
    //       srcA[i].s[3] * srcB[i].s[3] != dst[i])
    //     {
    //       printf ("FAIL\n");
    //       return -1;
    //     }
    }
#endif

  return 0;
}
