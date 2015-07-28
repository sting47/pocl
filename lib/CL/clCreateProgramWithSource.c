/* OpenCL runtime library: clCreateProgramWithSource()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include "pocl_cl.h"
#include "pocl_util.h"
#include <string.h>
#include <stdio.h>

CL_API_ENTRY cl_program CL_API_CALL
POname(clCreateProgramWithSource)(cl_context context,
                          cl_uint count,
                          const char **strings,
                          const size_t *lengths,
                          cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
#if 0
  // sting enqueue_kernel pre-parser
  int inpi = 0, sti = 0, si = 0;
  //size_t tt = lengths[0];
  char **strings;
  /*for (si = 0; si < 6; si++)
  {
    printf("Test: %u\n", strlen(strings_input[0]));
  }*/
  char *include_enq = strstr(strings_input[0], "enqueue_kernel");
  char *enq_start = strstr(include_enq+1, "enqueue_kernel");
  char *enq_end = strstr(enq_start+1, ";");
  char tmp[3000];
  // copy the previous part
  memcpy (tmp, strings_input[0], strlen(strings_input[0]) - strlen(enq_start) - 2);
  
  // store the enqueue line to a tmp storage
  char tmp_enq[300];
  memcpy (tmp_enq, enq_start, strlen(enq_start) - strlen(enq_end) + 2);
  tmp_enq[strlen(tmp_enq)-1] = NULL;

  // find the end quote, and copy the rest of the code
  char tmp_rest[3000];
  char *enq_rest = strstr(enq_end+1, "}");
  memcpy (tmp_rest, enq_end+1, strlen(enq_end) - strlen(enq_rest));
  tmp_rest[strlen(tmp_rest)-1] = NULL;
  tmp_rest[strlen(tmp_rest)-1] = NULL;

  // printf("The string: \n%s\n", tmp_rest);
  // printf("The size: %u, %c\n", strlen(tmp_enq), tmp_enq[strlen(tmp_enq)-2]);
  // printf("The count: %u\n", strlen(enq));
  // printf("TEST: %s\n", strings_input[0]);
  strings = (char **) malloc (sizeof(char *));
  strings[0] = (char *) malloc (strlen(strings_input[0])+100);
  memcpy (strings[0], tmp, strlen(tmp));
  strcat (strings[0], tmp_rest);
  strcat (strings[0], tmp_enq);
  strcat (strings[0], "}");
  // printf("The new kernel:\n %s\n", strings[0]);

  // free the input source
  /*int xxx = 0;
  for (; xxx < count; x++)
  {
    free(strings_input[xxx]);
  }
  free(strings_input);*/
#endif
  

  cl_program program = NULL;
  size_t size = 0;
  char *source = NULL;
  unsigned i;
  int errcode;

  POCL_GOTO_ERROR_COND((count == 0), CL_INVALID_VALUE);

  program = (cl_program) calloc(1, sizeof(struct _cl_program));
  if (program == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_OBJECT(program);

  for (i = 0; i < count; ++i)
    {
      POCL_GOTO_ERROR_ON((strings[i] == NULL), CL_INVALID_VALUE,
          "strings[%i] is NULL\n", i);

      if (lengths == NULL)
        size += strlen(strings[i]);
      else if (lengths[i] == 0)
        size += strlen(strings[i]);
      else
        size += lengths[i];
    }

  source = (char *) malloc(size + 1);
  if (source == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  program->source = source;
  program->compiler_options = NULL;

  for (i = 0; i < count; ++i)
    {
      if (lengths == NULL)
        {
          memcpy(source, strings[i], strlen(strings[i]));
          source += strlen(strings[i]);
        }
      else if (lengths[i] == 0)
        {
          memcpy(source, strings[i], strlen(strings[i]));
          source += strlen(strings[i]);
        }
      else
        {
          memcpy(source, strings[i], lengths[i]);
          source += lengths[i];
        }
    }

  *source = '\0';

  program->context = context;
  program->num_devices = context->num_devices;
  program->devices = context->devices;
  program->kernels = NULL;
  program->build_status = CL_BUILD_NONE;

  if ((program->binary_sizes =
       (size_t*) calloc (program->num_devices, sizeof(size_t))) == NULL ||
      (program->binaries = (unsigned char**)
       calloc (program->num_devices, sizeof(unsigned char*))) == NULL ||
      (program->build_log = (char**)
       calloc (program->num_devices, sizeof(char*))) == NULL ||
      ((program->llvm_irs =
        (void**) calloc (program->num_devices, sizeof(void*))) == NULL) ||
      ((program->build_hash = (SHA1_digest_t*)
        calloc (program->num_devices, sizeof(SHA1_digest_t))) == NULL))
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }


  POCL_RETAIN_OBJECT(context);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return program;

ERROR:
  if (program) {
    POCL_MEM_FREE(program->build_hash);
    POCL_MEM_FREE(program->llvm_irs);
    POCL_MEM_FREE(program->build_log);
    POCL_MEM_FREE(program->binaries);
    POCL_MEM_FREE(program->binary_sizes);
    POCL_MEM_FREE(program->source);
  }
  POCL_MEM_FREE(program);
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
  return NULL;
}
POsym(clCreateProgramWithSource)
