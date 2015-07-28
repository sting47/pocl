#include <stdlib.h>
#include <stdio.h>
#include <CL/opencl.h>
#include <poclu.h>
// #include "/home/sting47/Library/test/pocl/lib/get_default_queue.h"

#ifdef __cplusplus
extern "C" {
#endif

// queue_t *get_default_queue_host(void);

void 
delete_memobjs(cl_mem *memobjs, int n) 
{ 
  int i; 
  for (i=0; i<n; i++) 
    clReleaseMemObject(memobjs[i]); 
} 
 
int 
exec_dot_product_kernel(const char *program_source, 
                        int n, cl_float4 *srcA, cl_float4 *srcB, cl_float *dst) 
{ 
  cl_context  context; 
  cl_command_queue cmd_queue; 
  cl_device_id  *devices; 
  cl_program  program; 
  cl_kernel  kernel; 
  cl_mem       memobjs[4]; 
  size_t       global_work_size[1]; 
  size_t       local_work_size[1]; 
  size_t       cb; 
  cl_int       err; 
  int          i;
  context = poclu_create_any_context();
  if (context == (cl_context)0) 
    return -1; 
 
  // get the list of GPU devices associated with context 
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb); 
  devices = (cl_device_id *) malloc(cb); 
  clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL); 
 
  // create a command-queue 
  cmd_queue = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (cmd_queue == (cl_command_queue)0) 
    { 
      clReleaseContext(context); 
      free(devices); 
      return -1; 
    } 

  for (i = 0; i < n; ++i)
    {
       poclu_bswap_cl_float_array(devices[0], (cl_float*)&srcA[i], 4);
       poclu_bswap_cl_float_array(devices[0], (cl_float*)&srcB[i], 4);
    }

 
  // allocate the buffer memory objects 
  memobjs[0] = clCreateBuffer(context, 
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              sizeof(cl_float4) * n, srcA, NULL); 
  if (memobjs[0] == (cl_mem)0) 
    { 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  memobjs[1] = clCreateBuffer(context, 
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              sizeof(cl_float4) * n, srcB, NULL); 
  if (memobjs[1] == (cl_mem)0) 
    { 
      delete_memobjs(memobjs, 1); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1;
    } 
 
  memobjs[2] = clCreateBuffer(context, 
			      CL_MEM_READ_WRITE, 
			      sizeof(cl_float) * n, NULL, NULL); 
  if (memobjs[2] == (cl_mem)0) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 

  memobjs[3] = clCreateBuffer(context, 
			      CL_MEM_READ_WRITE, 
			      sizeof(cl_float4) * n, NULL, NULL); 
  if (memobjs[3] == (cl_mem)0) 
    { 
      delete_memobjs(memobjs, 3); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // create the program 
  program = clCreateProgramWithSource(context, 
				      1, (const char**)&program_source, NULL, NULL); 
  if (program == (cl_program)0) 
    { 
      delete_memobjs(memobjs, 4); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // queue_t *tmp = get_default_queue_host();
  // tmp->x = 8;
  // build the program 
  printf("before build program\n");
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      size_t len;
      char buffer[2048];
      printf("Build Error Code: %d\n", err);
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      delete_memobjs(memobjs, 3); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // create the kernel 
  kernel = clCreateKernel(program, "dot_product", NULL); 
  if (kernel == (cl_kernel)0) 
    { 
      printf("Create Kernel Fail\n");
      delete_memobjs(memobjs, 3); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // set the args values 
  err = clSetKernelArg(kernel,  0,  
		       sizeof(cl_mem), (void *) &memobjs[0]); 
  err |= clSetKernelArg(kernel, 1,  
			sizeof(cl_mem), (void *) &memobjs[1]); 
  err |= clSetKernelArg(kernel, 2,
			sizeof(cl_mem), (void *) &memobjs[2]); 
  err |= clSetKernelArg(kernel, 3,
			sizeof(cl_mem), (void *) &memobjs[3]); 
 
  if (err != CL_SUCCESS) 
    { 
      printf("Set Args Fail\n");
      delete_memobjs(memobjs, 4); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // set work-item dimensions 
  global_work_size[0] = n; 
  local_work_size[0]= 128; 
 
  // execute kernel 
  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, 
			       global_work_size, local_work_size,  
			       0, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      printf("Execute Kernel Fail\n");
      delete_memobjs(memobjs, 3); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // read output image 
  err = clEnqueueReadBuffer(cmd_queue, memobjs[2], CL_TRUE, 
			    0, n * sizeof(cl_float), dst, 
			    0, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      delete_memobjs(memobjs, 4); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
  for (i = 0; i < n; ++i)
    {
      poclu_bswap_cl_float_array(devices[0], (cl_float*)&dst[i], 1);
      poclu_bswap_cl_float_array(devices[0], (cl_float*)&srcA[i], 4);
      poclu_bswap_cl_float_array(devices[0], (cl_float*)&srcB[i], 4);
    }
  free(devices); 


  // release kernel, program, and memory objects 
  delete_memobjs(memobjs, 4); 
  clReleaseKernel(kernel); 
  clReleaseProgram(program); 
  clReleaseCommandQueue(cmd_queue); 
  clReleaseContext(context); 
  return 0; // success... 
}

#ifdef __cplusplus
}
#endif 

