/* pocl-hsa.c - driver for HSA supporteddevices

   Copyright (c) 2015 Pekka Jääskeläinen / Tampere University of Technology

   Short snippets borrowed from the MatrixMultiplication example in
   the HSA runtime library sources (c) 2014 HSA Foundation Inc.

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

#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "hsa_ext_amd.h"

#include "pocl-hsa.h"
#include "common.h"
#include "devices.h"
#include "pocl_cache.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

#ifndef _MSC_VER
#  include <sys/time.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#define max(a,b) (((a) > (b)) ? (a) : (b))

#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 128

/* TODO:
   - allocate buffers with hsa_memory_allocate() to ensure the driver
     works with HSA Base profile agents (assuming all memory is coherent
     requires the Full profile) -- or perhaps a separate hsabase driver
     for the simpler agents.
   - cleanup the leftover code from the pthread.c copy-paste
   - autoprobe for HSA devices
   - configure: --enable-hsa that detects that libclc etc. are found and
     setups the HSA driver
   - README.HSA that tells how to enable the HSA driver, the required
     external libraries, etc.
*/

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* The HSA kernel agent controlled by the device driver instance. */
  hsa_agent_t *agent;
  /* Queue for pushing work to the agent. */
  hsa_queue_t *queue;
};

void
pocl_hsa_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops (ops);

  /* TODO: more descriptive name from HSA probing the device */
  ops->device_name = "hsa";

  ops->init_device_infos = pocl_hsa_init_device_infos;
  ops->probe = pocl_hsa_probe;
  ops->uninit = pocl_hsa_uninit;
  ops->init = pocl_hsa_init;
  ops->compile_submitted_kernels = pocl_hsa_compile_submitted_kernels;
  ops->run = pocl_hsa_run;
}

void
pocl_hsa_init_device_infos(struct _cl_device_id* dev)
{
  pocl_basic_init_device_infos (dev);
  dev->type = CL_DEVICE_TYPE_GPU;
  dev->spmd = CL_TRUE;
  dev->llvm_target_triplet = "amdgcn--amdhsa";
  dev->llvm_cpu = "kaveri";
  dev->has_64bit_long = 1;
  /* TODO: probe from HSA */
  dev->max_mem_alloc_size = 512*1024*2014;
}

#define MAX_HSA_AGENTS 16

static hsa_agent_t hsa_agents[MAX_HSA_AGENTS];
static int found_hsa_agents = 0;

static hsa_status_t
pocl_hsa_get_agents(hsa_agent_t agent, void *data)
{
  hsa_device_type_t type;
  hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type != HSA_DEVICE_TYPE_GPU)
    return HSA_STATUS_SUCCESS;

  hsa_agents[found_hsa_agents] = agent;
  ++found_hsa_agents;
  return HSA_STATUS_SUCCESS;
}

unsigned int
pocl_hsa_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  POCL_MSG_PRINT_INFO("pocl-hsa: found %d env devices with %s.\n",
                      env_count, ops->device_name);

  /* No hsa env specified, the user did not request for HSA agents. */
  if (env_count <= 0)
    return 0;

  if (hsa_init() != HSA_STATUS_SUCCESS)
    {
      POCL_ABORT("pocl-hsa: hsa_init() failed.");
    }

  if (hsa_iterate_agents(pocl_hsa_get_agents, NULL) !=
      HSA_STATUS_SUCCESS)
    {
      assert (0 && "pocl-hsa: could not get agents.");
    }
  POCL_MSG_PRINT_INFO("pocl-hsa: found %d agents.\n", found_hsa_agents);
  return found_hsa_agents;
}

void
pocl_hsa_init (cl_device_id device, const char* parameters)
{
  struct data *d;
  static int global_mem_id;
  static int first_hsa_init = 1;
  hsa_device_type_t dev_type;
  hsa_status_t status;

  if (first_hsa_init)
    {
      first_hsa_init = 0;
      global_mem_id = device->dev_id;
    }
  device->global_mem_id = global_mem_id;

  d = (struct data *) calloc (1, sizeof (struct data));

  d->current_kernel = NULL;
  device->data = d;

  assert (found_hsa_agents > 0);

  /* TODO: support controlling multiple agents.
     Now all pocl devices control the same one. */
  d->agent = &hsa_agents[0];

  if (hsa_queue_create(*d->agent, 1, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                       &d->queue) != HSA_STATUS_SUCCESS)
    {
      POCL_ABORT("pocl-hsa: could not create the queue.");
    }

  /* TODO: replace with HSA calls: */
#if 0
  pocl_topology_detect_device_info(device);
  pocl_cpuinfo_detect_device_info(device);
#endif
  /* TODO: detect with HSA calls: */
  device->max_compute_units = 1;
}

void *
pocl_hsa_malloc (void *device_data, cl_mem_flags flags,
		    size_t size, void *host_ptr)
{
  void *b;

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      b = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, size);
      if (b != NULL)
        {
          memcpy(b, host_ptr, size);
          return b;
        }
      return NULL;
    }

  if (flags & CL_MEM_USE_HOST_PTR && host_ptr != NULL)
    {
      return host_ptr;
    }
  b = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, size);
  if (b != NULL)
    return b;
  return NULL;
}

#define MAX_KERNEL_ARG_WORDS 64

/* Setup the arguments according to the AMDGPU CC generated by llc.
    LLVM generates first arguments for passing the OpenCL id/size data.
    The rest of the args contain the actual kernel arguments.

    TODO: - Figure out the CC, are int32 pushed to the buffer directly,
            are the pointers 64b and pushed as two int32 to the buffer
            also? Aggregates via pointers?
*/

#pragma pack(push, 1)
typedef struct amdgpu_args_t {
  /* TODO: assign these */
  uint32_t wgs_x;
  uint32_t wgs_y;
  uint32_t wgs_z;
  uint32_t global_size_x;
  uint32_t global_size_y;
  uint32_t global_size_z;
  uint32_t workgroup_size_x;
  uint32_t workgroup_size_y;
  uint32_t workgroup_size_z;
  uint32_t kernel_args[MAX_KERNEL_ARG_WORDS];
} amdgpu_args_t;
#pragma pack(pop)

/* From HSA Runtime Specs: Used to find a region for passing
   kernel arguments. */
#if 0
  /* TODO: this is not supported in the reference runtime lib. */
static hsa_status_t
pocl_hsa_get_kernarg(hsa_region_t region, void* data)
{
  hsa_region_segment_t segment;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (segment != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }
  hsa_region_global_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    hsa_region_t* ret = (hsa_region_t*) data;
    *ret = region;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}
#endif

void
pocl_hsa_run
(void *data,
 _cl_command_node* cmd)
{
  struct data *d;
  struct pocl_argument *al;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  hsa_signal_value_t initial_value = 1;
#if 0
  /* Not yet supported by the reference library. */
  hsa_kernel_dispatch_packet_t kernel_packet;
#else
  hsa_dispatch_packet_t kernel_packet;
#endif
  hsa_signal_t kernel_completion_signal = 0;
  hsa_region_t region;
  int error;
  amdgpu_args_t *args;
  /* 32b word offset in the kernel arguments buffer we can push the next
     argument to. */
  int args_offset = 0;

  assert (data != NULL);
  d = (struct data *) data;

  d->current_kernel = kernel;

  memset (&kernel_packet, 0, sizeof (hsa_dispatch_packet_t));

#if 0
  /* TODO: not yet supported by the open source runtime implementation.
     Assume the HSA Full profile so we can simply use host malloc().
   */
  hsa_agent_iterate_regions(kernel_agent, pocl_hsa_get_kernarg, &region);

  if (hsa_memory_allocate(region, sizeof(amdgpu_args_t),
                          (void**)&args) !=
      HSA_STATUS_SUCCESS)
    {
      assert (0 && "hsa_memory_allocate() failed.");
    }
#else
  args = (amdgpu_args_t*)malloc(sizeof(amdgpu_args_t));
#endif

  kernel_packet.kernarg_address = (uint64_t)args;

  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local
     memory buffers, etc. */
  for (i = 0; i < kernel->num_args; ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      if (kernel->arg_info[i].is_local)
        {
          POCL_ABORT_UNIMPLEMENTED("pocl-hsa: local buffers not implemented.");
#if 0
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_hsa_malloc(data, 0, al->size, NULL);
#endif
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          if (args_offset + 1 >= MAX_KERNEL_ARG_WORDS)
            POCL_ABORT("pocl-hsa: too many kernel arguments!");
          /* Assuming the pointers are 64b (or actually the same as in
             host) due to HSA. TODO: the 32b profile. */
          if (al->value == NULL)
            {
              args->kernel_args[args_offset] = 0;
              args->kernel_args[args_offset + 1] = 0;
            }
          else
            {
              *(uint64_t*)&args->kernel_args[args_offset] =
                (uint64_t)(*(cl_mem *) (al->value))->
                device_ptrs[cmd->device->dev_id].mem_ptr;
            }
          args_offset += 2;

#if 0
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL)
            {
              arguments[i] = malloc (sizeof (void *));
              *(void **)arguments[i] = NULL;
            }
          else
            arguments[i] =
              &((*(cl_mem *) (al->value))->device_ptrs[cmd->device->dev_id].mem_ptr);
#endif
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          POCL_ABORT_UNIMPLEMENTED("hsa: image arguments not implemented.");
#if 0
          dev_image_t di;
          fill_dev_image_t (&di, al, cmd->device);

          void* devptr = pocl_hsa_malloc (data, 0, sizeof(dev_image_t), NULL);
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;
          pocl_hsa_write (data, &di, devptr, 0, sizeof(dev_image_t));
#endif
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          POCL_ABORT_UNIMPLEMENTED("hsa: sampler arguments not implemented.");
#if 0
          dev_sampler_t ds;
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_hsa_malloc
            (data, 0, sizeof(dev_sampler_t), NULL);
          pocl_hsa_write (data, &ds, *(void**)arguments[i], 0,
                            sizeof(dev_sampler_t));
#endif
        }
      else
        {
          if (args_offset >= MAX_KERNEL_ARG_WORDS)
            POCL_ABORT("pocl-hsa: too many kernel arguments!");

          /* Assuming the scalar fits to a 32b slot. TODO! */
          assert (al->size <= 4);
          args->kernel_args[args_offset] = *(uint32_t*)al->value;
          ++args_offset;
        }
    }

  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      POCL_ABORT_UNIMPLEMENTED("hsa: automatic local buffers not implemented.");
#if 0
      al = &(cmd->command.run.arguments[i]);
      arguments[i] = malloc (sizeof (void *));
      *(void **)(arguments[i]) = pocl_hsa_malloc (data, 0, al->size, NULL);
#endif
    }


  args->workgroup_size_x = kernel_packet.workgroup_size_x = cmd->command.run.local_x;
  args->workgroup_size_y = kernel_packet.workgroup_size_y = cmd->command.run.local_y;
  args->workgroup_size_z = kernel_packet.workgroup_size_z = cmd->command.run.local_z;

  kernel_packet.grid_size_x = pc->num_groups[0] * cmd->command.run.local_x;
  kernel_packet.grid_size_y = pc->num_groups[1] * cmd->command.run.local_y;
  kernel_packet.grid_size_z = pc->num_groups[2] * cmd->command.run.local_z;

  /* AMDGPU specific OpenCL argument data. */

  args->wgs_x = pc->num_groups[0];
  args->wgs_y = pc->num_groups[1];
  args->wgs_z = pc->num_groups[2];

  kernel_packet.dimensions = 1;
  if (cmd->command.run.local_y > 1) kernel_packet.dimensions = 2;
  if (cmd->command.run.local_z > 1) kernel_packet.dimensions = 3;

  kernel_packet.header.type = HSA_PACKET_TYPE_DISPATCH;
  kernel_packet.header.acquire_fence_scope = HSA_FENCE_SCOPE_SYSTEM;
  kernel_packet.header.release_fence_scope = HSA_FENCE_SCOPE_SYSTEM;
  kernel_packet.header.barrier = 1;

  kernel_packet.kernel_object_address =
    *(hsa_amd_code_t*)cmd->command.run.device_data[1];

  error =  hsa_signal_create(initial_value, 0, NULL, &kernel_completion_signal);
  assert (error == HSA_STATUS_SUCCESS);

  kernel_packet.completion_signal = kernel_completion_signal;

  {
    /* Launch the kernel by allocating a slot in the queue, writing the
       command to it, signaling the update with a door bell and finally,
       block waiting until finish signalled with the completion_signal. */
    const uint32_t queue_mask = d->queue->size - 1;
    uint64_t queue_index = hsa_queue_load_write_index_relaxed(d->queue);
    hsa_signal_value_t sigval;
    ((hsa_dispatch_packet_t*)(d->queue->base_address))[queue_index & queue_mask] =
      kernel_packet;
    hsa_queue_store_write_index_relaxed(d->queue, queue_index + 1);
    hsa_signal_store_relaxed(d->queue->doorbell_signal, queue_index);

    sigval = hsa_signal_wait_acquire(kernel_completion_signal, HSA_EQ, 0,
                                     (uint64_t)(-1), HSA_WAIT_EXPECTANCY_UNKNOWN);
  }

  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_info[i].is_local)
        {
#if 0
          pocl_hsa_free (data, 0, *(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
#endif
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
#if 0
          pocl_hsa_free (data, 0, *(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
#endif
        }
#if 0
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER ||
               (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER &&
                *(void**)args->kernel_args[i] == NULL))
        {
          POCL_MEM_FREE(arguments[i]);
        }
#endif
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
#if 0
      pocl_hsa_free(data, 0, *(void **)(arguments[i]));
      POCL_MEM_FREE(arguments[i]);
#endif
    }
  free(args);
}

void
pocl_hsa_uninit (cl_device_id device)
{
  struct data *d = (struct data*)device->data;
  POCL_MEM_FREE(d);
  device->data = NULL;
}

/* TODO: there's not much to do here, just build the kernel for HSA.
   Perhaps share the same function for all WG sizes in case it's an
   SPMD target. */
static void compile (_cl_command_node *cmd)
{
  int error;
  char bytecode[POCL_FILENAME_LENGTH];
  char objfile[POCL_FILENAME_LENGTH];
  FILE *file;
  char *elf_blob;
  size_t file_size, got_size;
  hsa_runtime_caller_t caller;

  error = snprintf (bytecode, POCL_FILENAME_LENGTH,
                    "%s/%s", cmd->command.run.tmp_dir,
                    POCL_PARALLEL_BC_FILENAME);
  assert (error >= 0);

  error = snprintf (objfile, POCL_FILENAME_LENGTH,
                    "%s/%s.o", cmd->command.run.tmp_dir,
                    POCL_PARALLEL_BC_FILENAME);
  assert (error >= 0);

  error = pocl_llvm_codegen (cmd->command.run.kernel, cmd->device, bytecode, objfile);
  assert (error == 0);

  /* Load the built AMDGPU ELF file. */
  file = fopen (objfile, "rb");
  assert (file != NULL);

  cmd->command.run.device_data = (void**)malloc (sizeof(void*)*2);
  cmd->command.run.device_data[0] = malloc (sizeof(hsa_amd_code_unit_t));
  cmd->command.run.device_data[1] = malloc (sizeof(hsa_amd_code_t));

  file_size = pocl_file_size (file);
  elf_blob = (char*)malloc (file_size);
  got_size = fread (elf_blob, 1, file_size, file);

  if (file_size != got_size)
    POCL_ABORT ("pocl-hsa: could not read the AMD ELF.");

  caller.caller = 0;
  if (hsa_ext_code_unit_load
      (caller, NULL, 0, elf_blob, file_size, NULL, NULL,
       (hsa_amd_code_unit_t*)cmd->command.run.device_data[0]) != HSA_STATUS_SUCCESS)
    {
      POCL_ABORT ("pocl-hsa: error while loading the built AMDGPU ELF binary.");
    }

  if (hsa_ext_code_unit_get_info
      (*(hsa_amd_code_unit_t*)cmd->command.run.device_data[0],
       HSA_EXT_CODE_UNIT_INFO_CODE_ENTITY_CODE, 0,
       (hsa_amd_code_t*)cmd->command.run.device_data[1]) != HSA_STATUS_SUCCESS)
    {
      POCL_ABORT ("pocl-hsa: unable to get the code handle to the kernel.");
    }
  free (elf_blob);
  fclose (file);
}

void
pocl_hsa_compile_submitted_kernels (_cl_command_node *cmd)
{
  if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    compile (cmd);
}
