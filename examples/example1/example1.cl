#define USE_VECTOR_DATATYPES
#include "/home/sting47/Library/pocl_enque/lib/kernel/enqueue_kernel.c"
// #include "/home/sting47/Library/test/pocl/lib/put_in.c"
// #include "/home/sting47/Library/test/pocl/lib/get_default_queue.c"

__kernel void
kenhaha (__global const float4 *a,  
         __global const float4 *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = 4.0;
}

__kernel void 
dot_product_child (__global const float4 *a,
	     __global const float4 *b, __global float *c, __global float4 *d) 
{ 
  int gid = get_global_id(0); 

#ifndef USE_VECTOR_DATATYPES
  /* This version is to smoke test the autovectorization.
     Tries to create parallel regions with nice memory
     access pattern etc. so it gets autovectorizer. */
  /* This parallel region does not vectorize with the
     loop vectorizer because it accesses vector datatypes.
     Perhaps with SLP/BB vectorizer.*/

  float ax = a[gid].x;
  float ay = a[gid].y; 
  float az = a[gid].z;
  float aw = a[gid].w;

  float bx = b[gid].x; 
  float by = b[gid].y; 
  float bz = b[gid].z; 
  float bw = b[gid].w;

  barrier(CLK_LOCAL_MEM_FENCE);

  /* This parallel region should vectorize. */
  c[gid] = ax * bx;
  c[gid] += ay * by;
  c[gid] += az * bz;
  c[gid] += aw * bw;

#else
  // queue_t in_queue = get_default_queue();
  float4 prod = a[gid] * d[gid];
  c[gid] = prod.x + prod.y + prod.z + prod.w;
  // c[gid] = in_queue->x;
#endif
 

}

__kernel void 
dot_product (__global const float4 *a,  
	     __global const float4 *b, __global float *c, __global float4 *d) 
{ 
  int gid = get_global_id(0); 

#ifndef USE_VECTOR_DATATYPES
  /* This version is to smoke test the autovectorization.
     Tries to create parallel regions with nice memory
     access pattern etc. so it gets autovectorizer. */
  /* This parallel region does not vectorize with the
     loop vectorizer because it accesses vector datatypes.
     Perhaps with SLP/BB vectorizer.*/

  float ax = a[gid].x;
  float ay = a[gid].y; 
  float az = a[gid].z;
  float aw = a[gid].w;

  float bx = b[gid].x; 
  float by = b[gid].y; 
  float bz = b[gid].z; 
  float bw = b[gid].w;

  barrier(CLK_LOCAL_MEM_FENCE);

  /* This parallel region should vectorize. */
  c[gid] = ax * bx;
  c[gid] += ay * by;
  c[gid] += az * bz;
  c[gid] += aw * bw;

#else
  // queue_t in_queue = get_default_queue();
  float4 prod = a[gid] * b[gid];
  barrier(CLK_LOCAL_MEM_FENCE);
  // d[gid] = prod;
  c[gid] = prod.x + prod.y + prod.z + prod.w;
  // void (^ken_wrapper)(void) = ^{dot_product_child(a, b, c, d);};
  // enqueue_kernel(ken_wrapper);
  // c[gid] = in_queue->x;
#endif
 

}

