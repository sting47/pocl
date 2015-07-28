#define USE_VECTOR_DATATYPES
#include "/home/sting47/Library/test/pocl/lib/kernel/enqueue_kernel.c"
// #include "/home/sting47/Library/test/pocl/lib/put_in.c"
// #include "/home/sting47/Library/test/pocl/lib/get_default_queue.c"

#if 0
__kernel void
kenhaha (__global const int *a,  
         __global const int *b, __global float *c)
{
  int gid = get_global_id(0);
  c[gid] = 4.0;
}
#endif

__kernel void 
dot_product_child (__global const int *a,
	     __global int *b, __global int *c) 
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
  // int res = b[gid-1] + b[gid] + b[gid+1];
  int res;
  int max = get_global_size(0);
  if(gid<128)
    res = b[gid] + b[gid+128];
  else if(gid>max-128)
    res = b[gid] + b[gid-128];
  else
    res = b[gid-128] + b[gid] + b[gid+128];
  // c[gid] = prod.x + prod.y + prod.z + prod.w;
  c[gid] = res;
  // c[gid] = in_queue->x;
#endif
 

}

__kernel void 
dot_product (__global const int *a,  
	     __global int *b, __global int *c, volatile __global int *counter) 
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
  int res;
  int id = get_local_id(0);
  int max = get_global_size(0);
  if(gid<128)
    res = a[gid] + a[gid+128];
  else if(gid>max-128)
    res = a[gid] + a[gid-128];
  else
    res = a[gid-128] + a[gid] + a[gid+128];
  // barrier(CLK_LOCAL_MEM_FENCE);
  // d[gid] = prod;
  // c[gid] = prod.x + prod.y + prod.z + prod.w;
  // res = a[gid] + a[gid+1] + a[gid+2];
  b[gid] = res;
  // c[gid] = res;
  // barrier(CLK_LOCAL_MEM_FENCE);
  if(id==0)
    counter[0] = counter[0] - 1;

  int tmp_test = 0;
  while(counter[0]!=0){
    tmp_test++;
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  // if(id==0)
    // barrier(CLK_GLOBAL_MEM_FENCE);
  barrier(CLK_GLOBAL_MEM_FENCE);
  void (^ken_wrapper)(void) = ^{dot_product_child(a, b, c);};
  enqueue_kernel(ken_wrapper);
  // c[gid] = in_queue->x;
#endif
 

}

