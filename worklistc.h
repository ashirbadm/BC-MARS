#ifndef WORKLISTC
#define WORKLISTC

#pragma once


#include "cutil_subset.h"

static int zero = 0;
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

struct Worklist {
  int *dwl, *wl;
  int length, *dnsize;
  int *dindex, *visited_count;

  Worklist (){
	 visited_count = dwl = wl = dnsize = dindex = NULL;
  }
  Worklist(size_t nsize)
  {
	//printf("Worlist defined\n");
    length = nsize;
    wl = (int *) calloc(nsize, sizeof(int));
    CUDA_SAFE_CALL(cudaMalloc((void **)&dwl, nsize * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&dnsize, 1 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&dindex, 1 * sizeof(int)));
	visited_count = NULL;
    CUDA_SAFE_CALL(cudaMemcpy(dnsize, &nsize, 1 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy((void *) dindex, &zero, 1 * sizeof(zero), cudaMemcpyHostToDevice));
  //  printf("dwl = %ld,dnsize = %ld\n",dwl,dnsize);
  }
void addMultiSourceSupport( size_t nodes)
{
        CUDA_SAFE_CALL(cudaMalloc((void **)&visited_count, (nodes+1) * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset((void *) visited_count, 0, (nodes+1) * sizeof(int)));
}

  ~Worklist()
  {
	//printf("Worlist destroyed\n");
	  /*
    if(wl!=NULL)free(wl);
    if(dwl!=NULL)CUDA_SAFE_CALL(cudaFree(dwl));
    if(dnsize!=NULL)CUDA_SAFE_CALL(cudaFree(dnsize));
    if(dindex!=NULL)CUDA_SAFE_CALL(cudaFree(dindex));
    */
  }
  void ensureSpace(int space)
  {
	int nsize = length;
	//CUDA_SAFE_CALL(cudaMemcpy(&nsize, (void *) dnsize, 1 * sizeof(nsize), cudaMemcpyDeviceToHost));
//	printf("nsize = %d\n",nsize);
	if (nsize<space)
	{
		nsize = space;
		length = space;
//		printf("Resetting\n");
		wl = (int *) calloc(nsize, sizeof(int));
		if(dwl!=NULL)	CUDA_SAFE_CALL(cudaFree(dwl));
		CUDA_SAFE_CALL(cudaMalloc((void **)&dwl, nsize * sizeof(int)));
//		checkCUDAError("Malloc\n");
		CUDA_SAFE_CALL(cudaMemcpy(dnsize, &nsize, 1 * sizeof(int), cudaMemcpyHostToDevice));
//		checkCUDAError("memcpy\n");
	}
	CUDA_SAFE_CALL(cudaMemcpy((void *) dindex, &zero, 1 * sizeof(zero), cudaMemcpyHostToDevice));
  }
  void update_cpu()
  {
    int nsize = nitems();
    CUDA_SAFE_CALL(cudaMemcpy(wl, dwl, nsize  * sizeof(int), cudaMemcpyDeviceToHost));
  }

  void display_items()
  {
    int nsize = nitems();
    CUDA_SAFE_CALL(cudaMemcpy(wl, dwl, nsize  * sizeof(int), cudaMemcpyDeviceToHost));

    printf("WL: ");
    for(int i = 0; i < nsize; i++)
      printf("%d %d, ", i, wl[i]);

    printf("\n");
    return;
  }

  void reset()
  {
    CUDA_SAFE_CALL(cudaMemcpy((void *) dindex, &zero, 1 * sizeof(zero), cudaMemcpyHostToDevice));
  }

  int nitems()
  {
    int index;
    CUDA_SAFE_CALL(cudaMemcpy(&index, (void *) dindex, 1 * sizeof(index), cudaMemcpyDeviceToHost));

    return index;
  }

  __device__ 
  int push(int item)
  {
    int lindex = atomicAdd((int *) dindex, 1);
    if(lindex >= *dnsize)
      return 0;

    dwl[lindex] = item;
    return 1;
  }

  __device__
  int pop(int &item)
  {
    int lindex = atomicSub((int *) dindex, 1);
    if(lindex <= 0)
      {
	*dindex = 0;
	return 0;
      }

    item = dwl[lindex - 1];
    return 1;
  }
//};

/*struct Worklist2: public Worklist {
  Worklist2(int nsize) : Worklist(nsize) {}
  ~Worklist2() {
	  //free(wl);
	  //    CUDA_SAFE_CALL(cudaFree(dwl));
	  //        //CUDA_SAFE_CALL(cudaFree(dnsize));
	  //            //CUDA_SAFE_CALL(cudaFree(dindex));
	  //
  }
*/


  template <typename T>
  __device__ __forceinline__
    int push_1item(int nitem, int item, int threads_per_block)
  {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ int queue_index;
    int total_items = 0;
    int thread_data = nitem;

    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

    if(threadIdx.x == 0)
      {	
	//if(debug) printf("t: %d\n", total_items);
	queue_index = atomicAdd((int *) dindex, total_items);
	//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
      }

    __syncthreads();

    if(nitem == 1)
      {
	if(queue_index + thread_data >= *dnsize)
	  {
	    printf("GPU: exceeded length: %d %d %d\n", queue_index, thread_data, *dnsize);
	    return 0;
	  }

	//dwl[queue_index + thread_data] = item;
	cub::ThreadStore<cub::STORE_CG>(dwl + queue_index + thread_data, item);
      }

    return total_items;
  }

  template <typename T>
  __device__ __forceinline__
  int push_nitems(int n_items, int *items, int threads_per_block)
  {
    __shared__ typename T::TempStorage temp_storage;
    __shared__ int queue_index;
    int total_items;

    int thread_data = n_items;

    T(temp_storage).ExclusiveSum(thread_data, thread_data, total_items);

    if(threadIdx.x == 0)
      {	
	queue_index = atomicAdd((int *) dindex, total_items);
	//printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
      }

    __syncthreads();

    
    for(int i = 0; i < n_items; i++)
      {
	//printf("pushing %d to %d\n", items[i], queue_index + thread_data + i);
	if(queue_index + thread_data + i >= *dnsize)
	  {
	    printf("GPU: exceeded length: %d %d %d %d\n", queue_index, thread_data, i, *dnsize);
	    return 0;
	  }

	dwl[queue_index + thread_data + i] = items[i];
      }

    return total_items;
  }

  __device__ 
  int pop_id(int id, unsigned &item)
  {
    if(id < *dindex)
      {
	item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
	//item = dwl[id];
	return 1;
      }
    
    return 0;
  }

	  __device__
  int pop_id(int id, unsigned &item, int level)
  {
        id += visited_count[level];
    if(id < visited_count[level+1])
      {
        item = cub::ThreadLoad<cub::LOAD_CG>(dwl + id);
        //printf("Pop %d\n", item);
        return 1;
      }

                          return 0;
   }

__device__
int countVerticesOnLevel(unsigned int level)
{
        return (visited_count[level+1]-visited_count[level]);
}
__device__
void level_complete(unsigned int &level)
{
	visited_count[level+1] = *dindex;
}


  
};
//Worklist wl1(2), wl2(2);
#endif
