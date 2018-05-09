#define BFS_VARIANT "worklistc"

//#define MAXDIST		100

#define AVGDEGREE	2.5
#define WORKPERTHREAD	1

unsigned int NVERTICES;

#include <cub/cub.cuh>
#include "worklistc.h"
#include "gbar.cuh"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "common.h"
#include "myutils.h"
#include "Structs.h"
//#include "worklist7.h"

const int BLKSIZE = 256;
//const int BLKSIZE = 256;
const int IN_CORE = 1;     // set this to zero to disable global-barrier version

//texture <unsigned, 1, cudaReadModeElementType> columns;
texture <unsigned, 1, cudaReadModeElementType> row_offsets;
/*
__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("ii=%d, nv=%d.\n", ii, *nv);
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}
*/
//rklist wl1(2), wl2(2);
/*void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}*/
__device__
foru processedge2(foru *dist, unsigned hnodes,unsigned *nodesigma,unsigned *edgesigma,unsigned *srcsrc,unsigned *edgesdst,unsigned iteration, unsigned edge, unsigned &dst, unsigned *border) {
  
  //dst = tex1Dfetch(columns, edge);
  dst = edgesdst[edge];
 unsigned src = srcsrc[edge];
  unsigned sigma_src = nodesigma[src],dist_dst;

  if (dst >= hnodes) return 0;
dist_dst = dist[dst];
 // foru wt = 1;	//graph.getWeight(src, ii);
  //if (wt >= MYINFINITY) return 0;

  //dist_src=cub::ThreadLoad<cub::LOAD_CG>(dist + src);
 // dist_src = dist[src];
  //dist_dst = cub::ThreadLoad<cub::LOAD_CG>(dist + dst);
 // dist_dst = dist[dst];
 // if((dist_src + wt ) < dist_dst)
  //  {
	    //if(dist_src==cub::ThreadLoad<cub::LOAD_CG>(dist + src))
	    //if(dist_dst==cub::ThreadLoad<cub::LOAD_CG>(dist + dst))
	    //{
      		//cub::ThreadStore<cub::STORE_CG>(dist + dst, dist_src + wt);
		//atomicExch(&dist[dst],dist_src+wt);
	//	atomicMin(&dist[dst], dist_src+wt);
		
	//	edgesigma[edge]=cub::ThreadLoad<cub::LOAD_CG>(nodesigma + src);
		//atomicExch(&dist[dst],dist[src]+wt);
	//	atomicExch(&nodesigma[dst],edgesigma[edge]);
      		//cub::ThreadStore<cub::STORE_CG>(nodesigma + dst,edgesigma+edge );
/*		if(dist_dst >= iteration || dist_dst == MYINFINITY)
		{//	printf("dist[%d] = %d\n",dst,iteration);
			dist[dst] = iteration;
			atomicAdd(nodesigma+dst,sigma_src);
      			printf("src = %d, sigma[%d] = %d\n",src,dst,nodesigma[dst]);
			return 1;
		}
		
*/
		if(atomicCAS(edgesigma+edge,0,1)==0 && dist_dst >= iteration)
		{
			dist[dst] = iteration;
			atomicAdd(nodesigma+dst,sigma_src);
		//	printf("src = %d, sigma[%d] = %d\n",src,dst,nodesigma[dst]);
                        if(border[dst]==0)
			return 1;
		}
	    //}
	    //return 0;
   // }
 //  else if((dist_src + wt ) == dist[dst]){
//	   atomicSub(&nodesigma[dst],edgesigma[edge]);
//	   edgesigma[edge]=cub::ThreadLoad<cub::LOAD_CG>(nodesigma + src);
//	   atomicAdd(&nodesigma[dst],edgesigma[edge]);
	   //return 1;
  // }
//  else
//	  edgesigma[edge] = 0;
  
  return 0;
}

/*
__device__
foru processedge2(foru *dist, unsigned hnodes,unsigned iteration, unsigned edge, unsigned &dst) {

  dst = tex1Dfetch(columns, edge);

  if (dst >= hnodes) return 0;

  foru wt = 1;  //graph.getWeight(src, ii);
  if (wt >= MYINFINITY) return 0;

  if(cub::ThreadLoad<cub::LOAD_CG>(dist + dst) == MYINFINITY)
  {
      cub::ThreadStore<cub::STORE_CG>(dist + dst, iteration);
      return MYINFINITY;
  }
    return 0;
}
*/
/*
__device__
foru processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//dst = graph.getDestination(src, ii);	

  // no bounds checking here
  dst = tex1Dfetch(columns, tex1Dfetch(row_offsets, graph.srcsrc[src]) + ii);

	//printf("%d %d %d %d\n", dst, tex1Dfetch(columns, tex1Dfetch(row_offsets, graph.srcsrc[src]) + ii));


	if (dst >= graph.nnodes) return 0;

	foru wt = 1;	//graph.getWeight(src, ii);
	if (wt >= MYINFINITY) return 0;

	//printf("%d %d %d %d\n", src, dst, dist[src], dist[dst]);

	foru altdist = cub::ThreadLoad<cub::LOAD_CG>(dist + src) + wt;
	 
	 if(cub::ThreadLoad<cub::LOAD_CG>(dist + dst) == MYINFINITY)
	   {
	     cub::ThreadStore<cub::STORE_CG>(dist + dst, altdist);
	     return MYINFINITY;
	   }
	 
	 // if (altdist < dist[dst]) { 
	 // 	foru olddist = atomicMin(&dist[dst], altdist); 
	 // 	if (altdist < olddist) { 
	 // 		return olddist; 
	 // 	}  
	 // 	// someone else updated distance to a lower value. 
	 // } 
	 return 0;
}

__device__ void expandByCTA(foru *dist, Graph &graph, Worklist2 &inwl, Worklist2 &outwl, unsigned iteration)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  int nn;

  typedef cub::BlockScan<int, BLKSIZE> BlockScan;

  __shared__ int owner;
  __shared__ int shnn;

  int total_inputs = (*inwl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);

  owner = -1;

  while(total_inputs-- > 0)
    {      
      int neighborsize = 0;
      int neighboroffset = 0;
      int nnsize = 0;

      if(inwl.pop_id(id, nn))
	{	  
	  neighborsize = nnsize = graph.getOutDegree(nn);
	  neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[nn]);	  
	}

      while(true)
	{
	  if(nnsize > BLKSIZE)
	    owner = threadIdx.x;

	  __syncthreads();
	  
	  if(owner == -1)
	    break;

	  if(owner == threadIdx.x)
	    {
	      shnn = nn;
	      cub::ThreadStore<cub::STORE_CG>(inwl.dwl + id, -1);
	      owner = -1;
	      nnsize = 0;
	    }

	  __syncthreads();

	  neighborsize = graph.getOutDegree(shnn);
	  neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[shnn]);
	  int xy = ((neighborsize + blockDim.x - 1) / blockDim.x) * blockDim.x;
	  
	  for(int i = threadIdx.x; i < xy; i+= blockDim.x)
	    {
	      int ncnt = 0;
	      unsigned to_push = 0;

	      if(i < neighborsize)
		if(processedge2(dist, graph, iteration, neighboroffset + i, to_push))
		  {
		    ncnt = 1;
		  }
	    
	      outwl.push_1item<BlockScan>(ncnt, (int) to_push, BLKSIZE);
	    }
	}

      id += gridDim.x * blockDim.x;
    }
}
*/

__device__
unsigned processnode2(foru *dist, unsigned hnodes,unsigned *nodesigma,unsigned *edgesigma,unsigned *srcsrc,unsigned *edgesdst,unsigned *noutgoing , Worklist &inwl, Worklist &outwl, unsigned iteration, unsigned *border) 
{
  //expandByCTA(dist, graph, inwl, outwl, iteration);

  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned nn,div =1;
;

  typedef cub::BlockScan<int, BLKSIZE> BlockScan;

  const int SCRATCHSIZE = BLKSIZE;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ int gather_offsets[SCRATCHSIZE];

  gather_offsets[threadIdx.x] = 0;

  int total_inputs = (*inwl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);
  
  while(total_inputs > 0)
    {      
	total_inputs -= (gridDim.x * blockDim.x)/div;
      int neighborsize = 0;
      int neighboroffset = 0;
      int scratch_offset = 0;
      int total_edges = 0;

	if(id%div == 0)
{
      if(inwl.pop_id(id/div, nn) )
	{	  
	  if(nn != -1)
	    {
	      neighborsize = noutgoing[nn];
	      neighboroffset = tex1Dfetch(row_offsets, nn);
	      //neighboroffset = psrc[nn];
	    }
	}
}
      BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
  
      int done = 0;
      int neighborsdone = 0;

       //if(total_edges) 
       	//if(threadIdx.x == 0) 
       	  //printf("total edges: %d\n", total_edges); 

      while(total_edges > 0)
	{
	  __syncthreads();

	  int i;
	  for(i = 0; neighborsdone + i < neighborsize && (scratch_offset + i - done) < SCRATCHSIZE; i++)
	    {
	      gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
	    }

	  neighborsdone += i;
	  scratch_offset += i;

	  __syncthreads();

	  int ncnt = 0;
	  unsigned to_push = 0;

	  if(threadIdx.x < total_edges)
	    {
	      if(processedge2(dist, hnodes, nodesigma,edgesigma,srcsrc,edgesdst,iteration, gather_offsets[threadIdx.x], to_push, border))
	      //if(processedge2(dist, hnodes, iteration, gather_offsets[threadIdx.x], to_push))
		{
		  ncnt = 1;
		}
	    }
	//  printf("%u ", to_push);

	  outwl.push_1item<BlockScan>(ncnt, (int) to_push, BLKSIZE);
      
	  total_edges -= BLKSIZE;
	  done += BLKSIZE;
	}

      id +=( blockDim.x * gridDim.x);
    }

  return 0;
}

/*
__device__
unsigned processnode(foru *dist, Graph &graph, Worklist2 &inwl, Worklist2 &outwl) 
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

  int neighbours[256];
  int ncnt = 0;
  int nn;

  if(inwl.pop_id(id, nn))
	{
	  unsigned neighborsize = graph.getOutDegree(nn);

	  if(neighborsize > 256)
	  	printf("whoa! out of local space");
	  
	  for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
		  //printf("%d pushing %d\n", nn, dst);
		  neighbours[ncnt] = dst;
		  ncnt++;
		}
      }
    }

  typedef cub::BlockScan<int, BLKSIZE> BlockScan;

  return outwl.push_nitems<BlockScan>(ncnt, neighbours, BLKSIZE) == 0 && ncnt > 0;
}
*/
__device__
void drelax(foru *dist, unsigned hnodes, unsigned *nodesigma,unsigned *edgesigma,unsigned *srcsrc, unsigned *edgesdst,unsigned *noutgoing,unsigned *gerrno, Worklist &inwl, Worklist& outwl, int iteration, unsigned *border) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
/*
	if(iteration == 0)
	  {
	    if(id == 0)
	      {
			int item = 0;
			inwl.push(item);
	      }
	    return;	    
	  }
	else
	  {
	 */
	    if(processnode2(dist, hnodes,nodesigma,edgesigma,srcsrc, edgesdst,noutgoing,inwl, outwl, iteration,border))
	      *gerrno = 1;
	  //}
}
__device__
void initialize(foru *dist, unsigned hnodes, int *sources,unsigned count,Worklist &inwl, Worklist &outwl) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id = 0;
	int item, ii;
//	inwl.dindex = 0;
	while(id < count)
	  //  if(id == 0)
	      {
			//for(ii = 0 ; ii < noutgoing[sources[id]] ; ii++){
			
				//item = psrc[sources[id]]+ii;
				item = sources[id];
				dist[item] = 0;
				inwl.push(item);
			//}
	    		//id += blockDim.x;
			++id;
	      }
	    return;	    
}

__global__ void drelax3(foru *dist, unsigned hnodes,int *d_sources,unsigned source_count,unsigned *nodesigma,unsigned *edgesigma,unsigned *srcsrc, unsigned *edgesdst,unsigned *noutgoing,unsigned *gerrno, Worklist inwl, Worklist outwl, int iteration)
{
	if(iteration == 0)
		initialize(dist, hnodes,d_sources,source_count,inwl, outwl);
//	else
//		drelax(dist, hnodes,nodesigma,edgesigma,srcsrc, edgesdst,noutgoing,gerrno, inwl, outwl, iteration);
}


__device__ int addSourcesOnCurrentLevel(foru *dist, int *d_sources, unsigned processed_count, Worklist &inwl, unsigned iteration, int *d_source_sigma, unsigned *nodesigma)
{
        int id = processed_count;
        int item = d_sources[id],dist_item;
	if(item == -iteration)
	{
		id++;
		item = d_sources[id];
		do
		{
			dist_item = dist[item];
			if(dist_item >= iteration || dist_item == MYINFINITY)
			{
				inwl.push(item);
				dist[item] = iteration;
				nodesigma[item] = d_source_sigma[id];
			}
	//		printf("push %d in iteration %d\n",item,iteration);
			id++;
			item = d_sources[id];
		}while(item >= 0);
	}
	return id;
}

__global__ void drelax2(foru *dist, unsigned hnodes,int *d_sources,unsigned source_count,unsigned *nodesigma,unsigned *edgesigma,unsigned *srcsrc, unsigned *edgesdst,unsigned *noutgoing,unsigned *gerrno, Worklist inwl, Worklist outwl, int iteration, GlobalBarrier gb, unsigned *border, int *d_source_sigma)
{
	unsigned processed_count = 0;
	Worklist *in;
        Worklist *out;
        Worklist *tmp;
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		processed_count = addSourcesOnCurrentLevel(dist, d_sources, processed_count, inwl, iteration, d_source_sigma, nodesigma); 
	}
//	if (threadIdx.x == 0 && blockIdx.x == 0) printf("[GPU1] iteration = %d\n",iteration);
	gb.Sync();
//	if (threadIdx.x == 0 && blockIdx.x == 0) printf("[GPU2] iteration = %d\n",iteration);
	in = &inwl; out = &outwl;

	while(*in->dindex > 0) // && iteration < 30)
        {
		iteration++;
		drelax(dist, hnodes,nodesigma,edgesigma,srcsrc, edgesdst,noutgoing,gerrno, *in, *out, iteration, border);

		//__threadfence_system();
//		if (threadIdx.x == 0 && blockIdx.x == 0) printf("[GPU3] iteration = %d\n",iteration);
	//	gb.Sync();
//		if (threadIdx.x == 0 && blockIdx.x == 0) printf("[GPU4] iteration = %d\n",iteration);
		if (threadIdx.x == 0 && blockIdx.x == 0)
		{
			if(processed_count+1 < source_count)
				processed_count = addSourcesOnCurrentLevel(dist, d_sources, processed_count, *out, iteration, d_source_sigma, nodesigma);
        	}
		gb.Sync();
//		if (threadIdx.x == 0 && blockIdx.x == 0) printf("[GPU5] iteration = %d\n",iteration);
          	tmp = in;
          	in = out;
          	out = tmp;

          	*out->dindex = 0;
          	//printf_array(dist,hnodes);
          	//iteration++;
        }
//if (threadIdx.x == 0 && blockIdx.x == 0) printf("[GPU3] iteration = %d\n",iteration);
}

__global__ void drelax2_SS(foru *dist, unsigned hnodes,int *d_sources,unsigned source_count,unsigned *nodesigma,unsigned *edgesigma,unsigned *srcsrc, unsigned *edgesdst,unsigned *noutgoing,unsigned *gerrno, Worklist inwl, Worklist outwl, int iteration, GlobalBarrier gb, unsigned *border)
{
       // clock_t start_time = clock(), stop_time;
  if(iteration == 0)
          initialize(dist, hnodes,d_sources,source_count,inwl, outwl);
    //drelax(dist, hnodes, srcsrc, noutgoing,gerrno, inwl, outwl, iteration);
  else{
      Worklist *in;
      Worklist *out;
      Worklist *tmp;

      in = &inwl; out = &outwl;

      while(*in->dindex > 0) // && iteration < 30)
        {
	//	iteration++;

          drelax(dist, hnodes,nodesigma,edgesigma,srcsrc, edgesdst,noutgoing,gerrno, *in, *out, iteration,border);
	//iteration++;

          //__threadfence_system();
         // gb.Sync();
       // if (threadIdx.x == 0 && blockIdx.x == 0)
       // {
       //         stop_time = clock();
         //       runtime[iteration] = (int)(stop_time - start_time);
           //     start_time = clock();
	//	printf("\n");
       // }
        gb.Sync();
          tmp = in;
          in = out;
          out = tmp;

          *out->dindex = 0;
          //printf_array(dist,hnodes);
          iteration++;
        }
    
//if (threadIdx.x == 0 && blockIdx.x == 0) printf("[GPU-SS] iteration = %d\n",iteration);
	}
//runtime[0] = iteration;
}
__global__ void print_array(int *a, int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    printf("%d %d\n", id, a[id]);
}

/*__global__ void print_texture(int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    printf("%d %d\n", id, tex1Dfetch(columns, id));
}*/

void lonestar_gpu(unsigned *psrc,unsigned *noutgoing, unsigned *d_psrc,unsigned *d_noutgoing,unsigned *d_edgessrc,unsigned *d_edgesdst,unsigned hnodes,unsigned hedges,unsigned *dist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,int *sources,cudaDeviceProp *deviceProp,bool BM_COMP,unsigned *nerr, unsigned *d_border, int *sigma)
//		Graph &graph, foru *dist)
{
	foru foruzero = 0;
	unsigned int NBLOCKS, FACTOR = 128;
	//bool *changed;
	int iteration = 0;
	//unsigned *nerr;
	int *d_sources,*d_source_sigma;
	unsigned ii;
	//source_count =1 ;
	double starttime, endtime;
	double runtime;

	//cudaDeviceProp deviceProp;
	//cudaGetDeviceProperties(&deviceProp, 0);
	NBLOCKS = deviceProp->multiProcessorCount;

	NVERTICES = hnodes;

	FACTOR = (NVERTICES + MAXBLOCKSIZE * NBLOCKS - 1) / (MAXBLOCKSIZE * NBLOCKS);

	//printf("initializing (nblocks=%d, blocksize=%d).\n", NBLOCKS*FACTOR, MAXBLOCKSIZE);
	//initialize <<<NBLOCKS*FACTOR, MAXBLOCKSIZE>>> (dist, graph.nnodes);
	//CudaTest("initializing failed");
	//cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);
//	starttime = rtclock();
	//if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");
	
//	if ( cudaMalloc((void **)&d_check, hnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating d_check failed");
	//starttime = rtclock() - starttime;
//	endtime = rtclock();
//	cudaMemset(d_check, 0, hnodes*sizeof(unsigned));
//	endtime = rtclock() - endtime;
//	printf("time_malloc = %.3lf, time_memset = %.3lf",starttime*1000,endtime*1000);
	//if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");


	if (cudaMalloc((void **)&d_sources, source_count * sizeof(int)) != cudaSuccess) CudaTest("allocating d_sources failed");
	cudaMemcpy(d_sources,sources,source_count * sizeof(int),cudaMemcpyHostToDevice);


	/*printf("sources are\n");
	for(ii=0;ii<source_count;ii++){
		printf("%u ",sources[ii]+1);
	}
	printf("\n");
	*/
//	cudaMemset(d_check, 0, hnodes*sizeof(unsigned));
	//printf("#Edges gpu partition %u",hedges);
	//starttime = rtclock();
	Worklist wl1(2*hedges), wl2(2*hedges);
	//wl1.ensureSpace(2*hedges);	wl2.ensureSpace(2*hedges);
	Worklist *inwl = &wl1, *outwl = &wl2;
	int nitems = 1;
//	endtime = rtclock();
  //      printf("worlist init Time : %f\n", 1000.0*(endtime - starttime));


	//cudaBindTexture(0, columns, d_edgesdst, (hedges + 1) * sizeof(unsigned));
	cudaBindTexture(0, row_offsets, d_psrc, (hnodes + 1) * sizeof(unsigned));

	//print_array<<<1, graph.nedges + 1>>>((int *) graph.edgessrcdst, graph.nedges + 1);
	//print_texture<<<1, graph.nedges + 1>>>(graph.nedges + 1);
	//return;


	//printf("solving.\n");
	//printf("starting...\n");
	//starttime = rtclock();

	if(IN_CORE) {
	 GlobalBarrierLifetime gb;
	 const size_t drelax2_max_blocks = maximum_residency(drelax2, BLKSIZE, 0);
	 gb.Setup(deviceProp->multiProcessorCount * drelax2_max_blocks);


	 //gb.Setup(26);
	//gb.Setup(1);

	  //printf("No of gpu blocks %d",(deviceProp->multiProcessorCount * drelax2_max_blocks));

	 // drelax2<<<1, BLKSIZE>>>(dist, hnodes,d_edgessrc,d_noutgoing, nerr, *inwl, *outwl, 0, gb);
	 // initialize<<<1, BLKSIZE>>>(dist, hnodes,d_edgessrc,d_noutgoing,d_sources,source_count,*inwl, *outwl);
	  /*
	  inwl->display_items();
	  printf("Sources are :-\n");
	  for(unsigned ii=0;ii < source_count ;ii++)
		  printf("%u ",sources[ii]);
	*/
	int iteration = 0;
//	cout<<"Hi "<<endl;
	if(source_count>1)
	{
		starttime = rtclock();
		iteration = -sources[0];
		if (cudaMalloc((void **)&d_source_sigma, source_count * sizeof(int)) != cudaSuccess) CudaTest("allocating d_source_sigma failed");
        	cudaMemcpy(d_source_sigma,sigma,source_count * sizeof(int),cudaMemcpyHostToDevice);
//		cout<<"Entering Kernal : iteration = "<<iteration<<" sources[0] = "<<sources[0]<<endl;
	//	for(int i=0;i<source_count;i++)
	//		cout<<sources[i]<<" ";
	//	cout<<endl;
	//	drelax2 <<<13, BLKSIZE>>> (dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, iteration, gb);
		drelax2 <<<13 * drelax2_max_blocks, BLKSIZE>>>(dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, iteration, gb,d_border,d_source_sigma);
		cudaThreadSynchronize();
		endtime = rtclock();
		printf("MultiS Time : %f\n", 1000.0*(endtime - starttime));
	        checkCUDAError("kernel invocation[>1]");
		cudaFree(d_source_sigma);

	}
	else
	{	//starttime = rtclock();
		drelax2_SS <<<1, 1>>> (dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, 0, gb,d_border);
        //	drelax2_SS <<<13, BLKSIZE>>> (dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, 1, gb);
		drelax2_SS <<<13 * drelax2_max_blocks, BLKSIZE>>>(dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, 1, gb,d_border);
        cudaThreadSynchronize();
	//endtime = rtclock();
        //printf("GPU-S Time : %f\n", 1000.0*(endtime - starttime));
	checkCUDAError("kernel invocation[=1]");
	 }
//	cudaThreadSynchronize();
//	checkCUDAError("kernel invocation");
//	cout<<"Iresh"<<endl;
	  //initialize<<<1, 1>>>(dist, hnodes,d_psrc,d_noutgoing,d_sources,source_count,*inwl, *outwl);
	  //inwl->display_items();
//	else
	//  drelax2 <<<1, 1>>> (dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, 0, gb,d_check);
	 // drelax2 <<<(deviceProp->multiProcessorCount * drelax2_max_blocks), BLKSIZE>>> (dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, iteration, gb,d_check);
	// drelax2 <<<12, BLKSIZE>>> (dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, iteration, gb,d_check);  
//drelax2 <<<26, BLKSIZE>>> (dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc,d_edgesdst,d_noutgoing, nerr, *inwl, *outwl, 1, gb);
	  //printf("\nCODE REACHING HERE\n");
	}

	
	else {
	/*  drelax3<<<1, 1>>>(dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc, d_edgesdst,d_noutgoing , nerr, *inwl, *outwl, 0);
	  nitems = inwl->nitems();

	  while(nitems > 0) {
	    ++iteration;
	    unsigned nblocks = (nitems + BLKSIZE - 1) / BLKSIZE; 
	    //printf("%d %d %d %d\n", nblocks, BLKSIZE, iteration, nitems);
	    //printf("ITERATION: %d\n", iteration);
	    //inwl->display_items();

	    //drelax3<<<nblocks, BLKSIZE>>>(dist, graph, nerr, *inwl, *outwl, iteration);
	  drelax3<<<nblocks, BLKSIZE>>>(dist, hnodes,d_sources,source_count,nodesigma,edgesigma,d_edgessrc, d_edgesdst,d_noutgoing , nerr, *inwl, *outwl, iteration);
  
	    nitems = outwl->nitems();

	    //printf("worklist size: %d\n", nitems);
		
	    Worklist *tmp = inwl;
	    inwl = outwl;
	    outwl = tmp;
	    
	    outwl->reset();
	  };  
*/	}
	
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	//CUDA_SAFE_CALL(cudaFree(changed));
	//endtime = rtclock();
	
	//printf("\titerations = %d.\n", iteration);
	//runtime = (1000.0f * (endtime - starttime));
	//printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
/*	free(wl1.wl);
	cudaFree(wl1.dwl);
	cudaFree(wl1.dnsize);
	cudaFree(wl1.dindex);
	free(wl2.wl);
	cudaFree(wl2.dwl);
	cudaFree(wl2.dnsize);
	cudaFree(wl2.dindex);
*/
	cudaFree(d_sources);
//	cudaFree(d_check);
	cudaUnbindTexture(row_offsets);
	return;
}
