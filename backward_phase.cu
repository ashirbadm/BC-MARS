#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <string.h>
#include <sys/time.h>
#include "Structs.h"
#include "myutils2.h"
#include "gbar.cuh"
//#include "graph28.h"

//__device__ bool *d_gpu_wait;
/*void cpu_component (unsigned *edgesrc,unsigned *edgedst,unsigned nnodes,unsigned nedges,unsigned *nodedist,unsigned *nodesigma,unsigned *edgesigma, float *nodedelta,bool *wait,int nthreads,unsigned *border,long level){
	//bool lchanged[nthreads];
#pragma omp parallel default(shared) firstprivate(level,nedges) num_threads(nthreads)
	{
		bool lchanged;
		unsigned src,dst,ii,tid;
		float delta_src,delta_dst;
		tid = omp_get_thread_num();
		//lchanged[tid] = false;
		lchanged = false;
#pragma omp for schedule(guided)
        for (ii = 0; ii < nedges; ii++) {
		// Don't process if already processed or not proccessable
		dst = edgedst[ii];
		if(nodedist[dst]!=level) continue;
		if(edgesigma[ii]==0) continue;
		if(border[dst]!=0){
			//lchanged[tid] = true;
			lchanged = true;
			continue;
		}// Don't process and wait if it is a border node
		src = edgesrc[ii];
		// wt = edgewt[ii];
		delta_dst = nodedelta[dst];
		delta_dst += 1;
		if(nodesigma[dst]!=0)
		delta_src = ((float)(nodesigma[src]/nodesigma[dst])*delta_dst);
#pragma omp atomic update
		nodedelta[src] += delta_src;
		edgesigma[ii]=0;
	}
	if(lchanged) *wait = true;
	//for(ii =2 ; ii < nthreads ;ii+=2){
	//	if(tid%ii==0)
	//		if(lchanged[tid] || lchanged[tid+(ii-1)]) lchanged[tid] = true;
	//#pragma omp barrier	
	//}
//#pragma omp master
//	*wait = lchanged[tid];
	//}
}*/
void cpu_component (unsigned *edgesrc,unsigned *edgedst,unsigned nnodes,unsigned nedges,unsigned *nodedist,unsigned *nodesigma,unsigned *edgesigma, float *nodedelta,bool *wait,int nthreads,unsigned *border,long level,unsigned *psrc, unsigned *noutgoing){
	bool lchanged[nthreads];
#ifdef _OPENMP
#pragma omp parallel default(shared) firstprivate(level,nnodes,edgedst,psrc,noutgoing) num_threads(nthreads)
	{
#endif
		//bool lchanged;
		unsigned src,dst,ii,tid,j, num_edges_v;
		float delta_src,delta_dst;
		tid = omp_get_thread_num();
		lchanged[tid] = false;
		//lchanged = false;
#ifdef _OPENMP
#pragma omp for schedule(guided) nowait
#endif
        for (ii = 0; ii < nnodes; ii++) {
		num_edges_v = psrc[ii];
		if(nodedist[ii]==level){
			for(j =  num_edges_v; j < (num_edges_v + noutgoing[ii]); j++){
				dst = edgedst[j];
				if(nodedist[dst]!=level+1) continue;
				if(edgesigma[j]==0) continue;
				if(border[dst]!=0){
					lchanged[tid] = true;
					//lchanged = true;
				continue;
				}// Don't process and wait if it is a border node
				src = ii;
				// wt = edgewt[ii];
				delta_dst = nodedelta[dst];
				delta_dst += 1;
				if(nodesigma[dst]!=0)
				delta_src = ((float)(nodesigma[src]/nodesigma[dst])*delta_dst);
#ifdef _OPENMP
		#pragma omp atomic
#endif
				nodedelta[src] += delta_src;
				edgesigma[ii]=0;
			}
		}
	}

	/*for(ii =2 ; ii < nthreads ;ii+=2){
		if(tid%ii==0)
			if(lchanged[tid] || lchanged[tid+(ii-1)]) lchanged[tid] = true;
	#pragma omp barrier	
	}
#pragma omp master
	*wait = lchanged[tid];
	*/
#ifdef _OPENMP
	}
#endif
		for(unsigned ii=0 ; ii < nthreads; ii++ )
			if(lchanged[ii]){ *wait = true; break;}
}
/*void cpu_single_component (unsigned *edgesrc,unsigned *edgedst,unsigned nnodes,unsigned nedges,unsigned *nodedist,unsigned *nodesigma,unsigned *edgesigma, float *nodedelta,int nthreads,long level){
	unsigned src,dst,ii;
	float delta_src,delta_dst;
//#pragma omp parallel for schedule(guided) private(ii,delta_src,delta_dst,src,dst,level) num_threads(nthreads)
#pragma omp parallel for schedule(guided) default(shared) private(ii,delta_src,delta_dst,src,dst,level,nedges) num_threads(nthreads)
        for (ii = 0; ii < nedges; ii++) {
		// Don't process if already processed or not proccessable
		dst = edgedst[ii];
		if(nodedist[dst]!=level) continue;
		if(edgesigma[ii]==0) continue;
		src = edgesrc[ii];
		// wt = edgewt[ii];
		delta_dst = nodedelta[dst];
		delta_dst++;
		if(nodesigma[dst]!=0)
		delta_src = ((float)(nodesigma[src]/nodesigma[dst])*delta_dst);
#pragma omp atomic update
		nodedelta[src] += delta_src;
		edgesigma[ii]=0;
	}
}*/
void cpu_single_component (unsigned *edgesrc,unsigned *edgedst,unsigned nnodes,unsigned nedges,unsigned *nodedist,unsigned *nodesigma,unsigned *edgesigma, float *nodedelta,int nthreads,long level,unsigned *psrc, unsigned *noutgoing){
#ifdef _OPENMP
#pragma omp parallel default(shared) firstprivate(level,nnodes,edgedst,psrc,noutgoing) num_threads(nthreads)
	{
#endif
		unsigned src,dst,ii,j, num_edges_v;
		float delta_src,delta_dst;
#ifdef _OPENMP
#pragma omp for schedule(guided) nowait
#endif
        for (ii = 0; ii < nnodes; ii++) {
		// Don't process if already processed or not proccessable
		num_edges_v = psrc[ii];
		if(nodedist[ii]==level){
			for(j =  num_edges_v; j < (num_edges_v + noutgoing[ii]); j++){
				dst = edgedst[j];
				if(nodedist[dst]!=level+1) continue;
				// mention the border
				if(edgesigma[j]==0) continue;
				// wt = edgewt[ii];
				delta_dst = nodedelta[dst];
				delta_dst += 1;
				if(nodesigma[dst]!=0)
				delta_src = ((float)(nodesigma[ii]/nodesigma[dst])*delta_dst);
#ifdef _OPENMP
#pragma omp atomic
#endif
				nodedelta[ii] += delta_src;
				edgesigma[j]=0;
			}
		}
	}
#ifdef _OPENMP
	}
#endif
}

/* function for backward traversal for cpu */
void cpu_backward(struct varto_cpu_part *var)
{
	
Graph *graph = var->graph;
unsigned numEdges_cpu,numNodes_cpu,borderIndex,borderIndex2,ii,borderSource;
int num_threads = var->num_threads;
Graph::DevicePartition *cpupart = var->partition;
Graph::Partition *borderInfo = var->borderInfo;
numEdges_cpu = cpupart->numEdges;
numNodes_cpu = cpupart->numNodes;
long * cpu_level = var->cpu_level ;
volatile long * gpu_level = var->gpu_level;
bool cpu_wait = false;
unsigned borderCount = borderInfo->borderCount[CPUPARTITION]; /* Border Count is of non GPU partition */
double starttime, endtime,tsttime,tendtime;
starttime = rtclock();
unsigned *edgesigma = cpupart->edgesigma;
float *nodedelta = cpupart->nodedelta;
float *gpunodelta = graph->devicePartition[GPUPARTITION].nodedelta;
//#pragma omp for
//for(int i = 0 ; i < graph->nnodes ; i++)
//	cpupart->nodedelta[i] = 0;
//#pragma omp parallel for schedule(static)
for(ii = 0 ; ii < numEdges_cpu ; ii++){
	if(graph->partition.part[cpupart->edgedst[ii]]==GPUPARTITION)
		cpupart->edgesigma[ii] = 0 ;
}
--*cpu_level;
while(*cpu_level > *(var->cpu_level_min)){
	cpu_wait =  false;
	//cpu_component (cpupart->edgesrc,cpupart->edgedst,graph->nnodes,numEdges_cpu,cpupart->nodedist,cpupart->nodesigma,edgesigma,nodedelta,&cpu_wait,num_threads,borderInfo->border,*cpu_level);
	tsttime = rtclock();
	cpu_component (cpupart->edgesrc,cpupart->edgedst,graph->nnodes,numEdges_cpu,cpupart->nodedist,cpupart->nodesigma,edgesigma,nodedelta,&cpu_wait,num_threads,borderInfo->border,*cpu_level,cpupart->psrc,cpupart->noutgoing);
	tendtime = rtclock();
	var->cpu_bck_knl_time += tendtime - tsttime;
	if(cpu_wait && *gpu_level > *(var->gpu_level_min)){
		while(*gpu_level > *cpu_level); // wait for gpu to catch up
	}
	if(cpu_wait){
		// copy border node data
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
#endif
		for(ii = 0 ; ii < borderCount; ii++){
			if(cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][ii]]==*cpu_level && graph->devicePartition[GPUPARTITION].nodedelta[borderInfo->borderNodes[CPUPARTITION][ii]] != cpupart->nodedelta[borderInfo->borderNodes[CPUPARTITION][ii]]){
				//float temp = cpupart->nodedelta[borderInfo->borderNodes[CPUPARTITION][ii]];
				//cpupart->nodedelta[borderInfo->borderNodes[CPUPARTITION][ii]] += graph->devicePartition[GPUPARTITION].nodedelta[borderInfo->borderNodes[CPUPARTITION][ii]];
				cpupart->nodedelta[borderInfo->borderNodes[CPUPARTITION][ii]] += gpunodelta[borderInfo->borderNodes[CPUPARTITION][ii]];
				//graph->devicePartition[GPUPARTITION].nodedelta[borderInfo->borderNodes[CPUPARTITION][ii]] += temp;
			}
		}
		//printf("CPU LEVEL %ld\n",*cpu_level);
//#pragma omp parallel shared(edgesigma,nodedelta) num_threads(num_threads)
//		{
        tsttime = rtclock();
	cpu_single_component (cpupart->edgesrc,cpupart->edgedst,graph->nnodes,numEdges_cpu,cpupart->nodedist,cpupart->nodesigma,edgesigma,nodedelta,num_threads,*cpu_level,cpupart->psrc,cpupart->noutgoing);
	tendtime = rtclock();
	var->cpu_bck_knl_time += tendtime - tsttime;
//		}
	}
	*cpu_level=*cpu_level-1;
	//printf("\nCODE REACHING cpu_level %d\n",*cpu_level);
}
endtime = rtclock();
printf("CPU Backward Phase, runtime = %.3lf ms\n", 1000*(endtime -starttime));
var->cpu_tot_bck_time += endtime -starttime;
}

__device__ void ArrayToBorder(float *Delta, float *borderDelta,unsigned *borderNodes,unsigned borderCount){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	while(id < borderCount){
		borderDelta[id] = Delta[borderNodes[id]];
		id += blockDim.x * gridDim.x;
	}
 }
__device__ void BorderToArray(float *Delta, float *borderDelta,unsigned *borderNodes,unsigned borderCount){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	while(id < borderCount){
		Delta[borderNodes[id]] =  borderDelta[id];
		id += blockDim.x * gridDim.x;
	}
 }

__global__
void gpu_component (unsigned *edgesrc,unsigned *edgedst,unsigned nnodes,unsigned nedges,unsigned *nodedist,unsigned *nodesigma,unsigned *edgesigma,float *nodedelta,int *wait,unsigned *border,float *borderDelta, unsigned borderCount,long level){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned nthreads = blockDim.x * gridDim.x;
	//unsigned edgesperthread = (nedges + nthreads - 1) / nthreads;
	//unsigned startedge = edgesperthread * id, endedge = edgesperthread * (id + 1);
        //if (endedge > nedges) endedge = nedges;
	bool lchanged=false;
	//__shared__ bool changed;
	//if(threadIdx.x==0)changed=false;
	unsigned src,dst;
	float delta_src,delta_dst;
        for (;id < nedges;id += gridDim.x*blockDim.x){
        //for (ii = startedge; ii < endedge; ii++) {
		// Don't process if already processed or not proccessable
		dst = edgedst[id];
		if(nodedist[dst]!=level) continue;
		if(edgesigma[id]==0) continue;
		if(border[dst]!=0){
			lchanged = true;
			continue;
		}// Don't process and wait if it is a border node
		src = edgesrc[id];
		// wt = edgewt[ii];
		delta_dst = nodedelta[dst];
		delta_dst++;
		//if(nodesigma[dst]!=0)
		delta_src = ((float)(nodesigma[src]/nodesigma[dst])*delta_dst);
		atomicAdd(&nodedelta[src],delta_src);
		edgesigma[id] = 0;
		//id += gridDim.x*blockDim.x;

	}
	if(lchanged) *wait =1 ;
	ArrayToBorder(nodedelta, borderDelta,border,borderCount);
	//__syncthreads();
	//if(lchanged) changed = true;
	//if(threadIdx.x==0)
	//	if(changed)
	//		*wait = 1;
}
/*
__global__
void gpu_component (unsigned *edgesrc,unsigned *edgedst,unsigned nnodes,unsigned nedges,unsigned *nodedist,unsigned *nodesigma,unsigned *edgesigma,float *nodedelta,int *wait,unsigned *border,float *borderDelta, unsigned borderCount,long level,unsigned *psrc, unsigned *noutgoing){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned nthreads = blockDim.x * gridDim.x;
	//unsigned edgesperthread = (nedges + nthreads - 1) / nthreads;
	//unsigned startedge = edgesperthread * id, endedge = edgesperthread * (id + 1);
        //if (endedge > nedges) endedge = nedges;
	bool lchanged=false;
	//__shared__ bool changed;
	//if(threadIdx.x==0)changed=false;
	unsigned src,dst,j;
	float delta_src,delta_dst;
        for (;id < nnodes;id += gridDim.x*blockDim.x){
        //for (ii = startedge; ii < endedge; ii++) {
		// Don't process if already processed or not proccessable
		if(nodedist[id]==level){
			for (j = psrc[id] ; j < (psrc[id] + noutgoing[id]) ; j++ ){
				dst = edgedst[j];
				if(nodedist[dst]!=level+1) continue;
				if(edgesigma[j]==0) continue;
				if(border[dst]!=0){
					lchanged = true;
					continue;
				}// Don't process and wait if it is a border node
				src = id;
				// wt = edgewt[ii];
				delta_dst = nodedelta[dst];
				delta_dst++;
				//if(nodesigma[dst]!=0)
				delta_src = ((float)(nodesigma[src]/nodesigma[dst])*delta_dst);
				atomicAdd(&nodedelta[src],delta_src);
				edgesigma[id] = 0;
			}
		}

	}
	if(lchanged) *wait =1 ;
	ArrayToBorder(nodedelta, borderDelta,border,borderCount);
	//__syncthreads();
	//if(lchanged) changed = true;
	//if(threadIdx.x==0)
	//	if(changed)
	//		*wait = 1;
}
*/
__global__
void gpu_single_component (unsigned *edgesrc,unsigned *edgedst,unsigned nnodes,unsigned nedges,unsigned *nodedist,unsigned *nodesigma,unsigned *edgesigma,float *nodedelta,unsigned *border,float *borderDelta, unsigned borderCount, long level){
	BorderToArray(nodedelta, borderDelta,border,borderCount);
	__syncthreads();
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned nthreads = blockDim.x * gridDim.x;
	//unsigned edgesperthread = (nedges + nthreads - 1) / nthreads;
	//unsigned startedge = edgesperthread * id, endedge = edgesperthread * (id + 1);
        //if (endedge > nedges) endedge = nedges;
	unsigned src,dst;
	float delta_src,delta_dst;
        for (;id < nedges;id += gridDim.x*blockDim.x){
        //for (ii = startedge; ii < endedge; ii++) {
		// Don't process if already processed or not proccessable
		dst = edgedst[id];
		if(nodedist[dst]!=level) continue;
		if(border[dst]==0) continue;
		if(edgesigma[id]==0) continue;
		src = edgesrc[id];
		// wt = edgewt[ii];
		delta_dst = nodedelta[dst];
		delta_dst++;
		//if(nodesigma[dst]!=0)
		delta_src = ((float)(nodesigma[src]/nodesigma[dst])*delta_dst);
		atomicAdd(&nodedelta[src],delta_src);
		edgesigma[id] = 0;
		//id += gridDim.x*blockDim.x;

	}
}
/* function for backward traversal for gpu */
void *gpu_backward(void *var){
struct varto_gpu_part *P = (struct varto_gpu_part *) var;
Graph *graph = P->graph;
unsigned numEdges_gpu,numNodes_gpu,borderIndex,borderIndex2,ii,borderSource;
Graph::DevicePartition *gpupart = P->gpupart;
Graph::Partition *borderInfo = P->borderInfo;
numEdges_gpu = gpupart->numEdges;
numNodes_gpu = gpupart->numNodes;
const volatile long *cpu_level = P->cpu_level ;
//volatile long cpu_level;
long * gpu_level = P->gpu_level;
int gpu_wait=0;
int *d_gpu_wait;// = P->d_gpu_wait; 
unsigned borderCount = borderInfo->borderCount[GPUPARTITION]; /* Border Count is of GPU partition */
unsigned *d_border = P->border,i;
float * cpunodelta, *gpuBorderDelta;
cpunodelta = graph->devicePartition[CPUPARTITION].nodedelta;
gpuBorderDelta = (float *) malloc (sizeof(float) * borderCount);
//long *d_gpu_level;
float temp;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
//if (
cudaMalloc((void **)&d_gpu_wait,  sizeof(int));// != cudaSuccess) CudaTest("allocating d_gpu_wait failed");
//if (
//cudaMalloc((void **)&d_gpu_level,  sizeof(long));// != cudaSuccess) CudaTest("allocating d_gpu_level failed");
//if (
//cudaMalloc((void **)&d_border,  graph->nnodes*sizeof(unsigned));// != cudaSuccess) CudaTest("allocating d_border failed");
//cudaMemcpy(d_border,borderInfo->border,(graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice); //copy from cpu to gpu
double starttime, endtime,tmpstarttime,tmpendtime,kerneltime=0;
float elapsedTime=0;
starttime = rtclock();
//cpu_level = *(P->cpu_level);
//gb.Setup(P->kconf->getNumberOfBlocks());
//for(int i = 0 ; i < graph->nnodes ; i++)
//	gpupart->nodedelta[i] = 0.0;

// Setting the edges of cpu border nodes as not processable
for(i = 0 ; i < numEdges_gpu ; i++){
	if(graph->partition.part[gpupart->edgedst[i]]==CPUPARTITION)
		gpupart->edgesigma[i] = 0 ;
}
for(i =0 ; i< borderCount ; i++)
	gpuBorderDelta[i] = gpupart->nodedelta[borderInfo->borderNodes[GPUPARTITION][i]];
//--*gpu_level;
while(*gpu_level > *(P->gpu_level_min)){
	gpu_wait = 0;
	//CUDACOPY(d_gpu_level,gpu_level,sizeof(long), cudaMemcpyHostToDevice,sone); //copy from cpu to gpu
	tmpstarttime = rtclock();
	cudaMemset(d_gpu_wait,0,sizeof(int));
	tmpendtime = rtclock();
	P->gpu_memcpy += tmpendtime - tmpstarttime ;
	 //cudaMemcpy(P->nodedelta,gpupart->nodedelta,(graph->nnodes) * sizeof(float), cudaMemcpyHostToDevice); //copy from cpu to gpu
	tmpstarttime = rtclock();
	//cudaEventRecord(start,0);
	gpu_component<<<P->kconf->getNumberOfBlocks(), P->kconf->getNumberOfBlockThreads()>>>(P->edgesrc,P->edgedst,graph->nnodes,numEdges_gpu,P->nodedist,P->nodesigma,P->edgesigma,P->nodedelta,d_gpu_wait,d_border,P->borderDelta,borderCount,*gpu_level);
	//gpu_component<<<26,704>>>(P->edgesrc,P->edgedst,graph->nnodes,numEdges_gpu,P->nodedist,P->nodesigma,P->edgesigma,P->nodedelta,d_gpu_wait,d_border,P->borderDelta,borderCount,*gpu_level,P->psrc,P->noutgoing);
//	gpu_component<<<26,704>>>(P->edgesrc,P->edgedst,graph->nnodes,numEdges_gpu,P->nodedist,P->nodesigma,P->edgesigma,P->nodedelta,d_gpu_wait,d_border,P->borderDelta,borderCount,*gpu_level);
	cudaDeviceSynchronize();
	tmpendtime = rtclock();
	//cudaEventRecord(stop,0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedTime, start,stop);
	//P->gpu_bck_knl_time +=  elapsedTime;
	P->gpu_bck_knl_time += tmpendtime - tmpstarttime ;
	//gpu_component<<<13,512>>>(P->edgesrc,P->edgedst,graph->nnodes,numEdges_gpu,P->nodedist,P->nodesigma,P->edgesigma,P->nodedelta,d_gpu_wait,d_border,d_gpu_level,gb);
	cudaMemcpy(&gpu_wait,d_gpu_wait,sizeof(int), cudaMemcpyDeviceToHost); // copy from gpu to cpu
	if(gpu_wait && *cpu_level > *(P->cpu_level_min)){
		//printf("\nCODE REACHING\n");
	tmpstarttime = rtclock();
		//cudaMemcpy(gpupart->nodedelta,P->nodedelta,(graph->nnodes) * sizeof(float), cudaMemcpyDeviceToHost); // copy from gpu to cpu
		cudaMemcpy(gpuBorderDelta,P->borderDelta,(borderCount) * sizeof(float), cudaMemcpyDeviceToHost); // copy from gpu to cpu
	tmpendtime = rtclock();
	kerneltime += tmpendtime - tmpstarttime ;
		//cpu_level = *(P->cpu_level);
		while(*cpu_level > *gpu_level); // wait for cpu to catch up
	}
	if(gpu_wait){
		//printf("GPU's CPU LEVEL %ld\n",*cpu_level);
		// copy border node data
//#pragma omp parallel for schedule(dynamic) private(i) 
		for(i = 0 ; i < borderCount; i++){
			if(gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][i]]==*gpu_level && graph->devicePartition[CPUPARTITION].nodedelta[borderInfo->borderNodes[GPUPARTITION][i]] != gpuBorderDelta[i] ){
				//temp = gpupart->nodedelta[borderInfo->borderNodes[GPUPARTITION][i]];
				//gpupart->nodedelta[borderInfo->borderNodes[GPUPARTITION][i]] += graph->devicePartition[CPUPARTITION].nodedelta[borderInfo->borderNodes[GPUPARTITION][i]];
				gpuBorderDelta[i] += cpunodelta[borderInfo->borderNodes[GPUPARTITION][i]];
				//graph->devicePartition[CPUPARTITION].nodedelta[borderInfo->borderNodes[GPUPARTITION][i]] += temp;
			}
		}
		//copy cpu to gpu
	//cudaMemcpy(P->nodedelta,gpupart->nodedelta,(graph->nnodes) * sizeof(float), cudaMemcpyHostToDevice); //copy from cpu to gpu
	tmpstarttime = rtclock();
	cudaMemcpy(P->borderDelta,gpuBorderDelta,(borderCount) * sizeof(float), cudaMemcpyHostToDevice); //copy from cpu to gpu
	tmpendtime = rtclock();
	P->gpu_memcpy += tmpendtime - tmpstarttime ;
	tmpstarttime = rtclock();
	//gpu_single_component<<<P->kconf->getNumberOfBlocks(), P->kconf->getNumberOfBlockThreads()>>>(P->edgesrc,P->edgedst,graph->nnodes,numEdges_gpu,P->nodedist,P->nodesigma,P->edgesigma,P->nodedelta,d_border,*gpu_level);
	//cudaEventRecord(start,0);
	gpu_single_component<<<26,704>>>(P->edgesrc,P->edgedst,graph->nnodes,numEdges_gpu,P->nodedist,P->nodesigma,P->edgesigma,P->nodedelta,d_border,P->borderDelta,borderCount,*gpu_level);
	cudaDeviceSynchronize();
	//cudaEventRecord(stop,0);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&elapsedTime, start,stop);
	//P->gpu_bck_knl_time +=  elapsedTime;
	tmpendtime = rtclock();
	P->gpu_bck_knl_time += tmpendtime - tmpstarttime ;
	//gpu_single_component<<<13,512>>>(P->edgesrc,P->edgedst,graph->nnodes,numEdges_gpu,P->nodedist,P->nodesigma,P->edgesigma,P->nodedelta,d_gpu_level);
	//cudaMemcpy(gpupart->nodedelta,P->nodedelta,(graph->nnodes) * sizeof(float), cudaMemcpyDeviceToHost); // copy from gpu to cpu
	}
	*gpu_level = *gpu_level-1;
	//printf("GPU LEVEL %u\n",*gpu_level);
}
	tmpstarttime = rtclock();
cudaMemcpy(gpupart->nodedelta,P->nodedelta,(graph->nnodes) * sizeof(float), cudaMemcpyDeviceToHost); // copy from gpu to cpu
	tmpendtime = rtclock();
	P->gpu_memcpy += tmpendtime - tmpstarttime ;
endtime = rtclock();
printf("GPU Backward Phase, runtime = %.3lf ms ;  kernel_time =  %.3lf ms \n", 1000*(endtime -starttime),1000*kerneltime);
P->gpu_tot_bck_time += endtime -starttime;
//cudaFree(d_gpu_wait);
//cudaFree(d_gpu_level);
//cudaFree(d_border);
//while(*cpu_level > 0);
free(gpuBorderDelta);
pthread_exit(NULL);
}
