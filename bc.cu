// 27 + texture: no temporal locality, so texture is not helpful.
/** Betweenness Centrality -*- CUDA -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Betweenness Centrality.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 */
#include "common.h"
//#include "metis.h"
#include "Structs.h"
#include "scheduler19.h"
#include "graph28.h"
#include "kernelconfig.h"
#include "list.h"
#include <cub/cub.cuh>
//#include "gbar.cuh"
#include "myutils.h"
#include "myutils2.h"

#define DIAMETER 221
//#define USE_DIA

//__device__ bool *gpu_wait;
double cpu_ratiotime,gpu_ratiotime;
struct d_s{
	unsigned dist;
	unsigned sig;
};
#ifndef USE_DIA
struct matrix_csr{
	unsigned row_size;
	vector<unsigned> dist;
	vector<unsigned> sig;
};
struct d_s find_matrix(struct matrix_csr *M,unsigned i, unsigned j){
	struct d_s s;
	s.dist = M->dist[i*(M->row_size)+j];
	s.sig =  M->sig[i*(M->row_size)+j];
	return s;
}
void modify_matrix(struct matrix_csr *M,unsigned i, unsigned j,unsigned dist,unsigned sig){
	M->dist.push_back(dist);
	M->sig.push_back(sig);
}
#else
struct matrix_csr{
	unsigned row_size;
	vector<bool> bits;
	vector<short> dist;
	vector<unsigned> sig;
};

struct d_s find_matrix(struct matrix_csr *M,unsigned i, unsigned j){
		struct d_s s;
		if(!M->bits[i*(M->row_size)+j]){
			s.dist=MYINFINITY;
			s.sig=0;
			return s;
		}
		else{
			int nthreads=omp_get_num_procs()/2;
			uint64_t ele[nthreads],ii,total_ele=0;
			for(ii=0;ii<nthreads;ii++)
				ele[ii]=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(ii) num_threads(nthreads)
#endif
			for(ii=0;ii <= (i*(M->row_size)+j) ; ii++)
				if(M->bits[ii])
					ele[omp_get_thread_num()]++;
			for(ii=0;ii<nthreads;ii++)
				total_ele +=ele[ii];

			s.dist=M->dist[total_ele];
			s.sig= M->sig[total_ele];

			return s;
		}
}
void modify_matrix(struct matrix_csr *M,unsigned i, unsigned j,unsigned dist,unsigned sig){
	if(dist > DIAMETER){
		//M->bits[i*(M->row_size)+j]=false;
		M->bits.push_back(false);
		return;
	}
	//cout<<"Reaching here "<<i<<" "<<M->row_size<<" "<<j<<"\n";
	//M->bits[i*(M->row_size)+j]=true;
	short s = (short)dist;
	M->bits.push_back(true);
	M->dist.push_back(s);
	M->sig.push_back(sig);
}
#endif

void initbc(dorf *bc, unsigned nv) {
	for (unsigned ii = 0; ii < nv; ++ii) {
		bc[ii] = 0.0;
	}
}

unsigned verify(dorf *bc, Graph &graph) {
	unsigned nerr = 1;
	printf("\tVerification not performed.\n");
	for (unsigned ii = 0; ii < 10; ++ii) {
		printf("\tbc[%d] = %lf.\n", ii, bc[ii]);
	}
	return nerr;
}

void initnodesigmadist(unsigned source, unsigned nodes, unsigned* nodesigma, unsigned* nodedist){
unsigned ii;
	for (ii = 0; ii < nodes; ii++) {
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
{
#ifdef _OPENMP
#pragma omp sections 
#endif
{	
	#ifdef _OPENMP
	#pragma omp section
	#endif
	{
		memset(graph.devicePartition[CPUPARTITION].edgesigma,0,((graph.devicePartition[CPUPARTITION].numEdges) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodedelta,0,((graph.nnodes) * sizeof(float)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].edgesigma,0,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodedelta,0,((graph.nnodes) * sizeof(float)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
	 /* Initializing all border Vectors */
	 for(borderIndex=0; borderIndex < borderCount_gpu; borderIndex++){
   		borderVector_gpu1[borderIndex] = borderVector_gpu2[borderIndex] = MYINFINITY;
		borderSigma_gpu[borderIndex] = 0;
		}
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
	  for(borderIndex=0; borderIndex < borderCount_cpu; borderIndex++){
		borderVector_cpu1[borderIndex] =  borderVector_cpu2[borderIndex] = MYINFINITY;
		borderSigma_cpu[borderIndex] = 0;
		}
	}}  {
	#ifdef _OPENMP
      //  #pragma omp section
        #endif
        {
               cudaStreamSynchronize(sone);
	       cudaStreamSynchronize(stwo);
     	       cudaStreamSynchronize(sthree);

        }
	}}
        }
        nodesigma[source] = 1;
        nodedist[source] = 0;
}
void initnodesigmadist_multisource(Graph *graph,unsigned *values, unsigned *sigma_values,unsigned nodes, unsigned* nodesigma, unsigned* nodedist,unsigned *sources,unsigned source_count,unsigned *psrc,unsigned *noutgoing,unsigned *edgedst,unsigned *border){
unsigned ii,j;#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
{
#ifdef _OPENMP
#pragma omp sections 
#endif
{	
	#ifdef _OPENMP
	#pragma omp section
	#endif
	{
		memset(graph.devicePartition[CPUPARTITION].edgesigma,0,((graph.devicePartition[CPUPARTITION].numEdges) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodedelta,0,((graph.nnodes) * sizeof(float)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].edgesigma,0,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodedelta,0,((graph.nnodes) * sizeof(float)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
	 /* Initializing all border Vectors */
	 for(borderIndex=0; borderIndex < borderCount_gpu; borderIndex++){
   		borderVector_gpu1[borderIndex] = borderVector_gpu2[borderIndex] = MYINFINITY;
		borderSigma_gpu[borderIndex] = 0;
		}
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
	  for(borderIndex=0; borderIndex < borderCount_cpu; borderIndex++){
		borderVector_cpu1[borderIndex] =  borderVector_cpu2[borderIndex] = MYINFINITY;
		borderSigma_cpu[borderIndex] = 0;
		}
	}}  {
	#ifdef _OPENMP
      //  #pragma omp section
        #endif
        {
               cudaStreamSynchronize(sone);
	       cudaStreamSynchronize(stwo);
     	       cudaStreamSynchronize(sthree);

        }
	}}

/*
	for (ii = 0; ii < nodes; ii++) {
//		if(graph->partition.border[ii]==0)
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;
        }
*/
	for ( ii = 0; ii < source_count; ii++) {
		unsigned v = sources[ii],w;
		unsigned num_edges_v = psrc[v];
		for (j = num_edges_v; j < (num_edges_v + noutgoing[v]) ; j++) {
			w = edgedst[j];
			if(border[w]==0)continue;
			nodedist[w]=MYINFINITY;
			nodesigma[w]=0;
		}
	}
	for(ii=0 ; ii < source_count ; ii++)
	{
	        nodedist[sources[ii]] = values[ii];
		nodesigma[sources[ii]] = sigma_values[ii];	   
	}

}
void initnodesigmadist_multisource_omp(Graph *graph,unsigned *values,unsigned *sigma_values,unsigned nodes, unsigned* nodesigma, unsigned* nodedist,unsigned *sources,unsigned source_count,int num_threads,unsigned *psrc,unsigned *noutgoing,unsigned *edgedst,unsigned *border){
	unsigned ii,j;

/*	
#pragma omp parallel for private(ii) schedule(static)
        for (ii = 0; ii < nodes; ii++) {
//		if(graph->partition.border[ii]==0)
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;
        }
*/
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(ii,j) num_threads(num_threads)
#endif
for ( ii = 0; ii < source_count; ii++) {
	unsigned v = sources[ii],w;
	unsigned num_edges_v = psrc[v];
	for (j = num_edges_v; j < (num_edges_v + noutgoing[v]) ; j++) {
		w = edgedst[j];
			if(border[w]==0)continue;
		nodedist[w]=MYINFINITY;#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
{
#ifdef _OPENMP
#pragma omp sections 
#endif
{	
	#ifdef _OPENMP
	#pragma omp section
	#endif
	{
		memset(graph.devicePartition[CPUPARTITION].edgesigma,0,((graph.devicePartition[CPUPARTITION].numEdges) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodedelta,0,((graph.nnodes) * sizeof(float)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].edgesigma,0,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodedelta,0,((graph.nnodes) * sizeof(float)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
	 /* Initializing all border Vectors */
	 for(borderIndex=0; borderIndex < borderCount_gpu; borderIndex++){
   		borderVector_gpu1[borderIndex] = borderVector_gpu2[borderIndex] = MYINFINITY;
		borderSigma_gpu[borderIndex] = 0;
		}
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
	  for(borderIndex=0; borderIndex < borderCount_cpu; borderIndex++){
		borderVector_cpu1[borderIndex] =  borderVector_cpu2[borderIndex] = MYINFINITY;
		borderSigma_cpu[borderIndex] = 0;
		}
	}}  {
	#ifdef _OPENMP
      //  #pragma omp section
        #endif
        {
               cudaStreamSynchronize(sone);
	       cudaStreamSynchronize(stwo);
     	       cudaStreamSynchronize(sthree);

        }
	}}
		nodesigma[w]=0;
	}
}
//#pragma omp parallel for private(ii) schedule(static)
	for(ii=0 ; ii < source_count ; ii++)
	{
	   nodedist[sources[ii]] = values[ii];
	   nodesigma[sources[ii]] =  sigma_values[ii];
	}
}
void initnodesigmadist_multisource_singlerelax(Graph *graph,unsigned *values, unsigned *sigma_values,unsigned nodes, unsigned* nodesigma, unsigned* nodedist,unsigned *sources,unsigned source_count){
unsigned ii,j;

/*
	for (ii = 0; ii < nodes; ii++) {
//		if(graph->partition.border[ii]==0)
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;
        }
*/
	for(ii=0 ; ii < source_count ; ii++)
	{
	        nodedist[sources[ii]] = values[ii];
		nodesigma[sources[ii]] = sigma_values[ii];	   
	}

}
void initnodesigmadist_multisource_omp_singlerelax(Graph *graph,unsigned *values,unsigned *sigma_values,unsigned nodes, unsigned* nodesigma, unsigned* nodedist,unsigned *sources,unsigned source_count,int num_threads){
	unsigned ii,j;

/*	
#pragma omp parallel for private(ii) schedule(static)
        for (ii = 0; ii < nodes; ii++) {
//		if(graph->partition.border[ii]==0)
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;
        }
*/
//#pragma omp parallel for private(ii) schedule(static)
#ifdef _OPENMP
#pragma omp for schedule(static) private(ii)
#endif
	for(ii=0 ; ii < source_count ; ii++)
	{
	   nodedist[sources[ii]] = values[ii];
	   nodesigma[sources[ii]] =  sigma_values[ii];
	}
}


void initnodesigmadist_omp(unsigned source, unsigned nodes, unsigned* nodesigma, unsigned* nodedist,int num_threads){
unsigned ii;
#ifdef _OPENMP
#pragma omp parallel for private(ii) schedule(guided) num_threads(num_threads)
#endif
	for (ii = 0; ii < nodes; ii++) {
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;
        }
        nodesigma[source] = 1;
        nodedist[source] = 0;
}

/*
 Comparision between Border Vector and Border Matrix 
 To find the lesser distances and increase the sigma
 of the corresponding change.
 */
void borderMatrixVector_comp (struct matrix_csr *M, unsigned *Vin,unsigned *Vout,unsigned *S_V,unsigned Bcount)
{
	unsigned *bV = (unsigned *)malloc (sizeof(unsigned) * Bcount);
	bool flag;
/*#ifdef _OPENMP
#pragma omp parallel shared(bV,flag)
	{
#endif*/
	unsigned borderIndex, borderIndex2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) 
#endif
	for (borderIndex=0;borderIndex < Bcount; borderIndex++)
		bV[borderIndex] = Vin[borderIndex];
		
	do{
//#pragma omp single
	flag = false;
	/* Border vector and border matrix comparision, modifying the borderVector for smaller distance values */
/*#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif*/
	for ( borderIndex=0;borderIndex < Bcount; borderIndex++){
          for(borderIndex2=0; borderIndex2 < Bcount; borderIndex2++){
		  struct d_s s=find_matrix(M,borderIndex,borderIndex2);

	    if(borderIndex!=borderIndex2 && (bV [borderIndex2] > s.dist + bV [borderIndex])){
		    bV [borderIndex2] = s.dist + bV[borderIndex];
		    S_V [borderIndex2] = s.sig-1 + S_V[borderIndex];
		    flag = true;
		}
	     }
	   }
/*#ifdef _OPENMP
#pragma omp barrier
#endif*/
	}while(flag);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
//#pragma simd
#endif
	for (borderIndex=0;borderIndex < Bcount; borderIndex++)
		Vout[borderIndex] = bV[borderIndex];
/*#ifdef _OPENMP
	}
#endif*/
	free(bV);
}

bool Equals (unsigned *V1, unsigned *V2, unsigned Bcount)
{
//	unsigned num_changes[omp_get_num_threads()] = 0;
	unsigned borderIndex;
//#ifdef _OPENMP
//#pragma omp parallel for schedule(static) private(borderIndex) shared(num_changes)
//#endif
	for(borderIndex = 0 ; borderIndex < Bcount ; borderIndex++)
	{
		if( V1 [borderIndex] != V2 [borderIndex])
			return false;
			//num_changes[omp_get_thread_num()]++;
	}
	return true;

/*	for(borderIndex =0 ; borderIndex < omp_get_num_threads() ; borderIndex++;)
	{		
		if(num_changes[borderIndex]==0)
			return true;
		else
			return false;
	}
	*/
}

__global__ void ArrayToBorder(unsigned *Dist, unsigned *Sigma, unsigned *borderDist, unsigned *borderSigma,unsigned *borderNodes,unsigned borderCount){
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	while(id < borderCount){
		borderDist[id] = Dist[borderNodes[id]];
		borderSigma[id] = Sigma[borderNodes[id]];
		id += blockDim.x * gridDim.x;
	}
}
void gpu_component (unsigned *psrc,unsigned *noutgoing,unsigned *d_psrc,unsigned *d_noutgoing,unsigned *edgesdstsrc,unsigned *edgessrcdst,unsigned hedges,unsigned hnodes,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,unsigned *sources,cudaDeviceProp *dp,bool BM_COMP,unsigned *nerr)
{
	//GlobalBarrierLifetime gb;
	lonestar_gpu(psrc,noutgoing,d_psrc,d_noutgoing,edgesdstsrc,edgessrcdst,hedges,hnodes,hdist,nodesigma,edgesigma,source_count,sources,dp,BM_COMP,nerr);
	//ananya_code_func(psrc,noutgoing,d_psrc,d_noutgoing,edgesdstsrc,edgessrcdst,hedges,hnodes,hdist,nodesigma,edgesigma,source_count,sources,dp,BM_COMP,nerr);
}
void cpu_component (unsigned *psrc,unsigned *noutgoing,unsigned *edgesdstsrc,unsigned *edgessrcdst,unsigned hnodes,unsigned hedges,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,unsigned *sources,omp_lock_t *lock,bool BM_COMP, int num_threads)
{
	betweenness_centrality_parallel(hnodes,hedges,psrc,edgessrcdst,edgesdstsrc,noutgoing,sources,source_count,hdist,nodesigma,edgesigma,lock,num_threads);
	//worklist_cpu(psrc,noutgoing,edgesdstsrc,edgessrcdst,hnodes,hedges,hdist,nodesigma,edgesigma,source_count,sources,lock,BM_COMP,num_threads);
}

void cpu_bfs_relax(unsigned nnodes, unsigned nedges, unsigned *psrc,unsigned *noutgoing,unsigned* edgesrc, unsigned* edgedst, unsigned* edgewt, unsigned *nodesigma, foru *nodedist,unsigned *edgesigma,int nthreads,unsigned *borderNodes, unsigned bcount,omp_lock_t *lock,unsigned *border) {
	unsigned i,j;
//#pragma omp parallel for schedule(guided) private(i,j) num_threads(nthreads)
	for ( i = 0; i < bcount; i++) {
		unsigned v = borderNodes[i],w;
		unsigned num_edges_v = psrc[v];
		foru ddist;
		foru wt=1;
		for (j = num_edges_v; j < (num_edges_v + noutgoing[v]) ; j++) {
			w = edgedst[j];
			if(border[w]==0)continue;

//		omp_set_lock(&(lock[w]));
		ddist = nodedist[w];
		if (ddist > (nodedist[v] + wt)) {
                        nodedist[w] = nodedist[v] + wt;
//			omp_unset_lock(&(lock[w]));
//#pragma omp atomic update
			edgesigma[j] = nodesigma[v];
//#pragma omp atomic write
			nodesigma[w] = edgesigma[j];
		}else if (ddist == (nodedist[v] + wt)){
//			omp_unset_lock(&(lock[w]));
//#pragma omp atomic update
			nodesigma[w] -= edgesigma[j];
			edgesigma[j] = nodesigma[v];
//#pragma omp atomic update
			nodesigma[w] += edgesigma[j];

		}
		else{
			omp_unset_lock(&(lock[w]));
                        edgesigma[j] = 0;
		}
		}
	}
}

void *cpu_BFS(void *P){
	  struct varto_cpu_part *var = (struct varto_cpu_part *)P;
	  Graph *graph = var->graph;
	  unsigned numEdges_src,numNodes_src,source = var->source,borderIndex,ii;
          int num_threads = var->num_threads;
	  double starttime, endtime;
	  //FILE *fp = fopen("CPU_INITIALBFS_VALUES.txt","w");
	  Graph::DevicePartition *srcpart = var->partition;
	  Graph::Partition *borderInfo = var->borderInfo;
          numEdges_src = srcpart->numEdges;
          numNodes_src = srcpart->numNodes;
	  //int convergence = 0;
	  unsigned borderCount = borderInfo->borderCount[CPUPARTITION]; /* Border Count is of non GPU partition */
          //printf("num edges of srcpartition: %d\n", numEdges_src);
	/* Do CPU BFS calculate border distance vector*/			
         initnodesigmadist_omp (source,graph->nnodes, srcpart->nodesigma, srcpart->nodedist,num_threads);
	  starttime = rtclock();
	   cpu_component (srcpart->psrc,srcpart->noutgoing,srcpart->edgesrc,srcpart->edgedst,graph->nnodes,numEdges_src,srcpart->nodedist,srcpart->nodesigma,srcpart->edgesigma,1,&source,var->lock,false,num_threads);
	endtime = rtclock ();
	/* Fill up borderVector */
//#pragma omp parallel for schedule(static) num_threads(num_threads)
	if(!var->single_relax)
            for(borderIndex=0; borderIndex < borderCount; borderIndex++){	
		    var->borderVector_cpu[borderIndex] = srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
		    var->borderSigma_cpu[borderIndex] = srcpart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
	//	    printf("%u:%d ",borderInfo->borderNodes[srcpartition][borderIndex],srcpart->nodedist[borderInfo->borderNodes[srcpartition][borderIndex]]);
	    }
	/*for(ii=0 ; ii < graph->nnodes ; ii++){
	fprintf(fp,"%u ",srcpart->nodedist[ii]);
	}
	fclose(fp);
	*/
        printf("For CPU BFS runtime = %.3lf ms\n", 1000*(endtime -starttime));
	var->cpu_F_I += endtime-starttime;
	cpu_ratiotime += endtime-starttime;
}

void *gpu_BFS(void *var){
	double starttime, endtime;
	struct varto_gpu_part *P = (struct varto_gpu_part *)var;
	unsigned borderIndex,borderIndex2;
	Graph *graph = P->graph;
        unsigned numEdges,numNodes,source = P->source,ii;
        //bool hchanged = false;
	Graph::DevicePartition *gpupart = P->gpupart;
	Graph::Partition *borderInfo = P->borderInfo;
	numEdges = gpupart->numEdges;
	numNodes = gpupart->numNodes;
	foru foruzero = 0, foruone=1;
	//int convergenceCount = 0;
	//FILE *fp = fopen("GPU_INITIALBFS_VALUES.txt","w");
	unsigned borderCount = borderInfo->borderCount[GPUPARTITION]; /* Border Count is of non GPU partition */
	cudaStream_t sone, stwo;
        cudaStreamCreate(&sone);
        cudaStreamCreate(&stwo);
	/* Peforming BFS from each border node to fill up the border Matrix */
	/*
	initnodesigmadist(source,graph->nnodes, gpupart->nodesigma, gpupart->nodedist);
        CUDACOPY(P->nodedist,gpupart->nodedist, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
        CUDACOPY(P->nodesigma,gpupart->nodesigma, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,stwo);
        //CUDACOPY(P->edgesigma,gpupart->edgesigma, (numEdges) * sizeof(unsigned), cudaMemcpyHostToDevice,stwo);
	cudaStreamSynchronize(sone);
	cudaStreamSynchronize(stwo);
	*/
	//cudaMemset(P->edgesigma,0,(numEdges) * sizeof(unsigned));
	//cudaMemset(P->nodesigma,0,(graph->nnodes)*sizeof(unsigned));
	//cudaMemset(P->nodedist,MYINFINITY,(graph->nnodes)*sizeof(unsigned));
	cudaMemcpy(&(P->nodedist[source]), &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);
	cudaMemcpy(&(P->nodesigma[source]), &foruone, sizeof(foruone), cudaMemcpyHostToDevice);
        starttime = rtclock();
	gpu_component (gpupart->psrc,gpupart->noutgoing,P->psrc,P->noutgoing,P->edgesrc,P->edgedst,numEdges,graph->nnodes,P->nodedist,P->nodesigma,P->edgesigma,1,&source,&(P->kconf->dp),false,P->nerr);
	endtime = rtclock ();
	    
	     CUDACOPY(gpupart->nodedist,P->nodedist,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost,sone);
            CUDACOPY(gpupart->nodesigma,P->nodesigma,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost,stwo);
            CUDACOPY(gpupart->edgesigma,P->edgesigma,(numEdges) * sizeof(unsigned), cudaMemcpyDeviceToHost,stwo);
	     cudaStreamSynchronize(sone);
            cudaStreamSynchronize(stwo);
	/* Fill up borderVector */
	    if(!P->single_relax)
        for(borderIndex=0; borderIndex < borderCount; borderIndex++){	
	    P->borderVector_gpu[borderIndex] = gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
	    P->borderSigma_gpu[borderIndex] = gpupart->nodesigma[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
	}
	//	    printf("%u:%d ",borderInfo->borderNodes[srcpartition][borderIndex],srcpart->nodedist[borderInfo->borderNodes[srcpartition][borderIndex]]);
	printf("For GPU BFS runtime = %.3lf ms\n", 1000*(endtime -starttime));
	P->gpu_F_I += endtime-starttime;
	gpu_ratiotime += endtime-starttime;
	//for(ii=0 ; ii < graph->nnodes ; ii++){
	//fprintf(fp,"%u ",gpupart->nodedist[ii]);
	//}
	cudaStreamDestroy(sone);
        cudaStreamDestroy(stwo);
	//fclose(fp);
}

void *cpu_Relax(void *P){
	  struct varto_cpu_part *var = (struct varto_cpu_part *)P;
	  Graph *graph = var->graph;
	  unsigned numEdges_src,numNodes_src,source = var->source,borderIndex,ii;
	  unsigned *sources,source_count;
          int num_threads = var->num_threads;
	 // bool hchanged_cpu = false;
	  double starttime, endtime;
	  Graph::DevicePartition *srcpart = var->partition;
	  Graph::Partition *borderInfo = var->borderInfo;
          numEdges_src = srcpart->numEdges;
          numNodes_src = srcpart->numNodes;
	  //int convergence = 0;
	  unsigned borderCount = borderInfo->borderCount[CPUPARTITION]; /* Border Count is of non GPU partition */
	  initnodesigmadist_multisource_omp(graph,var->borderVector_cpu,var->borderSigma_cpu,graph->nnodes, srcpart->nodesigma,srcpart->nodedist,borderInfo->borderNodes[CPUPARTITION],borderCount,num_threads,srcpart->psrc,srcpart->noutgoing,srcpart->edgedst,borderInfo->border); // sending the values of border node for a multi source bfs
          //for(borderIndex=0; borderIndex < borderCount; borderIndex++) // Copying from borderVector to the border nodes itself 
	   // srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = var->borderVector_cpu[borderIndex];
//#pragma omp parallel for schedule(static) num_threads(num_threads)
//	  for(int ii=0;ii < numEdges_src; ii++)
//	    srcpart->edgesigma[ii] = 0;
	  
 	 if(source!=MYINFINITY){
	  source_count = borderCount+1;
	  sources = (unsigned *)malloc ((borderCount+1)* sizeof(unsigned));
	  sources[borderCount] = source;
	  srcpart->nodedist[source] = 0;
	  srcpart->nodesigma[source] = 1;
	  }
	  else{
	  sources = (unsigned *)malloc ((borderCount)* sizeof(unsigned));
	  source_count = borderCount;
	  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
	  for(borderIndex=0; borderIndex < borderCount; borderIndex++)
	        sources[borderIndex] = borderInfo->borderNodes[CPUPARTITION][borderIndex];
	  starttime = rtclock();
	/* Multisource BFS*/
	  cpu_component (srcpart->psrc,srcpart->noutgoing,srcpart->edgesrc,srcpart->edgedst,graph->nnodes,numEdges_src,srcpart->nodedist,srcpart->nodesigma,srcpart->edgesigma,source_count,sources,var->lock,false,num_threads);
	endtime = rtclock ();
	/* Fill up borderVector */
//#pragma omp parallel for schedule(static) num_threads(num_threads)
         //   for(borderIndex=0; borderIndex < borderCount; borderIndex++){
	//	    var->borderVector_cpu[borderIndex] = srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
	//	    printf("%u:%d ",borderInfo->borderNodes[srcpartition][borderIndex],srcpart->nodedist[borderInfo->borderNodes[srcpartition][borderIndex]]);
//	    }
        printf("For CPU PART runtime = %.3lf ms\n", 1000*(endtime -starttime));
	var->cpu_F_R += endtime -starttime;
	free(sources);
}

void cpu_SingleRelax(void *P){
	  struct varto_cpu_part *var = (struct varto_cpu_part *)P;
	  Graph *graph = var->graph;
	  unsigned numEdges_src,numNodes_src,source = var->source,borderIndex,ii;
          int num_threads = var->num_threads;
	  //bool hchanged_cpu = false;
	  Graph::DevicePartition *srcpart = var->partition;
	  Graph::Partition *borderInfo = var->borderInfo;
          numEdges_src = srcpart->numEdges;
          numNodes_src = srcpart->numNodes;
	  //int convergence = 0;
	  unsigned borderCount = borderInfo->borderCount[CPUPARTITION]; /* Border Count is of non GPU partition */
#ifdef _OPENMP
#pragma omp parallel
	  {
#endif

	  initnodesigmadist_multisource_omp_singlerelax(graph,var->borderVector_cpu,var->borderSigma_cpu,graph->nnodes, srcpart->nodesigma,srcpart->nodedist,borderInfo->borderNodes[CPUPARTITION],borderCount,num_threads); // sending the values of border node for a multi source bfs
//#pragma omp parallel for schedule(static) num_threads(num_threads)
//	  for(int ii=0;ii < numEdges_src; ii++)
//	    srcpart->edgesigma[ii] = 0;
          //for(borderIndex=0; borderIndex < borderCount; borderIndex++) // Copying from borderVector to the border nodes itself 
	    //srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = var->borderVector_cpu[borderIndex];
	/*setting values of border nodes of CPU in GPU partition */
//#pragma omp parallel for schedule(static) num_threads(num_threads)
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
	  for(borderIndex=0 ; borderIndex < borderInfo->borderCount[GPUPARTITION] ; borderIndex++){
		  srcpart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]] = var->borderVector_gpu[borderIndex];
		  srcpart->nodesigma[borderInfo->borderNodes[GPUPARTITION][borderIndex]] = var->borderSigma_gpu[borderIndex];
	  }
#ifdef _OPENMP
}
#endif
	 if(source!=MYINFINITY){
	   srcpart->nodedist[source] = 0;
	   srcpart->nodesigma[source] = 1;
	 }
	/* 
#pragma omp parallel for schedule(guided) num_threads(num_threads)
	 for(ii=0;ii<graph->nnodes;ii++)
		srcpart->active[ii] = true;
          for(borderIndex=0; borderIndex < borderCount; borderIndex++) // Copying from borderVector to the border nodes itself 
	    srcpart->active[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = false;
	    */
	/* Multisource BFS*/
	 cpu_bfs_relax(numNodes_src, numEdges_src,srcpart->psrc,srcpart->noutgoing,srcpart->edgesrc, srcpart->edgedst, srcpart->edgewt, srcpart->nodesigma, srcpart->nodedist,srcpart->edgesigma,num_threads,borderInfo->borderNodes[CPUPARTITION],borderCount,var->lock,borderInfo->border);
	/* Fill up borderVector */

#ifdef _OPENMP
#pragma omp parallel 
	 {
#endif
//#pragma omp parallel for schedule(static) num_threads(num_threads)
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
         for(borderIndex=0; borderIndex < borderInfo->borderCount[GPUPARTITION]; borderIndex++){
	    var->borderVector_gpu[borderIndex] = srcpart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
	    var->borderSigma_gpu[borderIndex] = srcpart->nodesigma[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
	//	    printf("%u:%d ",borderInfo->borderNodes[srcpartition][borderIndex],srcpart->nodedist[borderInfo->borderNodes[srcpartition][borderIndex]]);
	 }
//#pragma omp parallel for schedule(static) num_threads(num_threads)
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
         for(borderIndex=0; borderIndex < borderCount; borderIndex++){
	    var->borderVector_cpu[borderIndex] = srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
	    var->borderSigma_cpu[borderIndex] = srcpart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
	//	    printf("%u:%d ",borderInfo->borderNodes[srcpartition][borderIndex],srcpart->nodedist[borderInfo->borderNodes[srcpartition][borderIndex]]);
	}
#ifdef _OPENMP
	 }
#endif
}
void gpu_SingleRelax(void *P){
	  struct varto_gpu_part *var = (struct varto_gpu_part *)P;
	  Graph *graph = var->graph;
	  unsigned numEdges_src,numNodes_src,source = var->source,borderIndex;
          int num_threads = var->num_threads;
	  //bool hchanged_cpu = false;
	  Graph::DevicePartition *gpupart = var->gpupart;
	  Graph::Partition *borderInfo = var->borderInfo;
          numEdges_src = gpupart->numEdges;
          numNodes_src = gpupart->numNodes;
	  //int convergence = 0;
	  unsigned borderCount = borderInfo->borderCount[GPUPARTITION]; /* Border Count is of non GPU partition */
#ifdef _OPENMP
#pragma omp parallel
	            {
#endif
	  initnodesigmadist_multisource_omp_singlerelax(graph,var->borderVector_gpu,var->borderSigma_gpu,graph->nnodes, gpupart->nodesigma,gpupart->nodedist,borderInfo->borderNodes[GPUPARTITION],borderCount,num_threads); // setting the values of border node for a multi source bfs
//#pragma omp parallel for schedule(static) num_threads(num_threads)
//	  for(int ii=0;ii < numEdges_src; ii++)
//	    srcpart->edgesigma[ii] = 0;
          //for(borderIndex=0; borderIndex < borderCount; borderIndex++) // Copying from borderVector to the border nodes itself 
	    //srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = var->borderVector_cpu[borderIndex];
	/*setting values of border nodes of CPU in GPU partition */
//#pragma omp parallel for schedule(static) num_threads(num_threads)
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
	  for(borderIndex=0 ; borderIndex < borderInfo->borderCount[CPUPARTITION] ; borderIndex++){
		  gpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = var->borderVector_cpu[borderIndex];
		  gpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = var->borderSigma_cpu[borderIndex];
	  }
#ifdef _OPENMP
	 }
#endif
	 if(source!=MYINFINITY){
	   gpupart->nodedist[source] = 0;
	   gpupart->nodesigma[source] = 1;
	 }
	/* Multisource BFS*/
	   cpu_bfs_relax(numNodes_src, numEdges_src,gpupart->psrc,gpupart->noutgoing,gpupart->edgesrc, gpupart->edgedst, gpupart->edgewt, gpupart->nodesigma, gpupart->nodedist,gpupart->edgesigma,num_threads,borderInfo->borderNodes[GPUPARTITION],borderCount,var->lock,borderInfo->border);
	/* Fill up borderVector */
	
//#pragma omp parallel for schedule(static) num_threads(num_threads)
#ifdef _OPENMP
#pragma omp parallel 
	 {
#endif
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
            for(borderIndex=0; borderIndex < borderCount; borderIndex++){
		    var->borderVector_gpu[borderIndex] = gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
		    var->borderSigma_gpu[borderIndex] = gpupart->nodesigma[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
	//	    printf("%u:%d ",borderInfo->borderNodes[srcpartition][borderIndex],srcpart->nodedist[borderInfo->borderNodes[srcpartition][borderIndex]]);
	    }
//#pragma omp parallel for schedule(static) num_threads(num_threads)
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
            for(borderIndex=0; borderIndex < borderInfo->borderCount[CPUPARTITION]; borderIndex++){
		    var->borderVector_cpu[borderIndex] = gpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
		    var->borderSigma_cpu[borderIndex] = gpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
	//	    printf("%u:%d ",borderInfo->borderNodes[srcpartition][borderIndex],srcpart->nodedist[borderInfo->borderNodes[srcpartition][borderIndex]]);
		}
#ifdef _OPENMP
	 }
#endif
}


void *cpu_BorderMatrix_comp(void *P){
	 unsigned nnz =0 ;
	  double starttime, endtime;
	  starttime = rtclock();
	  struct varto_cpu_part *var = (struct varto_cpu_part *)P;
	  Graph *graph = var->graph;
	  unsigned numEdges_cpu,numNodes_cpu,borderIndex,borderIndex2,ii,borderSource;
	  int num_threads = var->num_threads;
	  //FILE *fp = fopen("CPU_BORDERM_VALUES.txt","w");
	  Graph::DevicePartition *cpupart = var->partition;
	  Graph::Partition *borderInfo = var->borderInfo;
          numEdges_cpu = cpupart->numEdges;
          numNodes_cpu = cpupart->numNodes;
	  long long edges_BM=0, total_BM=0;
	  unsigned borderCount = borderInfo->borderCount[CPUPARTITION]; /* Border Count is of non GPU partition */
          printf("num edges of CPU PARTITION: %d\n", numEdges_cpu);
/* Peforming BFS from each border node to fill up the border Matrix */
	for(borderIndex = 0; borderIndex < borderCount; borderIndex++)	// For loop for each BFS/SSSP from each border node in GPU 
	 {
	    borderSource = borderInfo->borderNodes[CPUPARTITION][borderIndex]; // fill distance matrix of border vertices of border as source
	/* Do CPU BFS calculate border distance vector*/			
         initnodesigmadist_omp (borderSource,graph->nnodes, cpupart->nodesigma, cpupart->nodedist,num_threads);
	   
	   cpu_component (cpupart->psrc,cpupart->noutgoing,cpupart->edgesrc,cpupart->edgedst,graph->nnodes,numEdges_cpu,cpupart->nodedist,cpupart->nodesigma,cpupart->edgesigma,1,&borderSource,var->lock,false,num_threads);
           
	/* Fill up borderVector */
	 //  cout<<"BorderSource: "<<borderSource<<" "<<" BorderIndex: "<<borderIndex<<endl;
        for(borderIndex2=0; borderIndex2 < borderCount; borderIndex2++){	
	    //fprintf(fp,"%u ",cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex2]]);
	    modify_matrix(var->borderMatrix_cpu,borderIndex,borderIndex2, cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex2]],cpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex2]]);
	//	    printf("%u:%d ",borderInfo->borderNodes[srcpartition][borderIndex],srcpart->nodedist[borderInfo->borderNodes[srcpartition][borderIndex]]);
	}
	graph->progressPrint(borderCount,borderIndex);
      } // end of FOR loop for each border node
	endtime = rtclock ();
	//printf("BorderMatrix cpu nnz %u total elements %u\n",nnz,borderCount * borderCount);
        printf("CPU Border Matrix runtime = %.3lf ms\n", 1000*(endtime -starttime));
	//fclose(fp);
}


void *gpu_BorderMatrix_comp(void *var){
	unsigned nnz =0 ;
	double starttime, endtime;
        starttime = rtclock();
	struct varto_gpu_part *P = (struct varto_gpu_part *)var;
	unsigned borderIndex,borderIndex2;
	unsigned borderSource;
	Graph *graph = P->graph;
        unsigned numEdges,numNodes,source = P->source,ii;
	foru foruzero=0,foruone=1;
	Graph::DevicePartition *gpupart = P->gpupart;
	Graph::Partition *borderInfo = P->borderInfo;
	numEdges = gpupart->numEdges;
	numNodes = gpupart->numNodes;
	//int convergenceCount = 0;
	//FILE *fp = fopen("GPU_BORDERM_VALUES.txt","w");
	unsigned borderCount = borderInfo->borderCount[GPUPARTITION]; /* Border Count is of non GPU partition */
	long long edges_BM=0,total_BM=0;
	cudaStream_t sone, stwo;
        cudaStreamCreate(&sone);
        cudaStreamCreate(&stwo);
	unsigned *borderDist = (unsigned *)malloc(borderCount * sizeof(unsigned));
	unsigned *borderSigma = (unsigned *)malloc(borderCount * sizeof(unsigned));
	/* Peforming BFS from each border node to fill up the border Matrix */
	for(borderIndex = 0; borderIndex < borderCount; borderIndex++)	// For loop for each BFS/SSSP from each border node in GPU 
	 {
              borderSource = borderInfo->borderNodes[GPUPARTITION][borderIndex]; // fill distance matrix of border vertices of src
	     cudaMemset(P->edgesigma,0,(numEdges) * sizeof(unsigned));
	     cudaMemset(P->nodesigma,0,(graph->nnodes)*sizeof(unsigned));
	     cudaMemset(P->nodedist,MYINFINITY,(graph->nnodes)*sizeof(unsigned));
	     cudaMemcpy(&(P->nodedist[borderSource]), &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);
	     cudaMemcpy(&(P->nodesigma[borderSource]), &foruone, sizeof(foruone), cudaMemcpyHostToDevice);

   	     gpu_component (gpupart->psrc,gpupart->noutgoing,P->psrc,P->noutgoing,P->edgesrc,P->edgedst,numEdges,graph->nnodes,P->nodedist,P->nodesigma,P->edgesigma,1,&borderSource,&(P->kconf->dp),false,P->nerr);
	     ArrayToBorder <<<13,256>>>(P->nodedist,P->nodesigma,P->borderDist, P->borderSigma,P->borderNodes,borderCount);
		
	     CUDACOPY(borderDist,P->borderDist,(borderCount) * sizeof(unsigned), cudaMemcpyDeviceToHost,sone);
	     CUDACOPY(borderSigma,P->borderSigma,(borderCount) * sizeof(unsigned), cudaMemcpyDeviceToHost,stwo);
	     cudaStreamSynchronize(sone);
	     cudaStreamSynchronize(stwo);

		   // Fill the border row of borderMatrix correponding to the source
	     for(borderIndex2=0; borderIndex2 < borderCount; borderIndex2++)
	     {
			//fprintf(fp,"%u\n",gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex2]]);
		    modify_matrix(P->borderMatrix,borderIndex,borderIndex2,borderDist[borderIndex2],borderSigma[borderIndex2]);
		//	    if(gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex2]] > DIAMETER)
		//		                                    edges_BM++;
	     }
		  //  total_BM += borderCount;
		    //printf("\nGPUPART:- Paths greater than diameter: %lli,  Total elements in BM: %lli\n",edges_BM,total_BM);

		 /*   
		for(borderIndex2=0; borderIndex2 < borderCount; borderIndex2++){
			if(access_matrix(P->borderMatrix,borderIndex,borderIndex2)==MYINFINITY)
				nnz++;
		}
		*/
	    graph->progressPrint(borderCount,borderIndex);
	 }//end of FOR LOOP for sssp from each border vertex
	    endtime = rtclock();
	    printf("\n GPU border Matrix completion runtime = %.3lf ms\n", 1000*(endtime-starttime));
	  cudaStreamDestroy(sone);
	  cudaStreamDestroy(stwo);
	  free(borderDist);
	  free(borderSigma);
	//printf("BorderMatrix gpu nnz %u total elements %u\n",nnz,borderCount * borderCount);
		  //fclose(fp);
	  pthread_exit(NULL);
}


void *gpu_Relax(void *var){
	 struct varto_gpu_part *P = (struct varto_gpu_part *)var;
	 Graph *graph = P->graph;
	 unsigned numEdges,numNodes,source = P->source,borderIndex,ii;
	 bool hchanged = false;
	 double starttime, endtime;
	 Graph::Partition *borderInfo = P->borderInfo;
	 Graph::DevicePartition *nonsrcpart = P->gpupart;
	 numEdges = nonsrcpart->numEdges;
	 numNodes = nonsrcpart->numNodes;
	 unsigned *sources;
	 unsigned borderCount = borderInfo->borderCount[GPUPARTITION],bcount_temp; /* Border Count is of non GPU partition */
	 cudaStream_t sone, stwo;
	 cudaStreamCreate(&sone);
	 cudaStreamCreate(&stwo);
	 initnodesigmadist_multisource(graph,P->borderVector_gpu,P->borderSigma_gpu,graph->nnodes, nonsrcpart->nodesigma,nonsrcpart->nodedist,borderInfo->borderNodes[GPUPARTITION],borderCount,nonsrcpart->psrc,nonsrcpart->noutgoing,nonsrcpart->edgedst,borderInfo->border); // sending the values of border node for a multi source bfs
	 if(source!=MYINFINITY){
	  nonsrcpart->nodedist[source] = 0;
	  nonsrcpart->nodesigma[source] = 1;
	  sources = (unsigned *)malloc ((borderCount+1) * sizeof(unsigned));
	  sources[borderCount] = source;
	  bcount_temp = borderCount+1;
	 }
	 else{
	  sources = (unsigned *)malloc ((borderCount) * sizeof(unsigned));
	  bcount_temp = borderCount;
	 }
		  /*
		for(ii=0 ; ii < borderCount ; ii++)
		{
			nodedist[sources[ii]] = P->borderVector_gpu[ii];
			nodesigma[borderInfo->borderNodes[GPUPARTITION][ii]] = P->borderSigma_gpu[ii];	   
		}*/
	 for(borderIndex=0; borderIndex < borderCount; borderIndex++)
	  sources[borderIndex] = borderInfo->borderNodes[GPUPARTITION][borderIndex];
		  //if(P->single_relax){ // For update of edge list 
			  //setting values of border nodes of CPU in GPU partition 
		  
			  //for(borderIndex=0 ; borderIndex < borderInfo->borderCount[CPUPARTITION] ; borderIndex++)
			//	  nonsrcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = P->borderVector_cpu[borderIndex];
				  // Add sigma updation 
		  //}
	 CUDACOPY(P->nodedist,nonsrcpart->nodedist, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
	 CUDACOPY(P->nodesigma,nonsrcpart->nodesigma, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,stwo);
		  //CUDACOPY(P->edgesigma,nonsrcpart->edgesigma, (numEdges) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
	 cudaStreamSynchronize(sone);
	 cudaStreamSynchronize(stwo);
		  //cudaMemset(P->edgesigma,0,(numEdges) * sizeof(unsigned));
	 starttime = rtclock();

		  //for(borderIndex=0; borderIndex < borderInfo->borderCount[CPUPARTITION]; borderIndex++)
		//	    P->borderVector_cpu[borderIndex] = nonsrcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
	 gpu_component (nonsrcpart->psrc,nonsrcpart->noutgoing,P->psrc,P->noutgoing,P->edgesrc,P->edgedst,numEdges,graph->nnodes,P->nodedist,P->nodesigma,P->edgesigma,bcount_temp,sources,&(P->kconf->dp),false,P->nerr);
	 endtime = rtclock();
		     //CUDACOPY(nonsrcpart->nodedist,P->nodedist,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost);
	 cudaMemcpy(nonsrcpart->nodedist,P->nodedist,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost);
		     //CUDACOPY(nonsrcpart->nodesigma,P->nodesigma,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost,stwo);
		     //CUDACOPY(nonsrcpart->edgesigma,P->edgesigma,(numEdges) * sizeof(unsigned), cudaMemcpyDeviceToHost,sone);
		     //cudaStreamSynchronize(sone);
		     //cudaStreamSynchronize(stwo);
		      
		      /* Store in the borderVector_gpu */
		//if(!(P->single_relax)){
		//    for(borderIndex=0; borderIndex < borderCount; borderIndex++)
		//	    P->borderVector_gpu[borderIndex] = nonsrcpart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
	 printf("For GPU RELAX runtime = %.3lf ms\n", 1000*(endtime -starttime));
	 P->gpu_F_R += endtime -starttime;
		    /*
		  for(unsigned ii =0;ii<graph->nnodes;ii++)
			  printf("%u:%u ",ii+1,nonsrcpart->nodedist[ii]);
		      printf("\truntime = %.3lf ms\n", 1000*(endtime-starttime));
			*/
	cudaStreamDestroy(sone);
	cudaStreamDestroy(stwo);
		//if(!(P->single_relax))
	free (sources);
	pthread_exit(NULL);
		//else
		//;
}


int main(int argc, char *argv[]) {
	unsigned *nodesigma, *edgesrc, *edgedst, *nodedist, *edgewt,*psrc,*noutgoing,*edgesigma,*border,*nerr;
	unsigned *borderNodes,*borderDist, *borderSigma;
	float *borderDelta;
	float *nodedelta;
	double *BC;
	bool *gpu_wait;
	long *cpu_level,*cpu_level_min,*gpu_level,*gpu_level_min;
	Graph graph;
	KernelConfig kconf(0);
	pthread_t thread1;
	int srcpartition, nonsrcpartition;
	unsigned numEdges, numNodes;
	unsigned source;
	unsigned int borderCount_cpu, borderIndex, borderIndex2,borderCount_gpu; // assuming bordernodes are small
	struct matrix_csr borderMatrix_cpu,borderMatrix_gpu;
	unsigned int *borderVector_cpu1,*borderVector_cpu2,*borderVector_gpu1,*borderVector_gpu2;
	unsigned int *borderSigma_cpu,*borderSigma_gpu;
	bool hchanged ;
	int num_threads=32;
	struct varto_cpu_part P;
	struct varto_gpu_part data_gpu;
	unsigned *vernodedist;
	double starttime, endtime,Finalstarttime,Finalendtime,tmpsttime,tmpendtime,fwdph_starttime,totalIterativeTime,
	       total_fwd_time=0,F_R,total_bck_time=0,bckph_starttime,cpuratio=0,gpuratio=0,init_start,init_end;
	unsigned i=0;
		//cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferL1);
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	init_start = rtclock(); //Initialization start time
	char *inputfile = argv[1];
	unsigned weighted = 0;
	/*	if(argc > 2){
		num_threads = atoi (argv[2]);
		omp_set_num_threads(num_threads);
		}
	*/
		 /*
		    //setting the pthread to a single core for performance issues
		    int s,j;
		    pthread_t self = pthread_self();
		    cpu_set_t cpuset,cpuset_other;
		    CPU_ZERO(&cpuset);
		    CPU_ZERO(&cpuset_other);
		    CPU_SET(0, &cpuset);
		    for(j = 1 ; j < (num_threads+1); j++)
			    CPU_SET (j,&cpuset_other);
		    s = pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
		    if (s != 0)
			handle_error_en(s, "pthread_setaffinity_np");
		    s = pthread_setaffinity_np(thread1, sizeof(cpu_set_t), &cpuset_other);

		    // Check the actual affinity mask assigned to the thread 
		    s = pthread_getaffinity_np(self, sizeof(cpu_set_t), &cpuset);
		    if (s != 0)
			handle_error_en(s, "pthread_getaffinity_np");

		    printf("Set returned by pthread_getaffinity_np() contained:\n");
		    for (j = 0; j < CPU_SETSIZE; j++)
			 if (CPU_ISSET(j, &cpuset))
			    printf("    CPU %d\n", j);
	*/
		/* For performance issues modify for non hyperthreaded core*/
	if(omp_get_num_procs() <= 4)
		num_threads = omp_get_num_procs();
	else{
	//num_threads = omp_get_num_procs()/2;
		printf("No of CPUs %d\n",omp_get_num_procs());
		num_threads-=0;
		num_threads=16;
		//num_threads=1;
	}
	omp_set_num_threads(num_threads);
	printf("arguments: %s %d\n", inputfile,num_threads);
	char name[80]="";
	strcat(name,"./partition_metis.exe ");
	strcat(name,inputfile);
	system(name);
	graph.read(inputfile, weighted);
	graph.initFrom(graph);
 		//Using ratio.txt to get the ratios
		//string name (inputfile);
		//execl(name,inputfile,0);
		//wait();
		//cudaDeviceReset();
	/*	ifstream cfile;
		cfile.open("ratio.txt");
		cfile>>cpuratio>>gpuratio;
		cout<<"Cpuratio: "<<cpuratio<<" Gpuratio: "<<gpuratio<<endl;
		// Using patoh
		graph.usepatoh(graph,inputfile,cpuratio,gpuratio);
		//graph.usepatoh(graph,inputfile,0.50,0.50);
		//graph.formMetisPartitions(graph,&graph.partition);
		*/
	ifstream cfile;
	cfile.open("partitioninfo.txt");
	cfile>>graph.partition.edgecut;
	for(unsigned ii=0;ii<graph.nnodes;ii++)
		cfile>>graph.partition.part[ii];
	cfile.close();
	graph.fillBorderAndCount(graph,&graph.partition);
	graph.formDevicePartitions(graph);
	cout<<"Graph nnodes is : "<<graph.nnodes<<endl;
	if(argc > 2){
		vernodedist = (unsigned *)malloc(sizeof(unsigned) * (graph.nnodes));
		/* Inputting distance values from serial algo for Verification*/
	        ifstream ip;
	        ip.open(argv[2],ios::in);
		if (ip.is_open()) {
		   ip >> vernodedist[i];
		   while (!ip.eof()) {
		  	 i++;
			 ip >> vernodedist[i];
		   }
		   }
		ip.close();
	}
	cudaGetLastError();
		// VSS: The initFrom function has been modifed to form the metis partitions and to fill data structures regd. the partitions
		//hgraph.print1x1();
		// VSS: A new function to fill data structures that will be sent to GPU
	graph.num_threads = num_threads;
	printf("max node count: %d\n", graph.maxNodeCount);
	printf("max edge count: %d\n", graph.maxEdgeCount);
	if (cudaMalloc((void **)&edgesrc, (graph.maxEdgeCount) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating edgesrc failed");
	if (cudaMalloc((void **)&edgedst, (graph.maxEdgeCount) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating edgedst failed");
	//if (cudaMalloc((void **)&edgewt, (graph.maxEdgeCount) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating edgewt failed");
	if (cudaMalloc((void **)&edgesigma, (graph.maxEdgeCount) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating edgesigma failed");
	if (cudaMalloc((void **)&nodedist, (graph.nnodes) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
	if (cudaMalloc((void **)&nodedelta, (graph.nnodes) * sizeof(float)) != cudaSuccess) CudaTest("allocating nodedelta failed");
	if (cudaMalloc((void **)&nodesigma, (graph.nnodes) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodesigma failed");
		//if (cudaMalloc((void **)&active, (graph.maxEdgeCount) * sizeof(bool)) != cudaSuccess) CudaTest("allocating edgedstsigma failed");
		//if (cudaMalloc((void **)&localchanged, sizeof(bool)) != cudaSuccess) CudaTest("allocating localchanged failed");
	if (cudaMalloc((void **)&psrc, (graph.nnodes+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
	if (cudaMalloc((void **)&noutgoing, (graph.nnodes+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
	if (cudaMalloc((void **)&border, (graph.nnodes) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");// CAlculate no. of errors
	if (cudaMalloc((void **)&gpu_wait, sizeof(bool)) != cudaSuccess) CudaTest("allocating gpu_wait failed");// CAlculate no. of errors
	if (cudaMalloc((void **)&borderNodes, (graph.partition.borderCount[GPUPARTITION]) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
	if (cudaMalloc((void **)&borderDist, (graph.partition.borderCount[GPUPARTITION]) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
	if (cudaMalloc((void **)&borderSigma, (graph.partition.borderCount[GPUPARTITION]) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
	if (cudaMalloc((void **)&borderDelta, (graph.partition.borderCount[GPUPARTITION]) * sizeof(float)) != cudaSuccess) CudaTest("allocating nodedist failed");

	kconf.setMaxThreadsPerBlock();
	kconf.setProblemSize(graph.maxEdgeCount);

	if (!kconf.coversProblem()) {
			printf("The number of threads(%d) does not cover the problem(%d), number of items per thread=%d.\n", kconf.getNumberOfBlockThreads()*kconf.getNumberOfBlocks(), kconf.getProblemSize(), kconf.getProblemSize() / (kconf.getNumberOfBlockThreads()*kconf.getNumberOfBlocks())); 
	}

	printf("GPU backward phase #blocks %d, #blocksize %d\n",kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads());
		
	numEdges = graph.devicePartition[GPUPARTITION].numEdges;
	numNodes = graph.devicePartition[GPUPARTITION].numNodes;
	cudaStream_t sone, stwo,sthree,sfour;
	cudaStreamCreate(&sone);
	cudaStreamCreate(&stwo);
	cudaStreamCreate(&sthree);
	cudaStreamCreate(&sfour);
	CUDACOPY(edgesrc, graph.devicePartition[GPUPARTITION].edgesrc, (numEdges) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
	CUDACOPY(edgedst, graph.devicePartition[GPUPARTITION].edgedst, (numEdges) * sizeof(unsigned int), cudaMemcpyHostToDevice,stwo);
		//CUDACOPY(edgewt, graph.devicePartition[GPUPARTITION].edgewt, (numEdges) * sizeof(unsigned int), cudaMemcpyHostToDevice,sthree);
	CUDACOPY(psrc, graph.devicePartition[GPUPARTITION].psrc, (graph.nnodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice,sone);
	CUDACOPY(noutgoing, graph.devicePartition[GPUPARTITION].noutgoing, (graph.nnodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice,stwo);
	CUDACOPY(border, graph.partition.border, (graph.nnodes) * sizeof(unsigned int), cudaMemcpyHostToDevice,stwo);
	CUDACOPY(borderNodes, graph.partition.borderNodes[GPUPARTITION], (graph.partition.borderCount[GPUPARTITION]) * sizeof(unsigned int), cudaMemcpyHostToDevice,stwo);
	cudaStreamSynchronize(sone);
	cudaStreamSynchronize(stwo);
	cudaStreamSynchronize(sthree);

	omp_lock_t *writelock=(omp_lock_t *)malloc(graph.nnodes*sizeof(omp_lock_t));
	cpu_level = (long *) malloc (sizeof(long));
	gpu_level = (long *) malloc (sizeof(long));
	cpu_level_min = (long *) malloc (sizeof(long));
	gpu_level_min = (long *) malloc (sizeof(long));
		//cpu_wait = (bool *) malloc (sizeof(bool));
		//gpu_wait = (bool *) malloc (sizeof(bool));
	BC = (double *)malloc (sizeof(double) * graph.nnodes);
	memset(BC,0,((graph.nnodes) * sizeof(double)));

	for (unsigned i = 0;i < graph.nnodes ; i++)
		omp_init_lock(&writelock[i]);
		  
	/* Perform border matrix computation for both cpu and gpu simulatenously here */
	 /* Initializing variables for cpu border matrix compuation function */
	// P has the datastructures for CPU
	    P.partition = &(graph.devicePartition[CPUPARTITION]);
	    P.num_threads = num_threads;
	    P.graph = &graph;
	    P.borderInfo = &(graph.partition);
	    P.borderMatrix_cpu = &borderMatrix_cpu;
	    P.single_relax = true;
	    P.lock = writelock;
	    P.cpu_F_I=P.cpu_F_R=P.cpu_bck_knl_time=P.cpu_fwd_knl_time=P.cpu_tot_bck_time=0;
	/* Initializing variables for gpu_part function */
	    //data_gpu has the data st for GPU
	    data_gpu.gpupart = &(graph.devicePartition[GPUPARTITION]);
	    data_gpu.graph = &graph;
	    data_gpu.borderMatrix = &borderMatrix_gpu;
	    data_gpu.borderInfo = &(graph.partition);
	    data_gpu.nodesigma = nodesigma;
	    data_gpu.edgesrc = edgesrc;
	    data_gpu.edgedst = edgedst;
	    data_gpu.nodedist = nodedist;
	    data_gpu.edgewt = edgewt;
	    data_gpu.edgesigma = edgesigma;
	    data_gpu.nodedelta = nodedelta;
		    //data_gpu.active = active;
		    //data_gpu.localchanged = localchanged;
	    data_gpu.kconf = &kconf;
	    data_gpu.single_relax = true;
	    data_gpu.psrc = psrc;
	    data_gpu.noutgoing = noutgoing;
	    data_gpu.border = border;
	    data_gpu.borderNodes = borderNodes;
	    data_gpu.borderDist = borderDist;
	    data_gpu.borderSigma = borderSigma;
	    data_gpu.borderDelta = borderDelta;
	    data_gpu.nerr = nerr;    
	    data_gpu.num_threads = num_threads;
	    data_gpu.lock = writelock;
	    data_gpu.d_gpu_wait = gpu_wait; 
	    data_gpu.gpu_F_I=data_gpu.gpu_F_R=data_gpu.gpu_bck_knl_time=data_gpu.gpu_fwd_knl_time=data_gpu.gpu_tot_bck_time=0;

			
		/*Border Matrix and vector data structure*/
	    borderCount_cpu = graph.partition.borderCount[CPUPARTITION]; /* Border Count is of CPU partition */
	    borderCount_gpu = graph.partition.borderCount[GPUPARTITION]; /* Border Count is of GPU partition */
	    borderMatrix_gpu.row_size= borderCount_gpu;
	    borderMatrix_cpu.row_size = borderCount_cpu;
	    borderVector_cpu1 = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_cpu);
	    borderVector_cpu2 = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_cpu);
	    borderVector_gpu1 = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_gpu);
	    borderVector_gpu2 = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_gpu);
	    borderSigma_cpu = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_cpu);
	    borderSigma_gpu = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_gpu);

	    /*Preprocessing Step:- Executing border matrix functions */
	   /* Branching off a thread for gpu computation*/
	    pthread_create(&thread1,NULL,gpu_BorderMatrix_comp,&(data_gpu));
	    cpu_BorderMatrix_comp(&P);
	    pthread_join(thread1,NULL);
	
	    unsigned s_count_gpu=0,s_count_cpu=0;
	    unsigned num_srcs=1;
	    srand (time(NULL));
	init_end = rtclock(); // initialization end time
        printf("\nInitialization Runtime = %.3lf ms\n", 1000*(init_end-init_start));

	/* BC algorithm starts */
	Finalstarttime = rtclock(); // start timing for bfs	 

/* Iterating over each vertex in the graph considering it as the Source */	
for (int iter=0 ; iter < num_srcs ; iter++) { // num_srcs for the number of sources to perform BC on
//BRANDE's algo phase 1 Performing BFS/SSSP from each source
try{
	fwdph_starttime = rtclock(); // start timing for bfs	 
	printf("\nIteration# %d",iter);

	source = 0;
	// Selecting the sources
	/*
	if(s_count_cpu < num_srcs/2){
		s_count_cpu++;
		while(1){
		source = rand() % graph.nnodes;
		if(graph.partition.part[source]==CPUPARTITION) break;
		}
	}else if(s_count_gpu < num_srcs/2){
		s_count_gpu++;
		while(1){
		source = rand() % graph.nnodes;
		if(graph.partition.part[source]==GPUPARTITION)break;
	        }
	}else{break;}
	*/

	/*Initializing data structures*/
	//GPU data
	init_time = rtclock();
	cudaMemsetAsync(edgesigma,0,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(unsigned)),sone);
	cudaMemsetAsync(nodesigma,0,((graph.nnodes) * sizeof(unsigned)),stwo);
	cudaMemsetAsync(nodedelta,0,((graph.nnodes) * sizeof(float)),sthree);
	cudaMemset(nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
	// CPU data
#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
{
#ifdef _OPENMP
#pragma omp sections 
#endif
{	
	#ifdef _OPENMP
	#pragma omp section
	#endif
	{
		memset(graph.devicePartition[CPUPARTITION].edgesigma,0,((graph.devicePartition[CPUPARTITION].numEdges) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodedelta,0,((graph.nnodes) * sizeof(float)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[CPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].edgesigma,0,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodedelta,0,((graph.nnodes) * sizeof(float)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
		memset(graph.devicePartition[GPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
	 /* Initializing all border Vectors */
	 for(borderIndex=0; borderIndex < borderCount_gpu; borderIndex++){
   		borderVector_gpu1[borderIndex] = borderVector_gpu2[borderIndex] = MYINFINITY;
		borderSigma_gpu[borderIndex] = 0;
		}
        }
        #ifdef _OPENMP
        #pragma omp section
        #endif
        {
	  for(borderIndex=0; borderIndex < borderCount_cpu; borderIndex++){
		borderVector_cpu1[borderIndex] =  borderVector_cpu2[borderIndex] = MYINFINITY;
		borderSigma_cpu[borderIndex] = 0;
		}
	}}  {
	#ifdef _OPENMP
      //  #pragma omp section
        #endif
        {
               cudaStreamSynchronize(sone);
	       cudaStreamSynchronize(stwo);
     	       cudaStreamSynchronize(sthree);

        }
	}}

          srcpartition = graph.partition.part[source];
          nonsrcpartition = 1-srcpartition; // will work for 2 partitions;
	//printf("GPU partition is %d Border Count %d\n",nonsrcpartition,borderCount_cpu);
	    /* Initializing variables for cpu_part function */
	    P.partition = &(graph.devicePartition[CPUPARTITION]);
	    P.num_threads = num_threads;
	    P.graph = &graph;
	    P.borderInfo = &(graph.partition);
	    P.single_relax = false;
	/* Initializing variables for gpu_part function */
	    data_gpu.gpupart = &(graph.devicePartition[GPUPARTITION]);
	    data_gpu.graph = &graph;
	    data_gpu.srcpartition = srcpartition;
	    data_gpu.nonsrcpartition = nonsrcpartition;
	    data_gpu.graph = &graph;
	    data_gpu.borderInfo = &(graph.partition);
	    data_gpu.nodesigma = nodesigma;
	    data_gpu.edgesrc = edgesrc;
	    data_gpu.edgedst = edgedst;
	    data_gpu.nodedist = nodedist;
	    data_gpu.edgewt = edgewt;
	    data_gpu.single_relax = false;
	    //data_gpu.active = active;
	    //data_gpu.localchanged = localchanged;
	    data_gpu.kconf = &kconf;
	    data_gpu.psrc = psrc;
	    data_gpu.noutgoing = noutgoing;

	    /* FORWARD PHASE starts */
	    unsigned iterations=0;   
	    if(srcpartition==CPUPARTITION){ // CPU Forward phase
		 printf("\nSource %u in CPU Partition\n",source);
		 P.source = source;
		 data_gpu.source = MYINFINITY;
  		 P.borderVector_cpu = borderVector_cpu1;
  		 P.borderSigma_cpu = borderSigma_cpu;
	    	
		/* CPU initial step */
		cpu_BFS(&P);
		//for(int ii=0;ii < graph.nnodes ;ii++)
		//	printf("%d %d\n",ii+1,graph.devicePartition[graph.partition.part[CPUPARTITION]].nodesigma[ii]);
		starttime = rtclock(); // start timing for bfs	 

	// ITERATIVE step
   	      while(1){
		iterations++;
		/* Update Edge cut by CPU */
		P.borderVector_gpu = borderVector_gpu1;
		P.borderSigma_gpu = borderSigma_gpu;
		//tmpsttime=rtclock();
		cpu_SingleRelax(&P);
		//tmpendtime=rtclock();
           	//printf("\ncpu single relax Runtime = %.3lf ms\n", 1000*(tmpendtime-tmpsttime));
		/* Resolve */
		//tmpsttime=rtclock();
		borderMatrixVector_comp (&borderMatrix_gpu,borderVector_gpu1,borderVector_gpu2,borderSigma_gpu,borderCount_gpu);
		//tmpendtime=rtclock();
           	//printf("\nborder matrix 1 comp Runtime = %.3lf ms\n", 1000*(tmpendtime-tmpsttime));
		/* copying data from bvcpu1 to bvcpu2, else doesn't work for the first time */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(borderIndex)
#endif
		for(borderIndex=0; borderIndex < borderCount_cpu; borderIndex++){
			borderVector_cpu2[borderIndex] = borderVector_cpu1[borderIndex];
			}
		/* Update Edge cut by GPU */
		data_gpu.borderVector_cpu = borderVector_cpu2;
  	    	data_gpu.borderVector_gpu = borderVector_gpu2;
		data_gpu.borderSigma_cpu = borderSigma_cpu;
  	    	data_gpu.borderSigma_gpu = borderSigma_gpu;
		//tmpsttime=rtclock();
		gpu_SingleRelax (&data_gpu);	
		//tmpendtime=rtclock();
		/* Resolve */
		//tmpsttime=rtclock();
		borderMatrixVector_comp (&borderMatrix_cpu,borderVector_cpu2,borderVector_cpu1,borderSigma_cpu,borderCount_cpu);
		//tmpendtime=rtclock();
		/* Check stopping condition */
		if (Equals (borderVector_cpu1,borderVector_cpu2,borderCount_cpu) )
			break;
		}
           endtime = rtclock();
           printf("\nEnd of Iterative step: #Iterations %d Runtime = %.3lf ms\n CPU n GPU simul relax\n", iterations,1000*(endtime-starttime));
	   totalIterativeTime += endtime-starttime;
		P.borderVector_cpu = borderVector_cpu1;
		data_gpu.borderVector_gpu = borderVector_gpu2;
		P.borderSigma_cpu = borderSigma_cpu;
		data_gpu.borderSigma_gpu = borderSigma_gpu;
	        pthread_create(&thread1,NULL,gpu_Relax,&(data_gpu));
		cpu_Relax(&P);
	        pthread_join(thread1,NULL);
	   }
	   else{
		   // GPU FORWARD phase
		printf("\nSource %u in GPU Partition\n",source);
 		data_gpu.source = source;
		P.source = MYINFINITY;
  		data_gpu.borderVector_gpu = borderVector_gpu1; 		data_gpu.borderSigma_gpu = borderSigma_gpu;
		/* GPU initial step */
		gpu_BFS(&data_gpu);
		starttime = rtclock(); // start timing for bfs	 
		// ITERATIVE step
		while(1){
		iterations++;
		/* Update Edge cut by GPU */
		data_gpu.borderVector_cpu = borderVector_cpu1; 		data_gpu.borderSigma_cpu = borderSigma_cpu;
		gpu_SingleRelax(&data_gpu);
		/* Resolve */
		borderMatrixVector_comp (&borderMatrix_cpu,borderVector_cpu1,borderVector_cpu2,borderSigma_cpu,borderCount_cpu);
		/* copying data from bvgpu1 to bvgpu2, else doesn't work for the first time */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(borderIndex)
#endif
		for(borderIndex=0; borderIndex < borderCount_gpu; borderIndex++){
			borderVector_gpu2[borderIndex] = borderVector_gpu1[borderIndex];
			}
		/* Update Edge cut by CPU */
		P.borderVector_cpu = borderVector_cpu2;
  	    	P.borderVector_gpu = borderVector_gpu2;
		P.borderSigma_cpu = borderSigma_cpu;
  	    	P.borderSigma_gpu = borderSigma_gpu;
		cpu_SingleRelax (&P);	
		/* Resolve */
		borderMatrixVector_comp (&borderMatrix_gpu,borderVector_gpu2,borderVector_gpu1,borderSigma_gpu,borderCount_gpu);
		/* Check stopping condition */
		if (Equals (borderVector_gpu1,borderVector_gpu2,borderCount_gpu) || iterations==5)
			break;
		}
           endtime = rtclock();
           printf("\nEnd of Iterative step: #Iterations %d Runtime = %.3lf ms\n CPU n GPU simul relax\n", iterations,1000*(endtime-starttime));
	   totalIterativeTime += endtime-starttime;
	   P.borderVector_cpu = borderVector_cpu2;
	   data_gpu.borderVector_gpu = borderVector_gpu1;
	   P.borderSigma_cpu = borderSigma_cpu;
	   data_gpu.borderSigma_gpu = borderSigma_gpu;
	   pthread_create(&thread1,NULL,gpu_Relax,&(data_gpu));
	   cpu_Relax(&P);
	   pthread_join(thread1,NULL);
	   }
           Finalendtime = rtclock();
           printf("\nForward Phase Runtime = %.3lf ms\n", 1000*(Finalendtime-fwdph_starttime));
	   total_fwd_time += Finalendtime-fwdph_starttime;
	   /* Verfication of the BFS */
	   if(argc > 2){
	   printf("\nVerifying the BFS is correct or not:-\n");
           unsigned cnt=0;
	   int part;
	   foru dist;
	   ofstream op,sigop;
	   op.open("my_bfs.txt",ios::out);
	   sigop.open("my_sigma.txt",ios::out);
		   for(i = 0 ; i < graph.nnodes ; i++)
		   {
			   part = graph.partition.part[i];
	    		   dist = graph.devicePartition[part].nodedist[i];
			   if(vernodedist[i]!=dist){
				 //  if(graph.partition.border[i]!=0){
				   cout<<"node: "<<i+1<<" Partition: "<<part<<" Distance original: "<<vernodedist[i]<<" Algo distance: "<<dist<<"\n";
				   cnt++;
				 //  }
			   }
			   op<<dist<<"\n";
			   sigop<<graph.devicePartition[part].nodesigma[i]<<"\n";
		   }
		   if(cnt==0)
			   printf("BFS CORRECT\n");
		   else
			   printf("BFS NOT CORRECT, Difference in no of distance values %u\n",cnt);
	   op.close();
	   sigop.close();
	   }

// BACKWARD PHASE STARTS	
	   bckph_starttime = rtclock();
	   /* BACKWARD PHASE */
	   *cpu_level = *gpu_level = 0;
	   *cpu_level_min = *gpu_level_min = MYINFINITY;
	   //*cpu_wait = *gpu_wait = false;
	    P.cpu_level = cpu_level;
	    P.gpu_level = gpu_level;
	    P.cpu_level_min = cpu_level_min;
	    P.gpu_level_min = gpu_level_min;
	    data_gpu.cpu_level = cpu_level;
	    data_gpu.gpu_level = gpu_level;
	    data_gpu.cpu_level_min = cpu_level_min;
	    data_gpu.gpu_level_min = gpu_level_min;
	   // finding the largest level in each partition
#ifdef _OPENMP
#pragma omp parallel sections num_threads(2)
	   {
#pragma omp section
		   {
#endif
	   for(unsigned ii = 0 ; ii < graph.nnodes ; ii++){
		if (*cpu_level < graph.devicePartition[CPUPARTITION].nodedist[ii] && graph.devicePartition[CPUPARTITION].nodedist[ii]!=MYINFINITY)
			*cpu_level = graph.devicePartition[CPUPARTITION].nodedist[ii];
		if (*cpu_level_min > graph.devicePartition[CPUPARTITION].nodedist[ii])
			*cpu_level_min = graph.devicePartition[CPUPARTITION].nodedist[ii];
	   }
#ifdef _OPENMP
		   }
#pragma omp section
		   {
#endif
	   for(unsigned ii = 0 ; ii < graph.nnodes ; ii++){
		if (*gpu_level < graph.devicePartition[GPUPARTITION].nodedist[ii] && graph.devicePartition[GPUPARTITION].nodedist[ii]!=MYINFINITY)
			*gpu_level = graph.devicePartition[GPUPARTITION].nodedist[ii];
		if (*gpu_level_min > graph.devicePartition[GPUPARTITION].nodedist[ii])
			*gpu_level_min = graph.devicePartition[GPUPARTITION].nodedist[ii];
	   }
#ifdef _OPENMP
		   }
	   }
#endif
	   printf("\n CPU_level_max:%ld CPU_Level_min:%ld GPU_level_max:%ld GPU_level_min %ld\n",*cpu_level,*cpu_level_min,*gpu_level,*gpu_level_min);
	   pthread_create(&thread1,NULL,gpu_backward,&(data_gpu));
           cpu_backward(&P);
	   pthread_join(thread1,NULL);
	   // Finally adding delta values to the global BC
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	   for(unsigned i=0 ; i < graph.nnodes ; i++){
		   int part = graph.partition.part[i];
		   double delta = graph.devicePartition[part].nodedelta[i];
		   BC[i] += delta;
	   }
           Finalendtime = rtclock();
	   total_bck_time += Finalendtime-bckph_starttime;
           printf("Iteration Runtime = %.3lf ms\n", 1000*(Finalendtime-fwdph_starttime));
	
	  /*
	   printf("\nPrinting distance values of all nodes\n");
	   for(i = 0 ;i < graph.nnodes;i++){
		   part = graph.partition.part[i];
		   printf("%u:%u\n",i+1,graph.devicePartition[part].nodedist[i]);
	   }
	   
	  ofstream op;
	   op.open("my_bfs.txt",ios::out);
	   for(i=0;i<graph.nnodes;i++){
		   part = graph.partition.part[i];
		   dist = graph.devicePartition[part].nodedist[i];
		   op<<dist<<"\n";
	   }
	   op.close();
	   */

	/*// Test code for outputting a partition
   	ofstream op1,op2;
	op1.open("gpu_partition_data.edges",ios::out);
	//op2.open("gpu_partition_values.txt",ios::out);
	op1<<"#Nodes: "<<graph.nnodes<<"\n";
	op1<<"#Edges: "<<graph.devicePartition[GPUPARTITION].numEdges<<"\n";
	op1<<"#Directed"<<"\n";
	for (i = 0 ; i < graph.devicePartition[GPUPARTITION].numEdges ; i++){
		op1<<graph.devicePartition[GPUPARTITION].edgesrc[i]<<" "<<graph.devicePartition[GPUPARTITION].edgedst[i]<<" "<<graph.devicePartition[GPUPARTITION].edgewt[i]<<"\n";
	}
	op1.close();
	*/
	/*
	for (i = 0 ; i < graph.nnodes ; i++){
		if(graph.partition.part[i]==GPUPARTITION)
			op2<<graph.devicePartition[GPUPARTITION].nodedist[i]<<"\n";
		else
			op2<<"1000000000"<<"\n";
	}
	*/
	
	//op2.close();
}catch(...){
	P.cpu_fwd_knl_time = P.cpu_F_I + P.cpu_F_R;
	data_gpu.gpu_fwd_knl_time = data_gpu.gpu_F_I + data_gpu.gpu_F_R;
	F_R =  P.cpu_F_R + data_gpu.gpu_F_R;
 	printf("\n***FORWARD PHASE Runtime =  %.3lf ms ***:-\n cpu F_I runtime = %.3lf ms", 1000*total_fwd_time,1000*(P.cpu_F_I));
	printf("\n gpu F_I runtime = %.3lf ms", 1000*(data_gpu.gpu_F_I));
	printf("\n cpu F_R runtime = %.3lf ms", 1000*(P.cpu_F_R));
	printf("\n gpu F_R runtime = %.3lf ms", 1000*(data_gpu.gpu_F_R));
	printf("\n MAX F_R runtime = %.3lf ms", 1000*(F_R));
	printf("\n cpu total forward runtime = %.3lf ms", 1000*(P.cpu_fwd_knl_time));
	printf("\n gpu total forward runtime = %.3lf ms", 1000*(data_gpu.gpu_fwd_knl_time));
	double cpu_util_fwd = (P.cpu_fwd_knl_time/total_fwd_time)*100;
	double gpu_util_fwd = (data_gpu.gpu_fwd_knl_time/total_fwd_time)*100;
	printf("\nCPU Util=%.3lf GPU Util=%.3lf", cpu_util_fwd,gpu_util_fwd);
	printf("\n***Iterative step runtime = %.3lf ms***", 1000*(totalIterativeTime));
	printf("\n***BACKWARD PHASE Runtime =  %.3lf ms ***:-\n cpu Total backward runtime = %.3lf ms ; kernel time = %.3lf ms",P.cpu_tot_bck_time*1000,1000*total_bck_time,1000*(P.cpu_bck_knl_time));
	printf("\n gpu Total backward runtime = %.3lf ms ; kernel time = %.3lf ms", data_gpu.gpu_tot_bck_time*1000,1000*(data_gpu.gpu_bck_knl_time));
	double cpu_util_bck = (P.cpu_bck_knl_time/total_bck_time)*100;
	double gpu_util_bck = (data_gpu.gpu_bck_knl_time/total_bck_time)*100;
	printf("\nCPU Util=%.3lf GPU Util=%.3lf",cpu_util_bck,gpu_util_bck);
	Finalendtime = rtclock();
	printf("\n***Total Final runtime = %.3lf ms***\n", 1000*(Finalendtime - Finalstarttime));
}
} /* end of FOR LOOP which iterates for each vertex as source for brande's algo */
	P.cpu_fwd_knl_time = P.cpu_F_I + P.cpu_F_R;
	data_gpu.gpu_fwd_knl_time = data_gpu.gpu_F_I + data_gpu.gpu_F_R;
	F_R =  P.cpu_F_R + data_gpu.gpu_F_R;
 	printf("\n***FORWARD PHASE Runtime =  %.3lf ms ***:-\n cpu F_I runtime = %.3lf ms", 1000*total_fwd_time,1000*(P.cpu_F_I));
	printf("\n gpu F_I runtime = %.3lf ms", 1000*(data_gpu.gpu_F_I));
	printf("\n cpu F_R runtime = %.3lf ms", 1000*(P.cpu_F_R));
	printf("\n gpu F_R runtime = %.3lf ms", 1000*(data_gpu.gpu_F_R));
	printf("\n MAX F_R runtime = %.3lf ms", 1000*(F_R));
	printf("\n cpu total forward runtime = %.3lf ms", 1000*(P.cpu_fwd_knl_time));
	printf("\n gpu total forward runtime = %.3lf ms", 1000*(data_gpu.gpu_fwd_knl_time));
	double cpu_util_fwd = (P.cpu_fwd_knl_time/total_fwd_time)*100;
	double gpu_util_fwd = (data_gpu.gpu_fwd_knl_time/total_fwd_time)*100;
	printf("\nCPU Util=%.3lf GPU Util=%.3lf", cpu_util_fwd,gpu_util_fwd);
	printf("\n***Iterative step runtime = %.3lf ms***", 1000*(totalIterativeTime));
	printf("\n***BACKWARD PHASE Runtime =  %.3lf ms ***:-\n cpu Total backward runtime = %.3lf ms ; kernel time = %.3lf ms",1000*total_bck_time,P.cpu_tot_bck_time*1000,1000*(P.cpu_bck_knl_time));
	printf("\n gpu Total backward runtime = %.3lf ms ; kernel time = %.3lf ms", data_gpu.gpu_tot_bck_time*1000,1000*(data_gpu.gpu_bck_knl_time));
	printf("\ncudaMemcpy Time = %.3lf ms***", 1000*(data_gpu.gpu_memcpy));
	double cpu_util_bck = (P.cpu_bck_knl_time/total_bck_time)*100;
	double gpu_util_bck = (data_gpu.gpu_bck_knl_time/total_bck_time)*100;
	printf("\nCPU Util=%.3lf GPU Util=%.3lf",cpu_util_bck,gpu_util_bck);
	Finalendtime = rtclock();
	printf("\n***Total Final runtime = %.3lf ms***\n", 1000*(Finalendtime - Finalstarttime));

	/* Freeing up stuff*/
	cudaFree(edgesrc);
	cudaFree(edgedst);
	//cudaFree(edgewt);
	cudaFree(edgesigma);
	cudaFree(nodedist);
	cudaFree(nodesigma);
	cudaFree(nodedelta);
	//cudaFree(active);
	//cudaFree(localchanged);
	cudaFree(psrc);
	cudaFree(noutgoing);
	cudaFree(nerr);
	cudaFree(border);
	cudaFree(borderDist);
	cudaFree(borderSigma);
	cudaFree(borderDelta);
	cudaFree(gpu_wait);
	//free(hnodesigma);
	/*
	for(borderIndex=0; borderIndex < borderCount_cpu; borderIndex++){
              free(borderMatrix_cpu[borderIndex]);
        }
	for(borderIndex=0; borderIndex < borderCount_gpu; borderIndex++){
        	free(borderMatrix_gpu[borderIndex]);
	}
	*/
	//free(borderMatrix_cpu.rowptrs);	
	//free(borderMatrix_gpu.rowptrs);
	free(borderVector_cpu1);
	free(borderVector_gpu1);
	free(borderVector_cpu2);
	free(borderVector_gpu2);
	free(borderSigma_cpu);
	free(borderSigma_gpu);
	if(argc > 2)
	   free(vernodedist);
	for (unsigned i = 0;i < graph.nnodes ; i++)
		omp_destroy_lock(&writelock[i]);
	free(writelock);
	free(cpu_level);
	//free(cpu_wait);
	free(gpu_level);
	//free(gpu_wait);
	free(cpu_level_min);
	free(gpu_level_min);
	free(BC);

        return 0;
}
