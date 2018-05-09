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
#include <utility>
#include <queue>
//#include "gbar.cuh"
#include "myutils.h"
#include "myutils2.h"
//#include "worklistc.h"
#define DIAMETER 22100
typedef pair<unsigned, unsigned> iPair;
//#define USE_DIA
//Worklist wl1(2), wl2(2);
//__device__ bool *gpu_wait;
void checkMyCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

double cpu_ratiotime,gpu_ratiotime;
int BM_count = 0;
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
	if(dist==0)
                cout<<"Shit detected "<<i<<"->"<<j<<endl;

//	cout<<"pushed!\n";
}
void modify_borderGraph_adj(struct matrix_csr *M,unsigned i, unsigned j,unsigned dist,unsigned sig){
	M->dist[i*(M->row_size)+j] = dist;
        M->sig[i*(M->row_size)+j] = sig;
	if(dist==0)
		cout<<"Dist 0 detected "<<i<<"->"<<j<<endl;
//      cout<<"pushed!\n";
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
matrix_csr borderGraph_adj;
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
           nodedist[ii] = MYINFINITY;
        }
        nodesigma[source] = 1;
        nodedist[source] = 0;
}
void initnodesigmadist_multisource(Graph *graph,unsigned *values, unsigned *sigma_values,unsigned nodes, unsigned* nodesigma, unsigned* nodedist,unsigned *sources,unsigned source_count,unsigned *psrc,unsigned *noutgoing,unsigned *edgedst,unsigned *border){
unsigned ii,j;

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

	
#pragma omp parallel for private(ii) schedule(static)
        for (ii = 0; ii < nodes; ii++) {
//		if(graph->partition.border[ii]==0)
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;
        }

/*#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(ii,j) num_threads(num_threads)
#endif
for ( ii = 0; ii < source_count; ii++) {
	unsigned v = sources[ii],w;
	unsigned num_edges_v = psrc[v];
	for (j = num_edges_v; j < (num_edges_v + noutgoing[v]) ; j++) {
		w = edgedst[j];
			if(border[w]==0)continue;
		nodedist[w]=MYINFINITY;
		nodesigma[w]=0;
	}
}*/
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
//	   nodesigma[sources[ii]] =  sigma_values[ii];
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

void borderMatrixVector_comp (struct matrix_csr *M, unsigned *Vin,unsigned *Vout,unsigned *S_V,unsigned Bcount)
{
	bool flag;
/*#ifdef _OPENMP
#pragma omp parallel shared(bV,flag)
	{
#endif*/
	unsigned borderIndex, borderIndex2;
//cout<<"\nPrinting Vout(GPU):\n";
/*#ifdef _OPENMP
#pragma omp parallel for schedule(static) 
#endif
	for (borderIndex=0;borderIndex < Bcount; borderIndex++)
	{	Vout[borderIndex] = Vin[borderIndex];
//		printf("%u:%u ",borderIndex,Vout[borderIndex]);
	}	
*/
	do{
BM_count++;
//cout<<" A ";
//#pragma omp single
	flag = false;
	/* Border vector and border matrix comparision, modifying the borderVector for smaller distance values */
#ifdef _OPENMP
//#pragma omp for schedule(dynamic)
#endif
	for ( borderIndex=0;borderIndex < Bcount; borderIndex++){
#pragma omp parallel for schedule(static)
          for(borderIndex2=0; borderIndex2 < Bcount; borderIndex2++){
		  struct d_s s=find_matrix(M,borderIndex,borderIndex2);
//		if(borderIndex==9 && borderIndex2==10)
//			printf("BM_comp : %d->%d:%d, d[%d] = %d, d[%d] = %d\n",borderIndex,borderIndex2,s.dist,borderIndex,Vout[borderIndex],borderIndex2,Vout[borderIndex2]);
	    if(borderIndex!=borderIndex2 && (Vout [borderIndex2] > s.dist + Vout [borderIndex]) && (s.dist != MYINFINITY) && (Vout [borderIndex] != MYINFINITY)){
	//		printf("\n%u->%u:%u | Vout[borderIndex2] = %u | Vout[borderIndex] = %u",borderIndex,borderIndex2,s.dist,Vout [borderIndex2],Vout [borderIndex]);
		    Vout [borderIndex2] = s.dist + Vout[borderIndex];
//		    S_V [borderIndex2] = s.sig-1 + S_V[borderIndex];
		    flag = true;
		}
	     }
	   }
/*#ifdef _OPENMP
#pragma omp barrier
#endif*/
/*cout<<"\nPrinting Vout in borderMatrixVector_comp():\n";
for (borderIndex=0;borderIndex < Bcount; borderIndex++)
        {     
              printf("%u:%u ",borderIndex,Vout[borderIndex]);
        }*/

	}while(flag);
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
void gpu_component (unsigned *psrc,unsigned *noutgoing,unsigned *d_psrc,unsigned *d_noutgoing,unsigned *edgesdstsrc,unsigned *edgessrcdst,unsigned hedges,unsigned hnodes,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,int *sources,cudaDeviceProp *dp,bool BM_COMP,unsigned *nerr, unsigned *border, int *sigma)
{
	//GlobalBarrierLifetime gb;
	lonestar_gpu(psrc,noutgoing,d_psrc,d_noutgoing,edgesdstsrc,edgessrcdst,hnodes,hedges,hdist,nodesigma,edgesigma,source_count,sources,dp,BM_COMP,nerr,border,sigma);
	//ananya_code_func(psrc,noutgoing,d_psrc,d_noutgoing,edgesdstsrc,edgessrcdst,hedges,hnodes,hdist,nodesigma,edgesigma,source_count,sources,dp,BM_COMP,nerr);
}
void cpu_component (unsigned *psrc,unsigned *noutgoing,unsigned *edgesdstsrc,unsigned *edgessrcdst,unsigned hnodes,unsigned hedges,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,int *sources,omp_lock_t *lock,bool BM_COMP, int num_threads, Graph &graph)
{
	betweenness_centrality_parallel(hnodes,hedges,psrc,edgessrcdst,edgesdstsrc,noutgoing,sources,source_count,hdist,nodesigma,edgesigma,lock,num_threads, graph);
	//betweenness_centrality_parallel(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,unsigned *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *lock,int num_threads)
	//worklist_cpu(psrc,noutgoing,edgesdstsrc,edgessrcdst,hnodes,hedges,hdist,nodesigma,edgesigma,source_count,sources,lock,BM_COMP,num_threads);
}

void cpu_bfs_relax(unsigned nnodes, unsigned nedges, unsigned *psrc,unsigned *noutgoing,unsigned* edgesrc, unsigned* edgedst, unsigned* edgewt, unsigned *nodesigma, foru *nodedist,unsigned *edgesigma,int nthreads,unsigned *borderNodes, unsigned bcount,omp_lock_t *lock,unsigned *border) {
	unsigned i,j;
#pragma omp parallel for schedule(guided) private(i,j) num_threads(nthreads)
	for ( i = 0; i < bcount; i++) {
		unsigned v = borderNodes[i],w;
		unsigned num_edges_v = psrc[v];
		foru ddist;
		foru wt=1;
		if(nodedist[v]==MYINFINITY)	continue;
		for (j = num_edges_v; j < (num_edges_v + noutgoing[v]) ; j++) {
			w = edgedst[j];
			if(border[w]==0)continue;

		omp_set_lock(&(lock[w]));
		ddist = nodedist[w];
		if (ddist > (nodedist[v] + wt)) {
                        nodedist[w] = nodedist[v] + wt;
			omp_unset_lock(&(lock[w]));
//#pragma omp atomic update
//			edgesigma[j] = nodesigma[v];
//#pragma omp atomic write
//			nodesigma[w] = edgesigma[j];
		}else if (ddist == (nodedist[v] + wt)){
			omp_unset_lock(&(lock[w]));
//#pragma omp atomic update
//			nodesigma[w] -= edgesigma[j];
//			edgesigma[j] = nodesigma[v];
//#pragma omp atomic update
//			nodesigma[w] += edgesigma[j];

		}
		else{
			omp_unset_lock(&(lock[w]));
  //                      edgesigma[j] = 0;
		}
		}
	}
}

void *cpu_BFS(void *P){
	  struct varto_cpu_part *var = (struct varto_cpu_part *)P;
	  Graph *graph = var->graph;
	  unsigned numEdges_src,numNodes_src,borderIndex,ii;
          int num_threads = var->num_threads,source = var->source;
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
	   cpu_component (srcpart->psrc,srcpart->noutgoing,srcpart->edgesrc,srcpart->edgedst,graph->nnodes,numEdges_src,srcpart->nodedist,srcpart->nodesigma,srcpart->edgesigma,0,&source,var->lock,false,num_threads,*graph);
	endtime = rtclock ();
	/* Fill up borderVector */
//#pragma omp parallel for schedule(static) num_threads(num_threads)
	if(!var->single_relax)
	{
	#pragma omp parallel for schedule(static) num_threads(num_threads)	
            for(borderIndex=0; borderIndex < borderCount; borderIndex++){	
		    var->borderVector_cpu[borderIndex] = srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
		    var->borderSigma_cpu[borderIndex] = srcpart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
//		    printf("%u:%d ",borderInfo->borderNodes[CPUPARTITION][borderIndex],srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]]);
	    }
	}
//	for(ii=0 ; ii < graph->nnodes ; ii++){
//	printf("%d:%u,%u; ",ii,srcpart->nodedist[ii],srcpart->nodesigma[ii]);
//	}

        printf("For CPU BFS runtime = %.3lf ms\n", 1000*(endtime -starttime));
	var->cpu_F_I += endtime-starttime;
	cpu_ratiotime += endtime-starttime;
/*
	ofstream cpupart_dist;
	cpupart_dist.open("cpupart_dist.txt");
	for(int d=0;d<graph->nnodes;d++)
		cpupart_dist<<d<<" "<<srcpart->nodedist[d]<<" "<<srcpart->nodesigma[d]<<" "<<(var->graph)->partition.part[d]<<endl;
	cpupart_dist.close();
*/
}

void *gpu_BFS(void *var){
	double starttime, endtime;
	struct varto_gpu_part *P = (struct varto_gpu_part *)var;
	unsigned borderIndex,borderIndex2;
	Graph *graph = P->graph;
        unsigned numEdges,numNodes;
	int source = P->source,ii;
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
	gpu_component (gpupart->psrc,gpupart->noutgoing,P->psrc,P->noutgoing,P->edgesrc,P->edgedst,numEdges,graph->nnodes,P->nodedist,P->nodesigma,P->edgesigma,1,&source,&(P->kconf->dp),false,P->nerr,P->border,NULL);
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

struct src_dist_pair{
int id;
int distance;
int sigma;
};

int cmpfunc(const void *a,const void *b)
{
        return(((struct src_dist_pair*)a)->distance - ((struct src_dist_pair*)b)->distance);
}
void *cpu_Relax(void *P){
	  struct varto_cpu_part *var = (struct varto_cpu_part *)P;
	  Graph *graph = var->graph;
	  unsigned numEdges_src,numNodes_src,source = var->source,borderIndex,ii;
	  int *sources,source_count;
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
	  
/* if(source!=MYINFINITY){
	  source_count = borderCount+1;
	  sources = (unsigned *)malloc ((borderCount+1)* sizeof(unsigned));
	  sources[borderCount] = source;
	  srcpart->nodedist[source] = 0;
	  srcpart->nodesigma[source] = 1;
	  }
	  else{
*/
	  sources = (int *)malloc ((2*borderCount+1)* sizeof(int));
	  source_count = borderCount;
//	  }


        unsigned ID;
        struct src_dist_pair src_dist_pair_instance[source_count];
  //      ofstream bo;
    //       bo.open("border_ans.txt",ios::out);
         for(borderIndex=0; borderIndex < borderCount; borderIndex++)
        {       ID =  borderInfo->borderNodes[CPUPARTITION][borderIndex];
//              printf("GPUafter %u:%u\n",ID,nonsrcpart->nodedist[ID]);
//              sources[borderIndex] = ID;
                src_dist_pair_instance[borderIndex].id = ID;
                src_dist_pair_instance[borderIndex].distance = srcpart->nodedist[ID];//myarr[(borderIndex+source)%250];
               // bo<<ID<<" "<<nonsrcpart->nodedist[ID]<<endl;
//              nonsrcpart->nodedist[ID] = myarr[(borderIndex+source)%250];
        }


        //starttime = rtclock();
        qsort(src_dist_pair_instance,borderCount,sizeof(struct src_dist_pair),cmpfunc);

                int k = 0,p = 0;
        int my_dist, curr_dist = src_dist_pair_instance[0].distance;
/*      while( curr_dist == 0)
        {
                cout<<"Node = "<<src_dist_pair_instance[p].id<<" removed. d = "<<nonsrcpart->nodedist[src_dist_pair_instance[p].id]<<endl;
                p++;
                curr_dist = src_dist_pair_instance[p].distance;
        }*/
	if(source!=MYINFINITY){
        sources[k] = 0; sources[k+1] = source;	srcpart->nodesigma[source] = 1;
	k=2;}
	sources[k] = -curr_dist;
      //  cout<<sources[0]<<sources[1]<<sources[k]<<" ";
        k++;
         for(borderIndex=0; borderIndex < borderCount; borderIndex++)
        {
                my_dist = src_dist_pair_instance[borderIndex].distance;
                if(my_dist > curr_dist)
                {
                        curr_dist = my_dist;
                        sources[k] = -my_dist;
    //                    cout<<sources[k]<<" ";
                        k++;
                }
              sources[k] = src_dist_pair_instance[borderIndex].id;
  //              cout<<sources[k]<<" ";
                k++;
        }
	sources[k] = -100000;




/*#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads)
#endif
	  for(borderIndex=0; borderIndex < borderCount; borderIndex++)
	        sources[borderIndex] = borderInfo->borderNodes[CPUPARTITION][borderIndex];
	*/ 
	 starttime = rtclock();
	/* Multisource BFS*/
	  cpu_component (srcpart->psrc,srcpart->noutgoing,srcpart->edgesrc,srcpart->edgedst,graph->nnodes,numEdges_src,srcpart->nodedist,srcpart->nodesigma,srcpart->edgesigma,k+1,sources,var->lock,false,num_threads,*graph);
	
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
	  Graph::DevicePartition *srcpart = var->partition;
	  Graph::Partition *borderInfo = var->borderInfo;
          numEdges_src = srcpart->numEdges;
          numNodes_src = srcpart->numNodes;
	  unsigned borderCount = borderInfo->borderCount[CPUPARTITION]; /* Border Count is of non GPU partition */
#ifdef _OPENMP
#pragma omp parallel
	  {
#endif

	  initnodesigmadist_multisource_omp_singlerelax(graph,var->borderVector_cpu,var->borderSigma_cpu,graph->nnodes, srcpart->nodesigma,srcpart->nodedist,borderInfo->borderNodes[CPUPARTITION],borderCount,num_threads); // sending the values of border node for a multi source bfs
	/*setting values of border nodes of CPU in GPU partition */
//#pragma omp parallel for schedule(static) num_threads(num_threads)
//printf("\n\n:-  ");
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
	  for(borderIndex=0 ; borderIndex < borderInfo->borderCount[GPUPARTITION] ; borderIndex++){
		  srcpart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]] = var->borderVector_gpu[borderIndex];
//		  srcpart->nodesigma[borderInfo->borderNodes[GPUPARTITION][borderIndex]] = var->borderSigma_gpu[borderIndex];
//		printf(" %u:%u ",borderInfo->borderNodes[GPUPARTITION][borderIndex],var->borderVector_gpu[borderIndex]);
	  }
#ifdef _OPENMP
}
#endif
	 if(source!=MYINFINITY){
	   srcpart->nodedist[source] = 0;
	  // srcpart->nodesigma[source] = 1;
	 }
	/* Multisource BFS*/
	 cpu_bfs_relax(numNodes_src, numEdges_src,srcpart->psrc,srcpart->noutgoing,srcpart->edgesrc, srcpart->edgedst, srcpart->edgewt, srcpart->nodesigma, srcpart->nodedist,srcpart->edgesigma,num_threads,borderInfo->borderNodes[CPUPARTITION],borderCount,var->lock,borderInfo->border);
	/* Fill up borderVector */
//printf("\n\n:-  ");
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
//	    var->borderSigma_gpu[borderIndex] = srcpart->nodesigma[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
	//	    printf("%u:%u ",borderInfo->borderNodes[GPUPARTITION][borderIndex],srcpart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]]);
	 }

//#pragma omp parallel for schedule(static) num_threads(num_threads)
/*
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
//Iresh :: This is not needed as borderVector_cpu was already up to date.
         for(borderIndex=0; borderIndex < borderCount; borderIndex++){
	    var->borderVector_cpu[borderIndex] = srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
//	    var->borderSigma_cpu[borderIndex] = srcpart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
		    //printf("%u:%d ",borderInfo->borderNodes[CPUPARTITION][borderIndex],srcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]]);
	}
*/
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
//		  gpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = var->borderSigma_cpu[borderIndex];
	  }
#ifdef _OPENMP
	 }
#endif
	 if(source!=MYINFINITY){
	   gpupart->nodedist[source] = 0;
//	   gpupart->nodesigma[source] = 1;
	 }
	/* Multisource BFS*/
	   cpu_bfs_relax(numNodes_src, numEdges_src,gpupart->psrc,gpupart->noutgoing,gpupart->edgesrc, gpupart->edgedst, gpupart->edgewt, gpupart->nodesigma, gpupart->nodedist,gpupart->edgesigma,num_threads,borderInfo->borderNodes[GPUPARTITION],borderCount,var->lock,borderInfo->border);
	/* Fill up borderVector */
//	printf("\n\n:- ");
//#pragma omp parallel for schedule(static) num_threads(num_threads)
#ifdef _OPENMP
#pragma omp parallel 
	 {
#endif
/*
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
            for(borderIndex=0; borderIndex < borderCount; borderIndex++){
		    var->borderVector_gpu[borderIndex] = gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
//		    var->borderSigma_gpu[borderIndex] = gpupart->nodesigma[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
		   // printf("%u:%u ",borderInfo->borderNodes[GPUPARTITION][borderIndex],gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]]);
	    }
*/
//#pragma omp parallel for schedule(static) num_threads(num_threads)
#ifdef _OPENMP
#pragma omp for schedule(static) private(borderIndex)
#endif
            for(borderIndex=0; borderIndex < borderInfo->borderCount[CPUPARTITION]; borderIndex++){
		    var->borderVector_cpu[borderIndex] = gpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
//		    var->borderSigma_cpu[borderIndex] = gpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
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
	  unsigned numEdges_cpu,numNodes_cpu,borderIndex,borderIndex2,ii;
	  int num_threads = var->num_threads,borderSource;
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
//	printf("before initnodesigmadist_omp()\n");
         initnodesigmadist_omp (borderSource,graph->nnodes, cpupart->nodesigma, cpupart->nodedist,num_threads);
	  //printf("Here I am\n"); 
	   cpu_component (cpupart->psrc,cpupart->noutgoing,cpupart->edgesrc,cpupart->edgedst,graph->nnodes,numEdges_cpu,cpupart->nodedist,cpupart->nodesigma,cpupart->edgesigma,1,&borderSource,var->lock,false,num_threads,*graph);
	//printf("after cpu_component()\n");      
//	for(int i=0;i<18;i++)	printf("%d:%u ,",i,cpupart->nodedist[i]);
//	printf("\n");     
	/* Fill up borderVector */
//	  cout<<"BorderSource: "<<borderSource<<" "<<" BorderIndex: "<<borderIndex<<endl;
        for(borderIndex2=0; borderIndex2 < borderCount; borderIndex2++){
//	if(cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex2]] == 0 && borderIndex2!=borderIndex)	
//	    printf("(%u->%u)%u->%u:%d,%u\n",graph->partition.border[(borderInfo)->borderNodes[CPUPARTITION][borderIndex]]-1,graph->partition.border[(borderInfo)->borderNodes[CPUPARTITION][borderIndex2]]-1,borderInfo->borderNodes[CPUPARTITION][borderIndex],borderInfo->borderNodes[CPUPARTITION][borderIndex2],cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex2]],cpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex2]]);
//	    modify_matrix(var->borderMatrix_cpu,borderIndex,borderIndex2, cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex2]],cpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex2]]);
//	modify_borderGraph_adj(&borderGraph_adj,graph->partition.border[(borderInfo)->borderNodes[CPUPARTITION][borderIndex]]-1,graph->partition.border[(borderInfo)->borderNodes[CPUPARTITION][borderIndex2]]-1, cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex2]],cpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex2]]);	
	if(borderIndex2!=borderIndex)
	modify_borderGraph_adj(&borderGraph_adj,borderIndex,borderIndex2, cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex2]],cpupart->nodesigma[borderInfo->borderNodes[CPUPARTITION][borderIndex2]]);
	   // printf("%u:%d \n",borderInfo->borderNodes[CPUPARTITION][borderIndex],cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]]);
	//	printf("[%u->%u:%d]\n",borderIndex,borderIndex2,find_matrix(var->borderMatrix_cpu,borderIndex,borderIndex2).dist);
	}

/*	ofstream cpupart_dist;
        cpupart_dist.open("cpupart_dist.txt");
	for(borderIndex2=0; borderIndex2 < borderCount; borderIndex2++)
                cpupart_dist<<borderInfo->borderNodes[CPUPARTITION][borderIndex2]<<" "<<cpupart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex2]]<<endl;
        cpupart_dist.close();
	char name[300]="",str[50]="";
        strcat(name,"/home/iresh/parallel_graph_alg_madduri_modified/parallel_betweenness_centrality /home/iresh/parallel_graph_alg_madduri_modified/input_graphs/cpupart.bin 1 ");
        sprintf(str, "%d", borderInfo->borderNodes[CPUPARTITION][borderIndex]);
       strcat(name,str);
	system(name);
*/
//	system("/home/iresh/parallel_graph_alg_madduri_modified/parallel_betweenness_centrality /home/iresh/parallel_graph_alg_madduri_modified/input_graphs/cpupart.bin 1 "+borderInfo->borderNodes[CPUPARTITION][borderIndex]);
//	printf("BorderCount = %d\n",borderCount);
//	graph->progressPrint(borderCount,borderIndex);
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
	int borderSource;
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
//		printf("[Total : %d]GPU BM Source : %d\n",borderCount,borderSource);
	     cudaMemset(P->edgesigma,0,(numEdges) * sizeof(unsigned));
	     cudaMemset(P->nodesigma,0,(graph->nnodes)*sizeof(unsigned));
	     cudaMemset(P->nodedist,MYINFINITY,(graph->nnodes)*sizeof(unsigned));
	     cudaMemcpy(&(P->nodedist[borderSource]), &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);
	     cudaMemcpy(&(P->nodesigma[borderSource]), &foruone, sizeof(foruone), cudaMemcpyHostToDevice);

   	     gpu_component (gpupart->psrc,gpupart->noutgoing,P->psrc,P->noutgoing,P->edgesrc,P->edgedst,numEdges,graph->nnodes,P->nodedist,P->nodesigma,P->edgesigma,1,&borderSource,&(P->kconf->dp),false,P->nerr,P->border,NULL);
	     ArrayToBorder <<<13,256>>>(P->nodedist,P->nodesigma,P->borderDist, P->borderSigma,P->borderNodes,borderCount);
//	cudaMemcpy(gpupart->nodedist,P->nodedist,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost);
	
	     CUDACOPY(borderDist,P->borderDist,(borderCount) * sizeof(unsigned), cudaMemcpyDeviceToHost,sone);
	     CUDACOPY(borderSigma,P->borderSigma,(borderCount) * sizeof(unsigned), cudaMemcpyDeviceToHost,stwo);
	     cudaStreamSynchronize(sone);
	     cudaStreamSynchronize(stwo);
//		for(int i=0;i<18;i++)   printf("%d:%u ,",i,gpupart->nodedist[i]);
  //      printf("\n");

		   // Fill the border row of borderMatrix correponding to the source
	     for(borderIndex2=0; borderIndex2 < borderCount; borderIndex2++)
	     {
//		if(borderIndex2!=borderIndex && borderDist[borderIndex2]==0) 
//			printf("Here (%u->%u) : %u->%u:%d,%d\n",graph->partition.border[(borderInfo)->borderNodes[GPUPARTITION][borderIndex]]-1,graph->partition.border[(borderInfo)->borderNodes[GPUPARTITION][borderIndex2]]-1,borderInfo->borderNodes[GPUPARTITION][borderIndex],borderInfo->borderNodes[GPUPARTITION][borderIndex2],borderDist[borderIndex2],borderSigma[borderIndex2]);	
			//fprintf(fp,"%u\n",gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex2]]);
//		    modify_matrix(P->borderMatrix,borderIndex,borderIndex2,borderDist[borderIndex2],borderSigma[borderIndex2]);
//		modify_borderGraph_adj(&borderGraph_adj,graph->partition.border[(borderInfo)->borderNodes[GPUPARTITION][borderIndex]]-1,graph->partition.border[(borderInfo)->borderNodes[GPUPARTITION][borderIndex2]]-1,borderDist[borderIndex2],borderSigma[borderIndex2]);
               if(borderIndex2!=borderIndex)
			modify_borderGraph_adj(&borderGraph_adj,borderInfo->borderCount[CPUPARTITION]+borderIndex,borderInfo->borderCount[CPUPARTITION]+borderIndex2,borderDist[borderIndex2],borderSigma[borderIndex2]);
			//printf("\n%u->%u:%u ",borderIndex,borderIndex2,borderDist[borderIndex2]);
		//	    if(gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex2]] > DIAMETER)
		//		                                    edges_BM++;
	     }
	
/*		ofstream gpupart_dist;
        gpupart_dist.open("gpupart_dist.txt");
        for(borderIndex2=0; borderIndex2 < borderCount; borderIndex2++)
	        gpupart_dist<<borderInfo->borderNodes[GPUPARTITION][borderIndex2]<<" "<<borderDist[borderIndex2]<<" "<<borderSigma[borderIndex2]<<" 1"<<endl;
					//gpupart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex2]]<<endl;
        gpupart_dist.close();
        char name[300]="",str[50]="";
        strcat(name,"/home/iresh/parallel_graph_alg_madduri_modified/parallel_betweenness_centrality /home/iresh/parallel_graph_alg_madduri_modified/input_graphs/gpupart.bin 1 ");
        sprintf(str, "%d", borderInfo->borderNodes[GPUPARTITION][borderIndex]);
       strcat(name,str);
        system(name);

*/
	  //  total_BM += borderCount;
		    //printf("\nGPUPART:- Paths greater than diameter: %lli,  Total elements in BM: %lli\n",edges_BM,total_BM);

		 /*   
		for(borderIndex2=0; borderIndex2 < borderCount; borderIndex2++){
			if(access_matrix(P->borderMatrix,borderIndex,borderIndex2)==MYINFINITY)
				nnz++;
		}
		*/
   // graph->progressPrint(borderCount,borderIndex);
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
	 int *sources,*sigma;
	 unsigned borderCount = borderInfo->borderCount[GPUPARTITION],bcount_temp; /* Border Count is of non GPU partition */
//	 cudaStream_t sone, stwo;
//cudaStreamCreate(&sone);
//cudaStreamCreate(&stwo);
//for(borderIndex=0; borderIndex < borderCount; borderIndex++)
  //      {       printf("GPU %u:%u\n",borderIndex,P->borderVector_gpu[ID]);
//}
//	starttime = rtclock();
	 initnodesigmadist_multisource(graph,P->borderVector_gpu,P->borderSigma_gpu,graph->nnodes, nonsrcpart->nodesigma,nonsrcpart->nodedist,borderInfo->borderNodes[GPUPARTITION],borderCount,nonsrcpart->psrc,nonsrcpart->noutgoing,nonsrcpart->edgedst,borderInfo->border); // sending the values of border node for a multi source bfs
//endtime = rtclock();
//printf("[Iresh-MultiS-initnodesigmadist_multisource]runtime = %.3lf ms, source = %d\n", 1000*(endtime -starttime),source);
//	if(source!=MYINFINITY){
//          nonsrcpart->nodedist[source] = 0;
//          nonsrcpart->nodesigma[source] = 1;
          sources = (int *)malloc ((borderCount*2+1) * sizeof(int));
	  sigma = (int *)malloc ((borderCount*2+1) * sizeof(int));
//        sources[borderCount] = source;
          bcount_temp = borderCount;
//         }
//         else{
//          sources = (int *)malloc ((borderCount*2+1) * sizeof(int));
//         bcount_temp = borderCount;
//          }
		  
	/*	for(ii=0 ; ii < borderCount ; ii++)
		{
			nodedist[sources[ii]] = P->borderVector_gpu[ii];
			nodesigma[borderInfo->borderNodes[GPUPARTITION][ii]] = P->borderSigma_gpu[ii];	   
		}*/
//	starttime = rtclock();
	unsigned ID;
	struct src_dist_pair src_dist_pair_instance[bcount_temp];
//	ofstream bo;
          // bo.open("border_ans.txt",ios::out);
	 for(borderIndex=0; borderIndex < borderCount; borderIndex++)
	{	ID =  borderInfo->borderNodes[GPUPARTITION][borderIndex];
//		printf("GPUafter %u:%u\n",ID,nonsrcpart->nodedist[ID]);
//		sources[borderIndex] = ID;
		src_dist_pair_instance[borderIndex].id = ID;
		src_dist_pair_instance[borderIndex].distance = P->borderVector_gpu[borderIndex];//nonsrcpart->nodedist[ID];//myarr[(borderIndex+source)%250];
		src_dist_pair_instance[borderIndex].sigma = P->borderSigma_gpu[borderIndex];
	//	bo<<ID<<" "<<nonsrcpart->nodedist[ID]<<endl;
//		nonsrcpart->nodedist[ID] = myarr[(borderIndex+source)%250];
	}


	//starttime = rtclock();
	qsort(src_dist_pair_instance,borderCount,sizeof(struct src_dist_pair),cmpfunc);

	        int k = 0,p = 0, c = 0;
        int my_dist, curr_dist = src_dist_pair_instance[0].distance;
/*	while( curr_dist == 0)
	{
		cout<<"Node = "<<src_dist_pair_instance[p].id<<" removed. d = "<<nonsrcpart->nodedist[src_dist_pair_instance[p].id]<<endl;
		p++;
		curr_dist = src_dist_pair_instance[p].distance;
	}*/
        sources[k] = -curr_dist;
	sigma[k] = -curr_dist;
	//cout<<sources[k]<<" ";
	k++;
         for(borderIndex=0; borderIndex < bcount_temp; borderIndex++)
        {     
                my_dist = src_dist_pair_instance[borderIndex].distance;
                if(my_dist > curr_dist)
                {
	//		sources[p] = c;
	//		sigma[p] = c;
                        curr_dist = my_dist;
                        sources[k] = -my_dist;
			sigma[k] = -my_dist;
			k++;
	//		cout<<sources[k]<<" ";
         //               p = k+1; c = 0; k+=2;
                }
              sources[k] = src_dist_pair_instance[borderIndex].id;
		sigma[k] = src_dist_pair_instance[borderIndex].sigma;
	//	cout<<sources[k]<<" ";
		k++;
	//	c++;
        }
        sources[k] = -100000;
	sigma[k] = -100000;
        //starttime = rtclock() - starttime;
        //	printf("\nSort time = %.3lf \n ",starttime*1000);
//	if(src_dist_pair_instance[0].distance==0)
//	{ 
//		cout<<"\t\tERROR.."<<src_dist_pair_instance[0].id<<"\t\tERROR\n";
	//	return NULL;
//	}
       // sources[bcount_temp] = src_dist_pair_instance[0].distance;
                  //if(P->single_relax){ // For update of edge list 
                          //setting values of border nodes of CPU in GPU partition 

                          //for(borderIndex=0 ; borderIndex < borderInfo->borderCount[CPUPARTITION] ; borderIndex++)
                        //        nonsrcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]] = P->borderVector_cpu[borderIndex];
                                  // Add sigma updation 
                  //}
//	starttime = rtclock();
        //CUDACOPY(P->nodedist,nonsrcpart->nodedist, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
      //   CUDACOPY(P->nodesigma,nonsrcpart->nodesigma, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,stwo);
   //               CUDACOPY(P->edgesigma,nonsrcpart->edgesigma, (numEdges) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
//	CUDACOPY(P->nodedist,nonsrcpart->nodedist, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
//	        CUDACOPY(P->nodesigma,nonsrcpart->nodesigma, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,stwo);
    // cudaStreamSynchronize(sone);
    //     cudaStreamSynchronize(stwo);
                  //cudaMemset(P->edgesigma,0,(numEdges) * sizeof(unsigned));
//       starttime = rtclock();
//endtime = rtclock();
//printf("[Iresh-cudaMemcpyHostToDevice]runtime = %.3lf ms\n", 1000*(endtime -starttime));
/*
	CUDACOPY(P->nodedist,nonsrcpart->nodedist, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
        CUDACOPY(P->nodesigma,nonsrcpart->nodesigma, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,stwo);
     cudaStreamSynchronize(sone);
     cudaStreamSynchronize(stwo);

                  //for(borderIndex=0; borderIndex < borderInfo->borderCount[CPUPARTITION]; borderIndex++)
                //          P->borderVector_cpu[borderIndex] = nonsrcpart->nodedist[borderInfo->borderNodes[CPUPARTITION][borderIndex]];
starttime = rtclock();
	gpu_component (nonsrcpart->psrc,nonsrcpart->noutgoing,P->psrc,P->noutgoing,P->edgesrc,P->edgedst,numEdges,graph->nnodes,P->nodedist,P->nodesigma,P->edgesigma,1,sources+1,&(P->kconf->dp),false,P->nerr);
	endtime = rtclock();
*///	printf("[Iresh-SS]For GPU RELAX runtime = %.3lf ms\n", 1000*(endtime -starttime));
//	CUDACOPY(P->nodedist,nonsrcpart->nodedist, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
//        CUDACOPY(P->nodesigma,nonsrcpart->nodesigma, (graph->nnodes) * sizeof(unsigned), cudaMemcpyHostToDevice,stwo);
//                  CUDACOPY(P->edgesigma,nonsrcpart->edgesigma, (numEdges) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
//         cudaStreamSynchronize(sone);
 //        cudaStreamSynchronize(stwo);

	starttime = rtclock();
         gpu_component (nonsrcpart->psrc,nonsrcpart->noutgoing,P->psrc,P->noutgoing,P->edgesrc,P->edgedst,numEdges,graph->nnodes,P->nodedist,P->nodesigma,P->edgesigma,k+1,sources,&(P->kconf->dp),false,P->nerr,P->border,sigma);
//exit(5);
	 endtime = rtclock();
	printf("[Iresh-MultiS]For GPU RELAX runtime = %.3lf ms\n", 1000*(endtime -starttime));
/*	starttime = rtclock();
		     //CUDACOPY(nonsrcpart->nodedist,P->nodedist,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost);
	 cudaMemcpy(nonsrcpart->nodedist,P->nodedist,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost);
		     CUDACOPY(nonsrcpart->nodesigma,P->nodesigma,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost,stwo);
		     CUDACOPY(nonsrcpart->edgesigma,P->edgesigma,(numEdges) * sizeof(unsigned), cudaMemcpyDeviceToHost,sone);
		     cudaStreamSynchronize(sone);
		     cudaStreamSynchronize(stwo);
		      endtime = rtclock();
        printf("[Iresh-MultiS-cudaMemcpyDeviceToHost]runtime = %.3lf ms\n", 1000*(endtime -starttime));
*/
//starttime = rtclock();

		      /* Store in the borderVector_gpu */
		//if(!(P->single_relax)){
		//    for(borderIndex=0; borderIndex < borderCount; borderIndex++)
		//	    P->borderVector_gpu[borderIndex] = nonsrcpart->nodedist[borderInfo->borderNodes[GPUPARTITION][borderIndex]];
	// printf("For GPU RELAX runtime = %.3lf ms\n", 1000*(endtime -starttime));
//	 P->gpu_F_R += endtime -starttime;
		    
//		  for(unsigned ii =0;ii<graph->nnodes;ii++)
//			  printf("%u:%u ",ii,nonsrcpart->nodedist[ii]);
		     // printf("\truntime = %.3lf ms\n", 1000*(endtime-starttime));
			
//	cudaStreamDestroy(sone);
//	cudaStreamDestroy(stwo);
		//if(!(P->single_relax))
	free (sources);
	free(sigma);
	pthread_exit(NULL);
		//else
//endtime = rtclock();
//        printf("[Iresh-MultiS-other]runtime = %.3lf ms\n", 1000*(endtime -starttime));


}
struct data_SSSP
{
unsigned borderCount_cpu, borderCount_gpu, source;
int *cpu_sources, *gpu_sources, *cpu_sigma, *gpu_sigma, *d_sources, *d_sigma;
unsigned *SS_CPU_borderDist, *borderSigma_cpu;
vector<unsigned> *dist, *sigma;
Graph *graph;
int *gpu_flag;
cudaStream_t *s_sssp, *s_border_child;
struct data_MultiS_GPU *data_MultiS_gpu;
struct binary_semaphore *sem;
vector< vector<int> > *cpu_border_child;
int *gpu_border_child, *gpu_border_child_count, *d_gpu_border_child, *d_gpu_border_child_count;
};

struct data_MultiS_GPU
{
unsigned *d_psrc, *d_noutgoing, *d_edgessrc, *d_edgesdst;
unsigned hnodes, hedges;
unsigned *dist, *nodesigma, *edgesigma;
unsigned source_count;
int *d_sources;
cudaDeviceProp *deviceProp;
bool BM_COMP;
unsigned *nerr, *d_border;
int *d_sigma;
int *gpu_flag;
int *child;
unsigned *child_count;
};

void* SSSP(void *var)//struct data_SSSP data_sssp)
{
printf("[CPU] SSSP started\n");
struct data_SSSP &data_sssp = *(struct data_SSSP*)var;
priority_queue< iPair, vector <iPair> , greater<iPair> > pq;
cudaStream_t &s_sssp = *(data_sssp.s_sssp);
cudaStream_t &s_border_child = *(data_sssp.s_border_child);
//cudaStreamCreate(&s_sssp);
unsigned borderCount_cpu = data_sssp.borderCount_cpu;
unsigned borderCount_gpu = data_sssp.borderCount_gpu;
unsigned source = data_sssp.source;
int *cpu_sources = data_sssp.cpu_sources;
int *gpu_sources = data_sssp.gpu_sources;
int *cpu_sigma = data_sssp.cpu_sigma;
int *gpu_sigma = data_sssp.gpu_sigma;
int *d_sources = data_sssp.d_sources;
int *d_sigma = data_sssp.d_sigma;
unsigned *SS_CPU_borderDist = data_sssp.SS_CPU_borderDist;
unsigned *borderSigma_cpu = data_sssp.borderSigma_cpu;
vector<unsigned>& dist = *(data_sssp.dist);
vector<unsigned>& sigma = *(data_sssp.sigma);
Graph& graph = *(data_sssp.graph);
int *gpu_flag = data_sssp.gpu_flag;
struct data_MultiS_GPU &data_MultiS_gpu = *(data_sssp.data_MultiS_gpu);
struct binary_semaphore &sem = *(data_sssp.sem);
int borderIndex;
double starttime, endtime;
int true_flag = 1, false_flag = 0;
//vector<unsigned> dist(borderCount_cpu+borderCount_gpu, MYINFINITY);
//vector<unsigned> sigma(borderCount_cpu+borderCount_gpu, 0);
//cpu_sources = (int *)malloc ((borderCount_cpu*2+1) * sizeof(int));
//cpu_sigma = (int *)malloc ((borderCount_cpu*2+1) * sizeof(int));
//gpu_sources = (int *)malloc ((borderCount_gpu*2+1) * sizeof(int));
//gpu_sigma = (int *)malloc ((borderCount_gpu*2+1) * sizeof(int));
//if (cudaMalloc((void **)&d_sources, (borderCount_gpu*2+1) * sizeof(int)) != cudaSuccess) CudaTest("allocating d_sources failed");
//if (cudaMalloc((void **)&d_sigma, (borderCount_gpu*2+1) * sizeof(int)) != cudaSuccess) CudaTest("allocating d_sigma failed");
for(borderIndex=0; borderIndex < borderCount_cpu; borderIndex++){
        if(SS_CPU_borderDist[borderIndex] == MYINFINITY) continue;
        pq.push(make_pair(SS_CPU_borderDist[borderIndex],graph.partition.border[(graph.partition).borderNodes[CPUPARTITION][borderIndex]]-1));
//      cout<<"Push ("<<SS_CPU_borderDist[borderIndex]<<", "<<(P.borderInfo)->borderNodes[CPUPARTITION][borderIndex]<<") \n";
        dist[graph.partition.border[(graph.partition).borderNodes[CPUPARTITION][borderIndex]]-1] = SS_CPU_borderDist[borderIndex];

}
ofstream sssp;
sssp.open("sssp_debug.txt",ios::out);
int prev_d = 0, n = 0, cpu_curr_dist = 0, gpu_curr_dist = 0, cpu_k = 2, gpu_k = 0, prev_cpu_index = 0, prev_gpu_index = 0, temp_gpu = -99999, temp_cpu = 0, final_prev_gpu_index, final_gpu_k, last_gpu_d, first_d_gpu;
//cout<<"cpu_sources[1] = "<<cpu_sources[1]<<endl;
//cpu_sources[1] = source; cpu_sources[2] = -99999;
//cpu_sigma[0] = 0; cpu_sigma[1] = 1;

//cpu_sources[0] = 0;
bool myflag = 1;
//cudaMemcpy(d_sources, &temp, sizeof(int), cudaMemcpyHostToDevice);
//lonestar_gpu_MultiS(psrc,noutgoing,edgessrc,edgesdst,graph.nnodes,graph.devicePartition[GPUPARTITION].numEdges,nodedist,nodesigma,edgesigma,borderCount_gpu,d_sources,&(data_gpu.kconf->dp),false,data_gpu.nerr,border,d_sigma);
//P->nodedist,P->nodesigma,P->edgesigma,k+1,sources,&(P->kconf->dp),false,P->nerr,P->border,sigma);
starttime = rtclock();
while (!pq.empty())
{
        unsigned u = pq.top().second, d = pq.top().first;
        pq.pop();
        if(d != dist[u])        continue;
        if(u<borderCount_cpu && SS_CPU_borderDist[u]==d)
                sigma[u] += borderSigma_cpu[u];
	if(u<borderCount_cpu)
        {
                if(d > cpu_curr_dist)
                {
			cpu_sources[cpu_k] = -99999;	//cpu_k = 0 for GPU src
			pthread_mutex_lock(&sem.mutex);
				cpu_sources[prev_cpu_index] = temp_cpu; //temp_cpu = -99999 for GPU src
				pthread_cond_signal(&sem.cvar);
			pthread_mutex_unlock(&sem.mutex);
			cpu_curr_dist = d;
			prev_cpu_index = cpu_k;
                        temp_cpu = -d;
                        cpu_sigma[cpu_k] = -d;
                        cpu_k++;
                }
	/*	for(int j=borderCount_cpu;j<borderGraph_adj.row_size;j++)
		{
			if(dist[j]!=MYINFINITY && dist[j]+1==d)
				child_gpu
		}
        */ 
		cpu_sources[cpu_k] = (graph.partition).borderNodes[CPUPARTITION][u];
                cpu_sigma[cpu_k] = sigma[u];
                cpu_k++;
        }
        else
        {
                if(d > gpu_curr_dist)
                {
			if(myflag)
		        {
                		myflag = 0;
                		prev_d = d;
				first_d_gpu = d;
        		}
			if(d - prev_d > 50)// && (last_gpu_d - first_d_gpu + 1)%2==0)
			{
				prev_d = d;
				gpu_sources[gpu_k] = -99999;
				gpu_sigma[gpu_k] = -99999;
				gpu_k++;
				
				cudaStreamSynchronize(s_sssp);
				checkMyCUDAError("kernel invocation[>1(SSSP)]");	
			//	cout<<"NO ERROR\n";
				final_gpu_k = gpu_k;
				final_prev_gpu_index = prev_gpu_index;
			/*	cout<<"Send sources to GPU : prev_gpu_index = "<<prev_gpu_index<<", gpu_k = "<<gpu_k<<" : ";
				for(int i=prev_gpu_index;i<gpu_k;i++)
					cout<<gpu_sources[i]<<":"<<gpu_sigma[i]<<" ";
				cout<<endl;
			*/	//if(gpu_k-prev_gpu_index-1>0)
				cudaMemcpyAsync(d_sources+final_prev_gpu_index, gpu_sources+final_prev_gpu_index,(final_gpu_k-final_prev_gpu_index)*sizeof(int) ,cudaMemcpyHostToDevice, s_sssp);
				cudaMemcpyAsync(d_sigma+final_prev_gpu_index, gpu_sigma+final_prev_gpu_index,(final_gpu_k-final_prev_gpu_index)*sizeof(int) ,cudaMemcpyHostToDevice, s_sssp);
				cudaMemcpyAsync(gpu_flag, &false_flag,sizeof(int) ,cudaMemcpyHostToDevice, s_sssp);
				//cudaStreamSynchronize(s_sssp);
				if(final_prev_gpu_index!=0)
					lonestar_gpu_MultiS(data_MultiS_gpu.d_psrc,data_MultiS_gpu.d_noutgoing,data_MultiS_gpu.d_edgessrc,data_MultiS_gpu.d_edgesdst,data_MultiS_gpu.hnodes,data_MultiS_gpu.hedges,data_MultiS_gpu.dist,data_MultiS_gpu.nodesigma,data_MultiS_gpu.edgesigma,data_MultiS_gpu.source_count,data_MultiS_gpu.d_sources+final_prev_gpu_index-1,data_MultiS_gpu.deviceProp,data_MultiS_gpu.BM_COMP,data_MultiS_gpu.nerr, data_MultiS_gpu.d_border, data_MultiS_gpu.d_sigma+final_prev_gpu_index-1, data_MultiS_gpu.gpu_flag, &s_sssp, data_MultiS_gpu.child_count, data_MultiS_gpu.child);
				else
					lonestar_gpu_MultiS(data_MultiS_gpu.d_psrc,data_MultiS_gpu.d_noutgoing,data_MultiS_gpu.d_edgessrc,data_MultiS_gpu.d_edgesdst,data_MultiS_gpu.hnodes,data_MultiS_gpu.hedges,data_MultiS_gpu.dist,data_MultiS_gpu.nodesigma,data_MultiS_gpu.edgesigma,data_MultiS_gpu.source_count,data_MultiS_gpu.d_sources+final_prev_gpu_index,data_MultiS_gpu.deviceProp,data_MultiS_gpu.BM_COMP,data_MultiS_gpu.nerr, data_MultiS_gpu.d_border, data_MultiS_gpu.d_sigma+final_prev_gpu_index, data_MultiS_gpu.gpu_flag, &s_sssp, data_MultiS_gpu.child_count, data_MultiS_gpu.child);
				prev_gpu_index = gpu_k;
			}
                        gpu_curr_dist = d;
                        gpu_sources[gpu_k] = -d;
                        gpu_sigma[gpu_k] = -d;
                        gpu_k++;
			last_gpu_d = d;
                }
                gpu_sources[gpu_k] = (graph.partition).borderNodes[GPUPARTITION][u-borderCount_cpu];
                gpu_sigma[gpu_k] = sigma[u];
                gpu_k++;

        }

//	cout<<"Pop ("<<dist[u]<<", "<<sigma[u]<<", "<<(u<borderCount_cpu?(graph.partition).borderNodes[CPUPARTITION][u]:(graph.partition).borderNodes[GPUPARTITION][u-borderCount_cpu])<<")"<<endl;
        /*n++;
        if(d-prev > 50)
        {
                sssp<<1000*(rtclock()-starttime)<<" "<<dist[u]<<" "<<n<<endl;
                prev = d;
                n=0;
        }
        */
        for(int j=0; j<borderGraph_adj.row_size;j++)
        {
                if(dist[j] > dist[u] + borderGraph_adj.dist[u*borderGraph_adj.row_size+j] && borderGraph_adj.dist[u*borderGraph_adj.row_size+j]!=MYINFINITY)
                {
                        dist[j] = dist[u] + borderGraph_adj.dist[u*borderGraph_adj.row_size+j];
                        pq.push(make_pair(dist[j], j));
                        sigma[j] = sigma[u]*borderGraph_adj.sig[u*borderGraph_adj.row_size+j];
//			if(j == 834)
 //                               cout<<"sigma["<<(j<borderCount_cpu?(graph.partition).borderNodes[CPUPARTITION][j]:(graph.partition).borderNodes[GPUPARTITION][j-borderCount_cpu])<<"] updated by "<<sigma[u]*borderGraph_adj.sig[u*borderGraph_adj.row_size+j]<<". and is "<<sigma[j]<<"due to path from sigma["<<(u<borderCount_cpu?(graph.partition).borderNodes[CPUPARTITION][u]:(graph.partition).borderNodes[GPUPARTITION][u-borderCount_cpu])<<"] = "<<sigma[u]<<endl;

//                      cout<<"["<<(u<borderCount_cpu?(P.borderInfo)->borderNodes[CPUPARTITION][u]:(P.borderInfo)->borderNodes[GPUPARTITION][u-borderCount_cpu])<<"-"<<borderGraph_adj.dist[u*borderGraph_adj.row_size+j]<<">"<<(j<borderCount_cpu?(P.borderInfo)->borderNodes[CPUPARTITION][j]:(P.borderInfo)->borderNodes[GPUPARTITION][j-borderCount_cpu])<<"] Push ("<<dist[j]<<", "<<sigma[j]<<", "<<(j<borderCount_cpu?(P.borderInfo)->borderNodes[CPUPARTITION][j]:(P.borderInfo)->borderNodes[GPUPARTITION][j-borderCount_cpu])<<") \n";
                }
                else if(dist[j] == dist[u] + borderGraph_adj.dist[u*borderGraph_adj.row_size+j] && borderGraph_adj.dist[u*borderGraph_adj.row_size+j]!=MYINFINITY)
                {
                        sigma[j] += sigma[u]*borderGraph_adj.sig[u*borderGraph_adj.row_size+j];
//			if(j == 834)
//				cout<<"sigma["<<(j<borderCount_cpu?(graph.partition).borderNodes[CPUPARTITION][j]:(graph.partition).borderNodes[GPUPARTITION][j-borderCount_cpu])<<"] updated by "<<sigma[u]*borderGraph_adj.sig[u*borderGraph_adj.row_size+j]<<". and is "<<sigma[j]<<"due to path from sigma["<<(u<borderCount_cpu?(graph.partition).borderNodes[CPUPARTITION][u]:(graph.partition).borderNodes[GPUPARTITION][u-borderCount_cpu])<<"] = "<<sigma[u]<<endl;
//                      cout<<"["<<(u<borderCount_cpu?(P.borderInfo)->borderNodes[CPUPARTITION][u]:(P.borderInfo)->borderNodes[GPUPARTITION][u-borderCount_cpu])<<"->"<<(j<borderCount_cpu?(P.borderInfo)->borderNodes[CPUPARTITION][j]:(P.borderInfo)->borderNodes[GPUPARTITION][j-borderCount_cpu])<<"] Sigma update ("<<dist[j]<<", "<<sigma[j]<<", "<<(j<borderCount_cpu?(P.borderInfo)->borderNodes[CPUPARTITION][j]:(P.borderInfo)->borderNodes[GPUPARTITION][j-borderCount_cpu])<<") \n";
                }
        }
}
cpu_sources[cpu_k] = -100000;
cpu_sources[cpu_k+1] = 0; //is child_vector completed?
cpu_sigma[cpu_k] = -100000;
gpu_sources[gpu_k] = -100000;
gpu_sigma[gpu_k] = -100000;
gpu_k++;
pthread_mutex_lock(&sem.mutex);
	cpu_sources[prev_cpu_index] = temp_cpu;
	pthread_cond_signal(&sem.cvar);
pthread_mutex_unlock(&sem.mutex);

cudaStreamSynchronize(s_sssp);
final_gpu_k = gpu_k;
final_prev_gpu_index = prev_gpu_index;
/*cout<<"Send sources to GPU : prev_gpu_index = "<<prev_gpu_index<<", gpu_k = "<<gpu_k<<" : ";
for(int i=prev_gpu_index;i<gpu_k;i++)
	cout<<gpu_sources[i]<<" ";
cout<<endl;
*/                              //if(gpu_k-prev_gpu_index-1>0)
cudaMemcpyAsync(d_sources+final_prev_gpu_index, gpu_sources+final_prev_gpu_index,(final_gpu_k-final_prev_gpu_index)*sizeof(int) ,cudaMemcpyHostToDevice, s_sssp);
cudaMemcpyAsync(d_sigma+final_prev_gpu_index, gpu_sigma+final_prev_gpu_index,(final_gpu_k-final_prev_gpu_index)*sizeof(int) ,cudaMemcpyHostToDevice, s_sssp);
cudaMemcpyAsync(gpu_flag, &false_flag,sizeof(int) ,cudaMemcpyHostToDevice, s_sssp);
               //                 cudaStreamSynchronize(s_sssp);
		
if(final_prev_gpu_index!=0)
                                        lonestar_gpu_MultiS(data_MultiS_gpu.d_psrc,data_MultiS_gpu.d_noutgoing,data_MultiS_gpu.d_edgessrc,data_MultiS_gpu.d_edgesdst,data_MultiS_gpu.hnodes,data_MultiS_gpu.hedges,data_MultiS_gpu.dist,data_MultiS_gpu.nodesigma,data_MultiS_gpu.edgesigma,data_MultiS_gpu.source_count,data_MultiS_gpu.d_sources+final_prev_gpu_index-1,data_MultiS_gpu.deviceProp,data_MultiS_gpu.BM_COMP,data_MultiS_gpu.nerr, data_MultiS_gpu.d_border, data_MultiS_gpu.d_sigma+final_prev_gpu_index-1, data_MultiS_gpu.gpu_flag, &s_sssp, data_MultiS_gpu.child_count, data_MultiS_gpu.child);
                                else    
                                        lonestar_gpu_MultiS(data_MultiS_gpu.d_psrc,data_MultiS_gpu.d_noutgoing,data_MultiS_gpu.d_edgessrc,data_MultiS_gpu.d_edgesdst,data_MultiS_gpu.hnodes,data_MultiS_gpu.hedges,data_MultiS_gpu.dist,data_MultiS_gpu.nodesigma,data_MultiS_gpu.edgesigma,data_MultiS_gpu.source_count,data_MultiS_gpu.d_sources+final_prev_gpu_index,data_MultiS_gpu.deviceProp,data_MultiS_gpu.BM_COMP,data_MultiS_gpu.nerr, data_MultiS_gpu.d_border, data_MultiS_gpu.d_sigma+final_prev_gpu_index, data_MultiS_gpu.gpu_flag, &s_sssp, data_MultiS_gpu.child_count, data_MultiS_gpu.child);

endtime = rtclock();
           printf("\nEnd of SSSP step: Runtime = %.3lf ms\n",1000*(endtime-starttime));

//Calculate child for border nodes.
vector< vector<int> >& cpu_border_child = *(data_sssp.cpu_border_child);
int *gpu_border_child = data_sssp.gpu_border_child;
int *gpu_border_child_count = data_sssp.gpu_border_child_count;
int *d_gpu_border_child = data_sssp.d_gpu_border_child;
int *d_gpu_border_child_count = data_sssp.d_gpu_border_child_count;
int gpu_c = 0;
//gpu_border_child, gpu_border_child_count
//vector< vector<int> > cpu_border_child;
cpu_border_child.reserve(borderCount_cpu);
for(int j = borderCount_cpu; j < borderGraph_adj.row_size; j++)
{
	gpu_border_child_count[j-borderCount_cpu] = gpu_c;
	for(int u = 0; u < borderCount_cpu; u++)
	{
		if(borderGraph_adj.dist[u*borderGraph_adj.row_size+j]==1)
		{
			if(dist[u]+1==dist[j])          //u->j
                        {
				(cpu_border_child[u]).push_back((graph.partition).borderNodes[GPUPARTITION][j-borderCount_cpu]);
                        }
                        else if(dist[j]+1==dist[u])
                        {
				gpu_border_child[gpu_c] = (graph.partition).borderNodes[CPUPARTITION][u];;
				gpu_c++;
                        }

		}		
	}
}
gpu_border_child_count[borderCount_gpu] = gpu_c;
pthread_mutex_lock(&sem.mutex);
        cpu_sources[cpu_k+1] = 1;	//Border child vector ready.
        pthread_cond_signal(&sem.cvar);
pthread_mutex_unlock(&sem.mutex);
cudaMemcpyAsync(d_gpu_border_child_count, gpu_border_child_count,(borderCount_gpu+1)*sizeof(int) ,cudaMemcpyHostToDevice, s_border_child);
cudaMemcpyAsync(d_gpu_border_child, gpu_border_child,(gpu_c)*sizeof(int) ,cudaMemcpyHostToDevice, s_border_child);
cudaStreamSynchronize(s_border_child);
cudaStreamSynchronize(s_sssp);
checkMyCUDAError("kernel invocation[>1]");
//lonestar_gpu_MultiS(data_MultiS_gpu.d_psrc,data_MultiS_gpu.d_noutgoing,data_MultiS_gpu.d_edgessrc,data_MultiS_gpu.d_edgesdst,data_MultiS_gpu.hnodes,data_MultiS_gpu.hedges,data_MultiS_gpu.dist,data_MultiS_gpu.nodesigma,data_MultiS_gpu.edgesigma,data_MultiS_gpu.source_count,data_MultiS_gpu.d_sources+final_prev_gpu_index,data_MultiS_gpu.deviceProp,data_MultiS_gpu.BM_COMP,data_MultiS_gpu.nerr, data_MultiS_gpu.d_border, data_MultiS_gpu.d_sigma, data_MultiS_gpu.gpu_flag, &s_sssp);
//cudaStreamSynchronize(s_sssp);

endtime = rtclock();
           printf("\nEnd of SSSP+MultiS step: Runtime = %.3lf ms\n",1000*(endtime-starttime));
//debug<<1000*(endtime-starttime)<<"\t";
/*cout<<"CPU_sources : \n";
for(int i=0;i<=cpu_k;i++)
cout<<cpu_sources[i]<<" ";
cout<<endl;
*/
//Debug
ofstream bo;
           bo.open("border_ans.txt",ios::out);
for(borderIndex=0; borderIndex < borderCount_cpu; borderIndex++)
bo<<(graph.partition).borderNodes[CPUPARTITION][borderIndex]<<" "<<dist[graph.partition.border[(graph.partition).borderNodes[CPUPARTITION][borderIndex]]-1]<<" "<<sigma[graph.partition.border[(graph.partition).borderNodes[CPUPARTITION][borderIndex]]-1]<<endl;
for(borderIndex=0; borderIndex < borderCount_gpu; borderIndex++)
bo<<(graph.partition).borderNodes[GPUPARTITION][borderIndex]<<" "<<dist[graph.partition.border[(graph.partition).borderNodes[GPUPARTITION][borderIndex]]-1]<<" "<<sigma[graph.partition.border[(graph.partition).borderNodes[GPUPARTITION][borderIndex]]-1]<<endl;



cout<<"SSSP bbye\n";
pthread_exit(NULL);
}
/*
void *GPU_MultiS(void *var)
{
struct data_MultiS_GPU &data_MultiS_gpu = *(struct data_MultiS_GPU *)var;
printf("[CPU] GPU_MultiS started\n");
lonestar_gpu_MultiS(data_MultiS_gpu.d_psrc,data_MultiS_gpu.d_noutgoing,data_MultiS_gpu.d_edgessrc,data_MultiS_gpu.d_edgesdst,data_MultiS_gpu.hnodes,data_MultiS_gpu.hedges,data_MultiS_gpu.dist,data_MultiS_gpu.nodesigma,data_MultiS_gpu.edgesigma,data_MultiS_gpu.source_count,data_MultiS_gpu.d_sources,data_MultiS_gpu.deviceProp,data_MultiS_gpu.BM_COMP,data_MultiS_gpu.nerr, data_MultiS_gpu.d_border, data_MultiS_gpu.d_sigma, data_MultiS_gpu.gpu_flag, data_MultiS_gpu.s_multi);
pthread_exit(NULL);
}
*/

struct borderStruct{
int ID;
int borderID;
int distance;
int sigma;
bool part;
};
int borderStructcmpfunc(const void *a,const void *b)
{
        return(((struct borderStruct*)a)->distance - ((struct borderStruct*)b)->distance);
}

int main(int argc, char *argv[]) {
	unsigned *nodesigma, *edgesrc, *edgedst, *nodedist, *edgewt,*psrc,*noutgoing,*edgesigma,*border,*nerr, *child_count, *cpu_visited_vertices, *cpu_num_visited;
	unsigned *borderNodes,*borderDist, *borderSigma;
	int *d_sources, *d_sigma, *cpu_sources, *gpu_sources, *cpu_sigma, *gpu_sigma, *child, *gpu_border_child, *gpu_border_child_count, *d_gpu_border_child, *d_gpu_border_child_count;
	float *borderDelta;
	float *nodedelta;
	double *BC;
	bool *gpu_wait;
	long *cpu_level,*cpu_level_min,*gpu_level,*gpu_level_min;
	Graph graph;
	struct binary_semaphore sem;
	pthread_mutex_init(&(sem.mutex), NULL);
	pthread_cond_init(&(sem.cvar), NULL);
	KernelConfig kconf(0);
	pthread_t thread1, thread2;
	int srcpartition, nonsrcpartition;
	unsigned numEdges, numNodes;
	unsigned source;
	unsigned int borderCount_cpu, borderIndex, borderIndex2,borderCount_gpu; // assuming bordernodes are small
	struct matrix_csr borderMatrix_cpu,borderMatrix_gpu;
	unsigned int *borderVector_cpu1,*borderVector_cpu2,*borderVector_gpu1,*borderVector_gpu2,*SS_CPU_borderDist;
	unsigned int *borderSigma_cpu,*borderSigma_gpu;
	bool hchanged ;
	int *gpu_flag;
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
	cudaSetDevice(0);
	//printf("line 1020\n");cudaMalloc((void **)&edgesrc, (100) * sizeof(unsigned));
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
	//system(name);
	cout<<"Calling graph.read(inputfile, weighted);\n";
	int maxdegree = graph.read(inputfile, weighted);
	graph.initFrom(graph);

	//printf("line 1066\n");cudaMalloc((void **)&edgesrc, (100) * sizeof(unsigned));
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
	ofstream debug;
	debug.open("debug.txt");
	cfile.open("partitioninfo.txt");
	cfile>>graph.partition.edgecut;
	for(unsigned ii=0;ii<graph.nnodes;ii++)
		cfile>>graph.partition.part[ii];
	cfile.close();
	cout<<"Calling fillBorderAndCount()\n";
	graph.fillBorderAndCount(graph,&graph.partition);
	cout<<"fillBorderAndCount() over. Calling graph.formDevicePartitions()\n";
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
	//printf("line 1109\n");cudaMalloc((void **)&edgesrc, (100) * sizeof(unsigned));
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
	if (cudaMalloc((void **)&psrc, (graph.nnodes+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating psrc failed");
	if (cudaMalloc((void **)&noutgoing, (graph.nnodes+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating noutgoing failed");
	if (cudaMalloc((void **)&border, (graph.nnodes+2) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating border failed");
	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");// CAlculate no. of errors
	if (cudaMalloc((void **)&gpu_wait, sizeof(bool)) != cudaSuccess) CudaTest("allocating gpu_wait failed");// CAlculate no. of errors
	if (cudaMalloc((void **)&borderNodes, (graph.partition.borderCount[GPUPARTITION]) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating borderNodes failed");
	if (cudaMalloc((void **)&borderDist, (graph.partition.borderCount[GPUPARTITION]) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating borderDist failed");
	if (cudaMalloc((void **)&borderSigma, (graph.partition.borderCount[GPUPARTITION]) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating borderSigma failed");
	if (cudaMalloc((void **)&borderDelta, (graph.partition.borderCount[GPUPARTITION]) * sizeof(float)) != cudaSuccess) CudaTest("allocating borderDelta failed");
	if (cudaMalloc((void **)&child_count, (graph.nnodes+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating child_count failed");
	if (cudaMalloc((void **)&child, (graph.devicePartition[GPUPARTITION].numEdges) * sizeof(int)) != cudaSuccess) CudaTest("allocating child failed");
	if (cudaMalloc((void **)&d_gpu_border_child, (maxdegree*(graph.partition.borderCount[GPUPARTITION])) * sizeof(int)) != cudaSuccess) CudaTest("allocating d_gpu_border_child failed");
	if (cudaMalloc((void **)&d_gpu_border_child_count, ((graph.partition.borderCount[GPUPARTITION])+1) * sizeof(int)) != cudaSuccess) CudaTest("allocating d_gpu_border_child_count failed");

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
	CUDACOPY(border, graph.partition.border, (graph.nnodes+2) * sizeof(unsigned int), cudaMemcpyHostToDevice,stwo);
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
	SS_CPU_borderDist = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_cpu);
	    borderVector_gpu1 = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_gpu);
	    borderVector_gpu2 = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_gpu);
	    borderSigma_cpu = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_cpu);
	    borderSigma_gpu = (unsigned int *) malloc (sizeof(unsigned int) * borderCount_gpu);
		cpu_visited_vertices = (unsigned int *) malloc (sizeof(unsigned int) * graph.nnodes);
		cpu_num_visited = (unsigned int *) malloc (sizeof(unsigned int) * 10000); //#define EXPECTED_GRAPH_DIAMETER 10000
		borderGraph_adj.dist.reserve((borderCount_cpu+borderCount_gpu)*(borderCount_cpu+borderCount_gpu));
		borderGraph_adj.sig.reserve((borderCount_cpu+borderCount_gpu)*(borderCount_cpu+borderCount_gpu));
		borderGraph_adj.row_size = borderCount_cpu+borderCount_gpu;
	//	priority_queue< iPair, vector <iPair> , greater<iPair> > pq;
		vector<unsigned> dist(borderCount_cpu+borderCount_gpu, MYINFINITY);
		vector<unsigned> sigma(borderCount_cpu+borderCount_gpu, 0);
		vector< vector<int> > cpu_border_child(borderCount_cpu);
		cpu_sources = (int *)malloc ((borderCount_cpu*2+2) * sizeof(int));
		cpu_sigma = (int *)malloc ((borderCount_cpu*2+1) * sizeof(int));
		gpu_sources = (int *)malloc ((borderCount_gpu*3+1) * sizeof(int));
		gpu_sigma = (int *)malloc ((borderCount_gpu*3+1) * sizeof(int));
		gpu_border_child = (int *)malloc ((borderCount_gpu*maxdegree) * sizeof(int));
		gpu_border_child_count = (int *)malloc ((borderCount_gpu+1) * sizeof(int));
		if (cudaMalloc((void **)&d_sources, (borderCount_gpu*3+1) * sizeof(int)) != cudaSuccess) CudaTest("allocating d_sources failed");
		if (cudaMalloc((void **)&d_sigma, (borderCount_gpu*3+1) * sizeof(int)) != cudaSuccess) CudaTest("allocating d_sigma failed");
		if (cudaMalloc((void **)&gpu_flag, sizeof(int)) != cudaSuccess) CudaTest("allocating gpu_flag failed");
		struct data_SSSP data_sssp;
		data_sssp.borderCount_cpu = borderCount_cpu;
		data_sssp.borderCount_gpu = borderCount_gpu;
		data_sssp.cpu_sources = cpu_sources;
		data_sssp.gpu_sources = gpu_sources;
		data_sssp.cpu_sigma = cpu_sigma;
		data_sssp.gpu_sigma = gpu_sigma;
		data_sssp.d_sources = d_sources;
                data_sssp.d_sigma = d_sigma;
                data_sssp.SS_CPU_borderDist = SS_CPU_borderDist;
		data_sssp.borderSigma_cpu = borderSigma_cpu;
		data_sssp.dist = &dist;
		data_sssp.sigma = &sigma;
		data_sssp.graph = &graph;
		data_sssp.gpu_flag = gpu_flag;
		data_sssp.s_sssp = &sone;
		data_sssp.s_border_child = &stwo;
		data_sssp.sem = &sem;
		data_sssp.cpu_border_child = &cpu_border_child;
		data_sssp.gpu_border_child = gpu_border_child;
		data_sssp.gpu_border_child_count = gpu_border_child_count;
		data_sssp.d_gpu_border_child = d_gpu_border_child;
		data_sssp.d_gpu_border_child_count = d_gpu_border_child_count;
	//	data_sssp.data_MultiS_gpu = &data_MultiS_gpu; 
#pragma omp parallel for schedule(static) private(borderIndex,borderIndex2)
		        for ( borderIndex=0;borderIndex < borderGraph_adj.row_size; borderIndex++){
          			for(borderIndex2=0; borderIndex2 < borderGraph_adj.row_size; borderIndex2++){
					modify_borderGraph_adj(&borderGraph_adj,borderIndex,borderIndex2,MYINFINITY,0);			
											}
										}	

#pragma omp parallel for schedule(static) private(i) num_threads(16)
        for ( i = 0; i < borderCount_cpu; i++) {
                unsigned v = (graph.partition).borderNodes[CPUPARTITION][i],w;
		unsigned num_edges_v = graph.devicePartition[CPUPARTITION].psrc[v];
                foru ddist;
                foru wt=1;
               // if(nodedist[v]==MYINFINITY)     continue;
                for (int j = num_edges_v; j < (num_edges_v +  graph.devicePartition[CPUPARTITION].noutgoing[v]) ; j++) {
                        w =  graph.devicePartition[CPUPARTITION].edgedst[j];
			if(graph.partition.border[w]==0)continue;
			modify_borderGraph_adj(&borderGraph_adj,graph.partition.border[v]-1,graph.partition.border[w]-1,1,1);
			modify_borderGraph_adj(&borderGraph_adj,graph.partition.border[w]-1,graph.partition.border[v]-1,1,1);
                }
        }
	 /*Executing border matrix functions */
		/* Branching off a thread for gpu computation*/
	    pthread_create(&thread1,NULL,gpu_BorderMatrix_comp,&(data_gpu));
//printf("Start GPU BM comp.Enter int : ");
//gpu_BorderMatrix_comp(&data_gpu);
//printf("Start CPU BM comp.Enter int : ");
    cpu_BorderMatrix_comp(&P);
	
    pthread_join(thread1,NULL);


struct data_MultiS_GPU data_MultiS_gpu;
data_sssp.data_MultiS_gpu = &data_MultiS_gpu;
data_MultiS_gpu.d_psrc = psrc;
data_MultiS_gpu.d_noutgoing = noutgoing;
data_MultiS_gpu.d_edgessrc = edgesrc;
data_MultiS_gpu.d_edgesdst = edgedst;
data_MultiS_gpu.hnodes = graph.nnodes;
data_MultiS_gpu.hedges = graph.devicePartition[GPUPARTITION].numEdges;
data_MultiS_gpu.dist = nodedist;
data_MultiS_gpu.nodesigma = nodesigma;
data_MultiS_gpu.edgesigma = edgesigma;
data_MultiS_gpu.source_count = borderCount_gpu;
data_MultiS_gpu.d_sources = d_sources;
data_MultiS_gpu.deviceProp = &(data_gpu.kconf->dp);
data_MultiS_gpu.BM_COMP = false;
data_MultiS_gpu.nerr = data_gpu.nerr;
data_MultiS_gpu.d_border= border;
data_MultiS_gpu.d_sigma = d_sigma;
data_MultiS_gpu.gpu_flag = gpu_flag;
data_MultiS_gpu.child = child;
data_MultiS_gpu.child_count = child_count;
//data_MultiS_gpu.s_multi = &sone;
/*cout<<"BorderGraph_sigma Adj :\n";
for ( borderIndex=0;borderIndex < borderGraph_adj.row_size; borderIndex++){
                                for(borderIndex2=0; borderIndex2 < borderGraph_adj.row_size; borderIndex2++){
					//	cout<<borderGraph_adj.dist[borderIndex*borderGraph_adj.row_size+borderIndex2]<<" ";               
						if(borderGraph_adj.dist[borderIndex*borderGraph_adj.row_size+borderIndex2] == 0)
							cout<<"( "<<borderIndex<<"->"<<borderIndex2<<" ) "<<(borderIndex<borderCount_cpu?(P.borderInfo)->borderNodes[CPUPARTITION][borderIndex]:(P.borderInfo)->borderNodes[GPUPARTITION][borderIndex-borderCount_cpu])<<"->"<<(borderIndex2<borderCount_cpu?(P.borderInfo)->borderNodes[CPUPARTITION][borderIndex2]:(P.borderInfo)->borderNodes[GPUPARTITION][borderIndex2-borderCount_cpu])<<endl;	 
								}
		//		cout<<endl;
                                                                                }
//	cout<<endl;
*/
	    unsigned s_count_gpu=0,s_count_cpu=0;
	    unsigned num_srcs=1;
	
	    srand (time(NULL));
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
		unsigned vis_srcs[10];
	vis_srcs[0] = 	 12                ;
vis_srcs[1] = 11152514	                     ;
vis_srcs[2] = 1368925	                     ;
vis_srcs[3] = 1422892	                     ;
vis_srcs[4] = 617101	                     ;
vis_srcs[5] = 313133	                     ;
vis_srcs[6] = 1282947	             ;
vis_srcs[7] = 1095742	             ;
vis_srcs[8] = 1438330	             ;
vis_srcs[9] = 1363146	                ;

	init_end = rtclock(); // initialization end time
        printf("\nInitialization Runtime = %.3lf ms\n", 1000*(init_end-init_start));
	Finalstarttime = rtclock(); // start timing for bfs
	double mytime = rtclock(),init_time = 0.0;	 
//BRANDE's algo phase 1 Performing BFS/SSSP from each source
for (int iter=0 ; iter < num_srcs ; iter++) { // num_srcs for the number of sources to perform BC on
try{
	fwdph_starttime = rtclock(); // start timing for bfs	 
	printf("\nIteration# %d",iter);
//	source = 1462550;
	// Selecting the sources

	if(s_count_cpu < num_srcs){
		s_count_cpu++;
		while(1){
		source = 1706453;//rand() % graph.nnodes;//vis_srcs[iter];//rand() % graph.nnodes;
		if(graph.partition.part[source]==CPUPARTITION && graph.partition.border[source]==0) break;
		}
	}
	
/*	else if(s_count_gpu < num_srcs/2){
		s_count_gpu++;
		while(1){
		source = rand() % graph.nnodes;
		if(graph.partition.part[source]==GPUPARTITION)break;
	        }
	}*/else{break;}	
	/*Initializing data structures*/
	//GPU data
	init_time = rtclock();
	cudaMemsetAsync(edgesigma,0,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(unsigned)),sone);
	cudaMemsetAsync(nodesigma,0,((graph.nnodes) * sizeof(unsigned)),sthree);
	cudaMemsetAsync(child_count,0,((graph.nnodes+1) * sizeof(unsigned)),stwo);
	cudaMemsetAsync(d_sources,-99999,((borderCount_gpu*2+1) * sizeof(int)),stwo);
	cudaMemsetAsync(gpu_flag,0,sizeof(int),stwo);
	cudaMemsetAsync(nodedelta,0,((graph.nnodes) * sizeof(float)),sthree);
	cudaMemsetAsync(nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)),sthree);
	
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
                memset(graph.devicePartition[CPUPARTITION].child_count,0,((graph.nnodes+1) * sizeof(float)));
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
	    /* Initializing variables for cpu_part function */

		init_time = rtclock() - init_time;
		 printf("\nInit_time = %.3lf ms\n",1000*init_time);
		debug<<"\n"<<source<<"\t"<<1000*init_time<<"\t";    
	/* FORWARD PHASE starts */
	    unsigned iterations=0;  
		BM_count = 0; 
	    if(srcpartition==CPUPARTITION){ // CPU Forward phase
		printf("\nSource %u in CPU Partition\n",source);
		 data_sssp.source = source;
		init_time = rtclock();
		P.source = source;
		data_gpu.source = MYINFINITY;
  		P.borderVector_cpu = SS_CPU_borderDist;
  		P.borderSigma_cpu = borderSigma_cpu;
	    	
		/* CPU initial step */
		cpu_BFS(&P);
		init_time = rtclock() - init_time;
                printf("\nSS-BFS_time = %.3lf ms\n",1000*init_time);
		debug<<1000*init_time<<"\t";
		lonestar_gpu_MultiS_initialize(psrc,graph.nnodes,graph.devicePartition[GPUPARTITION].numEdges);
		int temp = -99999;
//		cudaMemcpy(d_sources, &temp, sizeof(int), cudaMemcpyHostToDevice);
		cpu_sources[2] = -99999; cpu_sources[0] = 0; cpu_sources[1] = source;
		cpu_sigma[2] = -99999; cpu_sigma[0] = 0; cpu_sigma[1] = 1;
//		pthread_create(&thread1,NULL,SSSP,&(data_sssp));
//lonestar_gpu_MultiS(psrc,noutgoing,edgesrc,edgedst,graph.nnodes,graph.devicePartition[GPUPARTITION].numEdges,nodedist,nodesigma,edgesigma,borderCount_gpu,d_sources,&(data_gpu.kconf->dp),false,data_gpu.nerr,border,d_sigma,sone);

//		pthread_create(&thread2,NULL,GPU_MultiS,&(data_MultiS_gpu));

		printf("[CPU] Set dist to MYINFINITY\n");
		int ii;
		Graph::DevicePartition *srcpart = P.partition;
		#pragma omp parallel for private(ii) schedule(static)   //TODO : Decrease num_threads.
                for (ii = 0; ii < graph.nnodes; ii++) {
//			printf("%d\n", ii);
                        srcpart->nodesigma[ii] = 0;
                        srcpart->nodedist[ii] = MYINFINITY;
                        }
		srcpart->nodesigma[source] = 1;	//As source is in CPU
//		printf("[CPU] Gonna create SSSP thread\n");
		pthread_create(&thread1,NULL,SSSP,&(data_sssp));
		init_time = rtclock();
		betweenness_centrality_parallel_MultiS(graph.nnodes,graph.devicePartition[CPUPARTITION].numEdges,srcpart->psrc,srcpart->edgedst,srcpart->edgesrc,srcpart->noutgoing,cpu_sources,borderCount_cpu,srcpart->nodedist,srcpart->nodesigma,srcpart->edgesigma,P.lock,num_threads,graph.partition.border, &sem, cpu_sigma, srcpart->child_count, srcpart->child, cpu_border_child, graph.partition.borderNodes[CPUPARTITION], cpu_visited_vertices, cpu_num_visited);
		init_time = rtclock() - init_time;
                printf("\nMultiS-BFS CPU_time = %.3lf ms\n",1000*init_time);		
//void betweenness_centrality_parallel_MultiS(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,int *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *lock,int num_threads, unsigned *border)
//betweenness_centrality_parallel_MultiS(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,int *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *lock,int num_threads, unsigned *border)
//cpu_component (srcpart->psrc,srcpart->noutgoing,srcpart->edgesrc,srcpart->edgedst,graph->nnodes,numEdges_src,srcpart->nodedist,srcpart->nodesigma,srcpart->edgesigma,k+1,sources,var->lock,false,num_threads,*graph);

cout<<"\nSource : "<<source<<endl;
pthread_join(thread1,NULL);
cout<<"SSSP thread joined\n";
updateBorderChildGPU(borderNodes, d_gpu_border_child, d_gpu_border_child_count, child_count, child, borderCount_gpu);
//pthread_join(thread2,NULL);
//return 0;
/*cudaMemcpy(graph.devicePartition[GPUPARTITION].child_count,child_count,((graph.nnodes)*sizeof(unsigned)), cudaMemcpyDeviceToHost);
cudaMemcpy(graph.devicePartition[GPUPARTITION].child,child,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(int)), cudaMemcpyDeviceToHost);
for (ii = 0; ii < graph.nnodes; ii++)
{
	if((srcpart->child_count)[ii]==0) continue;
	cout<<"["<<ii<<"] "<<(srcpart->child_count)[ii]<<" -> ";
	for(i = 0; i<(srcpart->child_count)[ii]; i++)
		cout<<srcpart->child[i+srcpart->psrc[ii]]<<", ";
	cout<<endl;
}
Graph::DevicePartition *nonsrcpart = data_gpu.gpupart;
for (ii = 0; ii < graph.nnodes; ii++)
{
        if((nonsrcpart->child_count)[ii]==0) continue;
        cout<<"["<<ii<<"] "<<(nonsrcpart->child_count)[ii]<<" -> ";
        for(i = 0; i<(nonsrcpart->child_count)[ii]; i++)
                cout<<nonsrcpart->child[i+nonsrcpart->psrc[ii]]<<", ";
        cout<<endl;
}
*/
Graph::DevicePartition *gpupart = data_gpu.gpupart;
//Graph::DevicePartition *srcpart = P.partition;
 	 CUDACOPY(graph.devicePartition[GPUPARTITION].nodesigma,nodesigma,(graph.nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost,stwo);
	cudaMemcpy(graph.devicePartition[GPUPARTITION].nodedist,nodedist,((graph.nnodes)*sizeof(unsigned)), cudaMemcpyDeviceToHost);
 //            cudaStreamSynchronize(sone);
            cudaStreamSynchronize(stwo);

/*	totalIterativeTime += endtime-starttime;
	 init_time = rtclock();
		P.borderVector_cpu = borderVector_cpu1;
		data_gpu.borderVector_gpu = borderVector_gpu1;
		P.borderSigma_cpu = borderSigma_cpu;
		data_gpu.borderSigma_gpu = borderSigma_gpu;
	        pthread_create(&thread1,NULL,gpu_Relax,&(data_gpu));
		#pragma omp parallel for private(ii) schedule(static)	//TODO : Decrease num_threads.
	        for (ii = 0; ii < graph.nnodes; ii++) {
           		nodesigma[ii] = 0;
           		nodedist[ii] = MYINFINITY;
        		}


		cpu_Relax(&P);
	        pthread_join(thread1,NULL);
init_time = rtclock() - init_time;
                 printf("\nMultiS-BFS_time = %.3lf ms\n",1000*init_time);
debug<<1000*init_time;
*/	   }
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
//#pragma omp parallel for schedule(static) private(borderIndex)
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
           printf("\nEnd of Iterative step: #reqd_Iterations %d Iterative step Runtime = %.3lf ms\n CPU n GPU simul relax\n", iterations,1000*(endtime-starttime));
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
	int part;
		for(i=0;i<0;i++)
		{
			part = graph.partition.part[i];
			cout<<"σ["<<i<<"] = "<<graph.devicePartition[part].nodesigma[i]<<", d["<<i<<"] = "<<graph.devicePartition[part].nodedist[i]<<endl;
		}
	cout<<"Enter source : ";
	i=-1;//cin>>i;
	while(i != -1)
	{
		part = graph.partition.part[i];
		cout<<"σ["<<i<<"] = "<<graph.devicePartition[part].nodesigma[i]<<", d["<<i<<"] = "<<graph.devicePartition[part].nodedist[i]<<endl;
		cin>>i;
	}
//Graph::DevicePartition *gpupart = data_gpu.gpupart;
//Graph::DevicePartition *srcpart = P.partition;
// CUDACOPY(gpupart->nodedist,nodedist,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost,sone);
 //           CUDACOPY(gpupart->nodesigma,nodesigma,(graph->nnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost,stwo);
   //          cudaStreamSynchronize(sone);
     //       cudaStreamSynchronize(stwo);

ofstream op;
foru dist;
int sig;
int mycount = 0,gpu_cor = 0;
op.open("my_bfs.txt",ios::out);
for(i = 0 ; i < graph.nnodes ; i++)
                   {
                           part = graph.partition.part[i];
				dist = MYINFINITY;
				if(part == GPUPARTITION)
				{	
					Graph::DevicePartition *nonsrcpart = data_gpu.gpupart;
					dist = nonsrcpart->nodedist[i];
					sig = nonsrcpart->nodesigma[i];
					if((unsigned)dist!=4294967295)
						gpu_cor++;
				}
				else if(part == CPUPARTITION)
				{
					Graph::DevicePartition *srcpart = P.partition;
					dist = srcpart->nodedist[i];
					sig = srcpart->nodesigma[i];
				}
                          // dist = graph.devicePartition[part].nodedist[i];
				if(dist!=MYINFINITY)
					mycount++;
				op<<i<<" "<<(unsigned)dist<<" "<<(unsigned)sig<<endl;
}
cout<<"eligible count = "<<mycount;
cout<<"GPU count = "<<gpu_cor;

	   if(argc > 2){
	   printf("\nVerifying the BFS is correct or not:-\n");
           unsigned cnt=0;
	
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
/*	
	   bckph_starttime = rtclock();
	   // BACKWARD PHASE 
	   *cpu_level = *gpu_level = 0;
	   *cpu_level_min = *gpu_level_min = MYINFINITY;
	// *cpu_wait = *gpu_wait = false;
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
*/	
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
	mytime = rtclock() - mytime;
	printf("\n mytime = %.3lf ms", 1000*(mytime));
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
