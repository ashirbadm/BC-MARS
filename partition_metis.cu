#include "common.h"
#include "Structs.h"
#include "scheduler19.h"
#include "graph28.h"
#include "kernelconfig.h"
#include "list.h"
#include <cub/cub.cuh>
#include "myutils.h"
#include "myutils2.h"
double cpu_ratiotime,gpu_ratiotime;
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

/*	
#pragma omp parallel for private(ii) schedule(static)
        for (ii = 0; ii < nodes; ii++) {
//		if(graph->partition.border[ii]==0)
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;
        }
*/
#pragma omp parallel for schedule(static) private(ii,j) num_threads(num_threads)
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
	for(ii=0 ; ii < source_count ; ii++)
	{
	   nodedist[sources[ii]] = values[ii];
	   nodesigma[sources[ii]] =  sigma_values[ii];
	}
}


void initnodesigmadist_omp(unsigned source, unsigned nodes, unsigned* nodesigma, unsigned* nodedist,int num_threads){
unsigned ii;
#pragma omp parallel for private(ii) schedule(guided) num_threads(num_threads)
	for (ii = 0; ii < nodes; ii++) {
           nodesigma[ii] = 0;
           nodedist[ii] = MYINFINITY;
        }
        nodesigma[source] = 1;
        nodedist[source] = 0;
}
void gpu_component (unsigned *psrc,unsigned *noutgoing,unsigned *d_psrc,unsigned *d_noutgoing,unsigned *edgesdstsrc,unsigned *edgessrcdst,unsigned hedges,unsigned hnodes,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,unsigned *sources,cudaDeviceProp *dp,bool BM_COMP,unsigned *nerr)
{
	lonestar_gpu(psrc,noutgoing,d_psrc,d_noutgoing,edgesdstsrc,edgessrcdst,hedges,hnodes,hdist,nodesigma,edgesigma,source_count,sources,dp,BM_COMP,nerr);
}
void cpu_component (unsigned *psrc,unsigned *noutgoing,unsigned *edgesdstsrc,unsigned *edgessrcdst,unsigned hnodes,unsigned hedges,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,unsigned *sources,omp_lock_t *lock,bool BM_COMP, int num_threads)
{
	betweenness_centrality_parallel(hnodes,hedges,psrc,edgessrcdst,edgesdstsrc,noutgoing,sources,source_count,hdist,nodesigma,edgesigma,lock,num_threads);
	//worklist_cpu(psrc,noutgoing,edgesdstsrc,edgessrcdst,hnodes,hedges,hdist,nodesigma,edgesigma,source_count,sources,lock,BM_COMP,num_threads);
}
void *cpu_BFS(void *P){
	  struct varto_cpu_part *var = (struct varto_cpu_part *)P;
	  Graph *graph = var->graph;
	  unsigned numEdges_src,numNodes_src,source = var->source,borderIndex,ii;
          int num_threads = var->num_threads;
	  double starttime, endtime;
	  Graph::DevicePartition *srcpart = var->partition;
	  Graph::Partition *borderInfo = var->borderInfo;
          numEdges_src = srcpart->numEdges;
          numNodes_src = srcpart->numNodes;
	  unsigned borderCount = borderInfo->borderCount[CPUPARTITION]; /* Border Count is of non GPU partition */
	/* Do CPU BFS calculate border distance vector*/			
         initnodesigmadist_omp (source,graph->nnodes, srcpart->nodesigma, srcpart->nodedist,num_threads);
	  starttime = rtclock();
	   cpu_component (srcpart->psrc,srcpart->noutgoing,srcpart->edgesrc,srcpart->edgedst,graph->nnodes,numEdges_src,srcpart->nodedist,srcpart->nodesigma,srcpart->edgesigma,1,&source,var->lock,false,num_threads);
	endtime = rtclock ();
	printf("For CPU BFS runtime = %.3lf ms\n", 1000*(endtime -starttime));
	cpu_ratiotime += endtime-starttime;
}

void *gpu_BFS(void *var){
	double starttime, endtime;
	struct varto_gpu_part *P = (struct varto_gpu_part *)var;
	unsigned borderIndex,borderIndex2;
	Graph *graph = P->graph;
        unsigned numEdges,numNodes,source = P->source,ii;
        Graph::DevicePartition *gpupart = P->gpupart;
	Graph::Partition *borderInfo = P->borderInfo;
	numEdges = gpupart->numEdges;
	numNodes = gpupart->numNodes;
	foru foruzero = 0, foruone=1;
	unsigned borderCount = borderInfo->borderCount[GPUPARTITION]; /* Border Count is of non GPU partition */
	cudaMemset(P->edgesigma,0,(numEdges) * sizeof(unsigned));
	cudaMemset(P->nodesigma,0,(graph->nnodes)*sizeof(unsigned));
	cudaMemset(P->nodedist,MYINFINITY,(graph->nnodes)*sizeof(unsigned));
	cudaMemcpy(&(P->nodedist[source]), &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);
	cudaMemcpy(&(P->nodesigma[source]), &foruone, sizeof(foruone), cudaMemcpyHostToDevice);
        starttime = rtclock();
	gpu_component (gpupart->psrc,gpupart->noutgoing,P->psrc,P->noutgoing,P->edgesrc,P->edgedst,numEdges,graph->nnodes,P->nodedist,P->nodesigma,P->edgesigma,1,&source,&(P->kconf->dp),false,P->nerr);
	cudaDeviceSynchronize    ();
	endtime = rtclock ();
	printf("For GPU BFS runtime = %.3lf ms\n", 1000*(endtime -starttime));
	gpu_ratiotime += endtime-starttime;
}

int main(int argc, char *argv[]){
 if (argc < 2) {
      printf("Usage: %s <graph>\n", argv[0]);
      exit(1);
 }
 char *inputfile = argv[1];
 unsigned weighted = 0,numEdges,numNodes;
 unsigned *nodesigma, *edgesrc, *edgedst, *nodedist, *edgewt,*psrc,*noutgoing,*edgesigma,*border,*nerr;
 int num_threads=16;
 Graph graph;
 cudaDeviceReset();
 KernelConfig kconf(1);
 cudaStream_t sone, stwo,sthree,sfour;
 struct varto_cpu_part P;
 struct varto_gpu_part data_gpu;
 pthread_t thread1;
 double starttime, endtime,Finalstarttime,Finalendtime,tmpsttime,tmpendtime,fwdph_starttime,totalIterativeTime,total_fwd_time=0,F_R,total_bck_time=0,bckph_starttime;
 cudaStreamCreate(&sone);
 cudaStreamCreate(&stwo);
 cudaStreamCreate(&sthree);
 cudaStreamCreate(&sfour);
 if(omp_get_num_procs() <= 4)
        num_threads = omp_get_num_procs();
 else{
        //num_threads = omp_get_num_procs()/2;
        printf("No of CPUs %d\n",omp_get_num_procs());
        num_threads-=0;
        num_threads=16;
 }
 
 std::ofstream cfile;
 cfile.open("ratio.txt");

 omp_set_num_threads(num_threads);
 graph.read(inputfile, weighted);
 graph.initFrom(graph);
 
 graph.formMetisPartitions(graph, &graph.partition);
 graph.formDevicePartitions(graph);

        srand (time(NULL));
double tstarttime = rtclock();
 graph.num_threads = num_threads;
        printf("max node count: %d\n", graph.maxNodeCount);
        printf("max edge count: %d\n", graph.maxEdgeCount);
        if (cudaMalloc((void **)&edgesrc, (graph.maxEdgeCount) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating edgesrc failed");
        if (cudaMalloc((void **)&edgedst, (graph.maxEdgeCount) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating edgedst failed");
        //if (cudaMalloc((void **)&edgewt, (graph.maxEdgeCount) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating edgewt failed");
        if (cudaMalloc((void **)&edgesigma, (graph.maxEdgeCount) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating edgesigma failed");
        if (cudaMalloc((void **)&nodedist, (graph.nnodes) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
        if (cudaMalloc((void **)&nodesigma, (graph.nnodes) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodesigma failed");
        //if (cudaMalloc((void **)&active, (graph.maxEdgeCount) * sizeof(bool)) != cudaSuccess) CudaTest("allocating edgedstsigma failed");
        //if (cudaMalloc((void **)&localchanged, sizeof(bool)) != cudaSuccess) CudaTest("allocating localchanged failed");
        if (cudaMalloc((void **)&psrc, (graph.nnodes+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
        if (cudaMalloc((void **)&noutgoing, (graph.nnodes+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
        if (cudaMalloc((void **)&border, (graph.nnodes) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nodedist failed");
        if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");// CAlculate no. of errors
 kconf.setMaxThreadsPerBlock();
//      kconf.setProblemSize(graph.maxEdgeCount);

        if (!kconf.coversProblem()) {
                printf("The number of threads(%d) does not cover the problem(%d), number of items per thread=%d.\n", kconf.getNumberOfBlockThreads()*kconf.getNumberOfBlocks(), kconf.getProblemSize(), kconf.getProblemSize() / (kconf.getNumberOfBlockThreads()*kconf.getNumberOfBlocks()));
        }

        CUDACOPY(edgesrc, graph.devicePartition[GPUPARTITION].edgesrc, (numEdges) * sizeof(unsigned), cudaMemcpyHostToDevice,sone);
        CUDACOPY(edgedst, graph.devicePartition[GPUPARTITION].edgedst, (numEdges) * sizeof(unsigned int), cudaMemcpyHostToDevice,stwo);
        //CUDACOPY(edgewt, graph.devicePartition[GPUPARTITION].edgewt, (numEdges) * sizeof(unsigned int), cudaMemcpyHostToDevice,sthree);
        CUDACOPY(psrc, graph.devicePartition[GPUPARTITION].psrc, (graph.nnodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice,sone);
        CUDACOPY(noutgoing, graph.devicePartition[GPUPARTITION].noutgoing, (graph.nnodes+1) * sizeof(unsigned int), cudaMemcpyHostToDevice,stwo);
        CUDACOPY(border, graph.partition.border, (graph.nnodes) * sizeof(unsigned int), cudaMemcpyHostToDevice,stwo);
        cudaStreamSynchronize(sone);
        cudaStreamSynchronize(stwo);
        cudaStreamSynchronize(sthree);
        omp_lock_t *writelock=(omp_lock_t *)malloc(graph.nnodes*sizeof(omp_lock_t));
// Perform border matrix computation for both cpu and gpu simulatenously here 
 // Initializing variables for cpu border matrix compuation function 
            P.partition = &(graph.devicePartition[CPUPARTITION]);
            P.num_threads = num_threads;
            P.graph = &graph;
            P.borderInfo = &(graph.partition);
            P.single_relax = false;
            P.lock = writelock;
            P.cpu_F_I=P.cpu_F_R=P.cpu_bck_knl_time=P.cpu_fwd_knl_time=P.cpu_tot_bck_time=0;
// Initializing variables for gpu_part function 
            data_gpu.gpupart = &(graph.devicePartition[GPUPARTITION]);
            data_gpu.graph = &graph;
            data_gpu.borderInfo = &(graph.partition);
            data_gpu.nodesigma = nodesigma;
            data_gpu.edgesrc = edgesrc;
            data_gpu.edgedst = edgedst;
            data_gpu.nodedist = nodedist;
            data_gpu.edgewt = edgewt;
            data_gpu.edgesigma = edgesigma;
            data_gpu.kconf = &kconf;
            data_gpu.single_relax = false;
            data_gpu.psrc = psrc;
            data_gpu.noutgoing = noutgoing;
            data_gpu.border = border;
            data_gpu.nerr = nerr;
            data_gpu.num_threads = num_threads;
            data_gpu.lock = writelock;
	    
    for(int ii=0;ii<5;ii++){
        //Initializing data structures
        //GPU data
        cudaMemset(edgesigma,0,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(unsigned)));
        cudaMemset(nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        cudaMemset(nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        // CPU data
        memset(graph.devicePartition[CPUPARTITION].edgesigma,0,((graph.devicePartition[CPUPARTITION].numEdges) * sizeof(unsigned)));
        memset(graph.devicePartition[CPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        memset(graph.devicePartition[CPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        memset(graph.devicePartition[GPUPARTITION].edgesigma,0,((graph.devicePartition[GPUPARTITION].numEdges) * sizeof(unsigned)));
        memset(graph.devicePartition[GPUPARTITION].nodesigma,0,((graph.nnodes) * sizeof(unsigned)));
        memset(graph.devicePartition[GPUPARTITION].nodedist,MYINFINITY,((graph.nnodes) * sizeof(unsigned)));
        while(1){
                data_gpu.source = rand() % graph.nnodes;
                if(graph.partition.part[data_gpu.source]==GPUPARTITION) break;
         }
        pthread_create(&thread1,NULL,gpu_BFS,&(data_gpu));
        while(1){
                P.source = rand() % graph.nnodes;
                if(graph.partition.part[P.source]==CPUPARTITION) break;
        }
        cpu_BFS(&P);
        pthread_join(thread1,NULL);
        }
	
	
	if(gpu_ratiotime/(cpu_ratiotime+gpu_ratiotime) > 0.10){
	       cfile<<gpu_ratiotime/(cpu_ratiotime+gpu_ratiotime)<<" "<<cpu_ratiotime/(cpu_ratiotime+gpu_ratiotime)<<endl;
	cout<<"Ratio for cpu and gpu are "<<gpu_ratiotime/(cpu_ratiotime+gpu_ratiotime)<<" "<<cpu_ratiotime/(cpu_ratiotime+gpu_ratiotime)<<std::endl; 
	}
	else{
		cout<<"Ratio for cpu and gpu are "<<10<<" "<<90;
		cfile<<10<<" "<<90;
	}

	//cout<<"Ratio for cpu and gpu are "<<35<<" "<<65;
	//cfile<<50<<" "<<50;
	//cfile<<40<<" "<<60;
	//cfile<<20<<" "<<80;
	cfile.close();

	char name[80]="";
	strcat(name,"./partition_patoh.exe ");
	strcat(name,inputfile);
	system(name);
	
	double tendtime = rtclock();
	printf("Ration calculation time = %.3lf ms\n",(tendtime-tstarttime)*1000);
/*	
	std::ofstream outfile;
	outfile.open("partitioninfo.txt");
	outfile<<graph.partition.edgecut<<std::endl;
	for(unsigned ii=0;ii<graph.nnodes;ii++)
		        outfile<<graph.partition.part[ii]<<std::endl;
	outfile.close();
*/	
	//cudaDeviceReset();
return 0;
}
