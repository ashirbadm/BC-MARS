/** Breadth-first search -*- CUDA -*-
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
 * Example breadth-first search application for demoing Galois system.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 */



#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <string.h>
#include <sys/time.h>
//#include "cuPrintf.cu"
using namespace std;

#include "common.h"
//#include "list.h"
//#include "metis.h"
//#include "graph28.h"
#include "Structs.h"
#include "worklist7.h"

#include "cudaKernels.h"
//#include "metrics.h"
//#include "convergence.h"
//#include "kernelconfig.h"
#include "myutils.h"
#include <omp.h>



//#define warpsize 32          // Take from GPU
//#define unsigned unsigned

//#define MYINFINITY 1000000000
//#define ENABLE_SERIAL_ALGO
#define VERIFY
//#define ENABLE_METRIC

#define SWAP(a,b) { tmp = a; a = b; b = tmp; }
//#define PRINT_BSM
//#define OPTIMIZE
unsigned hnerr;
void ananya_code_func(unsigned *psrc,unsigned *noutgoing,unsigned *d_psrc,unsigned *d_noutgoing,unsigned *d_edgessrc,unsigned *d_edgesdst,unsigned hnodes,unsigned hedges,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,unsigned *sources,cudaDeviceProp *dp,bool BM_COMP,unsigned *nerr){
//unsigned int NVERTICES;
unsigned NSM;
	unsigned KBLOCKSIZE;
	unsigned FACTOR;
	clock_t starttime, endtime;
	int runtime;
	int itr=0;
//	int dc;
	unsigned intzero = 0;
	struct timeval before,after,beforeo,aftero;
	unsigned floatzero = 0;
	//unsigned *dist;
	//unsigned *d_psrc,*d_noutgoing,*d_edgessrc,*d_edgesdst;
	//cudaDeviceReset();
	
	//cudaGetDeviceCount(&dc); //Get number of GPu's in the system
	//printf("dc=%d\n",dc);
	//cudaGetDeviceProperties(&deviceProp, 0);

	NSM = dp->multiProcessorCount; //Get MultiprocessorCount
	KBLOCKSIZE = dp->maxThreadsDim[0]; //Get Maximum dimension of Thread

	//printf("Device Multiprocessor Count: %d \n", NSM);
	//printf("Maximum Dimension of Thread: %d \n", KBLOCKSIZE);
	//printf("Device Name:%s\n",dp->name);
	Worklist inwl,outwl,*inwlptr, *outwlptr, *tmp;
	inwl.init();
	outwl.init();
	FACTOR=((hedges+ (KBLOCKSIZE * NSM)-1)/(KBLOCKSIZE * NSM));// FACTOR=621
	unsigned nblocks=FACTOR*NSM;
//	printf("assigned dist\n");
	//if (cudaMalloc((void **)&dist,(hnodes) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating dist failed");// Allocating destination distance
	//if (cudaMalloc((void **)&d_psrc,(hnodes+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating dist failed");
	//if (cudaMalloc((void **)&d_noutgoing,(hnodes) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating dist failed");
	//if (cudaMalloc((void **)&d_edgessrc,(hedges+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating dist failed");
	//if (cudaMalloc((void **)&d_edgesdst,(hedges+1) * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating dist failed");

	//cudaMemset((void *)&dist,MYINFINITY,hgraph->nnodes * sizeof(unsigned));
	/*
	printf("initializing.\n");
	initialize<<<nblocks,KBLOCKSIZE>>> (dist,hgraph->nnodes); 
	CudaTest("initializing failed");			           
	*/
	//cudaMemcpy(dist, hdist, sizeof(unsigned)*hnodes, cudaMemcpyHostToDevice); // Initialize
	//cudaMemcpy(&dist[set_parent_src], &floatzero, sizeof(floatzero), cudaMemcpyHostToDevice); // Source distance=0
	//cudaMemcpy(d_psrc, psrc, sizeof(unsigned)*(hnodes+1), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_noutgoing, noutgoing, sizeof(unsigned)*(hnodes), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_edgessrc, edgessrc, sizeof(unsigned)*(hedges+1), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_edgesdst, edgesdst, sizeof(unsigned)*(hedges+1), cudaMemcpyHostToDevice);
	//getchar();
	
	//printf("inwl capacity:%d\n",inwl.getCapacity());
	//printf("outwl capacity:%d\n",outwl.getCapacity());
	unsigned wlsz = 0;
	inwl.ensureSpace(13421760);
	outwl.ensureSpace(13421760);
	//inwl.ensureSpace(hedges);
	//outwl.ensureSpace(hedges);
	//inwl.ensureSpace(hedges);
	//outwl.ensureSpace(hedges);
	//printf("space ensured\n");
	//getchar();
	//unsigned hstart, hend;
//	inwl.getStartEnd(hstart, hend);
//	printf("inwl Start: %d End: %d\n",hstart,hend);
//	outwl.getStartEnd(hstart, hend);
//	printf("outwl Start: %d End: %d\n",hstart,hend);
	
	//gettimeofday(&beforeo, NULL);
	//double t1=beforeo.tv_sec+(beforeo.tv_usec/1000000.0);
	for(unsigned s=0;s < source_count ;s++)
	{
	unsigned nout=noutgoing[sources[s]];
	unsigned *k = (unsigned *) malloc (nout *sizeof(unsigned));
	//printf("hi3\n");
	for(unsigned i=0;i<nout;i++){
		k[i]=psrc[sources[s]]+i;
		//printf("eidst of source:%d\n",k[i]);
	}
	inwl.pushRangeHost(k,nout);
	free(k);
	}
//	gettimeofday(&aftero, NULL);
//	double t2=aftero.tv_sec+(aftero.tv_usec/1000000.0); 
	//printf("Time elapsed using gettimeofday(): %.6lf ms\n",(t2-t1)*1000.0f);
//	inwl.getStartEnd(hstart, hend);
//	printf("inwl Start: %d End: %d\n",hstart,hend);
//	outwl.getStartEnd(hstart, hend);
//	printf("outwl Start: %d End: %d\n",hstart,hend);
	
	inwlptr = &inwl;
	outwlptr = &outwl;
	//printf("worklist initialised\n");
	//getchar();
	//printf("solving.\n");
	//printf("do-while\n");
	
	//gettimeofday(&before, NULL);
	//t1=before.tv_sec+(before.tv_usec/1000000.0); 
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);
	
	do
	{				
		itr++;
	//	printf("itr:%d\n",itr);
		//cudaMemcpy(nerr, &intzero, sizeof(intzero), cudaMemcpyHostToDevice);
		cudaMemset(nerr,0,sizeof(unsigned));
		kernel_ddV3<<<(NSM),KBLOCKSIZE/2>>>(hdist,nodesigma,edgesigma,BM_COMP,d_edgessrc,d_edgesdst,d_psrc,d_noutgoing,*inwlptr,*outwlptr,nerr);		
		cudaMemcpy(&hnerr, nerr, sizeof(hnerr), cudaMemcpyDeviceToHost);
		wlsz = outwlptr->getSize();
	//	printf("WLSZ:%d\n",wlsz);
	//	outwlptr->getStartEnd(hstart, hend);
	//	printf("outwl in cpu after kernel Start: %d End: %d\n",hstart,hend);
		if (hnerr == 0) {
			SWAP(inwlptr, outwlptr);
			outwlptr->noverflows = inwlptr->noverflows;
		} else {	// error: currently only buffer oveflow.
			if (++outwlptr->noverflows == MAXOVERFLOWS) {
            //printf ("overflow\n");
				unsigned cap = inwlptr->getCapacity();
				//inwlptr->printHost();
				inwlptr->ensureSpace(2 * cap);	// double the capacity.
				//inwlptr->printHost();
				outwlptr->ensureSpace(2 * cap);
				inwlptr->appendHost(outwlptr);
				outwlptr->noverflows = 0;
			} else {
				// defer increasing worklist capacity.
				//printf("\tdeferred increasing worklist capacity.\n");
			}
			//printf("\tinwlsz=%d, outwlsz=%d.\n", inwlptr->getSize(), outwlptr->getSize());
		}
		outwlptr->clearHost();	// clear it whether overflow or not.
	}while(wlsz); 

	//gettimeofday(&after, NULL);
	//t2=after.tv_sec+(after.tv_usec/1000000.0); 
	//printf("Number of Iterations: %d\n",itr);
	//runtime = (int) (1000.0f * (endtime - starttime) / CLOCKS_PER_SEC);
	//printf("Time elapsed using gettimeofday(): %.6lf ms\n",(t2-t1)*1000.0f);
	//printf("using clock() %d ms.\n", runtime);
	//printf("Time(do-while loop):%.2f ms\n",time);
	//cudaMemcpy(hdist, dist,(hnodes) * sizeof(unsigned), cudaMemcpyDeviceToHost);// Copy Distance vector from device
	//for(int ii=0;ii < hgraph->nnodes;ii++) printf("%u\n",hdist[ii]);
	
	//cudaMemcpy(nerr, &intzero, sizeof(unsigned), cudaMemcpyHostToDevice);// Initialize no. of errors as 0
/*
	printf("verifying  serially\n");
	verifysolution(hgraph->edgessrcdst, hgraph->edgessrcwt, hgraph->noutgoing,hgraph->psrc, hdist, NVERTICES, hgraph->nedges, hgraph->srcsrc);
	ofstream op;
	char pname[100];
	sprintf(pname,"BFS_EdgeOP_%s.txt",myresult.graphfile);
	op.open(pname);
	op.precision(0);
	for (int i = 0; i < graph->nnodes; i++)
		op << hdist[i] << endl;

	op.close();

	printf("edge output created!\n");


	//getchar();
	unsigned output,cnt=0;
	int i=0;
	ifstream ip;
	ip.open(pname,ios::in);
	if (ip.is_open()) {
 		while (!ip.eof()) {
			ip >> output;
			if(output!=hdist[i])
			{
				cout<<"node:"<<i<<"\n";
			//	getchar();
				cnt++;
			}	
		i++;
		}
	}
	
	cout<<"count:"<<cnt<<"\n";
	ip.close();
*/
/*	printf("verifying parallely");
	dverifysolution<<<(hnodes + KBLOCKSIZE - 1) / KBLOCKSIZE,KBLOCKSIZE>>> (dist, graph, nerr);
	CudaTest("dverifysolution failed");
*/
	cudaFree(outwl.items);
	cudaFree(inwl.items);
	cudaFree(inwl.start);
	cudaFree(inwl.end);
	cudaFree(inwl.capacity);
	cudaFree(outwl.start);
	cudaFree(outwl.end);
	cudaFree(outwl.capacity);
	
}
