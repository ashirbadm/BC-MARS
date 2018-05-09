#include <stdio.h>
#include <omp.h>
#include "worklist_cpu.h"

#define SWAP(a,b) { tmp = a; a = b; b = tmp; }
unsigned cpu_nerr;

void cpu_ddV3(unsigned *dist,unsigned *nodesigma,unsigned *edgesigma,bool BM_COMP,unsigned *edgessrc,unsigned *edgesdst, unsigned *psrc,unsigned *noutgoing,Worklist_cpu &inwl, Worklist_cpu &outwl,unsigned *gerrno,omp_lock_t * lock)//,unsigned *dmaxdeg)//Add round Robin
{
	//printf("hi\n");
	unsigned tid=omp_get_thread_num(); //Get thread ID
	unsigned start =*inwl.start;
	unsigned end=*inwl.end;
	//unsigned d=(*dmaxdeg);unsigned *p= (unsigned*) malloc(d* sizeof(unsigned));
	//unsigned *q;

	unsigned src,dst,wt,alt,olddist;
	//inwl.myItems(start, end);
//	if(tid==0)
//		printf("inwl in kernel before start:%d,end:%d\n",start,end);
	unsigned work;
	//for (unsigned ii = start; ii < end; ++ii) {
	while((start+tid)<end){
		//work = inwl.getItem(start+tid);
		work=inwl.items[start+tid];
		src=edgessrc[work];
		dst=edgesdst[work];
		wt=1;	
		//if(tid==0||tid<8)
		//	printf("getitem:%d\n",work);
		alt=dist[src]+wt;// Alternate distance 
		olddist=dist[dst];
		omp_set_lock(&(lock[dst]));
		if(dist[dst]>alt){// check BFS Condition
//#pragma omp atomic write
			dist[dst] = alt;
			
			omp_unset_lock(&(lock[dst]));
			/*
			if(!BM_COMP){
//#pragma omp atomic update
//			nodesigma[dst] -= edgesigma[work];
#pragma omp atomic read
			edgesigma[work] = nodesigma[src];
#pragma omp atomic write
			nodesigma[dst] = edgesigma[work];
			}
			*/
//	dist[dst] = dist[dst] ^ ((alt ^ dist[dst]) & -(alt < dist[dst]));
//	dist[dst] = alt;
//if(dist[dst] > alt)
			//atomicMin(&dist[dst], alt);
			unsigned index=psrc[dst];
			unsigned nout=noutgoing[dst];
				if (outwl.pushRangeEdges(index,nout)) {	// buffer oveflow.
				// critical: reset the distance, since this node couldn't be added to the worklist.
//#pragma omp atomic write
					dist[dst] = olddist;
					*gerrno=1;
					//atomicMax(&dist[dst],alt);
					//inwl.noverflows++;
					break;
				}
		}
		if(dist[dst]==alt && !BM_COMP){
			omp_unset_lock(&(lock[dst]));
#pragma omp atomic update
			nodesigma[dst] -= edgesigma[work];
#pragma omp atomic read
			edgesigma[work] = nodesigma[src];
#pragma omp atomic update
			nodesigma[dst] += edgesigma[work];
		}else{
			omp_unset_lock(&(lock[dst]));
			if(!BM_COMP){
#pragma omp atomic update
			nodesigma[dst] -= edgesigma[work];
			edgesigma[work] = 0;
			}
		}
		//tid += gridDim.x*blockDim.x;
	tid += omp_get_num_threads();
	}
	//if(tid==0)
	//	printf("outwl in kernel after start:%d end:%d\n",*outwl.start,*outwl.end);
}

extern "C" void worklist_cpu(unsigned *psrc,unsigned *noutgoing,unsigned *edgessrc,unsigned *edgesdst,unsigned hnodes,unsigned hedges,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,unsigned *sources,omp_lock_t *writelock,bool BM_COMP,int num_threads){

	unsigned itr = 0;
	Worklist_cpu inwl,outwl,*inwlptr, *outwlptr, *tmp;
	inwl.init();
	outwl.init();
	//printf("inwl capacity:%d\n",inwl.getCapacity());
	//printf("outwl capacity:%d\n",outwl.getCapacity());
	unsigned wlsz = 0;
	inwl.ensureSpace(13421760);
	outwl.ensureSpace(13421760);
	//inwl.ensureSpace(hedges);
	//outwl.ensureSpace(hedges);
	//printf("space ensured\n");
	unsigned hstart, hend;
//	inwl.getStartEnd(hstart, hend);
//	printf("inwl Start: %d End: %d\n",hstart,hend);
//	outwl.getStartEnd(hstart, hend);
//	printf("outwl Start: %d End: %d\n",hstart,hend);
	
	for(unsigned s=0;s < source_count ;s++)
	{
	unsigned nout=noutgoing[sources[s]];
	unsigned *k = (unsigned *) malloc (nout *sizeof(unsigned));
	//printf("hi3\n");
	for(unsigned i=0;i<nout;i++){
		k[i]=psrc[sources[s]]+i;
		//printf("eidst of source:%d\n",k[i]+1);
	}
	inwl.pushRangeHost(k,nout);
	free(k);
	}
	unsigned *nerr = (unsigned *) malloc (sizeof(unsigned));
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
	do
	{				
		itr++;
	//	printf("itr:%d\n",itr);
		*nerr = 0;
#pragma omp parallel default(none) shared(hdist,nodesigma,edgesigma,BM_COMP,edgessrc,edgesdst,psrc,noutgoing,inwlptr,outwlptr,nerr,writelock) num_threads(num_threads)
		{
		cpu_ddV3(hdist,nodesigma,edgesigma,BM_COMP,edgessrc,edgesdst,psrc,noutgoing,*inwlptr,*outwlptr,nerr,writelock);		
		}
		//printf("\nThread id %u src %u dst %u dist_src %u dist_dst %u work %u\n",tid,src+1,dst+1,dist[src],dist[dst],work);
		//memcpy(&cpu_nerr, nerr, sizeof(cpu_nerr));
		cpu_nerr = *nerr;
		wlsz = outwlptr->getSize();
	//	printf("WLSZ:%d\n",wlsz);
	//	outwlptr->getStartEnd(hstart, hend);
	//	printf("outwl in cpu after kernel Start: %d End: %d\n",hstart,hend);
		if (cpu_nerr == 0) {
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
	free(outwl.items);
	free(inwl.items);
	free(inwl.start);
	free(inwl.end);
	free(inwl.capacity);
	free(outwl.start);
	free(outwl.end);
	free(outwl.capacity);
	free(nerr);
}

