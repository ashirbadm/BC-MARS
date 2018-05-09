/*
__global__
void initialize(unsigned *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}
*/
__global__
void kernel_ddV3(unsigned *dist,unsigned *nodesigma,unsigned *edgesigma,bool BM_COMP,unsigned *edgessrc,unsigned *edgesdst, unsigned *psrc,unsigned *noutgoing,Worklist inwl, Worklist outwl,unsigned *gerrno)//,unsigned *dmaxdeg)//Add round Robin
{
	//printf("hi\n");
	unsigned tid=blockDim.x*blockIdx.x+threadIdx.x; //Get thread ID
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
		if(olddist>alt){// check BFS Condition
			atomicMin(&dist[dst], alt);
			/*
			if(!BM_COMP){
//				atomicSub(&nodesigma[dst],edgesigma[work]);
				atomicExch(&edgesigma[work],nodesigma[src]);
				atomicExch(&nodesigma[dst],edgesigma[work]);
			}
			*/
			unsigned index=psrc[dst];
		//	unsigned nout=graph.getOutDegree(dst);
			unsigned nout=noutgoing[dst];
		//	for(unsigned i=0;i<nout;i++){
				if (outwl.pushRangeEdges(index,nout)) {	// buffer oveflow.
				// critical: reset the distance, since this node couldn't be added to the worklist.
					dist[dst] = olddist;
					*gerrno=1;
					//atomicMax(&dist[dst],alt);
					//inwl.noverflows++;
					break;
				}
		//	}
		}
		 if(dist[dst]==alt && !BM_COMP){
			 atomicSub(&nodesigma[dst],edgesigma[work]);
			 atomicExch(&edgesigma[work],nodesigma[src]);
			 atomicAdd(&nodesigma[dst],edgesigma[work]);
		}else{
			if(!BM_COMP){
			 atomicSub(&nodesigma[dst],edgesigma[work]);
			 edgesigma[work] = 0;
			}
																		                 	}
	tid += gridDim.x*blockDim.x;
	}
	//if(tid==0)
	//	printf("outwl in kernel after start:%d end:%d\n",*outwl.start,*outwl.end);
}
