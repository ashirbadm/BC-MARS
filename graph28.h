#ifndef LSG_GRAPH
#define LSG_GRAPH

#include "metis.h"

#define DISTANCETHRESHOLD	150
#define THRESHOLDDEGREE		256

#include "Structs.h"
static unsigned CudaTest(char *msg);

//__host__
__device__ unsigned Graph::getOutDegree(unsigned src) {
	if (src < nnodes) {
		//printf("src(%d) < nnodes(%d), nnodesresident = %d.\n", src, nnodes, nnodesresident);
		return noutgoing[src - getStartNodeFromNode(src)];
		// return tex2D<unsigned>(texnoutgoing, src - getStartNodeFromNode(src), 0);
	}
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes); 
	return 0;
}

__device__ unsigned Graph::getDestination(unsigned src, unsigned nthedge) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes && nthedge < getOutDegree(src)) {
		unsigned edge = getFirstEdge(src) + nthedge;
		if (edge < nedges) {
			//return edgessrcdst[edge % nedgesresident];
			unsigned startnode = getStartNodeFromNode(src);	//(src / nnodesresident) * nnodesresident;
			unsigned startedge = getFirstEdge(startnode);
			//unsigned startedge = srcsrc[startnode % nnodesresident];
			//printf("src=%d, nthedge=%d, startnode=%d, startedge=%d, edge=%d.\n", src, nthedge, startnode, startedge, edge);
			//return edgessrcdst[edge - srcsrc[startnode % nnodesresident]];
			return edgessrcdst[edge - startedge];
		}
		//printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, edge, nedges);
		return nnodes;
	}
	if (src < nnodes) {
		//printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nthedge, getOutDegree(src));
	} else {
		//printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	}
	return nnodes;
}
__device__ foru Graph::getWeight(unsigned src, unsigned nthedge) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes && nthedge < getOutDegree(src)) {
		unsigned edge = getFirstEdge(src) + nthedge;
		if (edge < nedges) {
			//return edgessrcwt[edge % nedgesresident];
			unsigned startnode = getStartNodeFromNode(src);	//(src / nnodesresident) * nnodesresident;
			unsigned startedge = getFirstEdge(startnode);
			//return edgessrcwt[edge - srcsrc[startnode % nnodesresident]];
			return edgessrcwt[edge - startedge];
		}
		////printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, edge, nedges + 1);
		return MYINFINITY;
	}
	if (src < nnodes) {
		//printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nthedge, getOutDegree(src));
	} else {
		//printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	}
	return MYINFINITY;
}

__device__ unsigned Graph::getFirstEdge(unsigned src) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes) {
		unsigned srcnout = getOutDegree(src);
		/*if (srcnout > 0 && srcsrc[src % nnodesresident] < nedges) {
			return srcsrc[src % nnodesresident];
		}*/
		if (srcnout > 0 && srcsrc[src - getStartNodeFromNode(src)] < nedges) {
			return srcsrc[src - getStartNodeFromNode(src)];
		}
		//printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, 0, srcnout);
		return 0;
	}
	//printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	return 0;
}
__device__ unsigned Graph::getFirstEdgeRandom(unsigned src) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes) {
		unsigned low = 0, high = nedgesresident - 1, mid = 0;
		for (unsigned mytry = 0; mytry < 5; ++mytry) {
			mid = (low + high) / 2;
			if (srcsrc[mid] == src) break;
			if (srcsrc[mid] >  src) { high = mid; }
			else 			{ low  = mid; }
		}
		return it * nedgesresident + mid;
	}
	//printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	return 0;
}
unsigned Graph::getFirstEdgeBSearch(unsigned src) {
	if (src < nnodes) {
		unsigned low = 0, high = nedgesresident - 1, mid = 0;
		for (unsigned mytry = 0; mytry < nedgesresident; ++mytry) {
			mid = (low + high) / 2;
			if (srcsrc[mid] == src) break;
			if (srcsrc[mid] >  src) { high = mid; }
			else 			{ low  = mid; }
			if (mid == low) break;
		}
		if (srcsrc[mid] == src)
			return it * nedgesresident + mid;
	}
	//printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	return 0;
}
__device__ unsigned Graph::getMinEdge(unsigned src) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < nnodes) {
		unsigned srcnout = getOutDegree(src);
		if (srcnout > 0) {
			unsigned minedge = 0;
			foru     minwt   = getWeight(src, 0);
			for (unsigned ii = 1; ii < srcnout; ++ii) {
				foru wt = getWeight(src, ii);
				if (wt < minwt) {
					minedge = ii;
					minwt = wt;
				}
			}
			return minedge;
		}
		printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, 0, srcnout);
		return 0;
	}
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
	return 0;
}
__device__ __host__ void Graph::getEdge(unsigned edge, unsigned &src, unsigned &dst, foru &wt) {
	if (edge < nedges) {
		src = srcsrc[edge % nedgesresident];
		dst = edgessrcdst[edge % nedgesresident];
		wt  = edgessrcwt[edge % nedgesresident];
		return;
	}
	/*unsigned id = blockIdx.x * blockDim.x + threadIdx.x;*/
	printf("Error: %s(%d): edge %d out of bounds %d.\n", __FILE__, __LINE__, edge, nedges);
	return;
}
__device__ dorf Graph::getInsum(dorf *insum, unsigned nn) {
	if (nn < nnodes) {
		return insum[nn - getStartNodeFromNode(nn)];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, nn, nnodes);
	return 0.0;
}
__device__ dorf Graph::getPR(dorf *pr, unsigned nn) {
	if (nn < nnodes) {
		return pr[nn - getStartNodeFromNode(nn)];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, nn, nnodes);
	return 0.0;
}
__device__ void Graph::setPR(dorf *pr, unsigned nn, dorf rr) {
	if (nn < nnodes) {
		//printf("setting pr[%d] = %lf.\n", nn - getStartNodeFromNode(nn), rr);
		pr[nn - getStartNodeFromNode(nn)] = rr;
		return;
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, nn, nnodes);
}
__device__ foru Graph::getDistance(foru *dist, unsigned nn) {
	if (nn < nnodes) {
		return dist[nn - getStartNodeFromNode(nn)];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, nn, nnodes);
	return 0;
}
__device__ foru Graph::getDestinationDistance(foru *dstdist, unsigned src, unsigned nthedge) {
	unsigned edge = getFirstEdge(src) + nthedge;
	if (edge < nedges) {
		unsigned startnode = getStartNodeFromNode(src);	//(src / nnodesresident) * nnodesresident;
		unsigned residentedge = edge - getFirstEdge(startnode);
		//unsigned residentedge = edge - srcsrc[startnode % nnodesresident];
		if (residentedge < nedgesresident)
			return dstdist[residentedge];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, edge, nnodes);
	return 0;
}
__device__ foru Graph::getDestinationDistance(foru *dstdist, unsigned edge) {
	if (edge < nedges) {
		//unsigned startnode = (src / nnodesresident) * nnodesresident;
		unsigned residentedge = edge % nedgesresident;
		return dstdist[residentedge];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, edge, nedges);
	return 0;
}
__device__ void Graph::setDestinationDistance(foru *dstdist, unsigned ee, foru altdist) {
	if (ee < nedges) {
		unsigned residentedge = ee % nedgesresident;
		dstdist[residentedge] = altdist;
		return;
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, ee, nedges);
}
__device__ unsigned Graph::addSigma(unsigned ee, unsigned *dstsigma, unsigned srcsigma) {
	if (ee < nedges) {
		unsigned residentedge = ee % nedgesresident;
		return atomicAdd(&dstsigma[residentedge], srcsigma);
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, ee, nedges);
	return 0;
}
__device__ unsigned Graph::getSigma(unsigned *sigma, unsigned nn) {
	if (nn < nnodes) {
		return sigma[nn - getStartNodeFromNode(nn)];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, nn, nnodes);
	return 0;
}
__device__ dorf Graph::getDestinationDelta(dorf *dstdelta, unsigned edge) {
	if (edge < nedges) {
		unsigned residentedge = edge % nedgesresident;
		return dstdelta[residentedge];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, edge, nedges);
	return 0;
}
__device__ unsigned Graph::getDestinationSigma(unsigned *dstsigma, unsigned edge) {
	if (edge < nedges) {
		unsigned residentedge = edge % nedgesresident;
		return dstsigma[residentedge];
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, edge, nedges);
	return 0;
}
__device__ dorf Graph::addDelta(dorf *delta, unsigned nn, dorf val) {
	if (nn < nnodes) {
		return atomicAdd((float *)&delta[nn - getStartNodeFromNode(nn)], (float)val);
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, nn, nnodes);
	return 0;
}
__device__ dorf Graph::addBC(dorf *bc, unsigned nn, dorf val) {
	if (nn < nnodes) {
		return atomicAdd((float *)&bc[nn - getStartNodeFromNode(nn)], (float)val);
	}
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, nn, nnodes);
	return 0;
}
__device__ void Graph::printResidentNode(unsigned ii) {
		unsigned nout = 0;	//getOutDegree(ii);
		printf("%d(%d): ", ii, nout);
		for (unsigned ee = 0; ee < nout; ++ee) {
			unsigned dst = getDestination(ii, ee);
			foru wt = getWeight(ii, ee);
			printf("(%d %d)", dst, wt);
		}
		printf("\n");

}
__device__ void Graph::printResident(unsigned it, unsigned nprintnodes/* = 100*/) {
	unsigned nprinted = 0;
	unsigned startnode = getStartNode(it);
	unsigned nnodestobecopied = (startnode + nnodesresident < nnodes ? nnodesresident : nnodes - startnode);

	printf("%10d(%d), startnode=%d\n", nnodesresident, nnodestobecopied, startnode);
	for (unsigned xx = 0, ii = startnode; xx < nnodestobecopied; ++ii, ++xx) {
		printResidentNode(ii);
		if (++nprinted == nprintnodes) break;
	}
	//printResidentNode(0);
	//printResidentNode(1);
	//printResidentNode(168);
	//printResidentNode(175);
}

__host__ __device__ void Graph::print1x1() {
	// hack.
	unsigned savednnodesresident = nnodesresident;
	unsigned savednedgesresident = nedgesresident;
	nnodesresident = nnodes;
	nedgesresident = nedges;

	unsigned lastsrc = nnodes;
	for (unsigned ee = 0; ee < nedges; ++ee) {
		if (lastsrc != srcsrc[ee]) {
			printf("\n%d(%d): ", srcsrc[ee], noutgoing[srcsrc[ee]]);
			lastsrc = srcsrc[ee];
		}
		printf("(%d %d) ", edgessrcdst[ee], edgessrcwt[ee]);
	}
	
	// restore hack.
	nnodesresident = savednnodesresident;
	nedgesresident = savednedgesresident;
}

unsigned Graph::init(unsigned percgraph) {
	noutgoing = NULL;
	srcsrc = edgessrcdst = NULL;
	edgessrcwt = NULL;
	//source = 0;
	nnodes = nedges = 0;
	nnodesresident = nedgesresident = percgraph;	// this is a temporary storage, actual value will be set when graph size is known.
	it = 0;
	memory = NotAllocated;

	maxOutDegree = NULL;
	diameter = 0;
	foundStats = false;

	return 0;
}

int Graph::formMetisPartitions(Graph &from, Partition* partition){
idx_t options[METIS_NOPTIONS];
idx_t ncon;
idx_t *xadj, *adjncy;
int *degree, *nodeoffset;
int ii, i;
unsigned int edgeindex, src, dst, srcindex, dstindex;
	
  ncon = 1;
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;
//  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
//  options[METIS_OPTION_NUMBERING] = 0;
//  options[METIS_OPTION_NITER] = 20;
//  options[METIS_OPTION_CCORDER] = 1;
  
  // Form xadj and adjncy needed for metis
  // Converting directed graph to CSR undirected format
  //
  xadj = (idx_t*)malloc(sizeof(idx_t)*(nnodes+1));
  degree = (int*)malloc(sizeof(int)*(nnodes+1));
  for(ii=0; ii<nnodes; ii++) degree[ii] = 0;
  for(edgeindex=0; edgeindex<nedges; edgeindex++){
    src = from.srcsrc[edgeindex];
    dst = from.edgessrcdst[edgeindex];
    degree[src]++;
    degree[dst]++;
  }
  for(ii=0; ii<nnodes+1; ii++){
    if(ii==0) xadj[ii] = 0;
    else xadj[ii] = degree[ii-1]+xadj[ii-1];
  }
 

  adjncy = (idx_t*)malloc(sizeof(idx_t)*(2*nedges));
  nodeoffset = (int*)malloc(sizeof(int)*nnodes);
  for(ii=0; ii<nnodes; ii++) nodeoffset[ii] = 0;
  for(edgeindex=0; edgeindex<nedges; edgeindex++){
    src = from.srcsrc[edgeindex];
    dst = from.edgessrcdst[edgeindex];
    srcindex = xadj[src]+nodeoffset[src];
    nodeoffset[src]++;
    adjncy[srcindex] = dst;
    dstindex = xadj[dst]+nodeoffset[dst];
    nodeoffset[dst]++;
    adjncy[dstindex] = src;
  }
/*
  adjncy = (idx_t*)malloc(sizeof(idx_t)*(nedges));
  nodeoffset = (int*)malloc(sizeof(int)*nnodes);
  for(ii=0; ii<nnodes; ii++) nodeoffset[ii] = 0;
  for(edgeindex=0; edgeindex<nedges; edgeindex++){
    src = from.srcsrc[edgeindex];
    dst = from.edgessrcdst[edgeindex];
    srcindex = xadj[src]+nodeoffset[src];
    nodeoffset[src]++;
    adjncy[srcindex] = dst;
  }
*/  
  // printf("nodes: %d\n edges:%d\n ncon:%d\n xadj[0]: %d xadj[nnodes]: %d\n adjncy[0]: %d adjncy[nedges]: %d\n nparts: %d\n",
  //              nnodes, nedges, ncon, xadj[0], xadj[nnodes],adjncy[0], adjncy[hgraph.nedges], partition.nparts); 
        METIS_PartGraphKway((idx_t*)(&nnodes),
                            &ncon, xadj, adjncy, NULL, NULL, NULL,
                            &(partition->nparts), NULL, NULL, options,
                            (idx_t*)(&(partition->edgecut)), (idx_t*)partition->part);
//   printf("Iresh : edgecut: %d\n", partition->edgecut);
  
  /*
  for(ii=0; ii<nnodes; ii++){
    if(partition->part[ii] != 0 && partition->part[ii] != 1){
      printf("Wrong partition: %d\n", partition->part[ii]);
      return 1;
    }
  }
  printf("Metis has given correct partitions\n");  
  */
  
	free(xadj);
	free(degree);
	free(adjncy);
	free(nodeoffset);
  fillBorderAndCount(from,partition);
  return 0;
}

// form device paritions of the graph
// A device parition is the graph partition that will be given to GPU
// Equal to the non-source partition coming from metis + the border nodes of the source partition
// For the current 2 partition method, can afford to form both such partitions well in advance
void Graph::formDevicePartitions(Graph &from){
unsigned edgeindex;
int partitionIndex;
unsigned partitionEdgeIndex;
int * tmpedgeindex, *temp;
unsigned int src, dst;
unsigned wt;
int srcpartition, dstpartition;

  devicePartition = (DevicePartition *)malloc(sizeof(DevicePartition) * partition.nparts);
  tmpedgeindex = (int *) malloc(sizeof(int) * partition.nparts);
  
  maxEdgeCount = 0;
  maxNodeCount = 0;
  for(partitionIndex=0 ; partitionIndex < partition.nparts ; partitionIndex++){
    tmpedgeindex[partitionIndex] = 0;
//   unsigned alpha_factor = (partition.borderCount[1-partitionIndex]/partition.nodeCount[1-partitionIndex]+1) * partition.edgeCount[1-partitionIndex];
    devicePartition[partitionIndex].numEdges = partition.edgeCount[partitionIndex] + (partition.edgecut);
    devicePartition[partitionIndex].numNodes = partition.nodeCount[partitionIndex] + partition.borderCount[1-partitionIndex]; // works for 2 partitions
	printf("numNodes[%d] = %d\n",partitionIndex,devicePartition[partitionIndex].numNodes);
	printf("numEdges[%d] = %d, edgeCount = %d, edgeCut = %d\n",partitionIndex,devicePartition[partitionIndex].numEdges,partition.edgeCount[partitionIndex],partition.edgecut);
//devicePartition[partitionIndex].numEdges *=2;
    if(devicePartition[partitionIndex].numEdges > maxEdgeCount){
      maxEdgeCount = devicePartition[partitionIndex].numEdges;
    }
    if(devicePartition[partitionIndex].numNodes > maxNodeCount){
      maxNodeCount = devicePartition[partitionIndex].numNodes;
    }
	
//	printf("line 446\n");cudaMalloc((void **)&temp, (100) * sizeof(unsigned));
    devicePartition[partitionIndex].edgesrc = (unsigned*)malloc((devicePartition[partitionIndex].numEdges) * sizeof(unsigned));
    devicePartition[partitionIndex].edgedst = (unsigned*)malloc((devicePartition[partitionIndex].numEdges) * sizeof(unsigned)); 
    devicePartition[partitionIndex].edgewt = (unsigned*)malloc((devicePartition[partitionIndex].numEdges) * sizeof(unsigned));
    devicePartition[partitionIndex].nodedist = (unsigned int *)malloc((from.nnodes) * sizeof(unsigned));
    devicePartition[partitionIndex].nodesigma = (unsigned int *)malloc((from.nnodes) * sizeof(unsigned));
    devicePartition[partitionIndex].nodedelta = (float *)malloc((from.nnodes) * sizeof(float));
    devicePartition[partitionIndex].psrc = (unsigned int *)calloc((from.nnodes+1), sizeof(unsigned));
    devicePartition[partitionIndex].psrc[nnodes] = devicePartition[partitionIndex].numEdges;
    devicePartition[partitionIndex].noutgoing = (unsigned int *)calloc((from.nnodes+1), sizeof(unsigned));
   // devicePartition[partitionIndex].edgedist = (unsigned int *)malloc((from.nnodes) * sizeof(unsigned));
    devicePartition[partitionIndex].edgesigma = (unsigned int *)malloc((devicePartition[partitionIndex].numEdges) * sizeof(unsigned));
    devicePartition[partitionIndex].child = (int *)malloc((devicePartition[partitionIndex].numEdges) * sizeof(int));
    devicePartition[partitionIndex].child_count = (unsigned int *)malloc((from.nnodes+1) * sizeof(unsigned));


    //devicePartition[partitionIndex].active = (bool*)malloc((from.nnodes) * sizeof(bool)); 
    //devicePartition[partitionIndex].edgeactive = (bool*)malloc((devicePartition[partitionIndex].numEdges) * sizeof(bool)); 
  }

//	printf("line 462\n");cudaMalloc((void **)&temp, (100) * sizeof(unsigned));
  /* For debugging output
  for(int i=0 ;i < from.nnodes;i++)
  {
	  printf("node: %d belongs to partition %d\t",i,partition.part[i]);
	  if(partition.border[i]!=0)
		  printf("node %d is border node",i);
	  printf("\n");
  }*/
//tmpedgeindex[0] = 0;
//tmpedgeindex[1] = 0;
  for(edgeindex = 0 ; edgeindex < nedges ; edgeindex++){
     src = from.srcsrc[edgeindex];
     dst = from.edgessrcdst[edgeindex];
     wt = from.edgessrcwt[edgeindex];
//	printf("[E%d]%d->%d\n",edgeindex,src,dst);
     srcpartition = partition.part[src];
     dstpartition = partition.part[dst];
     //printf("\nEdge Index %u, src %u, dst %u, srcpartitiob %d, dstpartition %d, tmpedgeindexsrc %u, tmpedgeindecdst %u\n",edgeindex,src,dst,srcpartition,dstpartition,tmpedgeindex[srcpartition],tmpedgeindex[dstpartition]);
     if(srcpartition == dstpartition){	/* Both vertice of the edge belong to the same partition */
       partitionIndex = srcpartition;
  //     printf("[%d]Same %d\n",partitionIndex,tmpedgeindex[partitionIndex]);
	partitionEdgeIndex = tmpedgeindex[partitionIndex];
       devicePartition[partitionIndex].edgesrc[partitionEdgeIndex] = src;
       devicePartition[partitionIndex].edgedst[partitionEdgeIndex] = dst;
       devicePartition[partitionIndex].edgewt[partitionEdgeIndex] = wt;
       devicePartition[partitionIndex].edgesigma[partitionEdgeIndex] = 0;
//       devicePartition[partitionIndex].active[partitionEdgeIndex] = false;
	// for psrc and noutgoing for all each partition
       if(devicePartition[partitionIndex].noutgoing[src]==0)devicePartition[partitionIndex].psrc[src]=partitionEdgeIndex;
       devicePartition[partitionIndex].noutgoing[src]++;
       tmpedgeindex[partitionIndex]++;
     }
     else{ /* Both vertices of the edge belong to difference partition */
       // Add the border nodes of one partition to the other. i.e., the boundary edges belong to both the partitions.
	partitionIndex = srcpartition;
    //   printf("[%d]Different %d\n",partitionIndex,tmpedgeindex[partitionIndex]);
	partitionEdgeIndex = tmpedgeindex[partitionIndex];
       devicePartition[partitionIndex].edgesrc[partitionEdgeIndex] = src;
       devicePartition[partitionIndex].edgedst[partitionEdgeIndex] = dst;
       devicePartition[partitionIndex].edgewt[partitionEdgeIndex] = wt;
//       devicePartition[partitionIndex].active[partitionEdgeIndex] = false;
       if(devicePartition[partitionIndex].noutgoing[src]==0)devicePartition[partitionIndex].psrc[src]=partitionEdgeIndex;
       devicePartition[partitionIndex].noutgoing[src]++;
       tmpedgeindex[partitionIndex]++;

       partitionIndex = dstpartition;
       partitionEdgeIndex = tmpedgeindex[partitionIndex];
       devicePartition[partitionIndex].edgesrc[partitionEdgeIndex] = src;
       devicePartition[partitionIndex].edgedst[partitionEdgeIndex] = dst;
       devicePartition[partitionIndex].edgewt[partitionEdgeIndex] = wt;
    //   devicePartition[partitionIndex].active[partitionEdgeIndex] = false;
	// for psrc and noutgoing for all each partition
       if(devicePartition[partitionIndex].noutgoing[src]==0)devicePartition[partitionIndex].psrc[src]=partitionEdgeIndex;
       devicePartition[partitionIndex].noutgoing[src]++;
       tmpedgeindex[partitionIndex]++;
     }
   }

   for(partitionIndex=0 ; partitionIndex < partition.nparts ; partitionIndex++){
	devicePartition[partitionIndex].noutgoing[from.nnodes] =  devicePartition[partitionIndex].noutgoing[from.nnodes-1];
   }



//printf("%d:%d %d:%d\n",partition.part[src],devicePartition[partition.part[src]].numEdges,partition.part[dst],devicePartition[partition.part[dst]].numEdges);

ofstream cpupart;
cpupart.open("cpupart.edges");
partitionIndex = CPUPARTITION;
printf("Make CPU Part graph %d\n",devicePartition[partitionIndex].numEdges);
for(edgeindex = 0 ; edgeindex < devicePartition[partitionIndex].numEdges ; edgeindex++){

	cpupart<<devicePartition[partitionIndex].edgesrc[edgeindex]<<" "<<devicePartition[partitionIndex].edgedst[edgeindex]<<endl;


}
cpupart.close();
char name[300]="",str[50]="";
        strcat(name,"/home/iresh/parallel_graph_alg_madduri_modified/graph_gen /home/iresh/parallel_graph_alg_madduri_modified/input/cpupart.grspec ");
	sprintf(str, "%d", devicePartition[partitionIndex].numEdges); 
       strcat(name,str);
  
      system(name);

/*

ofstream gpupart;
gpupart.open("gpupart.edges");
partitionIndex = GPUPARTITION;
printf("Make GPU Part graph %d\n",devicePartition[partitionIndex].numEdges);
for(edgeindex = 0 ; edgeindex < devicePartition[partitionIndex].numEdges ; edgeindex++){

        gpupart<<devicePartition[partitionIndex].edgesrc[edgeindex]<<" "<<devicePartition[partitionIndex].edgedst[edgeindex]<<endl;


}
gpupart.close();
char name2[300]="",str2[50]="";
        strcat(name2,"/home/iresh/parallel_graph_alg_madduri_modified/graph_gen /home/iresh/parallel_graph_alg_madduri_modified/input/gpupart.grspec ");
        sprintf(str2, "%d", devicePartition[partitionIndex].numEdges);
       strcat(name2,str2);

      system(name2);
*/

/*Print : 
for(partitionIndex=0 ; partitionIndex < partition.nparts ; partitionIndex++){
printf("\nPartition : %d\n",partitionIndex);
for(edgeindex = 0 ; edgeindex < devicePartition[partitionIndex].numEdges ; edgeindex++){
printf("%d->%d, ",devicePartition[partitionIndex].edgesrc[edgeindex],devicePartition[partitionIndex].edgedst[edgeindex]);
}
printf("\nnoutgoing\n");
for(edgeindex = 0 ; edgeindex <=from.nnodes ; edgeindex++){
printf("%d, ",devicePartition[partitionIndex].noutgoing[edgeindex]);

}
printf("\npsrc\n");
for(edgeindex = 0 ; edgeindex <=from.nnodes ; edgeindex++){
printf("%d, ",devicePartition[partitionIndex].psrc[edgeindex]);

}
}
printf("\n");
*/
   free(tmpedgeindex);
}
void Graph::removeDevicePartitions(Graph &from){
	for(int partitionIndex=0 ; partitionIndex < partition.nparts ; partitionIndex++){
		free(from.devicePartition[partitionIndex].edgesrc);
		free(from.devicePartition[partitionIndex].edgedst);
		free(from.devicePartition[partitionIndex].nodedist);
		free(from.devicePartition[partitionIndex].nodesigma);
		free(from.devicePartition[partitionIndex].nodedelta);
		//free(devicePartition[partitionIndex].edgedist);
		free(from.devicePartition[partitionIndex].edgesigma);
		free(from.devicePartition[partitionIndex].edgewt);
		free(from.devicePartition[partitionIndex].noutgoing);
		free(from.devicePartition[partitionIndex].psrc);
		//free(devicePartition[partitionIndex].active);
	}
	free(from.devicePartition);
}
        
int Graph::fillBorderAndCount(Graph &from, Partition *  partition){
        unsigned int edgeindex, borderindex;
        int partitionNumber;
        unsigned int src, dst, ii;
        int i;
	
        for(i=0 ; i < partition->nparts; i++){
          partition->borderCount[i] = 0;
          partition->nodeCount[i] = 0;
          partition->edgeCount[i] = 0;
        }
       
       // initialize border nodes
       // border[i] is 0, the node is not a border node, else contains the number of edges incident from this node to another partition       
        for(ii=0; ii<nnodes; ii++)
          partition->border[ii] = 0;
	printf("nedges : %d\n",nedges);  partition->edgecut = 0;
        for(edgeindex=0; edgeindex<nedges; edgeindex++){
          src = from.srcsrc[edgeindex];
          dst = from.edgessrcdst[edgeindex];
          if(partition->part[src] != partition->part[dst]){
	//	printf("%d:%d->%d\n",edgeindex,src,dst); 
		 partition->edgecut++;
            partition->border[src]++;
            partition->border[dst]++;
          }
          else{
            partitionNumber = partition->part[src];
            partition->edgeCount[partitionNumber]++;
          }
        }

	int prev = 0; 
        for(i=0; i<partition->nparts; i++)
          partition->borderCount[i] = 0;

        for(ii=0; ii<nnodes; ii++){
          partitionNumber = partition->part[ii];
          if(partition->border[ii] != 0){
//		prev++;
            borderindex = partition->borderCount[partitionNumber];
            partition->borderNodes[partitionNumber][borderindex] = ii;
            partition->borderCount[partitionNumber]++;
//		partition->border[ii] = prev;
          }
          else{
            partition->nodeCount[partitionNumber]++; // counting only non border nodes
          }
        }
	
	for(partitionNumber=0; partitionNumber<partition->nparts; partitionNumber++)
	{
		for(borderindex=0; borderindex<partition->borderCount[partitionNumber]; borderindex++)
		{
			prev++;
			partition->border[partition->borderNodes[partitionNumber][borderindex]] = prev;
		}
	}
	partition->border[nnodes] = partition->borderCount[0];
	partition->border[nnodes+1] = partition->borderCount[1];


        for(i=0; i<partition->nparts; i++){
          partitionNumber = i;
          partition->nodeCount[partitionNumber] += partition->borderCount[partitionNumber];
        }
    
        printf("Partition Counts:\n");
        for(i=0; i<partition->nparts; i++){
          printf("[%d]Total nodes: %d Border nodes: %d Edges: %d\n", i, partition->nodeCount[i], partition->borderCount[i], partition->edgeCount[i]);
        }

        printf("Eedge cut: %u\n", partition->edgecut);
/*	
 	printf("Border Nodes:\n");
        for(i=0; i<partition->nparts; i++){
		printf("%d : ",i);
		for(ii=0;ii<partition->borderCount[i];ii++)
		{
			printf(",%d", partition->borderNodes[i][ii]);
		}
		printf("\n");
        } 
	printf("All Nodes : \n");
	for(ii=0; ii<nnodes; ii++)
		printf("%d:%d,",ii,partition->part[ii]); 
 printf("\nBorder Nodes : \n");
        for(ii=0; ii<nnodes; ii++)
                printf("%d:%d,",ii,partition->border[ii]); 
        printf("\n");
*/
	return 0;
}
  
// VSS: Modified to form partitions
int Graph::initFrom(Graph &from) {
  	   
	nnodes = from.nnodes;
	nedges = from.nedges;
	
        partition.part = (int *)malloc(sizeof(int)*(from.nnodes));
	partition.nparts = 2;	
	//formMetisPartitions(from, &partition);
        partition.border = (unsigned int*)malloc(sizeof(unsigned int)*(nnodes+2));
	partition.borderCount = (unsigned int*)malloc(sizeof(unsigned int)* (partition.nparts) );
	partition.borderNodes = (unsigned int**)malloc(sizeof(unsigned int*)* (partition.nparts) );
	partition.nodeCount = (unsigned int*)malloc(sizeof(unsigned int)* (partition.nparts) );
	partition.edgeCount = (unsigned int*)malloc(sizeof(unsigned int)* (partition.nparts) );	
        for(int i=0; i<partition.nparts; i++)
          partition.borderNodes[i] = (unsigned int*)malloc(sizeof(unsigned int)*nnodes/2);
	
	return 0;
	
}

unsigned Graph::allocOnHost() {
	edgessrcdst = (unsigned int *)malloc((nedges) * sizeof(unsigned int));		// first entry acts as null.
	edgessrcwt = (foru *)malloc((nedges) * sizeof(foru));				// first entry acts as null.
	noutgoing = (unsigned int *)calloc(nnodes, sizeof(unsigned int));		// init to 0.
	psrc = (unsigned int *)calloc((nnodes+1), sizeof(unsigned int));		// init to 0.
	psrc[nnodes] = nedges;
	srcsrc = (unsigned int *)malloc((nedges+1) * sizeof(unsigned int));

	maxOutDegree = (unsigned *)malloc(sizeof(unsigned));
	*maxOutDegree = 0;

	memory = AllocatedOnHost;
	return 0;
}
unsigned Graph::allocOnDevice() {
	if (cudaMalloc((void **)&edgessrcdst, (nedges) * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating edgessrcdst failed");
	if (cudaMalloc((void **)&edgessrcwt, (nedgesresident) * sizeof(foru)) != cudaSuccess) 
		CudaTest("allocating edgessrcwt failed");
	if (cudaMalloc((void **)&noutgoing, nnodesresident * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating noutgoing failed");
	if (cudaMalloc((void **)&srcsrc, (nedgesresident) * sizeof(unsigned int)) != cudaSuccess) 
		CudaTest("allocating srcsrc failed");


	return 0;
}
unsigned Graph::deallocOnHost() {
	free(noutgoing);
	free(srcsrc);
	free(edgessrcdst);
	free(edgessrcwt);
	free(psrc);
	free(maxOutDegree);
	return 0;
}
unsigned Graph::deallocOnDevice() {
	cudaFree(noutgoing);
	cudaFree(srcsrc);
	cudaFree(edgessrcdst);
	cudaFree(edgessrcwt);

	cudaFree(maxOutDegree);
	return 0;
}
unsigned Graph::dealloc() {
	switch (memory) {
		case AllocatedOnHost:
			printf("dealloc on host.\n");
			deallocOnHost();
			break;
		case AllocatedOnDevice:
			printf("dealloc on device.\n");
			deallocOnDevice();
			break;
	}
	return 0;
}
unsigned Graph::dealloc_part() {
	free(partition.border);
	free(partition.borderCount);
	free(partition.nodeCount);
	free(partition.edgeCount);
	for(int i=0; i<partition.nparts; i++){
		free(partition.borderNodes[i]);
	//	for (unsigned ii = 0 ; ii < nnodes ; ii++)
	//		omp_destroy_lock(&devicePartition[i].nodelock[ii]);
	}
	free(partition.borderNodes);
	for(int partitionIndex=0 ; partitionIndex < partition.nparts ; partitionIndex++){
		free(devicePartition[partitionIndex].edgesrc);
		free(devicePartition[partitionIndex].edgedst);
		free(devicePartition[partitionIndex].nodedist);
		free(devicePartition[partitionIndex].nodesigma);
		free(devicePartition[partitionIndex].nodedelta);
		//free(devicePartition[partitionIndex].edgedist);
		free(devicePartition[partitionIndex].edgesigma);
		free(devicePartition[partitionIndex].edgewt);
		free(devicePartition[partitionIndex].noutgoing);
		free(devicePartition[partitionIndex].psrc);
		//free(devicePartition[partitionIndex].active);
		//free(devicePartition[partitionIndex].nodelock);
	}
	free(devicePartition);
	free(partition.part);
}

Graph::Graph(unsigned percgraph/* = 100*/) {
	init(percgraph);
}
Graph::~Graph() {
	//// The destructor seems to be getting called at unexpected times.
	dealloc_part();
	//init();
}

void Graph::progressPrint(unsigned maxii, unsigned ii) {
	const unsigned nsteps = 5;
	unsigned ineachstep = (int)(maxii / nsteps);
	//printf("ineachstep = %d,ii = %d\n",ineachstep,ii);
	/*if (ii == maxii) {
		printf("\t100%%\n");
	} else*/ if (ii % ineachstep == 0) {
		printf("\t%3d%%\r", ii*100/maxii + 1);
		fflush(stdout);
	}
}
unsigned Graph::readFromEdges(char file[]) {
        std::ifstream cfile;
        cfile.open(file);
        std::string str;
//      getline(cfile, str);
//      //      sscanf(str.c_str(), "%d %d", &nnodes, &nedges);
        char str_temp[30];
	unsigned base,pins;
        getline(cfile, str);
        sscanf(str.c_str(), "%d %u %u %u", &base,&nnodes,&nedges,&pins);
	//nedges *=2;
        printf("\nNum of nodes %u\n",nnodes);
        //getline(cfile, str);
        //sscanf(str.c_str(), "%s %u", str_temp, &nedges);
        printf("Num of edges %u\n",nedges);
       // getline(cfile, str);
        allocOnHost();
        for (unsigned ii = 0; ii < nnodes; ++ii) {
                srcsrc[ii] = ii;
        }
        unsigned int prevnode = 0;
        unsigned int tempsrcnode;
        unsigned int ncurroutgoing = 0;
        unsigned maxdegree = 0;
        for (unsigned ii = 0; ii < nedges; ++ii) {
                getline(cfile, str);
	        sscanf(str.c_str(), "%d %d %d", &srcsrc[ii], &edgessrcdst[ii], &edgessrcwt[ii]);
         //     sscanf(str.c_str(), "%d %d", &srcsrc[ii], &edgessrcdst[ii]);
         //     edgessrcwt[ii] = 1;
	 //     printf("%d %d %d\n",srcsrc[ii], edgessrcdst[ii], edgessrcwt[ii]);
	 
                tempsrcnode = srcsrc[ii];//For COO Representation
                if (prevnode == tempsrcnode) {
			  if (ii == 0) {
			            psrc[tempsrcnode] = ii + 1;
			    }
                        ++ncurroutgoing;
                } else {
			psrc[tempsrcnode] = ii + 1;
                        if (ncurroutgoing) {
                                noutgoing[prevnode] = ncurroutgoing;
                                maxdegree = (noutgoing[prevnode] > maxdegree ? noutgoing[prevnode] : maxdegree);
                        }
                        prevnode = tempsrcnode;
                        ncurroutgoing = 1;      // not 0.
                }
                   //   ++nincoming[edgessrcdst[ii+1]];
                progressPrint(nedges, ii);
	}
        noutgoing[prevnode] = ncurroutgoing;    // last entries.
        maxdegree = (noutgoing[prevnode] > maxdegree ? noutgoing[prevnode] : maxdegree);
        *maxOutDegree = maxdegree;
	cout<<"Max Degree is "<<maxdegree<<endl;
        cfile.close();
        return maxdegree;
}


unsigned Graph::readFromGR(char file[], bool weighted) {
	std::ifstream cfile;
	cfile.open(file);

	// copied from GaloisCpp/trunk/src/FileGraph.h
	int masterFD = open(file, O_RDONLY);
  	if (masterFD == -1) {
	printf("FileGraph::structureFromFile: unable to open %s.\n", file);
	return 1;
  	}

  	struct stat buf;
	int f = fstat(masterFD, &buf);
  	if (f == -1) {
    		printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
    		abort();
  	}
  	size_t masterLength = buf.st_size;

  	int _MAP_BASE = MAP_PRIVATE;
//#ifdef MAP_POPULATE
//  _MAP_BASE  |= MAP_POPULATE;
//#endif

  	void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  	if (m == MAP_FAILED) {
    		m = 0;
    		printf("FileGraph::structureFromFile: mmap failed.\n");
    		abort();
  	}

  	//parse file
  	uint64_t* fptr = (uint64_t*)m;
  	__attribute__((unused)) uint64_t version = le64toh(*fptr++);
  	assert(version == 1);
  	uint64_t sizeEdgeTy = le64toh(*fptr++);
  	uint64_t numNodes = le64toh(*fptr++);
  	uint64_t numEdges = le64toh(*fptr++);
  	uint64_t *outIdx = fptr;
  	fptr += numNodes;
  	uint32_t *fptr32 = (uint32_t*)fptr;
  	uint32_t *outs = fptr32; 
  	fptr32 += numEdges;
  	if (numEdges % 2) fptr32 += 1;
  	unsigned  *edgeData = (unsigned *)fptr32;

	// cuda.
	nnodes = numNodes;
	nedges = numEdges;

	printf("nnodes=%d, nedges=%d.\n", nnodes, nedges);
	allocOnHost();

	unsigned maxdegree = 0, maxdegreenode = 0;
	unsigned src, dst;
	for (unsigned ii = 0, edgeindex = 0; ii < nnodes; ++ii) {
		src = ii;
		// fill unsigned *noutgoing, *srcsrc, *edgessrcdst; foru *edgessrcwt;
		if (ii > 0) {
			psrc[ii] = le64toh(outIdx[ii - 1])+ 1;
			noutgoing[ii] = le64toh(outIdx[ii]) - le64toh(outIdx[ii - 1]);
			if (noutgoing[ii] > maxdegree) {
				maxdegree = noutgoing[ii];
				maxdegreenode = ii;
			}
		} else {
			psrc[0] = 1;
			noutgoing[0] = le64toh(outIdx[0]);
			maxdegree = (noutgoing[ii] > maxdegree ? noutgoing[ii] : maxdegree);
		}
		//printf("%d(%d)\n", ii, noutgoing[ii]);
		for (unsigned jj = 0; jj < noutgoing[ii]; ++jj, ++edgeindex) {
			dst = le32toh(outs[edgeindex]);
			//printf("nout[%d] = %d, edgeindex=%d, dst=%d.\n", ii, noutgoing[ii], edgeindex, dst);
			if (dst >= nnodes) printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);
			srcsrc[edgeindex] = src;
			edgessrcdst[edgeindex] = dst;
			edgessrcwt[edgeindex] = (weighted ? edgeData[edgeindex] : 1);
		}
		//printf("done\n");
		progressPrint(nedges, edgeindex);
	}
	printf("\n");
	*maxOutDegree = maxdegree;
	printf("%s(%d): maxdegree(%d) %s THRESHOLDDEGREE(%d), maxdegreenode = %d.\n", __FILE__, __LINE__, maxdegree, (maxdegree > THRESHOLDDEGREE ? ">" : "<="), THRESHOLDDEGREE, maxdegreenode);
	cfile.close();	// probably galois doesn't close its file due to mmap.

	//printEdges(0, 11);
	//printDegrees(0, 11);
	
  
	return 0;
}
unsigned Graph::read(char file[], bool weighted/* = true*/) {
	if (strstr(file, ".edges")) {
		return readFromEdges(file);
	} else if (strstr(file, ".gr")) {
		return readFromGR(file, weighted);
	}
	return 0;
}
void Graph::printOneEdge(unsigned ee) {
	printf("%d: %d -> %d\twt=%d.\n", ee, srcsrc[ee], edgessrcdst[ee], edgessrcwt[ee]);
}
void Graph::printEdges(unsigned startee, unsigned nee) {
	for (unsigned ee = 0; ee < nee; ++ee) {
		printOneEdge(startee + ee);
	}
}
void Graph::printEdgesOf(unsigned nn) {
	unsigned startedge = getFirstEdgeBSearch(nn);
	printf("Edges of %d = %d.\n", nn, noutgoing[nn]);
	printEdges(startedge, noutgoing[nn]);
}
void Graph::printDegrees(unsigned startnn, unsigned nnn) {
	for (unsigned node = 0; node < nnn; ++node) {
		printf("noutgoing[%d] = %d.\n", startnn + node, noutgoing[startnn + node]);
	}
}

__device__ void Graph::computeStats() {
	computeInOut();
	computeDiameter();
}
__device__ bool Graph::computeLevels() {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	bool changed = false;

	if (id < nnodes) {
		/*unsigned iilevel = levels[id];
		unsigned noutii = getOutDegree(id);

		//printf("level[%d] = %d.\n", id, iilevel);
		for (unsigned jj = 0; jj < noutii; ++jj) {
			unsigned dst = getDestination(id, jj);

			if (dst < nnodes && levels[dst] > iilevel + 1) {
				levels[dst] = iilevel + 1;
				changed = true;
			} else if (dst >= nnodes) {
				printf("\t%s(%d): dst %d >= nnodes %d.\n", __FILE__, __LINE__, dst, nnodes);
			}
		}*/
	}
	return changed;
}

#define MAX(a, b)	(a < nnodes && a > b ? a : b)

__device__ unsigned Graph::findMaxLevel() {
	unsigned maxlevel = 0;
	/*for (unsigned ii = 0; ii < nnodes; ++ii) {
		maxlevel = MAX(levels[ii], maxlevel);
	}*/
	return maxlevel;
}
__device__  void Graph::computeDiameter() {
	diameter = findMaxLevel();
}
__device__ void Graph::computeInOut() {
	for (unsigned ii = 0; ii < nnodes; ++ii) {
		// process outdegree.
		unsigned noutii = getOutDegree(ii);
		if (noutii > *maxOutDegree) {
			*maxOutDegree = noutii;
		}
	}
}

__device__ void Graph::printStats1x1() {	// 1x1.
	char prefix[] = "\t";

	computeStats();

	printf("%snnodes             = %d.\n",   prefix, nnodes);
	printf("%snedges             = %d.\n",   prefix, nedges);
	printf("%savg, max outdegree = %.2f, %d.\n", prefix, nedges*1.0 / nnodes, *maxOutDegree);
	printf("%sdiameter           = %d.\n",   prefix, diameter);
	return;
}
void Graph::allocLevels() {
	//if (cudaMalloc((void **)&levels, nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating levels failed");
}
void Graph::freeLevels() {
	//printf("freeing levels.\n");
	//cudaFree(levels);
}
__device__ void Graph::initLevels() {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	//if (id < nnodes) levels[id] = nnodes;
}
/*
 * Global functions.
 */
__global__ void dprintstats(Graph graph) {
	graph.printStats1x1();
}
__global__ void dcomputelevels(Graph graph, bool *changed) {
	if (graph.computeLevels()) {
		*changed = true;
	}
}
__global__ void dinitlevels(Graph graph) {
	graph.initLevels();
}
__global__ void dprint1x1(Graph graph) {
	graph.print1x1();
}
__global__ void dprintResident(Graph graph, unsigned it, unsigned nprintnodes) {
	graph.printResident(it, nprintnodes);
}
void Graph::printResidentDevice(unsigned it, unsigned nprintnodes/* = 100*/) {
	dprintResident<<<1,1>>>(*this, it, nprintnodes);
	CudaTest("dprintResident failed");
}
void Graph::print() {
	if (memory == AllocatedOnDevice) {
		dprint1x1<<<1,1>>>(*this);
		CudaTest("print1x1 failed");
	} else if (memory == AllocatedOnHost) {
		printf("memory allocated on host. hgraph.print not implemented.\n");
	} else {
		printf("\t%s(%d): graph not yet allocated.\n", __FILE__, __LINE__);
	}
}
unsigned Graph::printStats() {
	allocLevels();
	dinitlevels<<<(nnodes+MAXBLOCKSIZE-1)/MAXBLOCKSIZE, MAXBLOCKSIZE>>>(*this);
	CudaTest("dinitlevels failed");

	//unsigned intzero = 0;
	//cudaMemcpy(&levels[source], &intzero, sizeof(intzero), cudaMemcpyHostToDevice);
	bool *changed;
	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	printf("\tnot computing levels, diameter will be zero.\n");
	/*unsigned iteration = 0;
	bool hchanged;
	do {
		++iteration;
		hchanged = false;
		cudaMemcpy(changed, &hchanged, sizeof(bool), cudaMemcpyHostToDevice);
		printf("computelevels: iteration %d.\n", iteration);
		dcomputelevels<<<(nnodes+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(*this, changed);
		CudaTest("dcomputelevels failed");
		printf("computelevels: iteration %d over.\n", iteration);
		cudaMemcpy(&hchanged, changed, sizeof(bool), cudaMemcpyDeviceToHost);
	} while (hchanged);
	*/
	cudaFree(changed);

	dprintstats<<<1, 1>>>(*this);
	CudaTest("dprintstats failed");
	
	freeLevels();
	return 0;
}
/*
void Graph::copyNodesToEdges_dist(int partitionNumber){
unsigned dst;

  for(unsigned ee = 0; ee < devicePartition[partitionNumber].numEdges; ee++) {
     dst = devicePartition[partitionNumber].edgedst[ee];
     devicePartition[partitionNumber].edgedstdist[ee]  = devicePartition[partitionNumber].nodedist[dst];
  }
}

void Graph::copyNodesToEdges_distsigma(int partitionNumber){
unsigned dst;

  for(unsigned ee = 0; ee < devicePartition[partitionNumber].numEdges; ee++) {
     dst = devicePartition[partitionNumber].edgedst[ee];
     devicePartition[partitionNumber].edgedstdist[ee]  = devicePartition[partitionNumber].nodedist[dst];
     devicePartition[partitionNumber].edgedstsigma[ee]  = devicePartition[partitionNumber].nodesigma[dst];
  }
}

void Graph::resolveNodesFromEdges(int partitionNumber, unsigned nodes,unsigned int source=-1){
unsigned int ii, ee;
unsigned dst;

  // initialize all node distances to large values and all node sigmas to 0's
if(source!=-1)
{
  for(ii=0; ii<nnodes; ii++){
    devicePartition[partitionNumber].nodedist[ii] = MYINFINITY;
    devicePartition[partitionNumber].nodesigma[ii] = 0;
  }
  devicePartition[partitionNumber].nodedist[source] = 0;
  devicePartition[partitionNumber].nodesigma[source] = 1;
}
  // first find the minimum distances for all nodes
  for(ee=0; ee<devicePartition[partitionNumber].numEdges; ee++){
    dst = devicePartition[partitionNumber].edgedst[ee];
 //   if(dst != source){ // ugly
      if(devicePartition[partitionNumber].edgedstdist[ee] < devicePartition[partitionNumber].nodedist[dst]){
        devicePartition[partitionNumber].nodedist[dst] = devicePartition[partitionNumber].edgedstdist[ee];
      }
  //  }
  }

  // Now for those edges whose dst nodes have min distances, add up sigmas
  for(ee=0; ee<devicePartition[partitionNumber].numEdges; ee++){
    dst = devicePartition[partitionNumber].edgedst[ee];
  //  if(dst != source){ // ugly
      if(devicePartition[partitionNumber].active[ee]){
        if(devicePartition[partitionNumber].edgedstdist[ee] == devicePartition[partitionNumber].nodedist[dst]){
          devicePartition[partitionNumber].nodesigma[dst] += devicePartition[partitionNumber].edgedstsigma[ee];
        }
   //   }
    }
  }

}
void Graph::copyNodesToEdges_dist_omp(int partitionNumber){
unsigned dst,ee;

#pragma omp parallel for private (dst) schedule(static) num_threads(14)
  for(ee = 0; ee < devicePartition[partitionNumber].numEdges; ee++) {
     dst = devicePartition[partitionNumber].edgedst[ee];
     devicePartition[partitionNumber].edgedstdist[ee]  = devicePartition[partitionNumber].nodedist[dst];
  }
}

void Graph::copyNodesToEdges_distsigma_omp(int partitionNumber){
unsigned dst;

#pragma omp parallel for private (dst) schedule(static) num_threads(14)
  for(unsigned ee = 0; ee < devicePartition[partitionNumber].numEdges; ee++) {
     dst = devicePartition[partitionNumber].edgedst[ee];
     devicePartition[partitionNumber].edgedstdist[ee]  = devicePartition[partitionNumber].nodedist[dst];
     devicePartition[partitionNumber].edgedstsigma[ee]  = devicePartition[partitionNumber].nodesigma[dst];
  }
}

void Graph::activeNodesFromEdges_omp(int partitionNumber){
unsigned int ii, ee;
unsigned src;
#pragma omp parallel for private (src) schedule(guided) num_threads(14)
  for(ee=0; ee < devicePartition[partitionNumber].numEdges; ee++){
    src = devicePartition[partitionNumber].edgesrc[ee];
    if( devicePartition[partitionNumber].active[src]==false){
	     devicePartition[partitionNumber].edgeactive[ee] = false;
    }
  }
#pragma omp parallel for private (src) schedule(guided) num_threads(14)
  for(ee=0; ee < nnodes; ee++){
	  if(devicePartition[partitionNumber].active[ee] == false)
	  devicePartition[partitionNumber].active[ee] = true;
    }
}
*/
/*
void Graph::resolveNodesFromEdges_omp(int partitionNumber, unsigned nodes,unsigned source = -1){
unsigned int ii, ee;
unsigned dst;

  // initialize all node distances to large values and all node sigmas to 0's
 
if(source != -1)
{
#pragma omp parallel for private (dst) schedule(static)
  for(ii=0; ii<nnodes; ii++){
    devicePartition[partitionNumber].nodedist[ii] = MYINFINITY;
    devicePartition[partitionNumber].nodesigma[ii] = 0;
  }
#pragma omp single
  {
  devicePartition[partitionNumber].nodedist[source] = 0;
  devicePartition[partitionNumber].nodesigma[source] = 1;
  }
#pragma omp barrier
} 

  // first find the minimum distances for all nodes
#pragma omp parallel for private (dst) schedule(static) num_threads(14)
  for(ee=0; ee<devicePartition[partitionNumber].numEdges; ee++){
    dst = devicePartition[partitionNumber].edgedst[ee];
 //   if(dst != source){ // ugly
#pragma omp capture
    {
      if(devicePartition[partitionNumber].edgedstdist[ee] < devicePartition[partitionNumber].nodedist[dst])
        devicePartition[partitionNumber].nodedist[dst] = devicePartition[partitionNumber].edgedstdist[ee];
    } 
  //  }
  }
//#pragma omp barrier

  // Now for those edges whose dst nodes have min distances, add up sigmas
#pragma omp parallel for private (dst) schedule(static) num_threads(14)
  for(ee=0; ee<devicePartition[partitionNumber].numEdges; ee++){
    dst = devicePartition[partitionNumber].edgedst[ee];
  //  if(dst != source){ // ugly
      if(devicePartition[partitionNumber].active[ee]){
        if(devicePartition[partitionNumber].edgedstdist[ee] == devicePartition[partitionNumber].nodedist[dst]){
#pragma omp atomic
          devicePartition[partitionNumber].nodesigma[dst] += devicePartition[partitionNumber].edgedstsigma[ee];
        }
   //   }
    }
  }
}
*/
void Graph::findSerialBFS(unsigned src) {
	foru max_bfs;
	foru *localdist = (foru *) calloc(nnodes, sizeof(foru));
	for (unsigned i = 0; i < nnodes; i++)
		localdist[i] = MYINFINITY;
	
	localdist[src] = 0;
	for (unsigned i = 0; i < nnodes; i++) {
		int flag = 1;
		for (unsigned ver = 0; ver < nnodes; ver++) {
			foru tdist = localdist[ver];
			for (unsigned ed = psrc[ver]; ed < (psrc[ver] + noutgoing[ver]); ed++) {
				printf("\npsrc value %u\n",psrc[ver]);
				foru wt = 1;
				if (tdist + wt < localdist[srcsrc[ed]]) {
					flag = 0;
					localdist[srcsrc[ed]] = tdist + wt;
				}
			}
		}
		if (flag)
			break;
	}
	   printf("\nCODE REACHING HERE\n");

	max_bfs = 0;
	for (unsigned ver = 0; ver < nnodes; ver++) {
		if (localdist[ver] < MYINFINITY) {
			if (localdist[ver] > max_bfs) {
				max_bfs = localdist[ver];
			}
		}
	}
	ofstream op;
        op.open("EdgeOP.txt");
        op.precision(0);
        for (unsigned i = 0; i < nnodes; i++)
		op << localdist[i] << endl;
	op.close();
	free(localdist);
}


__device__ __host__
unsigned Graph::getStartNodeFromNode(unsigned nn) {
        return getStartNode(it);
}
__device__ __host__
unsigned Graph::getStartNode(unsigned it) {
        unsigned ee = getStartEdge(it);
        return srcsrc[ee % nedgesresident];
}
__device__ __host__
unsigned Graph::getEndNode(unsigned it) {
        unsigned ee = getEndEdge(it);
        return srcsrc[(ee - 1) % nedgesresident] + 1;   
}
unsigned Graph::getIterationFromNode(unsigned nn) {     
        return nn / nedgessliding;
}
__host__ __device__ unsigned Graph::getIterationFromEdge(unsigned ee) {
        return ee / nedgessliding;
}
__host__ __device__
unsigned Graph::getStartEdge(unsigned it) {
        return it * nedgessliding;
}
__host__ __device__
unsigned Graph::getEndEdge(unsigned it) {
        unsigned ee = getStartEdge(it + 1);
        return (ee < nedges ? ee : nedges);
}



#endif
