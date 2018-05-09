#ifndef ASHUUTILS_H
#define ASHUUTILS_H

#define IMYINFINITY		INT_MAX
#define MYINFINITY		UINT_MAX
#include "common.h"
#ifndef CPU_CODE
#include "metis.h"
#endif
//#include "kernelconfig.h"
#if USEASYNC == 1
        #define CUDACOPY(a,b,c,d,stream)        cudaMemcpyAsync(a,b,c,d,stream)
#else           
        #define CUDACOPY(a,b,c,d,stream)        cudaMemcpy(a,b,c,d)
#endif  


#define CPUPARTITION 0
#define GPUPARTITION 1

// GPU kernel configuration 
// from existing Rupesh's code
typedef struct KernelConfig {
	unsigned device;
	unsigned problemsize;
	unsigned nblocks, blocksize;
	cudaDeviceProp dp;

	KernelConfig(unsigned ldevice = 0);
	void	 init();
	unsigned setProblemSize(unsigned size);
	unsigned setNumberOfBlocks(unsigned lnblocks);
	unsigned setNumberOfBlocksPerSM(unsigned lnblocks);
	unsigned setNumberOfBlockThreads(unsigned lblocksize);
	unsigned setMaxThreadsPerBlock();
	unsigned getNumberOfBlocks();
	unsigned getNumberOfBlockThreads();
	unsigned getNumberOfTotalThreads();

	unsigned calculate();
	unsigned getMaxThreadsPerBlock();
	unsigned getMaxBlocks();
	unsigned getMaxSharedMemoryPerBlock();
	unsigned getNumberOfSMs();
	bool	 coversProblem(unsigned size = 0);
	unsigned getProblemSize();
} KernelConfig;

// Graph data storing all necessary code
// for graph accesing.
struct Graph {
	enum {NotAllocated, AllocatedOnHost, AllocatedOnDevice} memory;

	unsigned read(char file[], bool weighted = true);
	unsigned printStats();
	void     print();

	Graph(unsigned percgraph = 100);
	~Graph();
	unsigned init(unsigned percgraph);
	int initFrom(Graph &from);
	unsigned allocOnHost();
	unsigned allocOnDevice();
	unsigned dealloc();
	unsigned deallocOnHost();
	unsigned deallocOnDevice();
	unsigned dealloc_part();
	void allocLevels();
	void freeLevels();
	void progressPrint(unsigned maxii, unsigned ii);
	unsigned readFromEdges(char file[]);
	unsigned readFromGR(char file[], bool weighted);
	unsigned getResidentIterations();
	void printResidentDevice(unsigned it, unsigned nprintnodes = 100);
	void printEdges(unsigned startee, unsigned nee);
	void printOneEdge(unsigned ee);
	void printEdgesOf(unsigned nn);
	unsigned getFirstEdgeBSearch(unsigned nn);
	void printDegrees(unsigned startnn, unsigned nnn);
	void findSerialBFS(unsigned src);

        unsigned getIterationFromNode(unsigned nn);
        __host__ __device__ unsigned getIterationFromEdge(unsigned ee);

	__device__ void printStats1x1();
	__host__ __device__ void print1x1();
	__device__ void printResident(unsigned it, unsigned nprintnodes = 100);
	__device__ void printResidentNode(unsigned node);
	 __device__ unsigned getOutDegree(unsigned src);
	__device__ unsigned getInDegree(unsigned src);
	__device__ unsigned getDestination(unsigned src, unsigned nthedge);
	__device__ foru     getWeight(unsigned src, unsigned nthedge);
	__device__ unsigned getMinEdge(unsigned src);
	__device__ foru     getDistance(foru *dist, unsigned nn);
	__device__ foru     getDestinationDistance(foru *dist, unsigned src, unsigned ii);
	__device__ foru     getDestinationDistance(foru *dist, unsigned ee);
	__device__ void     setDestinationDistance(foru *dstdist, unsigned ee, foru altdist);
	__device__ __host__ void getEdge(unsigned ee, unsigned &src, unsigned &dst, foru &wt);

	__device__ unsigned     getSigma(unsigned *sigma, unsigned nn);
	__device__ unsigned     addSigma(unsigned ee, unsigned *dstsigma, unsigned srcsigma);
	__device__ dorf     addBC(dorf *bc, unsigned nn, dorf val);
	__device__ dorf     addDelta(dorf *delta, unsigned nn, dorf val);
	__device__ unsigned     getDestinationSigma(unsigned *dstsigma, unsigned edge);
	__device__ dorf     getDestinationDelta(dorf *dstdelta, unsigned edge);

	__device__ unsigned getFirstEdge(unsigned src);
	__device__ unsigned getFirstEdgeRandom(unsigned src);
	__device__ unsigned findStats();
	__device__ void computeStats();
	__device__ bool computeLevels();
	__device__ unsigned findMaxLevel();
	__device__ void computeDiameter();
	__device__ void computeInOut();
	__device__ void initLevels();
	__host__ __device__ unsigned getStartNode(unsigned it);
	__host__ __device__ unsigned getEndNode(unsigned it);
	__host__ __device__ unsigned getStartNodeFromNode(unsigned nn);
	__host__ __device__ unsigned getStartEdge(unsigned it);
	__host__ __device__ unsigned getEndEdge(unsigned it);


	__device__ dorf getInsum(dorf *insum, unsigned nn);
	__device__ dorf getPR(dorf *pr, unsigned nn);
	__device__ void setPR(dorf *pr, unsigned nn, dorf rr);

	unsigned nnodes, nedges; 
	unsigned nnodesresident, nedgesresident;
	unsigned nedgessliding;	// < nnodesresident.
	unsigned it;	// current iteration.
	int num_threads;


    // cudaTextureObject_t texnoutgoing;
	//struct texture<unsigned, 32, cudaReadModeElementType> texnoutgoing;
	unsigned *noutgoing, *srcsrc, *edgessrcdst,*edgedstsrc,*nincoming;
	foru *edgessrcwt;
	cudaArray *cuArray;
	//unsigned *levels;
	//unsigned source;

	unsigned *maxOutDegree;
	unsigned *psrc;
	unsigned diameter;
	bool foundStats;
	
	
	struct Partition{
	  int nparts;
	  int* part;
	  unsigned int* border;
          unsigned int* borderCount;
          unsigned int** borderNodes;
	  unsigned int* nodeCount;
	  unsigned int* edgeCount;
	  int edgecut; // since the edge cut is supposed to be small

	} partition;
	// Metis partitions
	int formMetisPartitions(Graph &from, Partition* partition);
        int fillBorderAndCount(Graph &from, Partition* partition);
	
	struct DevicePartition{
          unsigned int numEdges;
          unsigned int numNodes;
	  unsigned *edgesrc;
	  unsigned *edgedst;
	  unsigned *edgewt;
          unsigned * nodesigma;
          float * nodedelta;
          unsigned * nodedist;
          unsigned *psrc;
          unsigned *noutgoing;
          unsigned * edgesigma;
	unsigned *child_count;
	int *child;
          //unsigned *edgedist;
          //bool *active;
          //bool *edgeactive;
	}* devicePartition;

        unsigned int maxNodeCount;
        unsigned int maxEdgeCount;
        // VSS: Max node and edge count in device partitions

        void formDevicePartitions(Graph &from);
	void removeDevicePartitions(Graph &from);
	int usepatoh(Graph &,char *filename,float weight1,float weight2);

        void copyNodesToEdges_dist(int partitionNumber);
        void copyNodesToEdges_distsigma(int partitionNumber);
        void resolveNodesFromEdges(int partitionNumber, unsigned,unsigned int source);
        void copyNodesToEdges_dist_omp(int partitionNumber);
        void copyNodesToEdges_distsigma_omp(int partitionNumber);
        void resolveNodesFromEdges_omp(int partitionNumber, unsigned,unsigned int source);
	void activeNodesFromEdges_omp(int);
        

};

// Struct for lonestar code
struct Worklist_cpu {
	enum {NotAllocated, AllocatedOnHost, AllocatedOnDevice} memory;

	unsigned pushRange(unsigned *start, unsigned nitems);
	unsigned pushRangeEdges(unsigned startindex, unsigned nitems);
	unsigned push(unsigned work);
	unsigned popRange(unsigned *start, unsigned nitems);
	unsigned pop(unsigned &work);
	void clear();
	void myItems(unsigned &start, unsigned &end);
	unsigned getItem(unsigned at);
	unsigned getItemWithin(unsigned at, unsigned hsize);
	unsigned count();

	void init();
	void init(unsigned initialcapacity);
	void setSize(unsigned hsize);
	unsigned getSize();
	void setCapacity(unsigned hcapacity);
	unsigned getCapacity();
	void pushRangeHost(unsigned *start, unsigned nitems);
	void pushHost(unsigned work);
	void clearHost();
	void setInitialSize(unsigned hsize);
	unsigned calculateSize(unsigned hstart, unsigned hend);
	void setStartEnd(unsigned hstart, unsigned hend);
	void getStartEnd(unsigned &hstart, unsigned &hend);
	void copyOldToNew(unsigned *olditems, unsigned *newitems, unsigned oldsize, unsigned oldcapacity);
	void compressHost(unsigned wlsize, unsigned sentinel);
	void printHost();
	unsigned appendHost(Worklist_cpu *otherwl);

	Worklist_cpu();
	~Worklist_cpu();
	unsigned ensureSpace(unsigned space);
	unsigned *alloc(unsigned allocsize);
	unsigned realloc(unsigned space);
	unsigned dealloc();
	unsigned freeSize();

	unsigned *items;
	unsigned *start, *end, *capacity;	// since these change, we don't want their copies with threads, hence pointers.

	unsigned noverflows;


};

/*
typedef struct Worklist {
	enum {NotAllocated, AllocatedOnHost, AllocatedOnDevice} memory;

	__device__ unsigned pushRange(unsigned *start, unsigned nitems);
	__device__ unsigned pushRangeEdges(unsigned startindex, unsigned nitems);
	__device__ unsigned push(unsigned work);
	__device__ unsigned popRange(unsigned *start, unsigned nitems);
	__device__ unsigned pop(unsigned &work);
	__device__ void clear();
	__device__ void myItems(unsigned &start, unsigned &end);
	__device__ unsigned getItem(unsigned at);
	__device__ unsigned getItemWithin(unsigned at, unsigned hsize);
	__device__ unsigned count();

	void init();
	void init(unsigned initialcapacity);
	void setSize(unsigned hsize);
	unsigned getSize();
	void setCapacity(unsigned hcapacity);
	unsigned getCapacity();
	void pushRangeHost(unsigned *start, unsigned nitems);
	void pushHost(unsigned work);
	void clearHost();
	void setInitialSize(unsigned hsize);
	unsigned calculateSize(unsigned hstart, unsigned hend);
	void setStartEnd(unsigned hstart, unsigned hend);
	void getStartEnd(unsigned &hstart, unsigned &hend);
	void copyOldToNew(unsigned *olditems, unsigned *newitems, unsigned oldsize, unsigned oldcapacity);
	void compressHost(unsigned wlsize, unsigned sentinel);
	void printHost();
	unsigned appendHost(Worklist *otherwl);

	Worklist();
	~Worklist();
	unsigned ensureSpace(unsigned space);
	unsigned *alloc(unsigned allocsize);
	unsigned realloc(unsigned space);
	unsigned dealloc();
	unsigned freeSize();

	unsigned *items;
	unsigned *start, *end, *capacity;	// since these change, we don't want their copies with threads, hence pointers.

	unsigned noverflows;


} Worklist;
*/

// Struct for BC cpu part
// Ashirbad
struct varto_cpu_part{
	Graph *graph;
	Graph::DevicePartition *partition;
	Graph::Partition *borderInfo;
	unsigned source;
	int srcpartition, nonsrcpartition;
	int num_threads;
	unsigned int *borderVector_cpu,*borderVector_gpu;
	unsigned int *borderSigma_cpu,*borderSigma_gpu;
	unsigned *actual_dist;
	struct matrix_csr *borderMatrix_cpu, *borderMatrix_gpu;
	bool single_relax;
	unsigned *psrc, *noutgoing;
	omp_lock_t *lock;
	//for backward phase
	long *cpu_level;
	volatile long *gpu_level;
	long *cpu_level_min, *gpu_level_min;
	//bool *cpu_wait; 
	double cpu_F_I,cpu_F_R,cpu_bck_knl_time,cpu_fwd_knl_time,cpu_tot_bck_time;
};

// Struct for BC gpu part
// Ashirbad
struct varto_gpu_part{
	Graph *graph;
	Graph::DevicePartition *nonsrcpart;
	Graph::DevicePartition *srcpart;
        Graph::DevicePartition *gpupart;
	Graph::Partition *borderInfo;
	unsigned source;
	int srcpartition, nonsrcpartition;
	int num_threads;
	unsigned int *borderVector_cpu,*borderVector_gpu;
	unsigned int *borderSigma_cpu,*borderSigma_gpu;
        struct matrix_csr *borderMatrix;
	cudaStream_t sone, stwo,sthree,sfour;
	unsigned *nodesigma, *edgesigma, *edgesrc, *edgedst, *nodedist, *edgedstdist, *edgewt,*border;
	unsigned *borderNodes,*borderDist, *borderSigma;
	float *borderDelta;
	float *nodedelta;
        bool* active, *localchanged;
	KernelConfig *kconf;
	bool single_relax;
	unsigned *psrc, *noutgoing,*nerr;
	//for backward phase
	volatile long *cpu_level;
	long *gpu_level;
	long *cpu_level_min, *gpu_level_min;
	//bool *gpu_wait; 
	omp_lock_t *lock;
	bool *d_gpu_wait;
	double gpu_F_I,gpu_F_R,gpu_bck_knl_time,gpu_fwd_knl_time,gpu_tot_bck_time,gpu_memcpy;
};

struct binary_semaphore {
    pthread_mutex_t mutex;
    pthread_cond_t cvar;
};


#endif
