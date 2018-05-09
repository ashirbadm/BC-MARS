#ifndef _MYUTILS_H
#define _MYUTILS_H
#pragma once
#include "Structs.h"
/*
void verifysolution(unsigned int *edgessrcdst, foru *edgessrcwt, unsigned int *noutgoing, unsigned int *psrc, foru *dist, unsigned int nv, unsigned int nedges, unsigned *srcsrc) {
	unsigned int ii, nn;
	unsigned int nerr = 0;
	for (nn = 0; nn < nv; ++nn) {
		unsigned int start = psrc[nn];
		unsigned int nsrcedges = noutgoing[nn];
		for (ii = 0; ii < nsrcedges; ++ii) {
			unsigned int u = nn;
			unsigned int v = edgessrcdst[start + ii];
			//float wt = 1;
			foru wt = BFS_SSSP(start + ii); // had to put ifdef cause of error
			if (wt > 0 && dist[u] + wt < dist[v]) {
				++nerr;
			}
		}
	}
	printf("verifying: no of errors = %d.\n", nerr);
}

template <class myType1>
inline void AryPrint(char name[],myType1 &var,uint32_t size){
	cout<<name<<endl;
	uint32_t i;
	for(i=0;i<size;i++)
		cout<<var[i]<<" ";
	cout<<endl;
}

template <class myType1>
inline void AryPrintFile(char name[],myType1 &var,uint32_t size){
	ofstream op;
	op.open(name);
	uint32_t i;
	for(i=0;i<size;i++)
		op<<var[i]<<endl;
	op.close();
	cout<<name<<endl;
}

inline void AryFile(char *fname,float *&ary,unsigned &indexaryLen){
	uint32_t i=0;
	FILE  *opf;
	opf=fopen(fname,"w+t");
	while(i < indexaryLen){
		fprintf(opf ,"%f\n", ary[i]);
		i++;
	}
	fclose(opf);
}

inline void Ary2File(char *fname,foru *&ary,unsigned *&indexary,unsigned &indexaryLen){
	uint32_t i=0;
	FILE  *opf;
	opf=fopen(fname,"w+t");
	while(i < indexaryLen){
		if(indexary[i]){
			fprintf(opf ,"%u\n", (unsigned)ary[i]);
		}
		i++;
	}
	fclose(opf);
}

void mystringRev(char *str){
	int len=strlen(str);
	char *s1=str;
	char *s2=str+len-1;
	while(s2>s1){
		char temp=*s1;
		*s1=*s2;
		*s2=temp;
		s1++;
		s2--;
	}
}
void getGraphname(char *ans,char *src){
	char val[256];
	strcpy(val,src);
	mystringRev(val);
	char *s2=strchr(val,'/');
	strncpy(ans,val,s2-val);
	ans[s2-val]=0;
	mystringRev(ans);
}
*/
extern "C" void lonestar_gpu(unsigned *,unsigned *,unsigned *,unsigned *,unsigned *edgessrc,unsigned *edgesdst,unsigned hnodes,unsigned hedges,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned count,int *sources,cudaDeviceProp *,bool BM_COMP,unsigned *nerr, unsigned *, int *);
extern "C" void lonestar_gpu_MultiS(unsigned *d_psrc,unsigned *d_noutgoing,unsigned *d_edgessrc,unsigned *d_edgesdst,unsigned hnodes,unsigned hedges,unsigned *dist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,int *d_sources,cudaDeviceProp *deviceProp,bool BM_COMP,unsigned *nerr, unsigned *d_border, int *d_sigma, int *flag, cudaStream_t *s_multi, unsigned *child_count, int *child);
extern "C" void lonestar_gpu_MultiS_initialize(unsigned *d_psrc,unsigned hnodes,unsigned hedges);
extern "C" void updateBorderChildGPU(unsigned *borderNodes, int* d_gpu_border_child, int* d_gpu_border_child_count, unsigned *child_count, int* child, unsigned borderCount_gpu);
extern "C" void ananya_code_func(unsigned *,unsigned *,unsigned *,unsigned *,unsigned *edgessrc,unsigned *edgesdst,unsigned hnodes,unsigned hedges,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned count,unsigned *sources,cudaDeviceProp *,bool BM_COMP,unsigned *nerr);
int betweenness_centrality_parallel(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,int *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *,int num_threads, Graph &graph);
void betweenness_centrality_parallel_MultiS(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,int *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *lock,int num_threads, unsigned *border, struct binary_semaphore *sem, int *src_sigma, unsigned *child_count, int *child, vector< vector<int> >& cpu_border_child, unsigned *borderNodes, unsigned *cpu_visited_vertices, unsigned *cpu_num_visited);

//int betweenness_centrality_parallel1(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,unsigned *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *,int num_threads);
extern "C" void worklist_cpu(unsigned *psrc,unsigned *noutgoing,unsigned *edgessrc,unsigned *edgesdst,unsigned hnodes,unsigned hedges,unsigned *hdist,unsigned *nodesigma,unsigned *edgesigma,unsigned source_count,unsigned *sources,omp_lock_t *,bool,int);

/*
void writeResult(){

	FILE *fp;
	fp=fopen("result.vi","a+t");
	fprintf(fp,"%s,",myresult.exename);
	fprintf(fp,"%s,",myresult.algo);
	fprintf(fp,"%s,",myresult.graphfile);

	fprintf(fp,"%u,",myresult.bucketSize);

	fprintf(fp,"%u,",myresult.nnode);
	fprintf(fp,"%u,",myresult.nedge);
	fprintf(fp,"%u,",myresult.ns_nnode);
	fprintf(fp,"%u,",myresult.ns_nedge);

	fprintf(fp,"%u,",myresult.extranode);
	fprintf(fp,"%u,",myresult.maxdeg);
	fprintf(fp,"%u,",myresult.splitlevel);
	fprintf(fp,"%u,",myresult.MAX_EDGES_ALLOWED);
	fprintf(fp,"%u,",myresult.iterations);
	fprintf(fp,"%u,",myresult.runtime);
	fprintf(fp,"%f,",myresult.time);
	fprintf(fp,"%f,",myresult.ktime);
	fprintf(fp,"%f",myresult.sktime);

	fprintf(fp,"\n");
	fclose(fp);
}*/
#endif
