#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
//#ifdef _OPENMP
#include <omp.h>
//#endif
//#include "RngStream.h"
#include "Structs.h"
//#if TIME_RESULTS
//#endif
#include "EMS_x86_64.h"
#include<sys/time.h>
#include "utils.h"
#define DIAGNOSTIC 1
#define EXPECTED_GRAPH_DIAMETER 10000
//#define CPU_PARTITION 0
#define CACHELOG 7
#define NOSHARE(x) ((x)<<CACHELOG)
//#define MYINFINITY (0x7fffffff)
typedef unsigned vid_t;
typedef unsigned eid_t;

int betweenness_centrality_parallel(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,unsigned *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *lock,int num_threads) {

    /* stack of vertices in the order of non-decreasing 
     * distance from s. Also used to implicitly 
     * represent the BFS queue. 
     */
    vid_t* __restrict__ visited_vertices;    
    /* A struct to store information about the visited vertex 
    typedef struct {
        eid_t sigma;         // No. of shortest paths 
	uint64_t d;          // distance of vertex from source vertex 
	} S_info_t;
    
    S_info_t* vis_path_info;
	*/
   // vid_t* __restrict__ vis_srcs;  /* Randomly chosen set of vertices to initiate traversals from 
    unsigned * __restrict__ num_visited;
    unsigned *__restrict__ visited_counts;
        // Allocate memory for the data structures
        visited_vertices = (vid_t *) calloc(2*hedges , sizeof(vid_t));
        //vis_path_info = (S_info_t *) malloc(n * sizeof(S_info_t));
        assert(visited_vertices != NULL);
        //assert(vis_path_info != NULL);

        num_visited = (unsigned *) calloc(EXPECTED_GRAPH_DIAMETER,sizeof(unsigned));
        visited_counts = (unsigned *) calloc(NOSHARE(num_threads+1),sizeof(unsigned));

        assert(num_visited != NULL);
        assert(visited_counts != NULL);

    //omp_set_num_threads(omp_get_num_procs()/2);

  //  fprintf(stderr, "Beginning betweenness computation from %ld "
  //          "randomly-chosen source vertices ...\n", num_src_vertices);
#ifdef _OPENMP    
#pragma omp parallel default(none) shared(visited_vertices,num_visited,visited_counts,hnodes,hedges,rowptrs,columnindexes,edgessrc,noutgoing,sources,source_count,nodedist,nodesigma,edgesigma,lock,num_threads)
#endif
 {
    /* Vars to index local thread stacks */
    vid_t* __restrict__ p_vis;
    vid_t* __restrict__ p_vis_temp;
    unsigned p_vis_size, p_vis_count;
    unsigned phase_num;

    /* Other vars */
    unsigned i, j, k;
    unsigned v, w;
    
    uint64_t n, m;
    vid_t* __restrict__ adj;
    eid_t* __restrict__ num_edges;
    unsigned d_phase;
    unsigned tid, nthreads;
    unsigned ph_start, ph_end;
    unsigned sigma_v,dist_v;
    eid_t num_edges_v;
    unsigned offset;
    unsigned MAX_NUM_PHASES;
    MAX_NUM_PHASES = EXPECTED_GRAPH_DIAMETER;

    //S_info_t *vis_path_info_i, *vis_path_info_v, *vis_path_info_w;
    //eid_t vis_path_info_v_sigma;
    unsigned marked;

#if DIAGNOSTIC
    double elapsed_time_part;
#endif

//#ifdef _OPENMP
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
//#endif

#if DIAGNOSTIC
    if (tid == 0) {
        elapsed_time_part = timer();
    }
#endif

    n = hnodes;
    m = hedges;
    adj = columnindexes;
    num_edges = rowptrs;
        
#ifdef _OPENMP
#pragma omp barrier
#endif
        
    /* Start timing code from here */


   /* if (tid == 0) {
        // Allocate memory for the data structures
        visited_vertices = (vid_t *) calloc(m , sizeof(vid_t));
        //vis_path_info = (S_info_t *) malloc(n * sizeof(S_info_t));
        assert(visited_vertices != NULL);
        //assert(vis_path_info != NULL);

        num_visited = (unsigned *) calloc(MAX_NUM_PHASES,sizeof(unsigned));
        visited_counts = (unsigned *) calloc(NOSHARE(nthreads+1),sizeof(unsigned));

        assert(num_visited != NULL);
        assert(visited_counts != NULL);
    }*/

    /* local memory for each thread */
    p_vis_size = (2*m)/nthreads;
    p_vis = (vid_t *) malloc(p_vis_size*sizeof(vid_t));
    //assert(p_vis != NULL);
    p_vis_count = 0;


#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    for (i=0; i<(2*m); i++) {
        //vis_path_info_i = &vis_path_info[i];
        //vis_path_info_i->d = 0; // d == 0 => marked == 0
        //vis_path_info_i->sigma = 0;
        visited_vertices[i] = MYINFINITY;
    }

    // Start traversals
    //for (p = 0; p < n; p ++) {
        //s = vis_srcs[p];
	//s=0;

//#pragma omp single                
 //	 	printf("\nCODE REACHING HERE\n");
        phase_num = 0;
        
        if (tid == 0) {
	unsigned ii;
	for (ii = 0 ; ii < source_count ; ii++){
            visited_vertices[ii] = sources[ii];
	}
            //vis_path_info[s].sigma = 1;
            //vis_path_info[s].d = 1;
            num_visited[phase_num] = 0;
            num_visited[phase_num+1] = source_count;
            visited_counts[NOSHARE(0)] = 0;
        }

        visited_counts[NOSHARE(tid+1)] = 0;

        d_phase = 1;
        
#if DIAGNOSTIC
        if (tid == 0) 
            elapsed_time_part = timer();
#endif

#ifdef _OPENMP       
#pragma omp barrier
#endif

	while (num_visited[phase_num+1] - num_visited[phase_num] > 0) {

            if (phase_num >= MAX_NUM_PHASES - 1) {

                if (tid == 0)
                    printf( "Error! Increase EXPECTED_GRAPH_DIAMETER "
                            "setting in graph.h. Exiting\n");
                //assert(phase_num < MAX_NUM_PHASES - 1);
            }

            ph_start = num_visited[phase_num];
            ph_end = num_visited[phase_num+1];

            p_vis_count = 0;
            d_phase++;

//#pragma omp single
//	    printf("\nCODE REACHING HERE phase num %d\n",phase_num);
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(guided)
#endif
            for (i = ph_start; i < ph_end; i++) {
                v = visited_vertices[i];
		//if(v==MYINFINITY)continue;
                //vis_path_info_v = &(vis_path_info[v]);
                sigma_v = nodesigma[v];
                num_edges_v = num_edges[v];

                for (j = num_edges_v; j < (num_edges_v + noutgoing[v]) ; j++) {

		    //if(edgessrc[j]!=v)continue;                
                    w = adj[j];
                    //vis_path_info_w = &(vis_path_info[w]);
                    //marked = nodedist[w];
		    //dist_v = nodedist[v]+1;
		    omp_set_lock(&(lock[w]));
                    if (nodedist[w] > (nodedist[v]+1 )) {
			    //if ( __sync_bool_compare_and_swap(&nodedist[w], marked, dist_v) ) {
			    //if(nodedist[w]==marked){
//#pragma omp atomic write
                            nodedist[w] = nodedist[v] + 1;
			    omp_unset_lock(&(lock[w]));
/* FIRST */
#ifdef _OPENMP
			    edgesigma[j] = sigma_v;
                            (void)__sync_fetch_and_add(&nodesigma[w], sigma_v);
//#else
                            //nodesigma[w] += sigma_v;
#endif
                            /* Add w to local stack */
                            if (p_vis_count == p_vis_size) {
                                /* Resize p_vis */
                                p_vis_temp = (vid_t *) calloc(2*p_vis_size,sizeof(vid_t));
                                //assert(p_vis_temp != NULL);
                                memcpy(p_vis_temp, p_vis, p_vis_size*sizeof(vid_t));
                                free(p_vis);
                                p_vis = p_vis_temp;
                                p_vis_size = 2*p_vis_size;
                            }
                            p_vis[p_vis_count++] = w;
                        /*} else{ // this is the case where the distances have been set correct but the sigmas need to be added  
// ALMOST FIRST 		// Add here edgesigma calculation
			    edgesigma[j] = -edgesigma [j];
                            (void)__sync_fetch_and_add(&nodesigma[w], edgesigma[j]);
			    edgesigma[j] = sigma_v;
			    (void)__sync_fetch_and_add(&nodesigma[w], sigma_v);
			    
			    // Add w to local stack 
			    if (p_vis_count == p_vis_size) {
				    // Resize p_vis 
				    p_vis_temp = (vid_t *)malloc(2*p_vis_size*sizeof(vid_t));
				    assert(p_vis_temp != NULL);
				    memcpy(p_vis_temp, p_vis, p_vis_size*sizeof(vid_t));
				    free(p_vis);
				    p_vis = p_vis_temp;
				    p_vis_size = 2*p_vis_size;
			    }
			    p_vis[p_vis_count++] = w;
			    
                        }*/
		    }else if (nodedist[w] == (nodedist[v] + 1)) {
			    omp_unset_lock(&(lock[w]));
			    edgesigma[j] = -edgesigma [j];
                            (void)__sync_fetch_and_add(&nodesigma[w], edgesigma[j]);
			    edgesigma[j] = sigma_v;
			    (void)__sync_fetch_and_add(&nodesigma[w], sigma_v);
			    /*
			    // Add w to local stack 
			    if (p_vis_count == p_vis_size) {
				    // Resize p_vis 
				    p_vis_temp = (vid_t *)malloc(2*p_vis_size*sizeof(vid_t));
				    assert(p_vis_temp != NULL);
				    memcpy(p_vis_temp, p_vis, p_vis_size*sizeof(vid_t));
				    free(p_vis);
				    p_vis = p_vis_temp;
				    p_vis_size = 2*p_vis_size;
			    }
			    p_vis[p_vis_count++] = w;
			    */
		    
                    } else { /* This is the case where the the distance of w is lesser*/
			    omp_unset_lock(&(lock[w]));
			    edgesigma[j] = 0;
/* CLEARLY NOT FIRST 
 
                        d_val = vis_path_info_w->d;
                        if ((d_val == 0) || (d_val == d_phase)) {
                            child[num_edges_v+(vis_path_info_v->child_count)++] = w;
#ifdef _OPENMP
                            (void)__sync_fetch_and_add(&(vis_path_info_w->sigma), sigma_v);
#else
                            vis_path_info_w->sigma += sigma_v;
#endif

                        }
*/
                    }
                } /* End of inner for loop */
	    } /* End of outer for loop */
        
            /* Merge all local stacks for next iteration */
            phase_num++;

            visited_counts[NOSHARE(tid+1)] = p_vis_count;

#ifdef _OPENMP
#pragma omp barrier
#endif
 
            if (tid == 0) {
                visited_counts[NOSHARE(0)] = num_visited[phase_num];
                for(k=1; k<=nthreads; k++) {
                    visited_counts[NOSHARE(k)] += visited_counts[NOSHARE(k-1)];
                }
                num_visited[phase_num+1] = visited_counts[NOSHARE(nthreads)];
            }
       
           
#ifdef _OPENMP           
#pragma omp barrier
#endif
            for (k = visited_counts[NOSHARE(tid)]; k < visited_counts[NOSHARE(tid+1)]; k++) {
                offset = visited_counts[NOSHARE(tid)];
                visited_vertices[k] = p_vis[k-offset];
            } 
           
#ifdef _OPENMP            
#pragma omp barrier
#endif

	}// End of while loop
	    /* 
#if DIAGNOSTIC 
        if (tid == 0) {
             elapsed_time_part = timer() - elapsed_time_part;
             fprintf(stdout, "Traversal time: %9.6lf seconds, src %ld, " 
                     "num phases %ld\n",
                     elapsed_time_part, sources[0], phase_num);fflush(stdout);
            fprintf(stderr, "visited count %ld\n", num_visited[phase_num+1]);
        }
#endif*/
//#ifdef _OPENMP            
//#pragma omp barrier
//#endif
//#pragma omp single
//printf("\nCODE REACHING phase_num\n");
	
//#pragma omp critical
 	//{
 free(p_vis);
		/*if(tid==0){
		 free(visited_vertices);
		 free(num_visited);
		 free(visited_counts);
		}*/
 	//}
#ifdef _OPENMP
#pragma omp barrier
#endif
	} // End of parallel region
	
	free(visited_vertices);
	free(num_visited);
	free(visited_counts);
	
    return 0;
}

/*
int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	char *inputfile = argv[1];
	unsigned weighted = 0, source = 10,ii;
	Graph graph;
	graph.read(inputfile, weighted);
        graph.initFrom(graph);
	graph.formDevicePartitions(graph);
	for(ii = 0 ; ii < graph.nnodes ; ii++){
		graph.devicePartition[CPUPARTITION].nodedist[ii] = MYINFINITY;
		graph.devicePartition[CPUPARTITION].nodesigma[ii] = 0;
	}
	memset(graph.devicePartition[CPUPARTITION].edgesigma,0,((graph.devicePartition[CPUPARTITION].numEdges) * sizeof(unsigned)));
	for(ii = 0 ; ii < graph.nnodes ; ii++){
		source = ii;
	if(graph.partition.part[source]==CPUPARTITION){
	graph.devicePartition[CPUPARTITION].nodedist[source] = 0;
	graph.devicePartition[CPUPARTITION].nodesigma[source] = 1;
	betweenness_centrality_parallel(graph.nnodes,graph.devicePartition[CPUPARTITION].numEdges,graph.devicePartition[CPUPARTITION].psrc,graph.devicePartition[CPUPARTITION].edgedst, graph.devicePartition[CPUPARTITION].noutgoing,&source,1,graph.devicePartition[CPUPARTITION].nodedist,graph.devicePartition[CPUPARTITION].nodesigma,graph.devicePartition[CPUPARTITION].edgesigma);
	break;
	}
	else{
		printf("Source not in CPUPARTITION \n");
	}
	}

	return 0 ;
}
*/
