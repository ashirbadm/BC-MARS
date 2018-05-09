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


void betweenness_centrality_parallel_MultiS(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,int *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *lock,int num_threads, unsigned *border, struct binary_semaphore *sem, int *src_sigma, unsigned *child_count, int *child, vector< vector<int> >& cpu_border_child, unsigned *borderNodes, unsigned *cpu_visited_vertices, unsigned *cpu_num_visited)
{
//printf("[CPU] Starting MultiS fn\n");
unsigned* __restrict__ visited_vertices = cpu_visited_vertices;
    /* A struct to store information about the visited vertex 
    typedef struct {
        eid_t sigma;         // No. of shortest paths 
        uint64_t d;          // distance of vertex from source vertex 
        } S_info_t;
    
    S_info_t* vis_path_info;
        */
   // vid_t* __restrict__ vis_srcs;  /* Randomly chosen set of vertices to initiate traversals from 
    unsigned * __restrict__ num_visited = cpu_num_visited;
    unsigned *__restrict__ visited_counts;
        // Allocate memory for the data structures
       // visited_vertices = (vid_t *) calloc(2*hnodes , sizeof(vid_t));
        //vis_path_info = (S_info_t *) malloc(n * sizeof(S_info_t));
        assert(visited_vertices != NULL);
        //assert(vis_path_info != NULL);

      //  num_visited = (unsigned *) calloc(EXPECTED_GRAPH_DIAMETER,sizeof(unsigned));
        visited_counts = (unsigned *) calloc(NOSHARE(num_threads+1),sizeof(unsigned));

        assert(num_visited != NULL);
        assert(visited_counts != NULL);
        int processed_count = 0,d_phase = -1;
	omp_set_num_threads(num_threads-1);
//	printf("[CPU] No. of CPU threads = %d\n",num_threads-1);
#ifdef _OPENMP    
#pragma omp parallel default(none) shared(d_phase,processed_count,visited_vertices,num_visited,visited_counts,hnodes,hedges,rowptrs,columnindexes,edgessrc,noutgoing,sources,source_count,nodedist,nodesigma,edgesigma,lock,num_threads,border,sem,src_sigma, child_count,child,cpu_border_child,borderNodes)
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
   // unsigned d_phase;
    unsigned tid, nthreads;
    unsigned ph_start, ph_end;
    unsigned sigma_v,dist_v, sigma_w;
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
//      printf("Total Threads : %d",nthreads);
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
   /* local memory for each thread */
    p_vis_size = (2*n)/nthreads;
    p_vis = (vid_t *) malloc(p_vis_size*sizeof(vid_t));
    //assert(p_vis != NULL);
    p_vis_count = 0;


#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    for (i=0; i<(n); i++) {
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
 //             printf("\nCODE REACHING HERE\n");
        phase_num = 0;

        if (tid == 0) {
/*        unsigned ii = 0;
	while(sources[0] == -99999);
        d_phase = -sources[0];
        int item = -sources[processed_count],dist_item;
        if(item == d_phase)
        {
                processed_count++;
                item = sources[processed_count];
                do
                {
                        dist_item = nodedist[item];
                        if(dist_item >= d_phase || dist_item == MYINFINITY)
                        {
                                visited_vertices[ii] = item;
//                                printf("[%d]Push %d\n",d_phase,item);
                                ii++;
                                nodedist[item] = d_phase;
                        }
                        processed_count++;
                        item = sources[processed_count];
                }while(item>=0);
        }
                //printf("After first push : processed_count = %d, sources[%d] = %d, d_phase = %d\n",processed_count,processed_count,sources[processed_count],d_phase);
//      for (ii = 1 ; ii < source_count ; ii++){
  //          visited_vertices[ii] = sources[ii];
//      }
            //vis_path_info[s].sigma = 1;
            //vis_path_info[s].d = 1;
*/
            num_visited[phase_num] = 0;
            num_visited[phase_num+1] = 0;              // source_count;
            visited_counts[NOSHARE(0)] = 0;
        }


        visited_counts[NOSHARE(tid+1)] = 0;

     //   d_phase = 0;

#if DIAGNOSTIC
        if (tid == 0)
            elapsed_time_part = timer();
#endif

#ifdef _OPENMP       
#pragma omp barrier
#endif

        while (num_visited[phase_num+1] - num_visited[phase_num] > 0 ||  sources[processed_count]!=-100000 ) {

            if (phase_num >= MAX_NUM_PHASES - 1) {
                printf("ERROR : processed_count = %d, sources[%d] = %d, d_phase = %d\n",processed_count,processed_count,sources[processed_count],d_phase);
                if (tid == 0){
                    printf( "Error! Increase EXPECTED_GRAPH_DIAMETER "
                            "setting in graph.h. Exiting\n");
                        scanf("%d",&d_phase);
                }
//                assert(phase_num < MAX_NUM_PHASES - 1);
            }

            ph_start = num_visited[phase_num];
            ph_end = num_visited[phase_num+1];

            p_vis_count = 0;
        if(tid==0)
            d_phase++;
        if (tid == 0 && sources[processed_count]!=-100000) {
//        printf("[AddMoreSources]d_phase = %d, sources[processed_count] = %d\n", d_phase,sources[processed_count]);
        unsigned ii = 0;
//      scanf("%d",&ii);
       // d_phase = -sources[0];
	pthread_mutex_lock(&(sem->mutex));
	while(sources[processed_count] == -99999)	//Spin Lock. Waiting for more sources.
	{
		pthread_cond_wait(&(sem->cvar),&(sem->mutex));
		//while(spinlock_flag==1);
	//	printf("Waiting sources[%d] = %d\n",processed_count,sources[processed_count]);
	}
	pthread_mutex_unlock(&(sem->mutex));
//     	printf("[SpinLock over][AddMoreSources]d_phase = %d, sources[%d] = %d\n", d_phase,processed_count,sources[processed_count]);
	   int item = -sources[processed_count],dist_item;
        if(item == d_phase)
        {
                processed_count++;
                item = sources[processed_count];
                do
                {
                        dist_item = nodedist[item];
                        if(dist_item >= d_phase || dist_item == MYINFINITY)
                        {
                                nodedist[item] = d_phase;
				nodesigma[item] = src_sigma[processed_count];
                                if (p_vis_count == p_vis_size) {
                                /* Resize p_vis */
                                p_vis_temp = (vid_t *) calloc(2*p_vis_size,sizeof(vid_t));
                                //assert(p_vis_temp != NULL);
                                memcpy(p_vis_temp, p_vis, p_vis_size*sizeof(vid_t));
                                free(p_vis);
                                p_vis = p_vis_temp;
                                p_vis_size = 2*p_vis_size;
                            }
                            p_vis[p_vis_count++] = item;
//                            printf("[%d,%d]Push %d\n",d_phase,nodesigma[item],item);


                        }
                        processed_count++;
                        item = sources[processed_count];
                }while(item>=0);
        }

                }

//#pragma omp single
//          printf("\nCODE REACHING HERE phase num %d\n",phase_num);
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
			sigma_w = nodesigma[w];
		marked = nodedist[w];
                    dist_v = nodedist[v]+1;

		if(border[w] != 0)
		{
			if(marked==dist_v)	child[num_edges_v+child_count[v]++] = w;
			continue;
		}
                  //      if(w==45328 || w==45329 || w==1890994)
                   //             printf("\nIn Madduri's code : %d->%d, sigma[%d] = %d, sigma[%d] = %d\n",v,w,v,nodesigma[v],w,nodesigma[w]);
                //if(border[w] != 0)      continue;  
		  //vis_path_info_w = &(vis_path_info[w]);
                    //omp_set_lock(&(lock[w]));
                    //if (nodedist[w] > (nodedist[v]+1 )) {
                if(marked == MYINFINITY || marked>dist_v){
                            if ( __sync_bool_compare_and_swap(&nodedist[w], marked, dist_v) ) {
                            //if(nodedist[w]==marked){
        //                      if(v==89272 && w==89283)
      //                           printf("\n[SwapSuccess]In Madduri's code : %d->%d, d[%d] = %d\n",v,w,w,nodedist[w]);

//#pragma omp atomic write
                          //  nodedist[w] = nodedist[v] + 1;
                           // omp_unset_lock(&(lock[w]));
/* FIRST */
#ifdef _OPENMP
//                            edgesigma[j] = sigma_v;
                            (void)__sync_fetch_and_add(&nodesigma[w], sigma_v-sigma_w);
//#else
                            //nodesigma[w] += sigma_v;
#endif
				child[num_edges_v+child_count[v]++] = w; 
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
                }
                else {
			(void)__sync_fetch_and_add(&nodesigma[w], sigma_v);
			child[num_edges_v+child_count[v]++] = w;
			}//TODO:Iresh
                        /*} else{ // this is the case where the distances have been set correct but the sigmas need to be added  
// ALMOST FIRST                 // Add here edgesigma calculation
                            omp_unset_lock(&(lock[w]));
                            edgesigma[j] = 0;*/
                }
                else {
			int d_val = nodedist[w];
                        if(d_val == dist_v)
                        {
				child[num_edges_v+child_count[v]++] = w;
//                              edgesigma[j] = sigma_v;
                            (void)__sync_fetch_and_add(&nodesigma[w], sigma_v);
//				if (p_vis_count == p_vis_size) {
                                /* Resize p_vis */
/*                                p_vis_temp = (vid_t *) calloc(2*p_vis_size,sizeof(vid_t));
                                //assert(p_vis_temp != NULL);
                                memcpy(p_vis_temp, p_vis, p_vis_size*sizeof(vid_t));
                                free(p_vis);
                                p_vis = p_vis_temp;
                                p_vis_size = 2*p_vis_size;
                            }
                            p_vis[p_vis_count++] = w;
*/                        }
		}
	//	if( w==45328 || w==45329 || w==1890994)
          //                      printf("New sigma[%d] = %d\n",w,nodesigma[w]);


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
          //      if(v==89272 && w==89283)
            //                    printf("\n[After]In Madduri's code : %d->%d, d[%d] = %d\n",v,w,w,nodedist[w]);

                    //}
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
if(tid == 0)
{
	pthread_mutex_lock(&(sem->mutex));
        while(sources[processed_count+1] == 0)       //Spin Lock. Waiting for more sources.
        {
                pthread_cond_wait(&(sem->cvar),&(sem->mutex));
        }
        pthread_mutex_unlock(&(sem->mutex));
}
int borderCount_cpu = border[hnodes];
#ifdef _OPENMP
#pragma omp barrier
#pragma omp for schedule(guided)
#endif
	for (i = 0; i < borderCount_cpu; i++)
	{
		v = borderNodes[i]; 
		num_edges_v = num_edges[v];
		for(j = 0; j < cpu_border_child[i].size(); j++)
		{
			child[num_edges_v+child_count[v]++] = cpu_border_child[i][j];	
		}
	}


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
       // free(visited_vertices);
       // free(num_visited);
        free(visited_counts);

    //return 0;

}

void cpu_backward(unsigned *cpu_visited_vertices, unsigned *cpu_num_visited, unsigned *child_count, int *child, unsigned *nodesigma, unsigned *psrc, unsigned *delta, unsigned *BC, unsigned phase_num)
{
	unsigned ph_start, ph_end,j,k,v,w,vis_path_info_v_child_count,vis_path_info_v_delta,vis_path_info_v_sigma,num_edges_v;
	unsigned * __restrict__ visited_vertices = cpu_visited_vertices;
	unsigned * __restrict__ num_visited = cpu_num_visited;
	unsigned * __restrict__ num_edges = psrc;
	unsigned * __restrict__ sigma = nodesigma;
	 #ifdef _OPENMP
                #pragma omp parallel        \
                private(j,v,w,k,vis_path_info_v_child_count,vis_path_info_v_delta,vis_path_info_v_sigma,num_edges_v)    \
                shared(num_visited,visited_vertices, num_edges, BC, delta, child, sigma, child_count, ph_start, ph_end, phase_num)	\
                default(none)
        #endif
{
	while (phase_num > 0)
	{
		ph_start = num_visited[phase_num];
		ph_end = num_visited[phase_num+1];
		#ifdef _OPENMP
		#pragma omp parallel for  schedule(guided)
		#endif
		for (j = ph_start; j < ph_end; j++)
		{
                	v = visited_vertices[j];
                	vis_path_info_v_child_count = child_count[v];
                	vis_path_info_v_delta = 0.0;
                	vis_path_info_v_sigma = sigma[v];
                	num_edges_v = num_edges[v];
                	for (k = 0; k < vis_path_info_v_child_count; k++)
			{
                    		w = child[num_edges_v+k];
                    		vis_path_info_v_delta += vis_path_info_v_sigma*(1.0+delta[w])/sigma[w];
                	}
                	delta[w] = vis_path_info_v_delta;
                	BC[v] += vis_path_info_v_delta;
		}

		phase_num--;

		#ifdef _OPENMP
		#pragma omp barrier
		#endif
	}
}
}


int betweenness_centrality_parallel(unsigned hnodes,unsigned hedges,unsigned *rowptrs,unsigned *columnindexes, unsigned *edgessrc,unsigned *noutgoing,int *sources,unsigned source_count,unsigned *nodedist, unsigned *nodesigma, unsigned *edgesigma,omp_lock_t *lock,int num_threads, Graph &graph) {
//betweenness_centrality_parallel(hnodes,hedges,psrc,edgessrcdst,edgesdstsrc,noutgoing,sources,source_count,hdist,nodesigma,edgesigma,lock,num_threads);

    /* stack of vertices in the order of non-decreasing 
     * distance from s. Also used to implicitly 
     * represent the BFS queue. 
     */
int flag = 0;
if(source_count>1)
	return 0;//betweenness_centrality_parallel_MultiS(hnodes,hedges,rowptrs,columnindexes, edgessrc,noutgoing,sources,source_count,nodedist, nodesigma,edgesigma,lock,num_threads,graph.partition.border);
else if(source_count==0)
{	
	source_count = 1;
	flag = 1;
}
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
	unsigned *border = graph.partition.border;
        // Allocate memory for the data structures
        visited_vertices = (vid_t *) calloc(2*hnodes , sizeof(vid_t));
        //vis_path_info = (S_info_t *) malloc(n * sizeof(S_info_t));
        assert(visited_vertices != NULL);
        //assert(vis_path_info != NULL);

        num_visited = (unsigned *) calloc(EXPECTED_GRAPH_DIAMETER,sizeof(unsigned));
        visited_counts = (unsigned *) calloc(NOSHARE(num_threads+1),sizeof(unsigned));

        assert(num_visited != NULL);
        assert(visited_counts != NULL);
//	int processed_count = 0,d_phase;
omp_set_num_threads(num_threads);

  //  fprintf(stderr, "Beginning betweenness computation from %ld "
  //          "randomly-chosen source vertices ...\n", num_src_vertices);
#ifdef _OPENMP    
#pragma omp parallel default(none) shared(visited_vertices,num_visited,visited_counts,hnodes,hedges,rowptrs,columnindexes,edgessrc,noutgoing,sources,source_count,nodedist,nodesigma,edgesigma,lock,num_threads,border,flag)
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
//	printf("Total Threads : %d",nthreads);
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
    p_vis_size = (2*n)/nthreads;
    p_vis = (vid_t *) malloc(p_vis_size*sizeof(vid_t));
    //assert(p_vis != NULL);
    p_vis_count = 0;


#ifdef _OPENMP
#pragma omp barrier
#endif

#ifdef _OPENMP
#pragma omp for
#endif
    for (i=0; i<(n); i++) {
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

        d_phase = 0;
        
#if DIAGNOSTIC
        if (tid == 0) 
            elapsed_time_part = timer();
#endif

#ifdef _OPENMP       
#pragma omp barrier
#endif

	while (num_visited[phase_num+1] - num_visited[phase_num] > 0 ) {

            if (phase_num >= MAX_NUM_PHASES - 1) {
//printf("ERROR : processed_count = %d, sources[%d] = %d, d_phase = %d\n",processed_count,processed_count,sources[processed_count],d_phase);
                if (tid == 0){
                    printf( "Error! Increase EXPECTED_GRAPH_DIAMETER "
                            "setting in graph.h. Exiting\n");
//	scanf("%d",&d_phase);
		}
//                assert(phase_num < MAX_NUM_PHASES - 1);
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
                //sigma_v = 0;
	//	if(border[w] == 0)
		sigma_v = nodesigma[v];
//		if(flag == 1 && border[v]!=0)
//			sigma_v = 0;
		
                num_edges_v = num_edges[v];

                for (j = num_edges_v; j < (num_edges_v + noutgoing[v]) ; j++) {

		    //if(edgessrc[j]!=v)continue;                
                    w = adj[j];
//			if(v==89272 && w==89283)
//				printf("\nIn Madduri's code : %d->%d, d[%d] = %d\n",v,w,v,nodedist[v]);
                    //vis_path_info_w = &(vis_path_info[w]);
                    marked = nodedist[w];
		    dist_v = nodedist[v]+1;
		    //omp_set_lock(&(lock[w]));
                    //if (nodedist[w] > (nodedist[v]+1 )) {
		if(marked == MYINFINITY || marked>dist_v+1){
			    if ( __sync_bool_compare_and_swap(&nodedist[w], marked, dist_v) ) {
			    //if(nodedist[w]==marked){
  //                            if(v==89272 && w==89283)
//				 printf("\n[SwapSuccess]In Madduri's code : %d->%d, d[%d] = %d\n",v,w,w,nodedist[w]);
	//		if(w==1890994 || w==45328 || w==45329)
          //                      printf("\nIn Madduri's code : %d->%d, sigma[%d] = %d, sigma[%d] = %d\n",v,w,v,nodesigma[v],w,nodesigma[w]);

//#pragma omp atomic write
                          //  nodedist[w] = nodedist[v] + 1;
			   // omp_unset_lock(&(lock[w]));
/* FIRST */
#ifdef _OPENMP
//			    edgesigma[j] = sigma_v;
                            (void)__sync_fetch_and_add(&nodesigma[w], sigma_v);
//#else
                            //nodesigma[w] += sigma_v;
#endif
                            /* Add w to local stack */
//			if((border[w] == 0 && flag == 0) || flag == 1)
			if(border[w] == 0)
			{
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
			}
		}
		else {

//			edgesigma[j] = sigma_v;
                            (void)__sync_fetch_and_add(&nodesigma[w], sigma_v);



				}//TODO:Iresh
                        /*} else{ // this is the case where the distances have been set correct but the sigmas need to be added  
// ALMOST FIRST 		// Add here edgesigma calculation
			    omp_unset_lock(&(lock[w]));
			    edgesigma[j] = 0;*/
		}
		else {
			int d_val = nodedist[w];
			if(d_val == dist_v)
			{
//				edgesigma[j] = sigma_v;
                            (void)__sync_fetch_and_add(&nodesigma[w], sigma_v);
			}	
			}
	//		if( w==1890994 || w==45328 || w==45329)
          //                      printf("New sigma[%d] = %d\n",w,nodesigma[w]);
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
//		if(v==89272 && w==89283)
  //                              printf("\n[After]In Madduri's code : %d->%d, d[%d] = %d\n",v,w,w,nodedist[w]);

                    //}
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
