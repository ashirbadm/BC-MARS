#!/bin/bash
#PBS -N BC_job
#PBS -j oe
#PBS -l nodes=1:ppn=32
#PBS -V
#PBS -o out-pbs.log
#PBS -e err-pbs.log
#PBS -d /home/ashirbad/betweenness_centrality/utexas_modified/Mar_2016/FINAL_V9/

./BC ~/Partitioning/Patoh/graphs/delaunay_n25.edges 
