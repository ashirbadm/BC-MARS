#!/bin/bash
cd /home/ashirbad/betweenness_centrality/utexas_modified/Mar_2016/FINAL_V5
./BC ~/Partitioning/Patoh/graphs/USA-road-d.USA.edges > output_USA_USA
./BC ~/Partitioning/Patoh/graphs/USA-road-d.CTR.edges > output_USA_CTR
./BC ~/Partitioning/Patoh/graphs/delaunay_n25.edges > output_delaunay_n25
./BC ~/Partitioning/Patoh/graphs/nlpkkt240.edges > output_nlpkkt240
./BC ~/Partitioning/Patoh/graphs/delaunay_n24.edges > output_delaunay_n24
./BC ~/Partitioning/Patoh/graphs/wb-edu.edges > output_wb-edu 
./BC ~/Partitioning/Patoh/graphs/europe.osm.edges > output_europe
