#include "patoh.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
int patoh_partition(char *filename,int *partweights,float *targetweights)
{
PaToH_Parameters args;
int             _c, _n, _nconst, *cwghts, *nwghts, 
                *xpins, *pins, *partvec,cut; 

//float *targetweights;
struct timeval tv;
time_t curtime, prevtime;
gettimeofday(&tv, NULL);
prevtime = tv.tv_sec+(tv.tv_usec*1.0e-6);

PaToH_Read_Hypergraph(filename, &_c, &_n, &_nconst, &cwghts, 
                      &nwghts, &xpins, &pins);

 
prevtime = tv.tv_usec;
printf("Hypergraph %10s -- #Cells=%6d  #Nets=%6d  #Pins=%8d #Const=%2d\n",
       filename, _c, _n, xpins[_n], _nconst);

/* Modifying parameters for testing*/
args.MemMul_Pins=4;
//args.MemMul_CellNet=2;
PaToH_Initialize_Parameters(&args, PATOH_CUTPART, 
                            PATOH_SUGPARAM_SPEED);

args._k = 2; // atoi(argv[2]);
partvec = (int *) malloc(_c*sizeof(int));
PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, xpins, pins);
PaToH_Part(&args, _c, _n, _nconst, 0, cwghts, nwghts,
		           xpins, pins, targetweights, partvec, partweights, &cut);

printf("%d-way cutsize is: %d\n", args._k, cut);
gettimeofday(&tv, NULL);
curtime = tv.tv_sec+tv.tv_usec*1.0e-6;
printf("Patoh time: %.3lf msec \n",(curtime-prevtime)*1000);
std::ofstream outfile;
outfile.open("partitioninfo.txt");
outfile<<cut<<std::endl;
for(unsigned ii=0;ii<_c;ii++)
	outfile<<partvec[ii]<<std::endl;
outfile.close();

free(cwghts);      free(nwghts);
free(xpins);       free(pins);
//free(partweights); 
//free(targetweights); 
free(partvec);
PaToH_Free();
return 0;
}

int main(int argc, char *argv[]){
	if(argc < 1)
	{
		printf("Usage: %s <graph>\n", argv[0]);
		exit(1);
	}
	int *partweights = (int *)malloc(2*sizeof(int));
	float *targetweights = (float *)malloc(2*sizeof(float));
	char *filename = argv[1];
	std::ifstream cfile("ratio.txt");
	cfile>>targetweights[0]>>targetweights[1];
	patoh_partition(filename,partweights,targetweights);
	free(partweights);
	free(targetweights);
	return 0;
}
