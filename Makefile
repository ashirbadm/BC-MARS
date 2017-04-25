METISDIR = $(HOME)/metis
INCLUDEDIR = $(METISDIR)/include
CFLAGS = -O3 -ccbin=/opt/centos/devtoolset-1.1/root/usr/bin/c++ -Xcompiler -fopenmp -arch=sm_20 -I$(INCLUDEDIR) -I/home/ashirbad/softwares/cub-1.4.1/ -DCPU_CODE -w
#CFLAGS = -O3 -arch=sm_35 -Xcompiler -fopenmp -I$(INCLUDEDIR) -I/home/ashirbad/softwares/cub-1.4.1/ -DCPU_CODE -w
#CFLAGS = -O3 -arch=sm_35 -Xcompiler -fopenmp -Xcompiler -ftree-vectorize -Xcompiler -ftree-vectorizer-verbose=0 -I$(INCLUDEDIR) -I/home/ashirbad/softwares/cub-1.4.1/ -DCPU_CODE -w
CC1FLAGS = -O3 -fopenmp -I/usr/local/cuda/include -DCPU_CODE
LDFLAGS = -L$(METISDIR)/lib -lmetis -lgomp -L/usr/local/cuda/lib64 -lcudart -lpthread -lpatoh -lm #-L/home/ashirbad/softwares/electric-fence-2.1.13 -lefence
CC = nvcc
CC1 = /opt/centos/devtoolset-1.1/root/usr/bin/c++
OBJ= *.o
LD = 

all: main partition_metis.exe partition_patoh.exe

main: bfs_worklistc.o bc.o cpu_nodebased.o backward_phase.o utils.o 
	$(CC) $(LDFLAGS) $(OBJ) -o BC

#main: bfs_ddV3.o bc.o cpu_nodebased.o backward_phase.o utils.o
#	$(CC) $(LDFLAGS) $(OBJ) -o bc

bc.o: bc.cu
	$(CC) -c $(CFLAGS) bc.cu -o bc.o

bfs_worklistc.o: bfs_worklistc.cu
	$(CC) -c $(CFLAGS) -I/home/ashirbad/softwares/cub-1.4.1/ bfs_worklistc.cu -o bfs_worklistc.o

bfs_ddV3.o: bfs_ddV3.cu
	$(CC) -c $(CFLAGS) bfs_ddV3.cu -o bfs_ddV3.o

cpu_ddV3.o: cpu_ddV3.cu
	$(CC) -c $(CFLAGS) cpu_ddV3.cu -o cpu_ddV3.o

backward_phase.o: backward_phase.cu
	$(CC) -c $(CFLAGS) backward_phase.cu -o backward_phase.o

utils.o:
	g++ -Wall -O3 -c utils.c

cpu_nodebased.o: cpu_nodebased.cu
	$(CC) $(CFLAGS) -c cpu_nodebased.cu

partition_patoh.exe: partition_patoh.cu
	$(CC) $(CFLAGS)  partition_patoh.cu -o partition_patoh.exe -lpatoh

partition_metis.o: partition_metis.cu
	$(CC) $(CFLAGS) -c partition_metis.cu -o partition_metis.o

partition_metis.exe: partition_metis.o bfs_worklistc.o cpu_nodebased.o utils.o
	$(CC) $(LDFLAGS) partition_metis.o bfs_worklistc.o cpu_nodebased.o utils.o -o partition_metis.exe
clean:
	rm -f *.o BC bc *.exe

BUILD_SUBDIRS = .

TAGS_SUBDIRS = $(BUILD_SUBDIRS)
TAGS_SOURCES = find $(TAGS_SUBDIRS) -name '*.h' -o -name '*.cu' -o -name '*.cuh' -o -name '*.cpp'

tags::
	ctags --langmap=c++:+.cu -L cscope.files

cscope.files::
	$(TAGS_SOURCES) > cscope.files

cscope:: cscope.files
	cscope -b -q -k
