/* 
 * use atomicInc to automatically wrap around.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <string.h>
#include "Structs.h"

#define MINCAPACITY	65535
#define MAXOVERFLOWS	1


Worklist_cpu::Worklist_cpu() {
	init();
}
void Worklist_cpu::init() {
	init(0);
}
void Worklist_cpu::init(unsigned initialcapacity) {
	start = alloc(1);
	end = alloc(1);
	capacity = alloc(1);
	setCapacity(initialcapacity);
	setInitialSize(0);

	items = NULL;
	if (initialcapacity) items = alloc(initialcapacity);
	noverflows = 0;
}
unsigned *Worklist_cpu::alloc(unsigned allocsize) {
	unsigned *ptr = (unsigned *) malloc (allocsize * sizeof(unsigned));
	if (ptr==NULL) {
		//CudaTest("allocating ptr failed");
		printf("%s(%d): Allocating %d failed on CPU.\n", __FILE__, __LINE__, allocsize);
		return NULL;
	}
	return ptr;
}
unsigned Worklist_cpu::getCapacity() {
	return *capacity;
}
unsigned Worklist_cpu::calculateSize(unsigned hstart, unsigned hend) {
	if (hend >= hstart) {
		//printf("Size:%d\n",(hend-hstart));
		return hend - hstart;
	}
	// circular queue.
	unsigned hcapacity = getCapacity();
	//printf("Size:%d\n",( hend + (hcapacity - hstart + 1)));
	return hend + (hcapacity - hstart + 1);
}
void Worklist_cpu::getStartEnd(unsigned &hstart, unsigned &hend) {
	//cudaMemcpy(&hstart, start, sizeof(hstart), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&hend, end, sizeof(hend), cudaMemcpyDeviceToHost);
	hstart = *start;
	hend = *end;
}
unsigned Worklist_cpu::getSize() {
	unsigned hstart, hend;
	getStartEnd(hstart, hend);
	if (hstart != 0) { printf("\tNOTICE CPU: hstart = %d.\n", hstart); }
	return calculateSize(hstart, hend);
}
void Worklist_cpu::setStartEnd(unsigned hstart, unsigned hend) {
	//cudaMemcpy(start, &hstart, sizeof(hstart), cudaMemcpyHostToDevice);
	//cudaMemcpy(end, &hend, sizeof(hend), cudaMemcpyHostToDevice);
	*start = hstart;
	*end = hend;
}
void Worklist_cpu::setCapacity(unsigned hcapacity) {
	//cudaMemcpy(capacity, &hcapacity, sizeof(hcapacity), cudaMemcpyHostToDevice);
	*capacity = hcapacity;
}
void Worklist_cpu::setInitialSize(unsigned hsize) {
	setStartEnd(0, 0);
}
void Worklist_cpu::setSize(unsigned hsize) {
	unsigned hcapacity = getCapacity();
	if (hsize > hcapacity) {
		printf("%s(%d): buffer overflow, setting size=%d, when capacity=%d.\n", __FILE__, __LINE__, hsize, hcapacity);
		return;
	}
	unsigned hstart, hend;
	getStartEnd(hstart, hend);
	if (hstart + hsize < hcapacity) {
		hend   = hstart + hsize;
	} else {
		hsize -= hcapacity - hstart;
		hend   = hsize;
	}
	setStartEnd(hstart, hend);
}
void Worklist_cpu::copyOldToNew(unsigned *olditems, unsigned *newitems, unsigned oldsize, unsigned oldcapacity) {
	unsigned mystart, myend;
	getStartEnd(mystart, myend);

	if (mystart < myend) {	// no wrap-around.
		//cudaMemcpy(newitems, olditems + mystart, oldsize * sizeof(unsigned), cudaMemcpyDeviceToDevice);
		memcpy(newitems, olditems + mystart, oldsize * sizeof(unsigned));
	} else {
		//cudaMemcpy(newitems, olditems + mystart, (oldcapacity - mystart) * sizeof(unsigned), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(newitems + (oldcapacity - mystart), olditems, myend * sizeof(unsigned), cudaMemcpyDeviceToDevice);
		memcpy(newitems, olditems + mystart, (oldcapacity - mystart) * sizeof(unsigned));
		memcpy(newitems + (oldcapacity - mystart), olditems, myend * sizeof(unsigned));
	}
}
unsigned Worklist_cpu::realloc(unsigned space) {
	unsigned hcapacity = getCapacity();
	//printf("hcapacity=%d\n",hcapacity);
	unsigned newcapacity = (space > MINCAPACITY ? space : MINCAPACITY);
	if (hcapacity == 0) {
		//printf("in hcapacity=0\n");
		setCapacity(newcapacity);
		items = alloc(newcapacity);
		if (items == NULL) {
			return 1;
		}
	} else {
		unsigned *itemsrealloc = alloc(newcapacity);
		if (itemsrealloc == NULL) {
			return 1;
		}
		unsigned oldsize = getSize();
		//cudaMemcpy(itemsrealloc, items, getSize() * sizeof(unsigned), cudaMemcpyDeviceToDevice);
		copyOldToNew(items, itemsrealloc, oldsize, hcapacity);
		dealloc();
		items = itemsrealloc;
		setCapacity(newcapacity);
		setStartEnd(0, oldsize);
	}
	//printf("\tworklist capacity set to %d.\n", getCapacity());
	return 0;
}
unsigned Worklist_cpu::freeSize() {
	return getCapacity() - getSize();
}
unsigned Worklist_cpu::ensureSpace(unsigned space) {
//	printf("ensurespace\n");
	if (freeSize() >= space) {
//	printf("in free space\n");
		return 0;
	}
	
	realloc(space);
	// assert freeSize() >= space.
	return 1;
}
unsigned Worklist_cpu::dealloc() {
	 printf("\nDeallocating Worklist CPU");
	//cudaFree(items);
	free(items);
	setInitialSize(0);
	return 0;
}
Worklist_cpu::~Worklist_cpu() {
	 //dealloc();
	// init();
}
unsigned Worklist_cpu::pushRange(unsigned *copyfrom, unsigned nitems) {
	if (copyfrom == NULL || nitems == 0) return 0;
	//printf("hi\n");
	unsigned lcap = *capacity, offset;
#pragma omp atomic capture
	{
	offset = *end;
	*end += nitems;
	}
	if (offset >= lcap) {	// overflow.
#pragma omp atomic update
		*end -=nitems;
		//atomicSub(end, nitems);
		//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
		//printf("%s(%d): thread %d: buffer overflow, increase capacity beyond %d.\n", __FILE__, __LINE__, id, *capacity);
		return 1;
	}
	for (unsigned ii = 0; ii < nitems; ++ii) {
		items[(offset + ii) % lcap] = copyfrom[ii];
	}
//	printf("added %d items to WL\n",nitems);
	return 0;
}
unsigned Worklist_cpu::pushRangeEdges(unsigned startindex, unsigned nitems) {
	if (nitems == 0) return 0;
	//printf("hi\n");
	unsigned lcap = *capacity;
	unsigned offset;
#pragma omp atomic capture
	{
	offset = *end;
	*end += nitems;
	}
	//atomicAdd(end, nitems);
	if ((offset>=lcap)||((offset+nitems)>=lcap)) {	// overflow.
#pragma omp atomic update
		*end -= nitems;
		//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
		//printf("%s(%d): thread %d: buffer overflow, increase capacity beyond %d.\n", __FILE__, __LINE__, id, *capacity);
		return 1;
	}
	for (unsigned ii = 0; ii < nitems; ++ii) {
		items[(offset + ii) % lcap] = startindex+ii;
	}
	
//	printf("added %d items to WL\n",nitems);
	return 0;
}

unsigned Worklist_cpu::push(unsigned work) {
	return pushRange(&work, 1);
}
unsigned Worklist_cpu::popRange(unsigned *copyto, unsigned nitems) {
	unsigned currsize = count();
	if (currsize < nitems) {
		// popping fewer than requested number of items.
		nitems = currsize;
	}
	//unsigned offset = atomicSub(size, nitems);
	//unsigned offset = atomicCAS(size, currsize, currsize - nitems);
	//unsigned offset = atomicExch(size, currsize - nitems);
	unsigned offset = 0;
	unsigned lcap = *capacity;
	if (nitems) {
		if (*start + nitems < lcap) {
#pragma omp atomic capture
	{
			offset = *start;
			*start += nitems;
	}
		//	offset = atomicAdd(start, nitems);
		} else {
#pragma omp atomic capture
	{
			offset = *start;
			*start +=nitems - lcap;
	}
		}
	}
	// copy nitems starting from offset.
	for (unsigned ii = 0; ii < nitems; ++ii) {
		copyto[ii] = items[(offset + ii) % lcap];
	}
	return nitems;
}
unsigned Worklist_cpu::pop(unsigned &work) {
	return popRange(&work, 1);
}
void Worklist_cpu::pushRangeHost(unsigned *copyfrom, unsigned nitems) {
//	unsigned *t=(unsigned *)malloc(sizeof(unsigned));
	ensureSpace(nitems);
	unsigned hsize = getSize();
	//printf("size in pushRH:%d\n",hsize);
//	printf("work in pushRH:%d\n",(*copyfrom));
	//printf("work in pushRH:%d\n",sizeof(copyfrom));
	
	//cudaMemcpy(items + hsize , copyfrom, nitems * sizeof(unsigned), cudaMemcpyHostToDevice);
	memcpy(items + hsize , copyfrom, nitems * sizeof(unsigned));
//	for(int i=0;i<nitems;i++)
//		printf("element in pushRH:%d\n",t[i]);
	hsize += nitems;
	//printf("hsize after in pushRH:%d\n",hsize);
	setSize(hsize);
//	free(t);
}
void Worklist_cpu::pushHost(unsigned work) {
	pushRangeHost(&work, 1);
}
void Worklist_cpu::clear() {	// should be invoked by a single thread.
	*end = *start;
}
void Worklist_cpu::clearHost() {
	setSize(0);
}

void Worklist_cpu::myItems(unsigned &mystart, unsigned &myend) {
	unsigned id = omp_get_thread_num();
	unsigned hsize = count();
	unsigned nthreads = omp_get_num_threads();

	if (nthreads > hsize) {
		// each thread gets max 1 item.
		if (id < hsize) {
			mystart = id; 
			myend = mystart + 1;	// one item.
		} else {
			mystart = id; 
			myend = mystart;	// no items.
		}
	} else {
		unsigned nitemsperthread = hsize / nthreads;	// every thread gets at least these many.
		unsigned nitemsremaining = hsize % nthreads;	// initial some threads get one more.
		mystart = id * nitemsperthread; 
		myend = mystart + nitemsperthread;

		if (id < nitemsremaining) {
			mystart += id;			// initial few threads get one extra item, due to which
			myend   += id + 1;		// their assignment changes.
		} else {
			mystart += nitemsremaining;	// the others don't get anything extra, but
			myend   += nitemsremaining;	// their assignment changes.
		}
	}
}
unsigned Worklist_cpu::getItem(unsigned at) {
	unsigned hsize = count();
	//printf("hsize in get item:%d\n",hsize);
	return getItemWithin(at, hsize);
}
unsigned Worklist_cpu::getItemWithin(unsigned at, unsigned hsize) {
	if (at < hsize) {
		return items[at];
	}
	unsigned id = omp_get_thread_num();
	printf("%s(%d): thread %d: buffer overflow, extracting %d when buffer size is %d.\n", __FILE__, __LINE__, id, at, hsize);
	return 1;
}
unsigned Worklist_cpu::count() {
	if (*end >= *start) {
		return *end - *start;
	} else {
		return *end + (*capacity - *start + 1);
	}
}

#define SWAPDEV(a, b)	{ unsigned tmp = a; a = b; b = tmp; }
/*
void compress(Worklist wl, unsigned wlsize, unsigned sentinel) {
	unsigned id = omp_get_thread_num();
	unsigned shmem[MAXSHAREDUINT];

	// copy my elements to my ids in shmem.
	unsigned wlstart = MAXSHAREDUINT * blockIdx.x + SHAREDPERTHREAD * threadIdx.x;
	unsigned shstart = SHAREDPERTHREAD * threadIdx.x;

	for (unsigned ii = 0; ii < SHAREDPERTHREAD; ++ii) {
		if (wlstart + ii < wlsize && shstart + ii < MAXSHAREDUINT) {
			shmem[shstart + ii] = wl.getItemWithin(wlstart + ii, wlsize);
		}
	}
#pragma omp barrier
	
	// sort in shmem.
	for (unsigned s = blockDim.x / 2; s; s >>= 1) {
		if (threadIdx.x < s) {
			if (shmem[threadIdx.x] > shmem[threadIdx.x + s]) {
				SWAPDEV(shmem[threadIdx.x], shmem[threadIdx.x + s]);
			}
		}
		__syncthreads();
	}
#pragma omp barrier

	// uniq in shmem.
	// TODO: find out how to do uniq in a hierarchical manner.
	unsigned lastindex = 0;
	if (id == 0) {
		for (unsigned ii = 1; ii < MAXSHAREDUINT; ++ii) {
			if (shmem[ii] != shmem[lastindex]) {
				shmem[++lastindex] = shmem[ii];
			} else {
				shmem[ii] = sentinel;
			}
		}
	}
#pragma omp barrier

	// copy back elements to the worklist.
	for (unsigned ii = 0; ii < SHAREDPERTHREAD; ++ii) {
		if (wlstart + ii < wlsize) {
			//shmem[shstart + ii] = getItem(wlstart + ii);
			wl.items[wlstart + ii] = shmem[shstart + ii];
		}
	}
#pragma omp barrier

	// update worklist indices.
	if (id == 0) {
		*wl.start = 0;
		*wl.end = lastindex + 1;
	}
}
void Worklist::compressHost(unsigned wlsize, unsigned sentinel) {
	unsigned nblocks = (wlsize + MAXBLOCKSIZE - 1) / MAXBLOCKSIZE;
	compress<<<nblocks, MAXBLOCKSIZE>>>(*this, wlsize, sentinel);
	CudaTest("compress failed");
}
*/
void printWorklist(Worklist_cpu wl) {
	unsigned start, end;
	start = *wl.start;
	end = *wl.end;
	printf("\t");
	for (unsigned ii = start; ii < end; ++ii) {
		printf("%d,", wl.getItem(ii));
	}
	printf("\n");
}
void Worklist_cpu::printHost() {
#pragma omp single
	printWorklist(*this);
}
void appendWorklist(Worklist_cpu dst, Worklist_cpu src, unsigned dstsize) {
	unsigned start, end;
	src.myItems(start, end);

	for (unsigned ii = start; ii < end; ++ii) {
		dst.items[dstsize + ii] = src.items[ii];
	}
}
unsigned Worklist_cpu::appendHost(Worklist_cpu *otherlist) {
	unsigned otherlistsize = otherlist->getSize();
	appendWorklist(*this, *otherlist, getSize());
	unsigned hstart, hend;
	getStartEnd(hstart, hend);
	setStartEnd(hstart, hend + otherlistsize);

	return hend + otherlistsize;
}
