#include "utils.h"
#include <stdlib.h>
#include <sys/time.h>

double timer() {

    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + tp.tv_usec*1e-6);
}
