/* 
 * Routines to emulate xmt's full/empty bit operations with "lock cmpxchng:
 * and xmt's atomic fetch and add with "lock xadd"
 */ 

#include <stdint.h>
#include <assert.h>

#ifndef EMS_X86_64_H
#define EMS_X86_64_H

#define TAKEN(bool) __builtin_expect((bool),1)

static __inline int
pause(void *mem_loc) {
    __asm__ __volatile__ ("pause");
    return 1;
}

/* 
 * full empty bits operations:
 * readFE, readFF, readXX, writeEF, writeFF, writeXF, reset and isFull
 */

 /* int64_t */
typedef volatile int64_t sync_int64_t;

 /* Max Neg Int aka Integer Indefinite */
static const  int64_t empty_val_I8 = (int64_t)0x8000000000000000ULL;

 /* readFE -- wait until full, leave empty, return value */

static __inline int64_t
readFE_I8(sync_int64_t *mem_loc) {
    int64_t ret_val;
    __builtin_prefetch((void *)mem_loc,1,3);
    while(1) {
        while( pause((void *)mem_loc) ) /* while empty spin until full */
            if(  TAKEN(empty_val_I8 != (ret_val = *mem_loc)) )
                break;
        if ( TAKEN(__sync_bool_compare_and_swap(mem_loc,ret_val,empty_val_I8)) )
            break;
        /* if compare-swap is not successful then wait until full again */
    }
    return ret_val;
}

 /* readFF -- wait until full, leave full, return value */

static __inline int64_t
readFF_I8(sync_int64_t *mem_loc) {
    int64_t ret_val;
    __builtin_prefetch((void *)mem_loc,1,3);
    while( pause((void *)mem_loc) ) /* while empty spin until full */
        if ( TAKEN(empty_val_I8 != (ret_val = *mem_loc)) )
             break;
    return ret_val;
}

/* readXX -- non-blocking, return value */

static __inline int64_t
readXX_I8(sync_int64_t *mem_loc)
{ return *mem_loc; }

 /* writeEF -- wait until empty, leave full, set value */

static __inline void
writeEF_I8(sync_int64_t *mem_loc, int64_t mem_val) {
    assert(empty_val_I8 != mem_val);
    __builtin_prefetch((void *)mem_loc,1,3);
    while( pause((void *)mem_loc) ) /* while full spin until empty */
        if( TAKEN(__sync_bool_compare_and_swap(mem_loc,empty_val_I8,mem_val)) )
             break;
        /* if compare-swap is not successful then wait until empty again */
    return;
}

/* writeFF -- wait until full, leave full, set value */

static __inline void
writeFF_I8(sync_int64_t *mem_loc, int64_t mem_val) {
    int64_t ret_val;
    assert(empty_val_I8 != mem_val);
    __builtin_prefetch((void *)mem_loc,1,3);
    while(1) {
        while( pause((void *)mem_loc) ) /* while empty spin until full */
            if ( TAKEN(empty_val_I8 != (ret_val = *mem_loc)) )
                break;
        if ( TAKEN(__sync_bool_compare_and_swap(mem_loc,ret_val,mem_val)) )
            break;
        /* if compare-swap is not successful then wait until full again */
    }
    return;
}

/* writeXF -- non-blocking, leave full, set value */

static __inline void
writeXF_I8(sync_int64_t *mem_loc, int64_t mem_val)
{ assert(empty_val_I8 != mem_val); *mem_loc = mem_val; return; }

/* reset -- non-blocking, leave empty, reset value */

static __inline void
reset_I8(sync_int64_t *mem_loc)
{ *mem_loc = empty_val_I8; return; }

/* isFull -- non-blocking, return true if full else false */

static __inline int
isFull_I8(sync_int64_t *mem_loc)
{ return (empty_val_I8 != *mem_loc); } 

 /* int32_t */
typedef volatile int32_t sync_int32_t;


 /* uint32_t */
typedef volatile uint32_t sync_uint32_t;

 /* Max Neg Int aka Integer Indefinite */
static const  int32_t empty_val_I4  = (int32_t)0x8000000UL;

static const uint32_t empty_val_UI4 = (uint32_t)0xFFFFFFFFUL;

 /* readFE -- wait until full, leave empty, return value */

static __inline int32_t
readFE_I4(sync_int32_t *mem_loc) {
    int32_t ret_val;
    __builtin_prefetch((void *)mem_loc,1,3);
    while(1) {
        while( pause((void *)mem_loc) ) /* while empty spin until full */
            if(  TAKEN(empty_val_I4 != (ret_val = *mem_loc)) )
                break;
        if ( TAKEN(__sync_bool_compare_and_swap(mem_loc,ret_val,empty_val_I4)) )
            break;
        /* if compare-swap is not successful then wait until full again */
    }
    return ret_val;
}

static __inline uint32_t
readFE_UI4(sync_uint32_t *mem_loc) {
    uint32_t ret_val;
    __builtin_prefetch((void *)mem_loc,1,3);
    while(1) {
        while( pause((void *)mem_loc) ) /* while empty spin until full */
            if(  TAKEN(empty_val_UI4 != (ret_val = *mem_loc)) )
                break;
        if ( TAKEN(__sync_bool_compare_and_swap(mem_loc,ret_val,empty_val_UI4)) )
            break;
        /* if compare-swap is not successful then wait until full again */
    }
    return ret_val;
}

 /* readFF -- wait until full, leave full, return value */

static __inline int32_t
readFF_I4(sync_int32_t *mem_loc) {
    int32_t ret_val;
    __builtin_prefetch((void *)mem_loc,1,3);
    while( pause((void *)mem_loc) ) /* while empty spin until full */
        if ( TAKEN(empty_val_I4 != (ret_val = *mem_loc)) )
             break;
    return ret_val;
}

/* readXX -- non-blocking, return value */

static __inline int32_t
readXX_I4(sync_int32_t *mem_loc)
{ return *mem_loc; }

 /* writeEF -- wait until empty, leave full, set value */

static __inline void
writeEF_I4(sync_int32_t *mem_loc, int32_t mem_val) {
    assert(empty_val_I4 != mem_val);
    __builtin_prefetch((void *)mem_loc,1,3);
    while( pause((void *)mem_loc) ) /* while full spin until empty */
        if( TAKEN(__sync_bool_compare_and_swap(mem_loc,empty_val_I4,mem_val)) )
             break;
        /* if compare-swap is not successful then wait until empty again */
    return;
}

static __inline void
writeEF_UI4(sync_uint32_t *mem_loc, uint32_t mem_val) {
    assert(empty_val_UI4 != mem_val);
    __builtin_prefetch((void *)mem_loc,1,3);
    while( pause((void *)mem_loc) ) /* while full spin until empty */
        if( TAKEN(__sync_bool_compare_and_swap(mem_loc,empty_val_UI4,mem_val)) )
             break;
        /* if compare-swap is not successful then wait until empty again */
    return;
}

/* writeFF -- wait until full, leave full, set value */

static __inline void
writeFF_I4(sync_int32_t *mem_loc, int32_t mem_val) {
    int32_t ret_val;
    assert(empty_val_I4 != mem_val);
    __builtin_prefetch((void *)mem_loc,1,3);
    while(1) {
        while( pause((void *)mem_loc) ) /* while empty spin until full */
            if ( TAKEN(empty_val_I4 != (ret_val = *mem_loc)) )
                break;
        if ( TAKEN(__sync_bool_compare_and_swap(mem_loc,ret_val,mem_val)) )
            break;
        /* if compare-swap is not successful then wait until full again */
    }
    return;
}

/* writeXF -- non-blocking, leave full, set value */

static __inline void
writeXF_I4(sync_int32_t *mem_loc, int32_t mem_val)
{ assert(empty_val_I4 != mem_val); *mem_loc = mem_val; return; }

/* reset -- non-blocking, leave empty, reset value */

static __inline void
reset_I4(sync_int32_t *mem_loc)
{ *mem_loc = empty_val_I4; return; }

/* isFull -- non-blocking, return true if full else false */

static __inline int
isFull_I4(sync_int32_t *mem_loc)
{ return (empty_val_I4 != *mem_loc); } 

#endif // EMS_X86_64_H
