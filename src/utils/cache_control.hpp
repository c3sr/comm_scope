#pragma once

// https://gcc.gnu.org/onlinedocs/gcc/Simple-Constraints.html#Simple-Constraints

inline void flush(void * p) {
#ifdef __powerpc__

  /*
  PowerISA_V2.07B p. 773
  dcbf RA,RB,L
  
  effective address is RA|0 + RB
  this mnemonic has L=0, which is through all cache levels
  write block to storage and mark as invalid in all processors
  */

  asm volatile ( "dcbf %0,%1"
    : // no outputs
    : "r"(0), "r"(p)
    : // no clobbers
  );

#else
  #warning "flush not implemented"
  (void) p;
#endif
}


inline void flush_all(void *p, const size_t n) {
  for (size_t i = 0; i < n; i += 32) {
    char *c = static_cast<char*>(p);
    flush(&c[i]);
  }
}


inline void store(void * p) {
#ifdef __powerpc__

  /*
  PowerISA_V2.07B p. 773
  dcbst RA,RB
  if block is modified, write block to storage and mark as clean
  
  effective address is RA|0 + RB
  */

  asm volatile ( "dcbst %0,%1"
    : // no outputs
    : "r"(0), "r"(p)
    : // no clobbers
  );

#else
#warning "store not implemented"
(void) p;
#endif
}


