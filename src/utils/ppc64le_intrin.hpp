#pragma once

// https://gcc.gnu.org/onlinedocs/gcc/Simple-Constraints.html#Simple-Constraints

inline void flush(void * p) {
#ifdef __powerpc__

  /*
  PowerISA_V2.07B p. 773
  dcbf RA,RB,L
  
  effective address is RAI0 + RB
  this mnemonic has L=0, which is through all cache levels
  write block to storage and mark as invalid in all processors
  */

  asm volatile ( "dcbf %1,%0"
    : // no outputs
    : "r"(0), "r"(p)
    : // no clobbers
  );

#else
#warning "not implemented"
(void) p;
#endif
}

inline void store(void * p) {
#ifdef __powerpc__

  /*
  PowerISA_V2.07B p. 773
  dcbst RA,RB
  write block to storage and mark as modified
  
  effective address is RAI0 + RB
  */

  asm volatile ( "dcbst %1,%0"
    : // no outputs
    : "r"(0), "r"(p)
    : // no clobbers
  );

#else
#warning "not implemented"
(void) p;
#endif
}


