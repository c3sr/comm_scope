#pragma once

#define SMALL_ARGS() Ranges({{128,256}, {128, 256}, {128,256}})
#define TINY_ARGS() \
Args({128,128,128})\
->Args({512,512,8}) /*same size as 128x128x128*/ \
->Args({256,256,256})\
->Args({512,512,64})/*same size as 256x256x256*/ \
->Args({512,512,512})\
->Args({4,4,512})\
->Args({4,512,4})\
->Args({512,4,4})\
->Args({512,512,4})\
->Args({512,4,512})\
->Args({4,512,512})\
->Args({32,512,512}) /*32B L2 cache line*/ \
->Args({128,512,512}) /* 128B L1 cache line */
