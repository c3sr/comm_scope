# OLCF Summit

```
bsub -Is -W 0:10 -nnodes 1 -P csc465 $SHELL
```

```
cmake .. \
-DCMAKE_CUDA_ARCHITECTURES=70 \
-DSCOPE_USE_CUDA=ON \
-DSCOPE_USE_NVTX=ON
```

Example job submission scripts are in `scripts/summit`.

You may ignore messages like the following, where libscope fails to control CPU turbo and the governor.
On managed systems like Summit, this is not necessary.