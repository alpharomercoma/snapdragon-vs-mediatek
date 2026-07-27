// trainstep_bench.c — realistic transformer-layer TRAINING-STEP throughput.
// Port of ../../dimensity-9500s/gpu-matmul/vk_trainstep_bench.c for the
// Adreno 830 (OpenCL or Vulkan link, backend picked via the ggml registry).
//
// Measures the matmul FLOPs of one transformer layer's forward + backward:
//   QKV proj, output proj, FFN up, FFN down  (fwd)  + dX and dW for each (bwd).
// That is 3x the forward matmul FLOPs = 72*D^2*T per layer per step (the "6*N*T"
// training-compute rule, N = 12*D^2 params/layer). Reports achieved GFLOP/s.
//
// Difference from the original: dX/dW are kept alive as graph roots instead of
// being reduced through ggml_sum — the ggml OpenCL backend has no SUM kernel,
// and this program computes on a single backend with no CPU-fallback scheduler.
// The measured quantity (12 matmuls of fwd+bwd) is unchanged.
//
// Env:  BACKEND=gpu|cpu   PREC=f16|f32    D (d_model) / T (tokens) via -DD= -DT=.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef D
#define D 1024
#endif
#ifndef T
#define T 2048
#endif
#define WARMUP 2
#define ITERS  8

static int f32;
static double nowms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}

// one linear op of a training step: forward Y=W^T X, plus the two backward
// matmuls (dX and dW) of equal FLOP count, all expanded as graph roots.
static struct ggml_tensor* lin(struct ggml_context*c, struct ggml_cgraph*g,
                               struct ggml_tensor*W, struct ggml_tensor*X){
    struct ggml_tensor*Y=ggml_mul_mat(c,W,X);              // fwd  [N,T]
    if(f32) ggml_mul_mat_set_prec(Y,GGML_PREC_F32);
    struct ggml_tensor*dX=ggml_mul_mat(c,ggml_cont(c,ggml_transpose(c,W)),Y); // bwd dX [K,T]
    if(f32) ggml_mul_mat_set_prec(dX,GGML_PREC_F32);
    struct ggml_tensor*dW=ggml_mul_mat(c,ggml_cont(c,ggml_transpose(c,X)),
                                         ggml_cont(c,ggml_transpose(c,Y)));   // bwd dW [K,N]
    if(f32) ggml_mul_mat_set_prec(dW,GGML_PREC_F32);
    ggml_build_forward_expand(g,dX);
    ggml_build_forward_expand(g,dW);
    return Y;
}

int main(void){
    setvbuf(stdout, NULL, _IONBF, 0);
    const char*b=getenv("BACKEND"); int cpu=b&&!strcmp(b,"cpu");
    const char*p=getenv("PREC"); f32=p&&!strcmp(p,"f32");
    ggml_backend_t be;
    if (cpu) {
        be = ggml_backend_cpu_init();
    } else {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); // Adreno = integrated (UMA)
        be = dev ? ggml_backend_dev_init(dev, NULL) : NULL;
    }
    if(!be){printf("no backend\n");return 1;}
    printf("backend=%s prec=%s  D=%d T=%d\n",ggml_backend_name(be),f32?"f32":"f16",D,T);

    struct ggml_init_params ip={(size_t)64*1024*1024,NULL,true};
    struct ggml_context*c=ggml_init(ip);
    struct ggml_tensor*Wqkv=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,3*D);
    struct ggml_tensor*Wo  =ggml_new_tensor_2d(c,GGML_TYPE_F32,D,D);
    struct ggml_tensor*W1  =ggml_new_tensor_2d(c,GGML_TYPE_F32,D,4*D);
    struct ggml_tensor*W2  =ggml_new_tensor_2d(c,GGML_TYPE_F32,4*D,D);
    struct ggml_tensor*X   =ggml_new_tensor_2d(c,GGML_TYPE_F32,D,T);
    struct ggml_tensor*Xf  =ggml_new_tensor_2d(c,GGML_TYPE_F32,4*D,T);
    ggml_set_input(X); ggml_set_input(Xf);
    ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,be);
    (void)buf;
    size_t big=(size_t)4*D*(T>D?T:D); float*t=malloc(big*4);
    for(size_t i=0;i<big;i++) t[i]=((float)((i*2654435761u)>>20&1023)/512.0f-1.0f)*0.05f;
    ggml_backend_tensor_set(Wqkv,t,0,(size_t)D*3*D*4);
    ggml_backend_tensor_set(Wo,t,0,(size_t)D*D*4);
    ggml_backend_tensor_set(W1,t,0,(size_t)D*4*D*4);
    ggml_backend_tensor_set(W2,t,0,(size_t)4*D*D*4);
    ggml_backend_tensor_set(X,t,0,(size_t)D*T*4);
    ggml_backend_tensor_set(Xf,t,0,(size_t)4*D*T*4);
    free(t);

    // training-step graph: 4 linears, each fwd+dX+dW (12 matmuls total)
    struct ggml_cgraph*g=ggml_new_graph_custom(c,4096,false);
    lin(c,g,Wqkv,X);
    lin(c,g,Wo,  X);
    lin(c,g,W1,  X);
    lin(c,g,W2,  Xf);

    ggml_gallocr_t al=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(al,g);
    for(int i=0;i<WARMUP;i++) ggml_backend_graph_compute(be,g);
    ggml_backend_synchronize(be);
    double t0=nowms();
    for(int i=0;i<ITERS;i++) ggml_backend_graph_compute(be,g);
    ggml_backend_synchronize(be);
    double ms=(nowms()-t0)/ITERS;

    double flops=72.0*(double)D*D*T;          // per layer per step
    double gf=flops/(ms*1e6);
    double tok_per_s_1L=T/(ms/1e3);
    printf("step(1 layer) = %.1f ms   %.1f GFLOP/s   %.0f tok/s (1-layer)\n", ms, gf, tok_per_s_1L);
    printf("achieved training throughput: %.1f GFLOP/s\n", gf);
    return 0;
}
