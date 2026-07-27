// vk_trainstep_bench.c — realistic transformer-layer TRAINING-STEP throughput.
// Measures the matmul FLOPs of one transformer layer's forward + backward:
//   QKV proj, output proj, FFN up, FFN down  (fwd)  + dX and dW for each (bwd).
// That is 3x the forward matmul FLOPs = 72*D^2*T per layer per step (the "6*N*T"
// training-compute rule, N = 12*D^2 params/layer). Reports achieved GFLOP/s, from
// which tokens/sec for any model size = achieved_FLOPs / (6 * N_params).
//
// Env:  BACKEND=vk|cpu   PREC=f16|f32   GGML_VK_DISABLE_COOPMAT=1
//   D (d_model) and T (tokens = batch*seq) via -DD= / -DT=.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
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

// one linear op of a training step: forward Y=W^T X, plus the two backward matmuls
// (dX and dW) of equal FLOP count. W:[K,N] X:[K,T] -> add 3 matmuls to the graph.
static struct ggml_tensor* lin(struct ggml_context*c, struct ggml_tensor*W, struct ggml_tensor*X,
                               int K,int N,struct ggml_tensor**sink){
    struct ggml_tensor*Y=ggml_mul_mat(c,W,X);              // fwd  [N,T]
    if(f32) ggml_mul_mat_set_prec(Y,GGML_PREC_F32);
    struct ggml_tensor*dX=ggml_mul_mat(c,ggml_cont(c,ggml_transpose(c,W)),Y); // bwd dX [K,T]
    if(f32) ggml_mul_mat_set_prec(dX,GGML_PREC_F32);
    struct ggml_tensor*dW=ggml_mul_mat(c,ggml_cont(c,ggml_transpose(c,X)),
                                         ggml_cont(c,ggml_transpose(c,Y)));    // bwd dW [K,N]
    if(f32) ggml_mul_mat_set_prec(dW,GGML_PREC_F32);
    *sink = ggml_add(c, ggml_add(c, ggml_sum(c,dX), ggml_sum(c,dW)), *sink);
    return Y;
}

int main(void){
    const char*b=getenv("BACKEND"); int cpu=b&&!strcmp(b,"cpu");
    const char*p=getenv("PREC"); f32=p&&!strcmp(p,"f32");
    ggml_backend_t be=cpu?ggml_backend_cpu_init():ggml_backend_vk_init(0);
    if(!be){printf("no backend\n");return 1;}
    printf("backend=%s prec=%s  D=%d T=%d\n",ggml_backend_name(be),f32?"f32":"f16",D,T);

    struct ggml_init_params ip={(size_t)64*1024*1024,NULL,true};
    struct ggml_context*c=ggml_init(ip);
    // weights (persist on backend) — sizes for a transformer layer
    struct ggml_tensor*Wqkv=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,3*D);
    struct ggml_tensor*Wo  =ggml_new_tensor_2d(c,GGML_TYPE_F32,D,D);
    struct ggml_tensor*W1  =ggml_new_tensor_2d(c,GGML_TYPE_F32,D,4*D);
    struct ggml_tensor*W2  =ggml_new_tensor_2d(c,GGML_TYPE_F32,4*D,D);
    struct ggml_tensor*X   =ggml_new_tensor_2d(c,GGML_TYPE_F32,D,T);
    struct ggml_tensor*Xf  =ggml_new_tensor_2d(c,GGML_TYPE_F32,4*D,T);
    struct ggml_tensor*sink=ggml_new_tensor_1d(c,GGML_TYPE_F32,1);
    ggml_set_input(X); ggml_set_input(Xf); ggml_set_input(sink);
    ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(c,be);
    // init
    size_t big=(size_t)4*D*(T>D?T:D); float*t=malloc(big*4);
    for(size_t i=0;i<big;i++) t[i]=((float)((i*2654435761u)>>20&1023)/512.0f-1.0f)*0.05f;
    ggml_backend_tensor_set(Wqkv,t,0,(size_t)D*3*D*4);
    ggml_backend_tensor_set(Wo,t,0,(size_t)D*D*4);
    ggml_backend_tensor_set(W1,t,0,(size_t)D*4*D*4);
    ggml_backend_tensor_set(W2,t,0,(size_t)4*D*D*4);
    ggml_backend_tensor_set(X,t,0,(size_t)D*T*4);
    ggml_backend_tensor_set(Xf,t,0,(size_t)4*D*T*4);
    float z=0; ggml_backend_tensor_set(sink,&z,0,4);
    free(t);

    // build the training-step graph: 4 linears, each fwd+dX+dW
    struct ggml_cgraph*g=ggml_new_graph_custom(c,4096,false);
    struct ggml_tensor*s=sink;
    lin(c,Wqkv,X,D,3*D,&s);
    lin(c,Wo,  X,D,D,&s);
    lin(c,W1,  X,D,4*D,&s);
    lin(c,W2,  Xf,4*D,D,&s);
    ggml_build_forward_expand(g,s);

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
    double tok_per_s_1L=T/(ms/1e3);           // tokens/sec if the model were 1 layer
    printf("step(1 layer) = %.1f ms   %.1f GFLOP/s   %.0f tok/s (1-layer)\n", ms, gf, tok_per_s_1L);
    // tokens/sec for a full L-layer model = achieved_FLOPs / (6 * N_params)
    double achieved=flops/(ms/1e3);           // FLOP/s
    printf("achieved training throughput: %.1f GFLOP/s\n", achieved/1e9);
    return 0;
}
