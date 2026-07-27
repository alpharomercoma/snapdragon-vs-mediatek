// matmul_bench.c — GPU (Adreno 830, OpenCL) vs CPU matmul throughput.
// Port of gpu-matmul/vk_matmul_bench.c from the POCO (Mali/Vulkan) benchmark.
// C = A^T B (ggml_mul_mat), correctness checksum + GFLOP/s.
// Env: BACKEND=gpu|cpu (default gpu), PREC=f16|f32 (default f16), SZ via -DSZ.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef SZ
#define SZ 1024
#endif
#define M SZ
#define N SZ
#define K SZ
#define WARMUP 3
#define ITERS  20

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }

int main(void){
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *bsel = getenv("BACKEND"); int use_cpu = bsel && !strcmp(bsel,"cpu");
    const char *psel = getenv("PREC");    int f32 = psel && !strcmp(psel,"f32");

    ggml_backend_t backend;
    if (use_cpu) {
        backend = ggml_backend_cpu_init();
    } else {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); // Adreno registers as integrated (UMA)
        backend = dev ? ggml_backend_dev_init(dev, NULL) : NULL;
    }
    if(!backend){ printf("no backend\n"); return 1; }
    printf("backend=%s prec=%s\n", ggml_backend_name(backend), f32?"f32":"f16");

    struct ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor *B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor *C = ggml_mul_mat(ctx, A, B);          // C = [M,N]
    if(f32) ggml_mul_mat_set_prec(C, GGML_PREC_F32);          // force f32 accumulation
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    (void) buf;
    const char *ssel = getenv("SCALE"); float s = ssel ? atof(ssel) : 1.0f;
    int pos = s > 1.0f;
    float *a = malloc((size_t)K*M*4), *b = malloc((size_t)K*N*4);
    for(size_t i=0;i<(size_t)K*M;i++){ float v=(float)((i*1103515245u+12345u)%1000)/1000.0f; a[i] = pos ? v*s : (v*2-1); }
    for(size_t i=0;i<(size_t)K*N;i++){ float v=(float)((i*22695477u+1u)%1000)/1000.0f;    b[i] = pos ? v*s : (v*2-1); }
    ggml_backend_tensor_set(A,a,0,(size_t)K*M*4);
    ggml_backend_tensor_set(B,b,0,(size_t)K*N*4);
    free(a); free(b);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    for(int i=0;i<WARMUP;i++) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    double t0=now_ms();
    for(int i=0;i<ITERS;i++) ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    double ms=(now_ms()-t0)/ITERS;

    float *c = malloc((size_t)M*N*4);
    ggml_backend_tensor_get(C,c,0,(size_t)M*N*4);
    double sum=0; int nan=0;
    for(size_t i=0;i<(size_t)M*N;i++){ if(isnan(c[i])||isinf(c[i])) nan=1; else sum+=c[i]; }
    free(c);

    double gflops = (2.0*M*N*K)/(ms*1e6);
    printf("size=%d  latency=%.2f ms  %.1f GFLOP/s  checksum=%.3f  %s\n",
           SZ, ms, gflops, sum, nan?"*** NaN/Inf ***":"finite");
    return 0;
}
