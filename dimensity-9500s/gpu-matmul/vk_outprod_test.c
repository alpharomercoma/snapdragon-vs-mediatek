// vk_outprod_test.c — verify the identity  out_prod(a,b) == mul_mat(cont(T a), cont(T b))
// out_prod forward:  dst[i0,i1] = sum_k a[i0,k] * b[i1,k]   (contract dim1)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// a: [K, R]  b: [K, C]   (ne0=K, ne1=R / C)   out_prod(a,b) -> [R, C]
#define M 5
#define Kc 8
#define N 7

static float A[M*Kc], B[N*Kc];

static struct ggml_cgraph* build(struct ggml_context*ctx, struct ggml_tensor**po, struct ggml_tensor**pm){
    struct ggml_tensor*a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,M,Kc); ggml_set_name(a,"a"); ggml_set_input(a);
    struct ggml_tensor*b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,N,Kc); ggml_set_name(b,"b"); ggml_set_input(b);
    struct ggml_tensor*o=ggml_out_prod(ctx,a,b);                                  // reference
    struct ggml_tensor*m=ggml_mul_mat(ctx, ggml_cont(ctx,ggml_transpose(ctx,a)),
                                            ggml_cont(ctx,ggml_transpose(ctx,b))); // candidate
    ggml_set_output(o); ggml_set_output(m);
    *po=o; *pm=m;
    struct ggml_cgraph*g=ggml_new_graph(ctx); ggml_build_forward_expand(g,o); ggml_build_forward_expand(g,m);
    return g;
}

int main(void){
    for(int i=0;i<M*Kc;i++) A[i]=((i*7+3)%11)-5;
    for(int i=0;i<N*Kc;i++) B[i]=((i*5+1)%9)-4;
    ggml_backend_t be=(getenv("BK")&&!strcmp(getenv("BK"),"vk"))?ggml_backend_vk_init(0):ggml_backend_cpu_init();
    struct ggml_init_params ip={ggml_tensor_overhead()*32+ggml_graph_overhead(),NULL,true};
    struct ggml_context*ctx=ggml_init(ip);
    struct ggml_tensor*o,*m; struct ggml_cgraph*g=build(ctx,&o,&m);
    ggml_backend_buffer_t buf=ggml_backend_alloc_ctx_tensors(ctx,be);
    ggml_backend_tensor_set(ggml_graph_get_tensor(g,"a"),A,0,sizeof(A));
    ggml_backend_tensor_set(ggml_graph_get_tensor(g,"b"),B,0,sizeof(B));
    ggml_gallocr_t al=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(al,g);
    ggml_backend_graph_compute(be,g);
    printf("out_prod shape [%ld,%ld]  mul_mat shape [%ld,%ld]\n",o->ne[0],o->ne[1],m->ne[0],m->ne[1]);
    float od[M*N], md[M*N]; ggml_backend_tensor_get(o,od,0,sizeof(od)); ggml_backend_tensor_get(m,md,0,sizeof(md));
    double maxd=0; for(int i=0;i<M*N;i++){double d=fabs(od[i]-md[i]); if(d>maxd)maxd=d;}
    printf("max|out_prod - mul_mat| = %.6f  -> %s\n", maxd, maxd<1e-3?"IDENTITY HOLDS":"*** MISMATCH ***");
    printf("out_prod[0..3]: %.1f %.1f %.1f %.1f\n", od[0],od[1],od[2],od[3]);
    printf("mul_mat [0..3]: %.1f %.1f %.1f %.1f\n", md[0],md[1],md[2],md[3]);
    return 0;
}
