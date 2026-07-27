// train_bench.c — prove GPU training on the Adreno 830 via ggml + OpenCL.
// Port of training/gpu-vulkan/vk_train.c from the POCO (Mali/Vulkan) benchmark:
// trains a 2-layer MLP to fit y = sin(2*pi*x) with AdamW. Backend selection via
// TRAIN_BACKEND=cpu|gpu (default gpu = first GPU device in the ggml registry,
// i.e. OpenCL/Adreno in this build). The scheduler keeps a CPU fallback, so ops
// without a GPU kernel run on CPU — same setup as the original.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NDATA   1024      // training points
#define NIN     1         // input dim
#define NOUT    1         // output dim
#define NHID    128       // hidden width
#define NBATCH  128       // batch size
#define NEPOCH  60

static struct ggml_tensor *w1, *b1, *w2, *b2;

static struct ggml_opt_optimizer_params opt_pars(void *ud) {
    (void) ud;
    struct ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(NULL);
    p.adamw.alpha = 5e-3f;
    return p;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *bsel = getenv("TRAIN_BACKEND");
    int use_cpu_only = bsel && strcmp(bsel, "cpu") == 0;
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    ggml_backend_t backend;
    ggml_backend_sched_t sched;
    if (use_cpu_only) {
        backend = cpu_backend;
        ggml_backend_t backends[1] = { cpu_backend };
        sched = ggml_backend_sched_new(backends, NULL, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    } else {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); // Adreno registers as integrated (UMA)
        if (!dev) { printf("FAIL: no GPU device in ggml registry\n"); return 1; }
        backend = ggml_backend_dev_init(dev, NULL);
        if (!backend) { printf("FAIL: GPU backend init\n"); return 1; }
        ggml_backend_t backends[2] = { backend, cpu_backend };
        sched = ggml_backend_sched_new(backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    }
    printf("backend: %s\n", ggml_backend_name(backend));

    struct ggml_init_params wp = { ggml_tensor_overhead() * 8, NULL, true };
    struct ggml_context *wctx = ggml_init(wp);
    w1 = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, NIN,  NHID);
    b1 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, NHID);
    w2 = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, NHID, NOUT);
    b2 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, NOUT);
    for (struct ggml_tensor *t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t))
        ggml_set_param(t);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);

    srand(1234);
    #define FILL(t) do { int n = ggml_nelements(t); float *tmp = malloc(n*4); \
        for (int i=0;i<n;i++) tmp[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f; \
        ggml_backend_tensor_set(t, tmp, 0, n*4); free(tmp); } while(0)
    FILL(w1); FILL(b1); FILL(w2); FILL(b2);

    ggml_opt_dataset_t ds = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, NIN, NOUT, NDATA, NBATCH);
    float *xd = ggml_get_data_f32(ggml_opt_dataset_data(ds));
    float *yd = ggml_get_data_f32(ggml_opt_dataset_labels(ds));
    for (int i = 0; i < NDATA; i++) {
        float x = (float)i / NDATA;
        xd[i] = x;
        yd[i] = sinf(2.0f * 3.14159265f * x);
    }

    struct ggml_init_params cp = { (size_t)16*1024*1024, NULL, true };
    struct ggml_context *cctx = ggml_init(cp);
    struct ggml_tensor *inputs = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, NIN, NBATCH);
    ggml_set_name(inputs, "inputs");
    ggml_set_input(inputs);
    struct ggml_tensor *h = ggml_relu(cctx, ggml_add(cctx, ggml_mul_mat(cctx, w1, inputs), b1));
    struct ggml_tensor *out = ggml_add(cctx, ggml_mul_mat(cctx, w2, h), b2);
    ggml_set_name(out, "outputs");
    ggml_set_output(out);
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    (void) cbuf;

    printf("training MLP (%d params) on %s, %d epochs...\n",
           (int)(ggml_nelements(w1)+ggml_nelements(b1)+ggml_nelements(w2)+ggml_nelements(b2)),
           ggml_backend_name(backend), NEPOCH);
    ggml_opt_fit(sched, cctx, inputs, out, ds,
                 GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                 opt_pars, NEPOCH, NBATCH, 0.0f, false);

    printf("DONE: training completed on %s\n", ggml_backend_name(backend));

    ggml_opt_dataset_free(ds);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cctx); ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}
