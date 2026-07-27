// vk_train.c — prove GPU training on the Mali Immortalis-G925 via ggml + Vulkan.
// Trains a 2-layer MLP to fit y = sin(2*pi*x) with AdamW, running the forward AND
// backward + optimizer-step kernels on the Vulkan backend (GGML_OP_OPT_STEP_ADAMW,
// *_BACK ops). Loss should fall across epochs — that is on-device GPU training.
//
// Link against the prebuilt ggml shared libs from the llama.cpp Vulkan build.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "ggml-vulkan.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NDATA   1024      // training points
#define NIN     1         // input dim
#define NOUT    1         // output dim
#define NHID    128        // hidden width
#define NBATCH  128       // batch size
#define NEPOCH  60

static struct ggml_tensor *w1, *b1, *w2, *b2;

// optimizer params callback
static struct ggml_opt_optimizer_params opt_pars(void *ud) {
    (void) ud;
    struct ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(NULL);
    p.adamw.alpha = 5e-3f;
    return p;
}

int main(void) {
    // Backend selection: TRAIN_BACKEND=cpu forces CPU (isolation test);
    // default uses the Vulkan GPU with a CPU fallback backend.
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
        backend = ggml_backend_vk_init(0);
        if (!backend) { printf("FAIL: no Vulkan backend\n"); return 1; }
        // scheduler requires a CPU backend last (fallback); training kernels run on Vulkan.
        ggml_backend_t backends[2] = { backend, cpu_backend };
        sched = ggml_backend_sched_new(backends, NULL, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    }
    printf("backend: %s\n", ggml_backend_name(backend));

    // 2. model params (static ctx, weights live on the backend)
    struct ggml_init_params wp = { ggml_tensor_overhead() * 8, NULL, true };
    struct ggml_context *wctx = ggml_init(wp);
    w1 = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, NIN,  NHID);
    b1 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, NHID);
    w2 = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, NHID, NOUT);
    b2 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, NOUT);
    for (struct ggml_tensor *t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t))
        ggml_set_param(t);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);

    // random init on host, upload
    srand(1234);
    #define FILL(t) do { int n = ggml_nelements(t); float *tmp = malloc(n*4); \
        for (int i=0;i<n;i++) tmp[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f; \
        ggml_backend_tensor_set(t, tmp, 0, n*4); free(tmp); } while(0)
    FILL(w1); FILL(b1); FILL(w2); FILL(b2);

    // 3. dataset: x in [0,1], y = sin(2*pi*x)
    ggml_opt_dataset_t ds = ggml_opt_dataset_init(
        GGML_TYPE_F32, GGML_TYPE_F32, NIN, NOUT, NDATA, NBATCH);
    float *xd = ggml_get_data_f32(ggml_opt_dataset_data(ds));
    float *yd = ggml_get_data_f32(ggml_opt_dataset_labels(ds));
    for (int i = 0; i < NDATA; i++) {
        float x = (float)i / NDATA;
        xd[i] = x;
        yd[i] = sinf(2.0f * 3.14159265f * x);
    }

    // 4. compute context: inputs -> MLP -> outputs (built fresh by ggml_opt each graph)
    struct ggml_init_params cp = { (size_t)16*1024*1024, NULL, true };
    struct ggml_context *cctx = ggml_init(cp);
    struct ggml_tensor *inputs = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, NIN, NBATCH);
    ggml_set_name(inputs, "inputs");
    ggml_set_input(inputs);
    struct ggml_tensor *h = ggml_relu(cctx, ggml_add(cctx, ggml_mul_mat(cctx, w1, inputs), b1));
    struct ggml_tensor *out = ggml_add(cctx, ggml_mul_mat(cctx, w2, h), b2);
    ggml_set_name(out, "outputs");
    ggml_set_output(out);
    // inputs must be statically allocated for ggml_opt static graphs (UMA backend)
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);

    // 5. train on GPU
    printf("training MLP (%d params) on GPU, %d epochs...\n",
           (int)(ggml_nelements(w1)+ggml_nelements(b1)+ggml_nelements(w2)+ggml_nelements(b2)), NEPOCH);
    ggml_opt_fit(sched, cctx, inputs, out, ds,
                 GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR, GGML_OPT_OPTIMIZER_TYPE_ADAMW,
                 opt_pars, NEPOCH, NBATCH, 0.0f, false);

    printf("DONE: GPU training completed on %s\n", ggml_backend_name(backend));

    ggml_opt_dataset_free(ds);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wbuf);
    ggml_free(cctx); ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}
