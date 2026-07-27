// ort_npu_bench.c — run a quantized fully-connected workload on the Qualcomm
// Hexagon HTP (NPU) via ONNX Runtime's QNN execution provider, and measure
// throughput. Mirrors npu_matmul_bench.c (MediaTek MDLA) from the POCO repo:
// 8 chained FC layers, int8, M=32 K=N=2048.
//
// Usage: ort_npu_bench <model.onnx> [u8|f16] [htp|cpu]
//   u8  : feed uint8 input  (int8 QDQ bench model)
//   f16 : feed fp16 input   (fp16 capability test model)
// CPU fallback is DISABLED for htp runs — if the graph doesn't fully lower to
// the NPU, session creation fails. What runs, runs on the HTP.
#include "onnxruntime_c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define M 32
#define K 2048
#define N 2048
#define LAYERS 8
#define WARMUP 3
#define ITERS 30

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static const OrtApi *api;
#define CHECK(expr) do { OrtStatus *_s = (expr); if (_s) { \
    fprintf(stderr, "FAIL at line %d: %s\n", __LINE__, api->GetErrorMessage(_s)); \
    return 1; } } while (0)

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0); // QNN's atexit destructors crash; don't lose output
    if (argc < 2) { fprintf(stderr, "usage: %s model.onnx [u8|f16] [htp|cpu]\n", argv[0]); return 2; }
    const char *model_path = argv[1];
    int f16 = argc > 2 && strcmp(argv[2], "f16") == 0;
    int use_htp = !(argc > 3 && strcmp(argv[3], "cpu") == 0);

    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv *env; CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "npu_bench", &env));
    OrtSessionOptions *so; CHECK(api->CreateSessionOptions(&so));
    CHECK(api->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));

    if (use_htp) {
        // hard-fail instead of silently running layers on CPU
        CHECK(api->AddSessionConfigEntry(so, "session.disable_cpu_ep_fallback", "1"));
        const char *keys[] = { "backend_path", "htp_performance_mode", "htp_graph_finalization_optimization_mode" };
        const char *vals[] = { "libQnnHtp.so", "burst", "3" };
        CHECK(api->SessionOptionsAppendExecutionProvider(so, "QNN", keys, vals, 3));
    }

    double t0 = now_ms();
    OrtSession *sess;
    CHECK(api->CreateSession(env, model_path, so, &sess));
    printf("session created (compile+load): %.1f ms  [%s]\n", now_ms() - t0,
           use_htp ? "QNN HTP" : "CPU EP");

    OrtAllocator *alloc; CHECK(api->GetAllocatorWithDefaultOptions(&alloc));
    char *in_name, *out_name;
    CHECK(api->SessionGetInputName(sess, 0, alloc, &in_name));
    CHECK(api->SessionGetOutputName(sess, 0, alloc, &out_name));

    OrtMemoryInfo *mi; CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
    int64_t dims[2] = { M, K };
    OrtValue *in_val = NULL;
    size_t in_bytes = (size_t)M * K * (f16 ? 2 : 1);
    void *in_buf = malloc(in_bytes);
    if (f16) {
        uint16_t *p = (uint16_t *)in_buf; // fp16 bit patterns for small values
        for (size_t i = 0; i < (size_t)M * K; i++) p[i] = 0x3C00 + (i % 17); // ~1.0
    } else {
        uint8_t *p = (uint8_t *)in_buf;
        for (size_t i = 0; i < (size_t)M * K; i++) p[i] = i % 251;
    }
    CHECK(api->CreateTensorWithDataAsOrtValue(mi, in_buf, in_bytes, dims, 2,
        f16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &in_val));

    const char *in_names[] = { in_name };
    const char *out_names[] = { out_name };

    double best = 1e30, total = 0; int runs = 0;
    OrtValue *out_val = NULL;
    for (int it = 0; it < WARMUP + ITERS; it++) {
        if (out_val) { api->ReleaseValue(out_val); out_val = NULL; }
        double t = now_ms();
        CHECK(api->Run(sess, NULL, in_names, (const OrtValue *const *)&in_val, 1,
                       out_names, 1, &out_val));
        double dt = now_ms() - t;
        if (it >= WARMUP) { total += dt; runs++; if (dt < best) best = dt; }
    }
    void *out_data; CHECK(api->GetTensorMutableData(out_val, &out_data));
    int layers = f16 ? 1 : LAYERS;
    double gops = (2.0 * M * K * (double)N * layers) / (best * 1e6);
    printf("checksum byte: %u\n", ((uint8_t *)out_data)[0]);
    printf("avg latency: %.2f ms  best: %.2f ms  (%d runs)\n", total / runs, best, runs);
    printf("throughput (best): %.1f GOPS %s on %s\n", gops, f16 ? "fp16" : "int8",
           use_htp ? "Hexagon HTP (QNN)" : "CPU EP");
    fflush(stdout);
    _exit(0); // skip libQnnHtpPrepare.so's broken exit-time destructors
}
