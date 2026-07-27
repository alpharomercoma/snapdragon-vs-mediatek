// npu_matmul_bench.c — run a real quantized fully-connected workload on the
// MediaTek MDLA (NPU) via the Neuron Adapter API and measure throughput.
// Usage: npu_matmul_bench [device_name] (default mtk-mdla; try mtk-dsp, mtk-gpu)
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

// NNAPI-compatible constants (Neuron Adapter mirrors NNAPI)
#define NEURON_TENSOR_INT32        4
#define NEURON_TENSOR_QUANT8_ASYMM 5
#define NEURON_INT32               1
#define NEURON_FULLY_CONNECTED     9
#define NEURON_NO_ERROR            0

typedef struct {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t *dimensions;
    float scale;
    int32_t zeroPoint;
} NeuronOperandType;

// batch, size: M x K matmul with K x N weights, N units
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

#define LOAD(h, name) name = (typeof(name))dlsym(h, #name); \
    if (!name) { printf("missing symbol %s\n", #name); return 1; }
#define CHECK(call) do { int _rc = (call); if (_rc != NEURON_NO_ERROR) { \
    printf("FAIL rc=%d at %s (line %d)\n", _rc, #call, __LINE__); return 1; } } while (0)

int main(int argc, char **argv) {
    const char *want = argc > 1 ? argv[1] : "mtk-mdla";
    void *h = dlopen("libneuronusdk_adapter.mtk.so", RTLD_NOW);
    if (!h) { printf("dlopen failed: %s\n", dlerror()); return 1; }

    int (*Neuron_getDeviceCount)(uint32_t *);
    int (*Neuron_getDevice)(uint32_t, void **);
    int (*NeuronDevice_getName)(const void *, const char **);
    int (*NeuronModel_create)(void **);
    int (*NeuronModel_addOperand)(void *, const NeuronOperandType *);
    int (*NeuronModel_setOperandValue)(void *, int32_t, const void *, size_t);
    int (*NeuronModel_addOperation)(void *, int32_t, uint32_t, const uint32_t *, uint32_t, const uint32_t *);
    int (*NeuronModel_identifyInputsAndOutputs)(void *, uint32_t, const uint32_t *, uint32_t, const uint32_t *);
    int (*NeuronModel_finish)(void *);
    int (*NeuronCompilation_createForDevices)(void *, const void *const *, uint32_t, void **);
    int (*NeuronCompilation_finish)(void *);
    int (*NeuronExecution_create)(void *, void **);
    int (*NeuronExecution_setInput)(void *, int32_t, const void *, const void *, size_t);
    int (*NeuronExecution_setOutput)(void *, int32_t, const void *, void *, size_t);
    int (*NeuronExecution_compute)(void *);
    LOAD(h, Neuron_getDeviceCount); LOAD(h, Neuron_getDevice); LOAD(h, NeuronDevice_getName);
    LOAD(h, NeuronModel_create); LOAD(h, NeuronModel_addOperand); LOAD(h, NeuronModel_setOperandValue);
    LOAD(h, NeuronModel_addOperation); LOAD(h, NeuronModel_identifyInputsAndOutputs);
    LOAD(h, NeuronModel_finish); LOAD(h, NeuronCompilation_createForDevices);
    LOAD(h, NeuronCompilation_finish); LOAD(h, NeuronExecution_create);
    LOAD(h, NeuronExecution_setInput); LOAD(h, NeuronExecution_setOutput); LOAD(h, NeuronExecution_compute);

    // pick device
    uint32_t ndev = 0; Neuron_getDeviceCount(&ndev);
    void *dev = NULL; const char *devname = NULL;
    for (uint32_t i = 0; i < ndev; i++) {
        void *d; const char *nm;
        Neuron_getDevice(i, &d); NeuronDevice_getName(d, &nm);
        if (strstr(nm, want)) { dev = d; devname = nm; }
    }
    if (!dev) { printf("device %s not found\n", want); return 1; }
    printf("target device: %s\n", devname);

    // build model: LAYERS x FC(quant8) chained, [M,K] -> [M,N] with N==K
    void *model = NULL; CHECK(NeuronModel_create(&model));
    static int8_t weights[N * K];
    static int32_t bias[N];
    for (size_t i = 0; i < sizeof(weights); i++) weights[i] = (int8_t)(i * 31 % 7 - 3);
    for (int i = 0; i < N; i++) bias[i] = i % 100;

    uint32_t dimsIn[2] = {M, K}, dimsW[2] = {N, K}, dimsB[1] = {N}, dimsOut[2] = {M, N};
    NeuronOperandType tIn  = {NEURON_TENSOR_QUANT8_ASYMM, 2, dimsIn, 0.5f, 0};
    NeuronOperandType tW   = {NEURON_TENSOR_QUANT8_ASYMM, 2, dimsW, 0.5f, 0};
    NeuronOperandType tB   = {NEURON_TENSOR_INT32, 1, dimsB, 0.25f, 0};
    NeuronOperandType tAct = {NEURON_INT32, 0, NULL, 0.0f, 0};
    NeuronOperandType tOut = {NEURON_TENSOR_QUANT8_ASYMM, 2, dimsOut, 8.0f, 0};

    int32_t relu = 1; // fused RELU
    uint32_t prev = 0;
    CHECK(NeuronModel_addOperand(model, &tIn)); // operand 0 = input
    for (int l = 0; l < LAYERS; l++) {
        uint32_t wIdx, bIdx, actIdx, outIdx;
        CHECK(NeuronModel_addOperand(model, &tW));   wIdx  = 1 + l * 4;
        CHECK(NeuronModel_addOperand(model, &tB));   bIdx  = wIdx + 1;
        CHECK(NeuronModel_addOperand(model, &tAct)); actIdx = wIdx + 2;
        CHECK(NeuronModel_addOperand(model, &tOut)); outIdx = wIdx + 3;
        CHECK(NeuronModel_setOperandValue(model, wIdx, weights, sizeof(weights)));
        CHECK(NeuronModel_setOperandValue(model, bIdx, bias, sizeof(bias)));
        CHECK(NeuronModel_setOperandValue(model, actIdx, &relu, sizeof(relu)));
        uint32_t ins[4] = {prev, wIdx, bIdx, actIdx}, outs[1] = {outIdx};
        CHECK(NeuronModel_addOperation(model, NEURON_FULLY_CONNECTED, 4, ins, 1, outs));
        prev = outIdx;
        // NOTE: all hidden layers reuse tOut's scale; input of next layer has scale 8.0
        tIn = tOut;
    }
    uint32_t modelIn[1] = {0}, modelOut[1] = {prev};
    CHECK(NeuronModel_identifyInputsAndOutputs(model, 1, modelIn, 1, modelOut));
    CHECK(NeuronModel_finish(model));

    double t0 = now_ms();
    void *comp = NULL;
    const void *devs[1] = {dev};
    CHECK(NeuronCompilation_createForDevices(model, devs, 1, &comp));
    CHECK(NeuronCompilation_finish(comp));
    printf("compile time: %.1f ms\n", now_ms() - t0);

    static uint8_t inBuf[M * K], outBuf[M * N];
    for (size_t i = 0; i < sizeof(inBuf); i++) inBuf[i] = i % 251;

    // one execution object per iteration is allowed; reuse is faster
    double best = 1e30, total = 0; int runs = 0;
    for (int it = 0; it < WARMUP + ITERS; it++) {
        void *exec = NULL;
        CHECK(NeuronExecution_create(comp, &exec));
        CHECK(NeuronExecution_setInput(exec, 0, NULL, inBuf, sizeof(inBuf)));
        CHECK(NeuronExecution_setOutput(exec, 0, NULL, outBuf, sizeof(outBuf)));
        double t = now_ms();
        CHECK(NeuronExecution_compute(exec));
        double dt = now_ms() - t;
        if (it >= WARMUP) { total += dt; runs++; if (dt < best) best = dt; }
    }
    double gops = (2.0 * M * K * (double)N * LAYERS) / (best * 1e6);
    printf("checksum byte: %u\n", outBuf[0]);
    printf("avg latency: %.2f ms  best: %.2f ms  (%d runs)\n", total / runs, best, runs);
    printf("throughput (best): %.1f GOPS int8 on %s\n", gops, devname);
    return 0;
}
