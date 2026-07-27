// qnn_probe.c — probe Qualcomm QNN (AI Engine Direct) runtime accessibility
// on Snapdragon 8 Elite (SM8750, Hexagon V79 HTP) from a shell context.
// Mirrors npu_probe.c (MediaTek NeuroPilot probe) from the POCO benchmark.
//
// The QNN host libraries are NOT preinstalled on this device; they are the
// publicly redistributed ones from Maven Central (com.qualcomm.qti:qnn-runtime).
// The FastRPC transport (libcdsprpc.so) IS a public vendor library on-device.
//
// QnnInterface_t layout was verified empirically on-device (qnn_probe2.c):
//   [0] u32 backendId          [1] const char* providerName
//   [2..4] Qnn_ApiVersion_t    [5+] function table (property, backendCreate,
//   setConfig, getApiVersion, getBuildId, ...)
// The probe cross-checks the layout by comparing backendGetApiVersion()'s
// output against the struct's version fields before trusting anything else.
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef uint64_t Qnn_ErrorHandle_t;
typedef struct { uint32_t major, minor, patch; } Qnn_Version_t;
typedef struct { Qnn_Version_t coreApiVersion; Qnn_Version_t backendApiVersion; } Qnn_ApiVersion_t;
typedef void *Qnn_BackendHandle_t;

typedef Qnn_ErrorHandle_t (*QnnProperty_HasCapabilityFn_t)(uint32_t);
typedef Qnn_ErrorHandle_t (*QnnBackend_CreateFn_t)(void *logger, const void **config, Qnn_BackendHandle_t *);
typedef Qnn_ErrorHandle_t (*QnnBackend_SetConfigFn_t)(Qnn_BackendHandle_t, const void **);
typedef Qnn_ErrorHandle_t (*QnnBackend_GetApiVersionFn_t)(Qnn_ApiVersion_t *);
typedef Qnn_ErrorHandle_t (*QnnBackend_GetBuildIdFn_t)(const char **);

typedef struct {
    uint32_t backendId;
    uint32_t reserved;
    const char *providerName;
    Qnn_ApiVersion_t apiVersion;
    QnnProperty_HasCapabilityFn_t propertyHasCapability;
    QnnBackend_CreateFn_t         backendCreate;
    QnnBackend_SetConfigFn_t      backendSetConfig;
    QnnBackend_GetApiVersionFn_t  backendGetApiVersion;
    QnnBackend_GetBuildIdFn_t     backendGetBuildId;
    // ... more entries follow; not needed for the probe
} QnnInterface_t;

typedef Qnn_ErrorHandle_t (*QnnInterface_getProvidersFn_t)(const QnnInterface_t ***, uint32_t *);

static void *try_open(const char *name) {
    void *h = dlopen(name, RTLD_NOW);
    printf("dlopen(%-28s) -> %s\n", name, h ? "OK" : dlerror());
    return h;
}

static int probe_backend(const char *lib) {
    void *h = try_open(lib);
    if (!h) return 1;
    QnnInterface_getProvidersFn_t getProviders =
        (QnnInterface_getProvidersFn_t)dlsym(h, "QnnInterface_getProviders");
    if (!getProviders) { printf("  QnnInterface_getProviders: not found\n"); return 2; }
    const QnnInterface_t **provs = NULL; uint32_t n = 0;
    Qnn_ErrorHandle_t rc = getProviders(&provs, &n);
    printf("  QnnInterface_getProviders rc=%llu -> %u provider(s)\n", (unsigned long long)rc, n);
    if (rc != 0 || n == 0) return 3;

    const QnnInterface_t *p = provs[0];
    printf("  provider[0] = %s (backendId %u, core API %u.%u.%u, backend API %u.%u.%u)\n",
           p->providerName ? p->providerName : "?", p->backendId,
           p->apiVersion.coreApiVersion.major, p->apiVersion.coreApiVersion.minor,
           p->apiVersion.coreApiVersion.patch,
           p->apiVersion.backendApiVersion.major, p->apiVersion.backendApiVersion.minor,
           p->apiVersion.backendApiVersion.patch);

    // layout self-check: the function-table getApiVersion must agree with the
    // struct fields, otherwise we are misreading the interface
    Qnn_ApiVersion_t v = {0};
    if (p->backendGetApiVersion && p->backendGetApiVersion(&v) == 0 &&
        v.coreApiVersion.major == p->apiVersion.coreApiVersion.major &&
        v.coreApiVersion.minor == p->apiVersion.coreApiVersion.minor) {
        printf("  backendGetApiVersion rc=0 (matches struct -> layout verified)\n");
    } else {
        printf("  layout self-check FAILED, not calling further entries\n");
        return 4;
    }
    const char *bid = NULL;
    if (p->backendGetBuildId && p->backendGetBuildId(&bid) == 0 && bid)
        printf("  build id: %s\n", bid);
    Qnn_BackendHandle_t bh = NULL;
    Qnn_ErrorHandle_t brc = p->backendCreate(NULL, NULL, &bh);
    printf("  backendCreate rc=%llu -> %s\n", (unsigned long long)brc,
           brc == 0 ? "BACKEND CREATED" : "failed");
    return brc == 0 ? 0 : 5;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== transport / GPU libraries (on-device, vendor) ===\n");
    try_open("libcdsprpc.so");              // FastRPC to compute DSP (public lib)
    try_open("/vendor/lib64/libOpenCL.so"); // Adreno OpenCL (public lib)
    printf("\n=== QNN backends (host libs from Maven qnn-runtime 2.48) ===\n");
    int rc_htp = probe_backend("libQnnHtp.so");
    printf("\n");
    int rc_gpu = probe_backend("libQnnGpu.so");
    printf("\n");
    if (rc_htp == 0)
        printf("RESULT: QNN HTP (Hexagon NPU) RUNTIME ACCESSIBLE, backend initialized\n");
    else
        printf("RESULT: HTP probe rc=%d gpu rc=%d\n", rc_htp, rc_gpu);
    return rc_htp;
}
