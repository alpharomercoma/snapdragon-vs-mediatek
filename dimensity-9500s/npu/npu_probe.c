// npu_probe.c — probe MediaTek Neuron NPU runtime accessibility from an
// untrusted app context (Termux) on Dimensity 9500s / mt6991.
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>

typedef struct { uint8_t major, minor, patch; } NeuronRuntimeVersion;

static void *try_open(const char *name) {
    void *h = dlopen(name, RTLD_NOW);
    printf("dlopen(%-32s) -> %s\n", name, h ? "OK" : dlerror());
    return h;
}

int main(void) {
    // Candidate libraries, most preferred first.
    const char *libs[] = {
        "libneuronusdk_adapter.mtk.so",
        "libneuron_adapter_mgvi.so",
        "libneuron_adapter.so",
        "libneuron_runtime.so",
        "libneuron_runtime.8.so",
    };
    void *adapter = NULL;
    for (int i = 0; i < 5; i++) {
        void *h = try_open(libs[i]);
        if (h && !adapter) adapter = h;
    }
    if (!adapter) { printf("RESULT: no neuron library loadable\n"); return 1; }

    // Neuron Adapter API (NNAPI-like, from MediaTek NeuroPilot)
    int (*Neuron_getVersion)(NeuronRuntimeVersion *) = dlsym(adapter, "Neuron_getVersion");
    int (*Neuron_getDeviceCount)(uint32_t *) = dlsym(adapter, "Neuron_getDeviceCount");
    int (*Neuron_getDevice)(uint32_t, void **) = dlsym(adapter, "Neuron_getDevice");
    int (*NeuronDevice_getName)(const void *, const char **) = dlsym(adapter, "NeuronDevice_getName");
    int (*Neuron_getL1MemorySizeKb)(uint32_t *) = dlsym(adapter, "Neuron_getL1MemorySizeKb");

    if (Neuron_getVersion) {
        NeuronRuntimeVersion v = {0};
        int rc = Neuron_getVersion(&v);
        printf("Neuron_getVersion rc=%d -> NeuroPilot runtime %u.%u.%u\n", rc, v.major, v.minor, v.patch);
    } else {
        printf("Neuron_getVersion: symbol not found\n");
    }
    if (Neuron_getL1MemorySizeKb) {
        uint32_t kb = 0;
        int rc = Neuron_getL1MemorySizeKb(&kb);
        printf("Neuron_getL1MemorySizeKb rc=%d -> %u KB APU L1\n", rc, kb);
    }
    if (Neuron_getDeviceCount && Neuron_getDevice && NeuronDevice_getName) {
        uint32_t n = 0;
        int rc = Neuron_getDeviceCount(&n);
        printf("Neuron_getDeviceCount rc=%d -> %u device(s)\n", rc, n);
        for (uint32_t i = 0; i < n; i++) {
            void *dev = NULL; const char *name = "?";
            if (Neuron_getDevice(i, &dev) == 0 && NeuronDevice_getName(dev, &name) == 0)
                printf("  device[%u] = %s\n", i, name);
        }
        if (n > 0) { printf("RESULT: NPU RUNTIME ACCESSIBLE, %u accelerator device(s) visible\n", n); return 0; }
    } else {
        printf("device enumeration symbols not found\n");
    }
    printf("RESULT: library loaded but device enumeration unavailable\n");
    return 2;
}
