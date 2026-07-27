// cltest.c — OpenCL reachability probe on Snapdragon 8 Elite (Adreno 830).
// Mirrors npu/cltest.c from the POCO (Mali) benchmark, where OpenCL was BLOCKED.
// Tries: vendor frontend libOpenCL.so, then the ICD libOpenCL_adreno.so directly.
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef uint64_t cl_ulong;

#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002

typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, uint64_t, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (*clGetDeviceInfo_t)(cl_device_id, cl_uint, size_t, void *, size_t *);
typedef cl_int (*clGetPlatformInfo_t)(cl_platform_id, cl_uint, size_t, void *, size_t *);

static int probe(const char *libname, const char *entry) {
    printf("--- %s (entry: %s) ---\n", libname, entry);
    void *h = dlopen(libname, RTLD_NOW);
    if (!h) { printf("dlopen -> %s\n", dlerror()); return 1; }
    printf("dlopen -> OK\n");
    clGetPlatformIDs_t getPlat = (clGetPlatformIDs_t)dlsym(h, entry);
    if (!getPlat) { printf("dlsym(%s) -> not found\n", entry); return 2; }
    cl_uint n = 0;
    cl_int rc = getPlat(0, NULL, &n);
    printf("%s(0,NULL,&n) rc=%d nplatforms=%u\n", entry, rc, n);
    if (rc != 0 || n == 0) return 3;
    cl_platform_id plat;
    getPlat(1, &plat, NULL);
    clGetDeviceIDs_t getDev = (clGetDeviceIDs_t)dlsym(h, "clGetDeviceIDs");
    clGetDeviceInfo_t devInfo = (clGetDeviceInfo_t)dlsym(h, "clGetDeviceInfo");
    if (!getDev || !devInfo) { printf("device query symbols missing\n"); return 0; }
    cl_uint nd = 0;
    rc = getDev(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &nd);
    printf("clGetDeviceIDs rc=%d ndevices=%u\n", rc, nd);
    for (cl_uint i = 0; i < nd && i < 4; i++) {
        cl_device_id dev;
        getDev(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);
        char name[256] = {0}, ver[256] = {0};
        cl_ulong mem = 0; cl_uint cus = 0;
        devInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
        devInfo(dev, CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
        devInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
        devInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
        printf("  device[%u]: %s | %s | %u CUs | %.1f GiB global mem\n",
               i, name, ver, cus, mem / 1073741824.0);
    }
    printf("RESULT: OpenCL ACCESSIBLE via %s\n", libname);
    return 0;
}

int main(void) {
    int a = probe("/vendor/lib64/libOpenCL.so", "clGetPlatformIDs");
    printf("\n");
    int b = probe("/vendor/lib64/libOpenCL_adreno.so", "clIcdGetPlatformIDsKHR");
    return (a == 0 || b == 0) ? 0 : 1;
}
