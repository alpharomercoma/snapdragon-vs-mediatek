#include <dlfcn.h>
#include <stdio.h>
typedef int (*clGetPlatformIDs_t)(unsigned, void **, unsigned *);
int main(void) {
    const char *paths[] = {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so"};
    void *h = 0;
    for (int i = 0; i < 3; i++) {
        h = dlopen(paths[i], RTLD_NOW);
        printf("dlopen(%-32s) -> %s\n", paths[i], h ? "OK" : dlerror());
        if (h) break;
    }
    if (!h) { printf("RESULT: libOpenCL not loadable from Termux native namespace\n"); return 1; }
    clGetPlatformIDs_t p = (clGetPlatformIDs_t)dlsym(h, "clGetPlatformIDs");
    if (!p) { printf("no clGetPlatformIDs symbol\n"); return 2; }
    unsigned n = 0;
    int rc = p(0, 0, &n);
    printf("clGetPlatformIDs rc=%d platforms=%u\n", rc, n);
    printf("RESULT: OpenCL REACHABLE, %u platform(s)\n", n);
    return 0;
}
