// qnn_probe2.c — dump the QnnInterface_t provider struct layout empirically:
// find the providerName string pointer and version fields without trusting
// a hardcoded header layout.
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <unistd.h>
#include <fcntl.h>

typedef uint64_t (*getProvidersFn)(const void ***, uint32_t *);

static int looks_like_string(const void *p) {
    // probe readability via write() to a pipe (copies from user memory,
    // returns EFAULT on bad pointers — unlike /dev/null which skips the copy)
    static int pfd[2] = { -1, -1 };
    if (pfd[0] < 0 && pipe(pfd) != 0) return 0;
    if (write(pfd[1], p, 8) != 8) return 0;
    char drain[8]; (void)!read(pfd[0], drain, 8);
    const char *s = (const char *)p;
    int n = 0;
    while (n < 64 && s[n]) { if (!isprint((unsigned char)s[n])) return 0; n++; }
    return n >= 3 && n < 64;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    void *h = dlopen("libQnnHtp.so", RTLD_NOW);
    if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
    getProvidersFn gp = (getProvidersFn)dlsym(h, "QnnInterface_getProviders");
    const void **provs = NULL; uint32_t n = 0;
    uint64_t rc = gp(&provs, &n);
    printf("getProviders rc=%llu n=%u\n", (unsigned long long)rc, n);
    if (rc || !n) return 1;
    const uint64_t *w = (const uint64_t *)provs[0];
    printf("provider[0] @ %p, first 24 words:\n", (void *)w);
    for (int i = 0; i < 24; i++) {
        const void *ptr = (const void *)w[i];
        printf("  [%2d] 0x%016llx", i, (unsigned long long)w[i]);
        if (w[i] && looks_like_string(ptr)) printf("  -> string: \"%s\"", (const char *)ptr);
        else {
            Dl_info info;
            if (w[i] && dladdr((void *)w[i], &info) && info.dli_sname)
                printf("  -> func: %s", info.dli_sname);
            else if (w[i] && dladdr((void *)w[i], &info) && info.dli_fname)
                printf("  -> in: %s", info.dli_fname);
        }
        printf("\n");
    }
    return 0;
}
