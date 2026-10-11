// usage: weaksym -p 'Py*' -p '_Py*' <input.so> <output.so>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fnmatch.h>
#include <elf.h>

static char *globs[64]; static int nglob = 0;

int main(int argc, char **argv) {
    int a = 1;
    for (; a < argc && strcmp(argv[a], "-p") == 0; a += 2) {
        if (a + 1 >= argc) { fprintf(stderr, "-p needs a pattern\n"); return 1; }
        globs[nglob++] = argv[a + 1];
    }
    if (nglob == 0 || a + 2 != argc) {
        fprintf(stderr, "usage: %s -p <glob> [-p <glob>...] <input.so> <output.so>\n", argv[0]);
        return 1;
    }
    const char *in = argv[a], *out = argv[a + 1];

    FILE *f = fopen(in, "rb");
    if (!f) { perror(in); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) { perror("read"); fclose(f); free(buf); return 1; }
    fclose(f);

    Elf64_Ehdr *eh = (Elf64_Ehdr *)buf;
    Elf64_Shdr *sh = (Elf64_Shdr *)(buf + eh->e_shoff);
    for (int i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_type != SHT_DYNSYM) continue;
        char *str = (char *)(buf + sh[sh[i].sh_link].sh_offset);
        Elf64_Sym *sym = (Elf64_Sym *)(buf + sh[i].sh_offset);
        int cnt = sh[i].sh_size / sizeof(Elf64_Sym);
        for (int j = 0; j < cnt; j++) {
            char *name = str + sym[j].st_name;
            for (int g = 0; g < nglob; g++)                       // test each pattern here
                if (fnmatch(globs[g], name, 0) == 0) {
                    sym[j].st_info = (STB_WEAK << 4) | ELF64_ST_TYPE(sym[j].st_info);
                    break;                                        // matched; next symbol
                }
        }
    }

    FILE *o = fopen(out, "wb");
    if (!o) { perror(out); free(buf); return 1; }
    fwrite(buf, 1, n, o);
    fclose(o); free(buf);
    return 0;
}
