#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_escaped(FILE *out, const char *value) {
    for (const unsigned char *p = (const unsigned char *)value; *p != '\0'; ++p) {
        switch (*p) {
            case '\\':
                fputs("\\\\", out);
                break;
            case '"':
                fputs("\\\"", out);
                break;
            case '\n':
                fputs("\\n", out);
                break;
            case '\r':
                fputs("\\r", out);
                break;
            case '\t':
                fputs("\\t", out);
                break;
            default:
                fputc(*p, out);
                break;
        }
    }
}

static int process_stream(FILE *out, FILE *in, const char *path) {
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;

    while ((len = getline(&line, &cap, in)) != -1) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }

        if (len == 0) {
            continue;
        }

        char *space = strchr(line, ' ');
        if (space == NULL) {
            continue;
        }

        *space = '\0';
        const char *name = line;
        const char *value = space + 1;

        if (*name == '\0') {
            continue;
        }

        fputs("const char ", out);
        fputs(name, out);
        fputs("[] = \"", out);
        write_escaped(out, value);
        fputs("\";\n", out);
    }

    if (ferror(in)) {
        fprintf(stderr, "genstamp: failed while reading %s\n", path);
        free(line);
        return 1;
    }

    free(line);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <output-file> <status-file> [<status-file> ...]\n", argv[0]);
        return 1;
    }

    const char *output_path = argv[1];
    FILE *out = fopen(output_path, "w");
    if (out == NULL) {
        fprintf(stderr, "genstamp: failed to open %s for writing\n", output_path);
        return 1;
    }

    for (int i = 2; i < argc; ++i) {
        const char *path = argv[i];
        FILE *in = fopen(path, "r");
        if (in == NULL) {
            fprintf(stderr, "genstamp: failed to open %s\n", path);
            fclose(out);
            return 1;
        }

        int rc = process_stream(out, in, path);
        if (fclose(in) != 0) {
            fprintf(stderr, "genstamp: failed to close %s\n", path);
            fclose(out);
            return 1;
        }

        if (rc != 0) {
            fclose(out);
            return rc;
        }
    }

    if (fflush(out) != 0) {
        fprintf(stderr, "genstamp: failed to flush %s\n", output_path);
        fclose(out);
        return 1;
    }

    if (fclose(out) != 0) {
        fprintf(stderr, "genstamp: failed to close %s\n", output_path);
        return 1;
    }

    return 0;
}