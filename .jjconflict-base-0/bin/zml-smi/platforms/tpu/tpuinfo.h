#ifndef TPUINFO_H_
#define TPUINFO_H_

#define TPUINFO_MAX_DEVICES 64

#ifdef __cplusplus
extern "C" {
#endif

int tpu_query_int(const char* address, const char* metric_name,
                  long long* device_ids, long long* values, int max_n);

int tpu_query_double(const char* address, const char* metric_name,
                     long long* device_ids, double* values, int max_n);

#ifdef __cplusplus
}
#endif

#endif /* TPUINFO_H_ */
