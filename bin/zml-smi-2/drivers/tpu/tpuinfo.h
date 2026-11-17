#ifndef TPUINFO_H_
#define TPUINFO_H_

#define TPUINFO_MAX_DEVICES 64

#ifdef __cplusplus
extern "C" {
#endif

// Pass exactly one values array, and provide TPUINFO_MAX_DEVICES slots in both
// outputs.
int tpu_query(const char* address, const char* metric_name,
              long long* device_ids, long long* integers, double* reals);

#ifdef __cplusplus
}
#endif

#endif /* TPUINFO_H_ */
