#include "tpuinfo.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <map>

#include "tpu_metric_service.grpc.pb.h"
#include "tpu_metric_service.pb.h"

using tpu::monitoring::runtime::MetricRequest;
using tpu::monitoring::runtime::MetricResponse;
using tpu::monitoring::runtime::RuntimeMetricService;

template <typename Value, typename ReadValue>
static int QueryMetric(const char* address, const char* metric_name,
                       long long* device_ids, Value* values, int max_n,
                       ReadValue read_value) {
  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto stub = RuntimeMetricService::NewStub(channel);

  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));

  MetricRequest req;
  req.set_metric_name(metric_name);

  MetricResponse resp;
  if (!stub->GetRuntimeMetric(&ctx, req, &resp).ok()) return -1;

  std::map<int, Value> sorted;
  for (const auto& metric : resp.metric().metrics()) {
    int device = static_cast<int>(metric.attribute().value().int_attr());
    sorted[device] = read_value(metric.gauge());
  }

  int count = 0;
  for (const auto& [device, value] : sorted) {
    if (count >= max_n) break;
    device_ids[count] = device;
    values[count] = value;
    ++count;
  }
  return count;
}

extern "C" int tpu_query_int(const char* address, const char* metric_name,
                             long long* device_ids, long long* values,
                             int max_n) {
  return QueryMetric(address, metric_name, device_ids, values, max_n,
                     [](const auto& gauge) { return gauge.as_int(); });
}

extern "C" int tpu_query_double(const char* address, const char* metric_name,
                                long long* device_ids, double* values,
                                int max_n) {
  return QueryMetric(address, metric_name, device_ids, values, max_n,
                     [](const auto& gauge) { return gauge.as_double(); });
}
