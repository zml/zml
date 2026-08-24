#include "tpuinfo.h"

#include <chrono>
#include <map>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "tpu_metric_service.grpc.pb.h"
#include "tpu_metric_service.pb.h"

using tpu::monitoring::runtime::MetricRequest;
using tpu::monitoring::runtime::MetricResponse;
using tpu::monitoring::runtime::RuntimeMetricService;

static MetricResponse FetchMetric(const char* address, const char* metric_name,
                                  bool* ok) {
  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto stub = RuntimeMetricService::NewStub(channel);
  
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));
  
  MetricRequest req;
  req.set_metric_name(metric_name);
  
  MetricResponse resp;
  auto status = stub->GetRuntimeMetric(&ctx, req, &resp);
  *ok = status.ok();
  
  return resp;
}

extern "C" int tpu_query_int(const char* address, const char* metric_name,
                              long long* device_ids, long long* values,
                              int max_n) {
  bool ok;
  auto resp = FetchMetric(address, metric_name, &ok);
  if (!ok) return -1;

  std::map<int, long long> sorted;
  for (const auto& m : resp.metric().metrics()) {
    int id = static_cast<int>(m.attribute().value().int_attr());
    sorted[id] = m.gauge().as_int();
  }

  int i = 0;
  for (const auto& [id, val] : sorted) {
    if (i >= max_n) break;
    
    device_ids[i] = id;
    values[i] = val;
    ++i;
  }
  
  return i;
}

extern "C" int tpu_query_double(const char* address, const char* metric_name,
                                 long long* device_ids, double* values,
                                 int max_n) {
  bool ok;
  auto resp = FetchMetric(address, metric_name, &ok);
  if (!ok) return -1;

  std::map<int, double> sorted;
  for (const auto& m : resp.metric().metrics()) {
    int id = static_cast<int>(m.attribute().value().int_attr());
    sorted[id] = m.gauge().as_double();
  }

  int i = 0;
  for (const auto& [id, val] : sorted) {
    if (i >= max_n) break;
    
    device_ids[i] = id;
    values[i] = val;
    ++i;
  }
  
  return i;
}
