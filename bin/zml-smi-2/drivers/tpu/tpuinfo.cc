#include "tpuinfo.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <map>

#include "tpu_metric_service.grpc.pb.h"
#include "tpu_metric_service.pb.h"

namespace monitoring = tpu::monitoring::runtime;

int tpu_query(const char* address, const char* metric_name,
              long long* device_ids, long long* integers, double* reals) {
  auto stub = monitoring::RuntimeMetricService::NewStub(
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));

  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));

  monitoring::MetricRequest req;
  req.set_metric_name(metric_name);

  monitoring::MetricResponse resp;
  if (!stub->GetRuntimeMetric(&ctx, req, &resp).ok()) return -1;

  std::map<int, const monitoring::Gauge*> sorted;
  for (const auto& metric : resp.metric().metrics()) {
    int device = static_cast<int>(metric.attribute().value().int_attr());
    sorted[device] = &metric.gauge();
  }

  int count = 0;
  for (const auto& [device, gauge] : sorted) {
    if (count == TPUINFO_MAX_DEVICES) break;
    device_ids[count] = device;
    if (integers) {
      integers[count] = gauge->as_int();
    } else {
      reals[count] = gauge->as_double();
    }
    ++count;
  }
  return count;
}
