const std = @import("std");
const tf_op_utils = @import("tf_op_utils.zig");

pub const kXlaAsyncOpLineName = "Async XLA Ops";

pub const kHostThreadsPlaneName = "/host:CPU";
pub const kGpuPlanePrefix = "/device:GPU:";
pub const kTpuPlanePrefix = "/device:TPU:";
pub const kCustomPlanePrefix = "/device:CUSTOM:";

pub const HostEventType = enum(u16) {
    kUnknownHostEventType = 0,
    kTraceContext,
    kSessionRun,
    kFunctionRun,
    kRunGraph,
    kRunGraphDone,
    kTfOpRun,
    kEagerKernelExecute,
    kExecutorStateProcess,
    kExecutorDoneCallback,
    kMemoryAllocation,
    kMemoryDeallocation,
    // Performance counter related.
    kRemotePerf,
    // tf.data captured function events.
    kTfDataCapturedFunctionRun,
    kTfDataCapturedFunctionRunWithBorrowedArgs,
    kTfDataCapturedFunctionRunInstantiated,
    kTfDataCapturedFunctionRunAsync,
    // Loop ops.
    kParallelForOp,
    kForeverOp,
    kWhileOpEvalCond,
    kWhileOpStartBody,
    kForOp,
    // tf.data related.
    kIteratorGetNextOp,
    kIteratorGetNextAsOptionalOp,
    kIterator,
    kDeviceInputPipelineSecondIterator,
    kPrefetchProduce,
    kPrefetchConsume,
    kParallelInterleaveProduce,
    kParallelInterleaveConsume,
    kParallelInterleaveInitializedInput,
    kParallelMapProduce,
    kParallelMapConsume,
    kMapAndBatchProduce,
    kMapAndBatchConsume,
    kParseExampleProduce,
    kParseExampleConsume,
    kParallelBatchProduce,
    kParallelBatchConsume,
    // Batching related.
    kBatchingSessionRun,
    kProcessBatch,
    kBrainSessionRun,
    kConcatInputTensors,
    kMergeInputTensors,
    kScheduleWithoutSplit,
    kScheduleWithSplit,
    kScheduleWithEagerSplit,
    kASBSQueueSchedule,
    // TFRT related.
    kTfrtModelRun,
    // Serving related.
    kServingModelRun,
    // GPU related.
    kKernelLaunch,
    kKernelExecute,
    // TPU related
    kEnqueueRequestLocked,
    kRunProgramRequest,
    kHostCallbackRequest,
    kTransferH2DRequest,
    kTransferPreprocessedH2DRequest,
    kTransferD2HRequest,
    kOnDeviceSendRequest,
    kOnDeviceRecvRequest,
    kOnDeviceSendRecvLocalRequest,
    kCustomWait,
    kOnDeviceSendRequestMulti,
    kOnDeviceRecvRequestMulti,
    kPjrtAsyncWait,
    kDoEnqueueProgram,
    kDoEnqueueContinuationProgram,
    kWriteHbm,
    kReadHbm,
    kTpuExecuteOp,
    kCompleteCallbacks,
    kTransferToDeviceIssueEvent,
    kTransferToDeviceDone,
    kTransferFromDeviceIssueEvent,
    kTransferFromDeviceDone,
    kTpuSystemExecute,
    kTpuPartitionedCallOpInitializeVarOnTpu,
    kTpuPartitionedCallOpExecuteRemote,
    kTpuPartitionedCallOpExecuteLocal,
    kLinearize,
    kDelinearize,
    kTransferBufferFromDeviceFastPath,

    pub fn fromString(event_name: []const u8) ?HostEventType {
        if (HostEventTypeMap.get(event_name)) |event_type| return event_type;
        return null;
    }

    pub fn fromTfOpEventType(event_name: []const u8) ?HostEventType {
        return switch (tf_op_utils.parseTfOpFullName(event_name).category) {
            .kTensorFlow => .kTfOpRun,
            .kTfData => .kIterator,
            else => null,
        };
    }

    pub fn isInternalEvent(event_type: HostEventType) bool {
        // TODO(b/162102421): Introduce a prefix for internal event names.
        return switch (event_type) {
            .kMemoryAllocation,
            .kMemoryDeallocation,
            .kPrefetchProduce,
            .kPrefetchConsume,
            .kParallelInterleaveProduce,
            .kParallelInterleaveConsume,
            .kParallelInterleaveInitializedInput,
            .kParallelMapProduce,
            .kParallelMapConsume,
            .kMapAndBatchProduce,
            .kMapAndBatchConsume,
            .kParseExampleProduce,
            .kParseExampleConsume,
            => true,
            else => false,
        };
    }

    pub const kFirstHostEventType = 0;
    pub const kLastHostEventType = @intFromEnum(HostEventType.kTransferBufferFromDeviceFastPath);
};

pub const StatType = enum(u16) {
    kUnknownStatType = 0,
    // TraceMe arguments.
    kStepId,
    kDeviceOrdinal,
    kChipOrdinal,
    kNodeOrdinal,
    kModelId,
    kQueueId,
    kQueueAddr,
    kRequestId,
    kRunId,
    kReplicaId,
    kGraphType,
    kStepNum,
    kIterNum,
    kIndexOnHost,
    kAllocatorName,
    kBytesReserved,
    kBytesAllocated,
    kBytesAvailable,
    kFragmentation,
    kPeakBytesInUse,
    kRequestedBytes,
    kAllocationBytes,
    kAddress,
    kRegionType,
    kDataType,
    kTensorShapes,
    kTensorLayout,
    kKpiName,
    kKpiValue,
    kElementId,
    kParentId,
    kCoreType,
    // XPlane semantics related.
    kProducerType,
    kConsumerType,
    kProducerId,
    kConsumerId,
    kIsRoot,
    kIsAsync,
    // Device trace arguments.
    kDeviceId,
    kDeviceTypeString,
    kContextId,
    kCorrelationId,
    // TODO(b/176137043): These "details" should differentiate between activity
    // and API event sources.
    kMemcpyDetails,
    kMemallocDetails,
    kMemFreeDetails,
    kMemsetDetails,
    kMemoryResidencyDetails,
    kNVTXRange,
    kKernelDetails,
    kStream,
    // Stats added when processing traces.
    kGroupId,
    kFlow,
    kStepName,
    kTfOp,
    kHloOp,
    kDeduplicatedName,
    kHloCategory,
    kHloModule,
    kProgramId,
    kEquation,
    kIsEager,
    kIsFunc,
    kTfFunctionCall,
    kTfFunctionTracingCount,
    kFlops,
    kModelFlops,
    kBytesAccessed,
    kMemoryAccessBreakdown,
    kSourceInfo,
    kModelName,
    kModelVersion,
    kBytesTransferred,
    kDmaQueue,
    kDcnCollectiveInfo,
    // Performance counter related.
    kRawValue,
    kScaledValue,
    kThreadId,
    kMatrixUnitUtilizationPercent,
    // XLA metadata map related.
    kHloProto,
    // Device capability related.
    kDevCapClockRateKHz,
    // For GPU, this is the number of SMs.
    kDevCapCoreCount,
    kDevCapMemoryBandwidth,
    kDevCapMemorySize,
    kDevCapComputeCapMajor,
    kDevCapComputeCapMinor,
    kDevCapPeakTeraflopsPerSecond,
    kDevCapPeakHbmBwGigabytesPerSecond,
    kDevCapPeakSramRdBwGigabytesPerSecond,
    kDevCapPeakSramWrBwGigabytesPerSecond,
    kDevVendor,
    // Batching related.
    kBatchSizeAfterPadding,
    kPaddingAmount,
    kBatchingInputTaskSize,
    // GPU occupancy metrics
    kTheoreticalOccupancyPct,
    kOccupancyMinGridSize,
    kOccupancySuggestedBlockSize,
    // Aggregated Stats
    kSelfDurationPs,
    kMinDurationPs,
    kTotalProfileDurationPs,
    kMaxIterationNum,
    kDeviceType,
    kUsesMegaCore,
    kSymbolId,
    kTfOpName,
    kDmaStallDurationPs,
    kKey,
    kPayloadSizeBytes,
    kDuration,
    kBufferSize,
    kTransfers,
    // Dcn message Stats
    kDcnLabel,
    kDcnSourceSliceId,
    kDcnSourcePerSliceDeviceId,
    kDcnDestinationSliceId,
    kDcnDestinationPerSliceDeviceId,
    kDcnChunk,
    kDcnLoopIndex,
    kEdgeTpuModelInfo,
    kEdgeTpuModelProfileInfo,
    kEdgeTpuMlir,
    kDroppedTraces,
    kCudaGraphId,
    // Many events have kCudaGraphId, such as graph sub events when tracing is in
    // node level. Yet kCudaGraphExecId is used only for CudaGraphExecution events
    // on the GPU device when tracing is in graph level.
    kCudaGraphExecId,
    kCudaGraphOrigId,
    kStepIdleTimePs,
    kGpuDeviceName,
    kSourceStack,
    kDeviceOffsetPs,
    kDeviceDurationPs,

    pub const kFirstStatType = 0;
    pub const kLastStatType = @intFromEnum(StatType.kDeviceDurationPs);

    pub fn fromString(stat_name: []const u8) ?StatType {
        if (StatTypeMap.get(stat_name)) |stat_type| return stat_type;
        return null;
    }

    pub fn isInternalStat(stat_type: StatType) bool {
        return switch (stat_type) {
            .kKernelDetails,
            .kProducerType,
            .kProducerId,
            .kConsumerType,
            .kConsumerId,
            .kIsRoot,
            .kFlops,
            .kBytesAccessed,
            .kProgramId,
            .kSymbolId,
            => true,
            else => false,
        };
    }
};

const HostEventTypeMap = std.StaticStringMap(HostEventType).initComptime(&.{
    .{ "UnknownHostEventType", .kUnknownHostEventType },
    .{ "TraceContext", .kTraceContext },
    .{ "SessionRun", .kSessionRun },
    .{ "FunctionRun", .kFunctionRun },
    .{ "RunGraph", .kRunGraph },
    .{ "RunGraphDone", .kRunGraphDone },
    .{ "TfOpRun", .kTfOpRun },
    .{ "EagerExecute", .kEagerKernelExecute },
    .{ "ExecutorState::Process", .kExecutorStateProcess },
    .{ "ExecutorDoneCallback", .kExecutorDoneCallback },
    .{ "MemoryAllocation", .kMemoryAllocation },
    .{ "MemoryDeallocation", .kMemoryDeallocation },
    // Performance counter related.
    .{ "RemotePerfCounter", .kRemotePerf },
    // tf data captured function events.
    .{ "InstantiatedCapturedFunction::Run", .kTfDataCapturedFunctionRun },
    .{ "InstantiatedCapturedFunction::RunWithBorrowedArgs", .kTfDataCapturedFunctionRunWithBorrowedArgs },
    .{ "InstantiatedCapturedFunction::RunInstantiated", .kTfDataCapturedFunctionRunInstantiated },
    .{ "InstantiatedCapturedFunction::RunAsync", .kTfDataCapturedFunctionRunAsync },
    // Loop ops.
    .{ "ParallelForOp", .kParallelForOp },
    .{ "ForeverOp", .kForeverOp },
    .{ "WhileOp-EvalCond", .kWhileOpEvalCond },
    .{ "WhileOp-StartBody", .kWhileOpStartBody },
    .{ "ForOp", .kForOp },
    // tf.data related.
    .{ "IteratorGetNextOp::DoCompute", .kIteratorGetNextOp },
    .{ "IteratorGetNextAsOptionalOp::DoCompute", .kIteratorGetNextAsOptionalOp },
    .{ "Iterator", .kIterator },
    .{ "Iterator::Prefetch::Generator", .kDeviceInputPipelineSecondIterator },
    .{ "PrefetchProduce", .kPrefetchProduce },
    .{ "PrefetchConsume", .kPrefetchConsume },
    .{ "ParallelInterleaveProduce", .kParallelInterleaveProduce },
    .{ "ParallelInterleaveConsume", .kParallelInterleaveConsume },
    .{ "ParallelInterleaveInitializeInput", .kParallelInterleaveInitializedInput },
    .{ "ParallelMapProduce", .kParallelMapProduce },
    .{ "ParallelMapConsume", .kParallelMapConsume },
    .{ "MapAndBatchProduce", .kMapAndBatchProduce },
    .{ "MapAndBatchConsume", .kMapAndBatchConsume },
    .{ "ParseExampleProduce", .kParseExampleProduce },
    .{ "ParseExampleConsume", .kParseExampleConsume },
    .{ "ParallelBatchProduce", .kParallelBatchProduce },
    .{ "ParallelBatchConsume", .kParallelBatchConsume },
    // Batching related.
    .{ "BatchingSessionRun", .kBatchingSessionRun },
    .{ "ProcessBatch", .kProcessBatch },
    .{ "BrainSessionRun", .kBrainSessionRun },
    .{ "ConcatInputTensors", .kConcatInputTensors },
    .{ "MergeInputTensors", .kMergeInputTensors },
    .{ "ScheduleWithoutSplit", .kScheduleWithoutSplit },
    .{ "ScheduleWithSplit", .kScheduleWithSplit },
    .{ "ScheduleWithEagerSplit", .kScheduleWithEagerSplit },
    .{ "ASBSQueue::Schedule", .kASBSQueueSchedule },
    // TFRT related.
    .{ "TfrtModelRun", .kTfrtModelRun },
    // Serving related.
    .{ "ServingModelRun", .kServingModelRun },
    // GPU related.
    .{ "KernelLaunch", .kKernelLaunch },
    .{ "KernelExecute", .kKernelExecute },
    // TPU related.
    .{ "EnqueueRequestLocked", .kEnqueueRequestLocked },
    .{ "RunProgramRequest", .kRunProgramRequest },
    .{ "HostCallbackRequest", .kHostCallbackRequest },
    .{ "TransferH2DRequest", .kTransferH2DRequest },
    .{ "TransferPreprocessedH2DRequest", .kTransferPreprocessedH2DRequest },
    .{ "TransferD2HRequest", .kTransferD2HRequest },
    .{ "OnDeviceSendRequest", .kOnDeviceSendRequest },
    .{ "OnDeviceRecvRequest", .kOnDeviceRecvRequest },
    .{ "OnDeviceSendRecvLocalRequest", .kOnDeviceSendRecvLocalRequest },
    .{ "CustomWait", .kCustomWait },
    .{ "OnDeviceSendRequestMulti", .kOnDeviceSendRequestMulti },
    .{ "OnDeviceRecvRequestMulti", .kOnDeviceRecvRequestMulti },
    .{ "PjrtAsyncWait", .kPjrtAsyncWait },
    .{ "DoEnqueueProgram", .kDoEnqueueProgram },
    .{ "DoEnqueueContinuationProgram", .kDoEnqueueContinuationProgram },
    .{ "WriteHbm", .kWriteHbm },
    .{ "ReadHbm", .kReadHbm },
    .{ "TpuExecuteOp", .kTpuExecuteOp },
    .{ "CompleteCallbacks", .kCompleteCallbacks },
    .{ "TPUPartitionedCallOp-InitializeVarOnTPU", .kTpuPartitionedCallOpInitializeVarOnTpu },
    .{ "TPUPartitionedCallOp-ExecuteRemote", .kTpuPartitionedCallOpExecuteRemote },
    .{ "TPUPartitionedCallOp-ExecuteLocal", .kTpuPartitionedCallOpExecuteLocal },
    .{ "Linearize", .kLinearize },
    .{ "Delinearize", .kDelinearize },
    .{ "TransferBufferFromDevice-FastPath", .kTransferBufferFromDeviceFastPath },
    .{ "tpu::System::TransferToDevice=>IssueEvent", .kTransferToDeviceIssueEvent },
    .{ "tpu::System::TransferToDevice=>IssueEvent=>Done", .kTransferToDeviceDone },
    .{ "tpu::System::TransferFromDevice=>IssueEvent", .kTransferFromDeviceIssueEvent },
    .{ "tpu::System::TransferFromDevice=>IssueEvent=>Done", .kTransferFromDeviceDone },
    .{ "tpu::System::Execute", .kTpuSystemExecute },
});

pub const StatTypeMap = std.StaticStringMap(StatType).initComptime(&.{
    .{ "UnknownStatType", .kUnknownStatType },
    // TraceMe arguments.
    .{ "id", .kStepId },
    .{ "device_ordinal", .kDeviceOrdinal },
    .{ "chip_ordinal", .kChipOrdinal },
    .{ "node_ordinal", .kNodeOrdinal },
    .{ "model_id", .kModelId },
    .{ "queue_addr", .kQueueAddr },
    .{ "queue_id", .kQueueId },
    .{ "request_id", .kRequestId },
    .{ "run_id", .kRunId },
    .{ "replica_id", .kReplicaId },
    .{ "graph_type", .kGraphType },
    .{ "step_num", .kStepNum },
    .{ "iter_num", .kIterNum },
    .{ "index_on_host", .kIndexOnHost },
    .{ "allocator_name", .kAllocatorName },
    .{ "bytes_reserved", .kBytesReserved },
    .{ "bytes_allocated", .kBytesAllocated },
    .{ "bytes_available", .kBytesAvailable },
    .{ "fragmentation", .kFragmentation },
    .{ "peak_bytes_in_use", .kPeakBytesInUse },
    .{ "requested_bytes", .kRequestedBytes },
    .{ "allocation_bytes", .kAllocationBytes },
    .{ "addr", .kAddress },
    .{ "region_type", .kRegionType },
    .{ "data_type", .kDataType },
    .{ "shape", .kTensorShapes },
    .{ "layout", .kTensorLayout },
    .{ "kpi_name", .kKpiName },
    .{ "kpi_value", .kKpiValue },
    .{ "element_id", .kElementId },
    .{ "parent_id", .kParentId },
    .{ "core_type", .kCoreType },
    // XPlane semantics related.
    .{ "_pt", .kProducerType },
    .{ "_ct", .kConsumerType },
    .{ "_p", .kProducerId },
    .{ "_c", .kConsumerId },
    .{ "_r", .kIsRoot },
    .{ "_a", .kIsAsync },
    // Device trace arguments.
    .{ "device_id", .kDeviceId },
    .{ "device_type_string", .kDeviceTypeString },
    .{ "context_id", .kContextId },
    .{ "correlation_id", .kCorrelationId },
    .{ "memcpy_details", .kMemcpyDetails },
    .{ "memalloc_details", .kMemallocDetails },
    .{ "MemFree_details", .kMemFreeDetails },
    .{ "Memset_details", .kMemsetDetails },
    .{ "MemoryResidency_details", .kMemoryResidencyDetails },
    .{ "kernel_details", .kKernelDetails },
    .{ "nvtx_range", .kNVTXRange },
    .{ "stream", .kStream },
    // Stats added when processing traces.
    .{ "group_id", .kGroupId },
    .{ "flow", .kFlow },
    .{ "step_name", .kStepName },
    .{ "tf_op", .kTfOp },
    .{ "hlo_op", .kHloOp },
    .{ "deduplicated_name", .kDeduplicatedName },
    .{ "hlo_category", .kHloCategory },
    .{ "hlo_module", .kHloModule },
    .{ "program_id", .kProgramId },
    .{ "equation", .kEquation },
    .{ "is_eager", .kIsEager },
    .{ "is_func", .kIsFunc },
    .{ "tf_function_call", .kTfFunctionCall },
    .{ "tracing_count", .kTfFunctionTracingCount },
    .{ "flops", .kFlops },
    .{ "model_flops", .kModelFlops },
    .{ "bytes_accessed", .kBytesAccessed },
    .{ "memory_access_breakdown", .kMemoryAccessBreakdown },
    .{ "source", .kSourceInfo },
    .{ "model_name", .kModelName },
    .{ "model_version", .kModelVersion },
    .{ "bytes_transferred", .kBytesTransferred },
    .{ "queue", .kDmaQueue },
    .{ "dcn_collective_info", .kDcnCollectiveInfo },
    // Performance counter related.
    .{ "Raw Value", .kRawValue },
    .{ "Scaled Value", .kScaledValue },
    .{ "Thread Id", .kThreadId },
    .{ "matrix_unit_utilization_percent", .kMatrixUnitUtilizationPercent },
    // XLA metadata map related.
    .{ "Hlo Proto", .kHloProto },
    .{ "EdgeTPU Model information", .kEdgeTpuModelInfo },
    .{ "EdgeTPU Model Profile information", .kEdgeTpuModelProfileInfo },
    .{ "EdgeTPU MLIR", .kEdgeTpuMlir },
    // Device capability related.
    .{ "clock_rate", .kDevCapClockRateKHz },
    .{ "core_count", .kDevCapCoreCount },
    .{ "memory_bandwidth", .kDevCapMemoryBandwidth },
    .{ "memory_size", .kDevCapMemorySize },
    .{ "compute_cap_major", .kDevCapComputeCapMajor },
    .{ "compute_cap_minor", .kDevCapComputeCapMinor },
    .{ "peak_teraflops_per_second", .kDevCapPeakTeraflopsPerSecond },
    .{ "peak_hbm_bw_gigabytes_per_second", .kDevCapPeakHbmBwGigabytesPerSecond },
    .{ "peak_sram_rd_bw_gigabytes_per_second", .kDevCapPeakSramRdBwGigabytesPerSecond },
    .{ "peak_sram_wr_bw_gigabytes_per_second", .kDevCapPeakSramWrBwGigabytesPerSecond },
    .{ "device_vendor", .kDevVendor },
    // Batching related.
    .{ "batch_size_after_padding", .kBatchSizeAfterPadding },
    .{ "padding_amount", .kPaddingAmount },
    .{ "batching_input_task_size", .kBatchingInputTaskSize },
    // GPU related metrics.
    .{ "theoretical_occupancy_pct", .kTheoreticalOccupancyPct },
    .{ "occupancy_min_grid_size", .kOccupancyMinGridSize },
    .{ "occupancy_suggested_block_size", .kOccupancySuggestedBlockSize },
    // Aggregated Stat
    .{ "self_duration_ps", .kSelfDurationPs },
    .{ "min_duration_ps", .kMinDurationPs },
    .{ "total_profile_duration_ps", .kTotalProfileDurationPs },
    .{ "max_iteration_num", .kMaxIterationNum },
    .{ "device_type", .kDeviceType },
    .{ "uses_megacore", .kUsesMegaCore },
    .{ "symbol_id", .kSymbolId },
    .{ "hlo_category", .kHloCategory },
    .{ "tf_op_name", .kTfOpName },
    .{ "dma_stall_duration_ps", .kDmaStallDurationPs },
    .{ "key", .kKey },
    .{ "payload_size_bytes", .kPayloadSizeBytes },
    .{ "duration_us", .kDuration },
    .{ "buffer_size", .kBufferSize },
    .{ "transfers", .kTransfers },
    // Dcn message Stats
    .{ "dcn_label", .kDcnLabel },
    .{ "dcn_source_slice_id", .kDcnSourceSliceId },
    .{ "dcn_source_per_slice_device_id", .kDcnSourcePerSliceDeviceId },
    .{ "dcn_destination_slice_id", .kDcnDestinationSliceId },
    .{ "dcn_destination_per_slice_device_id", .kDcnDestinationPerSliceDeviceId },
    .{ "dcn_chunk", .kDcnChunk },
    .{ "dcn_loop_index", .kDcnLoopIndex },
    .{ "dropped_traces", .kDroppedTraces },
    .{ "cuda_graph_id", .kCudaGraphId },
    .{ "cuda_graph_exec_id", .kCudaGraphExecId },
    .{ "cuda_graph_orig_id", .kCudaGraphOrigId },
    .{ "step_idle_time_ps", .kStepIdleTimePs },
    .{ "gpu_device_name", .kGpuDeviceName },
    .{ "source_stack", .kSourceStack },
    .{ "device_offset_ps", .kDeviceOffsetPs },
    .{ "device_duration_ps", .kDeviceDurationPs },
});
