const std = @import("std");
const tf_op_utils = @import("tf_op_utils.zig");

// `HostEventType` uses the unconventional casing/formatting
// so that the string representation of the enum  used in the
// protobuf encoding directly maps to the zig enum tag name.
pub const HostEventType = enum(u16) {
    UnknownHostEventType = 0,
    TraceContext,
    SessionRun,
    FunctionRun,
    RunGraph,
    RunGraphDone,
    TfOpRun,
    EagerExecute,
    @"ExecutorState::Process",
    ExecutorDoneCallback,
    MemoryAllocation,
    MemoryDeallocation,
    // Performance counter related.
    RemotePerfCounter,
    // tf.data captured function events.
    @"InstantiatedCapturedFunction::Run",
    @"InstantiatedCapturedFunction::RunWithBorrowedArgs",
    @"InstantiatedCapturedFunction::RunInstantiated",
    @"InstantiatedCapturedFunction::RunAsync",
    // Loop ops.
    ParallelForOp,
    ForeverOp,
    @"WhileOp-EvalCond",
    @"WhileOp-StartBody",
    ForOp,
    // tf.data related.
    @"IteratorGetNextOp::DoCompute",
    @"IteratorGetNextAsOptionalOp::DoCompute",
    Iterator,
    @"Iterator::Prefetch::Generator",
    PrefetchProduce,
    PrefetchConsume,
    ParallelInterleaveProduce,
    ParallelInterleaveConsume,
    ParallelInterleaveInitializeInput,
    ParallelMapProduce,
    ParallelMapConsume,
    MapAndBatchProduce,
    MapAndBatchConsume,
    ParseExampleProduce,
    ParseExampleConsume,
    ParallelBatchProduce,
    ParallelBatchConsume,
    // Batching related.
    BatchingSessionRun,
    ProcessBatch,
    BrainSessionRun,
    ConcatInputTensors,
    MergeInputTensors,
    ScheduleWithoutSplit,
    ScheduleWithSplit,
    ScheduleWithEagerSplit,
    @"ASBSQueue::Schedule",
    // TFRT related.
    TfrtModelRun,
    // Serving related.
    ServingModelRun,
    // GPU related.
    KernelLaunch,
    KernelExecute,
    // TPU related
    EnqueueRequestLocked,
    RunProgramRequest,
    HostCallbackRequest,
    TransferH2DRequest,
    TransferPreprocessedH2DRequest,
    TransferD2HRequest,
    OnDeviceSendRequest,
    OnDeviceRecvRequest,
    OnDeviceSendRecvLocalRequest,
    CustomWait,
    OnDeviceSendRequestMulti,
    OnDeviceRecvRequestMulti,
    PjrtAsyncWait,
    DoEnqueueProgram,
    DoEnqueueContinuationProgram,
    WriteHbm,
    ReadHbm,
    TpuExecuteOp,
    CompleteCallbacks,
    @"tpu::System::TransferToDevice=>IssueEvent",
    @"tpu::System::TransferToDevice=>IssueEvent=>Done",
    @"tpu::System::TransferFromDevice=>IssueEvent",
    @"tpu::System::TransferFromDevice=>IssueEvent=>Done",
    @"tpu::System::Execute",
    @"TPUPartitionedCallOp-InitializeVarOnTPU",
    @"TPUPartitionedCallOp-ExecuteRemote",
    @"TPUPartitionedCallOp-ExecuteLocal",
    Linearize,
    Delinearize,
    @"TransferBufferFromDevice-FastPath",

    pub fn fromString(event_name: []const u8) ?HostEventType {
        return std.meta.stringToEnum(HostEventType, event_name);
    }

    pub fn fromTfOpEventType(event_name: []const u8) ?HostEventType {
        return switch (tf_op_utils.parseTfOpCategory(event_name)) {
            .tensorflow => .TfOpRun,
            .tf_data => .Iterator,
            else => null,
        };
    }

    pub fn isInternalEvent(event_type: HostEventType) bool {
        // TODO(b/162102421): Introduce a prefix for internal event names.
        return switch (event_type) {
            .MemoryAllocation,
            .MemoryDeallocation,
            .PrefetchProduce,
            .PrefetchConsume,
            .ParallelInterleaveProduce,
            .ParallelInterleaveConsume,
            .ParallelInterleaveInitializeInput,
            .ParallelMapProduce,
            .ParallelMapConsume,
            .MapAndBatchProduce,
            .MapAndBatchConsume,
            .ParseExampleProduce,
            .ParseExampleConsume,
            => true,
            else => false,
        };
    }
};

// `StatType` uses the unconventional casing/formatting
// so that the string representation of the enum  used in the
// protobuf encoding directly maps to the zig enum tag name.
pub const StatType = enum(u16) {
    UnknownStatType = 0,
    // TraceMe arguments.
    id,
    device_ordinal,
    chip_ordinal,
    node_ordinal,
    model_id,
    queue_id,
    queue_addr,
    request_id,
    run_id,
    replica_id,
    graph_type,
    step_num,
    iter_num,
    index_on_host,
    allocator_name,
    bytes_reserved,
    bytes_allocated,
    bytes_available,
    fragmentation,
    peak_bytes_in_use,
    requested_bytes,
    allocation_bytes,
    addr,
    region_type,
    data_type,
    shape,
    layout,
    kpi_name,
    kpi_value,
    element_id,
    parent_id,
    core_type,
    // XPlane semantics related.
    _pt,
    _ct,
    _p,
    _c,
    _r,
    _a,
    // Device trace arguments.
    device_id,
    device_type_string,
    context_id,
    correlation_id,
    // TODO(b/176137043): These "details" should differentiate between activity
    // and API event sources.
    memcpy_details,
    memalloc_details,
    MemFree_details,
    Memset_details,
    MemoryResidency_details,
    nvtx_range,
    kernel_details,
    stream,
    // Stats added when processing traces.
    group_id,
    flow,
    step_name,
    tf_op,
    hlo_op,
    deduplicated_name,
    hlo_category,
    hlo_module,
    program_id,
    equation,
    is_eager,
    is_func,
    tf_function_call,
    tracing_count,
    flops,
    model_flops,
    bytes_accessed,
    memory_access_breakdown,
    source,
    model_name,
    model_version,
    bytes_transferred,
    queue,
    dcn_collective_info,
    // Performance counter related.
    @"Raw Value",
    @"Scaled Value",
    @"Thread Id",
    matrix_unit_utilization_percent,
    // XLA metadata map related.
    @"Hlo Proto",
    // Device capability related.
    clock_rate,
    // For GPU, this is the number of SMs.
    core_count,
    memory_bandwidth,
    memory_size,
    compute_cap_major,
    compute_cap_minor,
    peak_teraflops_per_second,
    peak_hbm_bw_gigabytes_per_second,
    peak_sram_rd_bw_gigabytes_per_second,
    peak_sram_wr_bw_gigabytes_per_second,
    device_vendor,
    // Batching related.
    batch_size_after_padding,
    padding_amount,
    batching_input_task_size,
    // GPU occupancy metrics
    theoretical_occupancy_pct,
    occupancy_min_grid_size,
    occupancy_suggested_block_size,
    // Aggregated Stats
    self_duration_ps,
    min_duration_ps,
    total_profile_duration_ps,
    max_iteration_num,
    device_type,
    uses_megacore,
    symbol_id,
    tf_op_name,
    dma_stall_duration_ps,
    key,
    payload_size_bytes,
    duration_us,
    buffer_size,
    transfers,
    // Dcn message Stats
    dcn_label,
    dcn_source_slice_id,
    dcn_source_per_slice_device_id,
    dcn_destination_slice_id,
    dcn_destination_per_slice_device_id,
    dcn_chunk,
    dcn_loop_index,
    @"EdgeTPU Model information",
    @"EdgeTPU Model Profile information",
    @"EdgeTPU MLIR",
    dropped_traces,
    cuda_graph_id,
    // Many events have `.cuda_graph_id`, such as graph sub events when tracing is in
    // node level. Yet `.cuda_graph_exec_id` is used only for CudaGraphExecution events
    // on the GPU device when tracing is in graph level.
    cuda_graph_exec_id,
    cuda_graph_orig_id,
    step_idle_time_ps,
    gpu_device_name,
    source_stack,
    device_offset_ps,
    device_duration_ps,

    pub fn fromString(stat_name: []const u8) ?StatType {
        return std.meta.stringToEnum(StatType, stat_name);
    }

    pub fn isInternalStat(stat_type: StatType) bool {
        return switch (stat_type) {
            .kernel_details,
            ._pt,
            ._p,
            ._ct,
            ._c,
            ._r,
            .flops,
            .bytes_accessed,
            .program_id,
            .symbol_id,
            => true,
            else => false,
        };
    }
};
