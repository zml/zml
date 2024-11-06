// Constants used as trace_viewer PID (device_id in trace_events.proto).
// PID 0 is unused.
// Support up to 500 accelerator devices.
pub const kFirstDeviceId = 1;
const kLastDeviceId = 500;
// Support Upto 200 custom planes as fake devices (i.e., planes with a
// "/custom:" prefix). See `<project_name>::kCustomPlanePrefix` for more
// information
const kFirstCustomPlaneDeviceId = kLastDeviceId + 1;
const kMaxCustomPlaneDevicesPerHost = 200;
const kLastCustomPlaneDeviceId = kFirstCustomPlaneDeviceId + kMaxCustomPlaneDevicesPerHost - 1;

// Host threads are shown as a single fake device.
pub const kHostThreadsDeviceId = kLastCustomPlaneDeviceId + 1;
