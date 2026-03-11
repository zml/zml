/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __AMDSMI_H__
#define __AMDSMI_H__

/**
 * @file amdsmi.h
 * @brief AMD System Management Interface API
 */

#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#include <cstdint>
#else  // __cplusplus
#include <stdint.h>
#endif // __cplusplus

/**
 * @brief Initialization flags
 *
 * Initialization flags may be OR'd together and passed to ::amdsmi_init().
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{cpu_bm} @tag{guest_windows} @endcond
 */
typedef enum {
    AMDSMI_INIT_ALL_PROCESSORS = 0xFFFFFFFF,  //!< Initialize all processors
    AMDSMI_INIT_AMD_CPUS       = (1 << 0),    //!< Initialize AMD CPUS
    AMDSMI_INIT_AMD_GPUS       = (1 << 1),    //!< Initialize AMD GPUS
    AMDSMI_INIT_NON_AMD_CPUS   = (1 << 2),    //!< Initialize Non-AMD CPUS
    AMDSMI_INIT_NON_AMD_GPUS   = (1 << 3),    //!< Initialize Non-AMD GPUS
    AMDSMI_INIT_AMD_APUS       = (AMDSMI_INIT_AMD_CPUS | AMDSMI_INIT_AMD_GPUS) /**< Initialize AMD CPUS and GPUS
                                                                                    (Default option) */
} amdsmi_init_flags_t;

/**
 * @brief Maximum size definitions
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
#define AMDSMI_MAX_MM_IP_COUNT              8  //!< Maximum number of multimedia IP blocks
#define AMDSMI_MAX_STRING_LENGTH          256  //!< Maximum length for string buffers
#define AMDSMI_MAX_DEVICES                 32  //!< Maximum number of devices supported
#define AMDSMI_MAX_CACHE_TYPES             10  //!< Maximum number of cache types
#define AMDSMI_MAX_ACCELERATOR_PROFILE     32  //!< Maximum number of accelerator profiles
#define AMDSMI_MAX_CP_PROFILE_RESOURCES    32  //!< Maximum number of compute profile resources
#define AMDSMI_MAX_ACCELERATOR_PARTITIONS   8  //!< Maximum number of accelerator partitions
#define AMDSMI_MAX_NUM_NUMA_NODES          32  //!< Maximum number of NUMA nodes
#define AMDSMI_GPU_UUID_SIZE               38  //!< Size of GPU UUID string

/**
 * @brief Common defines
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
#define AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK 64  //!< Maximum number of XGMI physical links
#define AMDSMI_MAX_CONTAINER_TYPE          2  //!< Maximum number of container types

/**
 * @brief The following structure holds the gpu metrics values for a device.
 */

/**
 * @brief Unit conversion factor for HBM temperatures
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define CENTRIGRADE_TO_MILLI_CENTIGRADE 1000

/**
 * @brief This should match NUM_HBM_INSTANCES
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_NUM_HBM_INSTANCES 4

/**
 * @brief This should match MAX_NUM_VCN
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_MAX_NUM_VCN 4

/**
 * @brief This should match MAX_NUM_CLKS
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_MAX_NUM_CLKS 4

/**
 * @brief This should match MAX_NUM_XGMI_LINKS
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_MAX_NUM_XGMI_LINKS 8

/**
 * @brief This should match MAX_NUM_GFX_CLKS
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_MAX_NUM_GFX_CLKS 8

/**
 * @brief This should match AMDSMI_MAX_AID
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_MAX_AID 4

/**
 * @brief This should match AMDSMI_MAX_ENGINES
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_MAX_ENGINES 8

/**
 * @brief This should match AMDSMI_MAX_NUM_JPEG (8*4=32)
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_MAX_NUM_JPEG 32

/**
 * @brief Introduced in gpu metrics v1.8, document presents NUM_JPEG_ENG_V1
 * but will change to AMDSMI_MAX_NUM_JPEG_ENG_V1 for continuity
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_MAX_NUM_JPEG_ENG_V1 40

/**
 * @brief This should match AMDSMI_MAX_NUM_XCC;
 * XCC - Accelerated Compute Core, the collection of compute units,
 * ACE (Asynchronous Compute Engines), caches,
 * and global resources organized as one unit.
 *
 * Refer to amd.com documentation for more detail:
 * https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @endcond
 */
#define AMDSMI_MAX_NUM_XCC 8

/**
 * @brief This should match AMDSMI_MAX_NUM_XCP;
 * XCP - Accelerated Compute Processor,
 * also referred to as the Graphics Compute Partitions.
 * Each physical gpu could have a maximum of 8 separate partitions
 * associated with each (depending on ASIC support).
 *
 * Refer to amd.com documentation for more detail:
 * https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @endcond
 */
#define AMDSMI_MAX_NUM_XCP 8

/**
 * @brief Max Number of AFIDs that will be inside one cper entry
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
#define MAX_NUMBER_OF_AFIDS_PER_RECORD 12 //!< Maximum AFIDs per CPER record

/**
 * @brief String format
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
#define AMDSMI_TIME_FORMAT "%02d:%02d:%02d.%03d"                //!< Time format string
#define AMDSMI_DATE_FORMAT "%04d-%02d-%02d:%02d:%02d:%02d.%03d" //!< Date format string

/**
 * @brief library versioning
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */

//! Major version should be changed for every header change that breaks ABI
//! Such as adding/deleting APIs, changing names, fields of structures, etc.
#define AMDSMI_LIB_VERSION_MAJOR 26

//! Minor version should be updated for each API change, but without changing headers
#define AMDSMI_LIB_VERSION_MINOR 1

//! Release version should be set to 0 as default and can be updated by the PMs for each CSP point release
#define AMDSMI_LIB_VERSION_RELEASE 0

#define AMDSMI_LIB_VERSION_CREATE_STRING(MAJOR, MINOR, RELEASE) (#MAJOR "." #MINOR "." #RELEASE)
#define AMDSMI_LIB_VERSION_EXPAND_PARTS(MAJOR_STR, MINOR_STR, RELEASE_STR) AMDSMI_LIB_VERSION_CREATE_STRING(MAJOR_STR, MINOR_STR, RELEASE_STR)
#define AMDSMI_LIB_VERSION_STRING AMDSMI_LIB_VERSION_EXPAND_PARTS(AMDSMI_LIB_VERSION_MAJOR, AMDSMI_LIB_VERSION_MINOR, AMDSMI_LIB_VERSION_RELEASE)

/**
 * @brief GPU Capability info
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_MM_UVD,  //!< Multi-Media Unified Video Decoder
    AMDSMI_MM_VCE,  //!< Multi-Media Video Coding Engine
    AMDSMI_MM_VCN,  //!< Multi-Media Video Core Next
    AMDSMI_MM__MAX
} amdsmi_mm_ip_t;

/**
 * @brief Container
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_CONTAINER_LXC,     //!< Linux containers
    AMDSMI_CONTAINER_DOCKER   //!< Docker containers
} amdsmi_container_types_t;

/**
 * @brief opaque handler point to underlying implementation
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{cpu_bm} @tag{guest_windows} @endcond
 */
typedef void *amdsmi_processor_handle;
typedef void *amdsmi_socket_handle;

/**
 * @brief opaque handler point to underlying implementation
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef void *amdsmi_node_handle;

#ifdef ENABLE_ESMI_LIB

/**
 * @brief opaque handler point to underlying implementation
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef void *amdsmi_cpusocket_handle;

/**
 * @brief This structure holds HSMP Driver version information.
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
    uint32_t major;  //!< Major version number
    uint32_t minor;  //!< Minor version number
} amdsmi_hsmp_driver_version_t;

#endif

/**
 * @brief Processor types detectable by AMD SMI
 *
 * AMDSMI_PROCESSOR_TYPE_AMD_CPU      - CPU Socket is a physical component that holds the CPU.
 * AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE - CPU Cores are number of individual processing units within the CPU.
 * AMDSMI_PROCESSOR_TYPE_AMD_APU      - Combination of AMDSMI_PROCESSOR_TYPE_AMD_CPU and integrated GPU on single die
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{cpu_bm} @tag{guest_windows} @endcond
 */
typedef enum {
    AMDSMI_PROCESSOR_TYPE_UNKNOWN = 0,  //!< Unknown processor type
    AMDSMI_PROCESSOR_TYPE_AMD_GPU,      //!< AMD Graphics processor type
    AMDSMI_PROCESSOR_TYPE_AMD_CPU,      //!< AMD CPU processor type
    AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU,  //!< Non-AMD Graphics processor type
    AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU,  //!< Non-AMD CPU processor type
    AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE, //!< AMD CPU-Core processor type
    AMDSMI_PROCESSOR_TYPE_AMD_APU       //!< AMD Accelerated processor type (GPU and CPU)
} processor_type_t;

/**
 * @brief Error codes returned by amdsmi functions
 *
 * Please avoid status codes that are multiples of 256 (256, 512, etc..)
 * Return values in the shell get modulo 256 applied, meaning any multiple of 256 ends up as 0
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{cpu_bm} @tag{guest_windows} @endcond
 */
typedef enum {
    AMDSMI_STATUS_SUCCESS = 0,              //!< Call succeeded
    // Library usage errors
    AMDSMI_STATUS_INVAL = 1,                //!< Invalid parameters
    AMDSMI_STATUS_NOT_SUPPORTED = 2,        //!< Command not supported
    AMDSMI_STATUS_NOT_YET_IMPLEMENTED = 3,  //!< Not implemented yet
    AMDSMI_STATUS_FAIL_LOAD_MODULE = 4,     //!< Fail to load lib
    AMDSMI_STATUS_FAIL_LOAD_SYMBOL = 5,     //!< Fail to load symbol
    AMDSMI_STATUS_DRM_ERROR = 6,            //!< Error when call libdrm
    AMDSMI_STATUS_API_FAILED = 7,           //!< API call failed
    AMDSMI_STATUS_TIMEOUT = 8,              //!< Timeout in API call
    AMDSMI_STATUS_RETRY = 9,                //!< Retry operation
    AMDSMI_STATUS_NO_PERM = 10,             //!< Permission Denied
    AMDSMI_STATUS_INTERRUPT = 11,           //!< An interrupt occurred during execution of function
    AMDSMI_STATUS_IO = 12,                  //!< I/O Error
    AMDSMI_STATUS_ADDRESS_FAULT = 13,       //!< Bad address
    AMDSMI_STATUS_FILE_ERROR = 14,          //!< Problem accessing a file
    AMDSMI_STATUS_OUT_OF_RESOURCES = 15,    //!< Not enough memory
    AMDSMI_STATUS_INTERNAL_EXCEPTION = 16,  //!< An internal exception was caught
    AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS = 17, //!< The provided input is out of allowable or safe range
    AMDSMI_STATUS_INIT_ERROR = 18,          //!< An error occurred when initializing internal data structures
    AMDSMI_STATUS_REFCOUNT_OVERFLOW = 19,   //!< An internal reference counter exceeded INT32_MAX
    AMDSMI_STATUS_DIRECTORY_NOT_FOUND = 20, //!< Error when a directory is not found, maps to ENOTDIR
    // Processor related errors
    AMDSMI_STATUS_BUSY = 30,                //!< Processor busy
    AMDSMI_STATUS_NOT_FOUND = 31,           //!< Processor Not found
    AMDSMI_STATUS_NOT_INIT = 32,            //!< Processor not initialized
    AMDSMI_STATUS_NO_SLOT = 33,             //!< No more free slot
    AMDSMI_STATUS_DRIVER_NOT_LOADED = 34,   //!< Processor driver not loaded
    // Data and size errors
    AMDSMI_STATUS_MORE_DATA = 39,           //!< There is more data than the buffer size the user passed
    AMDSMI_STATUS_NO_DATA = 40,             //!< No data was found for a given input
    AMDSMI_STATUS_INSUFFICIENT_SIZE = 41,   //!< Not enough resources were available for the operation
    AMDSMI_STATUS_UNEXPECTED_SIZE = 42,     //!< An unexpected amount of data was read
    AMDSMI_STATUS_UNEXPECTED_DATA = 43,     //!< The data read or provided to function is not what was expected
    //esmi errors
    AMDSMI_STATUS_NON_AMD_CPU = 44,         //!< System has different cpu than AMD
    AMDSMI_STATUS_NO_ENERGY_DRV = 45,       //!< Energy driver not found
    AMDSMI_STATUS_NO_MSR_DRV = 46,          //!< MSR driver not found
    AMDSMI_STATUS_NO_HSMP_DRV = 47,         //!< HSMP driver not found
    AMDSMI_STATUS_NO_HSMP_SUP = 48,         //!< HSMP not supported
    AMDSMI_STATUS_NO_HSMP_MSG_SUP = 49,     //!< HSMP message/feature not supported
    AMDSMI_STATUS_HSMP_TIMEOUT = 50,        //!< HSMP message timed out
    AMDSMI_STATUS_NO_DRV = 51,              //!< No Energy and HSMP driver present
    AMDSMI_STATUS_FILE_NOT_FOUND = 52,      //!< file or directory not found
    AMDSMI_STATUS_ARG_PTR_NULL = 53,        //!< Parsed argument is invalid
    AMDSMI_STATUS_AMDGPU_RESTART_ERR = 54,  //!< AMDGPU restart failed
    AMDSMI_STATUS_SETTING_UNAVAILABLE = 55, //!< Setting is not available
    AMDSMI_STATUS_CORRUPTED_EEPROM = 56,    //!< EEPROM is corrupted
    // General errors
    AMDSMI_STATUS_MAP_ERROR = 0xFFFFFFFE,     //!< The internal library error did not map to a status code
    AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF  //!< An unknown error occurred
} amdsmi_status_t;

/**
 * @brief Clock types
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
typedef enum {
    AMDSMI_CLK_TYPE_SYS = 0x0,  //!< System clock
    AMDSMI_CLK_TYPE_FIRST = AMDSMI_CLK_TYPE_SYS,
    AMDSMI_CLK_TYPE_GFX = AMDSMI_CLK_TYPE_SYS,  //!< Graphics clock
    AMDSMI_CLK_TYPE_DF,         /**< Data Fabric clock (for ASICs
                                     running on a separate clock) */
    AMDSMI_CLK_TYPE_DCEF,       /**< Display Controller Engine Front clock,
                                     timing/bandwidth signals to display */
    AMDSMI_CLK_TYPE_SOC,        //!< System On Chip clock, integrated circuit frequency
    AMDSMI_CLK_TYPE_MEM,        //!< Memory clock speed, system operating frequency
    AMDSMI_CLK_TYPE_PCIE,       //!< PCI Express clock, high bandwidth peripherals
    AMDSMI_CLK_TYPE_VCLK0,      //!< Video 0 clock, video processing units
    AMDSMI_CLK_TYPE_VCLK1,      //!< Video 1 clock, video processing units
    AMDSMI_CLK_TYPE_DCLK0,      //!< Display 1 clock, timing signals for display output
    AMDSMI_CLK_TYPE_DCLK1,      //!< Display 2 clock, timing signals for display output
    AMDSMI_CLK_TYPE__MAX = AMDSMI_CLK_TYPE_DCLK1
} amdsmi_clk_type_t;

/**
 * @brief Accelerator Partition
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_ACCELERATOR_PARTITION_INVALID = 0,  //!< Invalid accelerator partition type
    AMDSMI_ACCELERATOR_PARTITION_SPX,          /**< Single GPU mode (SPX)- All XCCs work
                                                    together with shared memory */
    AMDSMI_ACCELERATOR_PARTITION_DPX,          /**< Dual GPU mode (DPX)- Half XCCs work
                                                    together with shared memory */
    AMDSMI_ACCELERATOR_PARTITION_TPX,          /**< Triple GPU mode (TPX)- One-third XCCs
                                                    work together with shared memory */
    AMDSMI_ACCELERATOR_PARTITION_QPX,          /**< Quad GPU mode (QPX)- Quarter XCCs
                                                    work together with shared memory */
    AMDSMI_ACCELERATOR_PARTITION_CPX,          /**< Core mode (CPX)- Per-chip XCC with
                                                    shared memory */
    AMDSMI_ACCELERATOR_PARTITION_MAX
} amdsmi_accelerator_partition_type_t;

/**
 * @brief Accelerator Partition Resource Types
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_ACCELERATOR_XCC,      //!< Compute complex or stream processors
    AMDSMI_ACCELERATOR_ENCODER,  //!< Video encoding
    AMDSMI_ACCELERATOR_DECODER,  //!< Video decoding
    AMDSMI_ACCELERATOR_DMA,      //!< Direct Memory Access, high speed data transfers
    AMDSMI_ACCELERATOR_JPEG,     //!< Encoding and Decoding jpeg engines
    AMDSMI_ACCELERATOR_MAX
} amdsmi_accelerator_partition_resource_type_t;

/**
 * @brief Compute Partition. This enum is used to identify
 * various compute partitioning settings.
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @endcond
 */
typedef enum {
    AMDSMI_COMPUTE_PARTITION_INVALID = 0,  //!< Invalid compute partition type
    AMDSMI_COMPUTE_PARTITION_SPX,  /**< Single GPU mode (SPX)- All XCCs work
                                        together with shared memory */
    AMDSMI_COMPUTE_PARTITION_DPX,  /**< Dual GPU mode (DPX)- Half XCCs work
                                        together with shared memory */
    AMDSMI_COMPUTE_PARTITION_TPX,  /**< Triple GPU mode (TPX)- One-third XCCs
                                        work together with shared memory */
    AMDSMI_COMPUTE_PARTITION_QPX,  /**< Quad GPU mode (QPX)- Quarter XCCs
                                        work together with shared memory */
    AMDSMI_COMPUTE_PARTITION_CPX   /**< Core mode (CPX)- Per-chip XCC with
                                        shared memory */
} amdsmi_compute_partition_type_t;

/**
 * @brief Memory Partitions
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_MEMORY_PARTITION_UNKNOWN = 0,
    AMDSMI_MEMORY_PARTITION_NPS1 = 1,  /**< NPS1 - All CCD & XCD data is interleaved
                                            across all 8 HBM stacks (all stacks/1) */
    AMDSMI_MEMORY_PARTITION_NPS2 = 2,  /**< NPS2 - 2 sets of CCDs or 4 XCD interleaved
                                            across the 4 HBM stacks per AID pair
                                            (8 stacks/2) */
    AMDSMI_MEMORY_PARTITION_NPS4 = 4,  /**< NPS4 - Each XCD data is interleaved
                                            across 2 (or single) HBM stacks
                                            (8 stacks/8 or 8 stacks/4) */
    AMDSMI_MEMORY_PARTITION_NPS8 = 8,  /**< NPS8 - Each XCD uses a single HBM stack
                                            (8 stacks/8). Or each XCD uses a single
                                            HBM stack & CCDs share 2 non-interleaved
                                            HBM stacks on its AID
                                            (AID[1,2,3] = 6 stacks/6) */
} amdsmi_memory_partition_type_t;

/**
 * @brief This enumeration is used to indicate from which part of the processor a
 * temperature reading should be obtained.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
typedef enum {
    AMDSMI_TEMPERATURE_TYPE_EDGE,    //!< Edge temperature
    AMDSMI_TEMPERATURE_TYPE_FIRST = AMDSMI_TEMPERATURE_TYPE_EDGE,
    AMDSMI_TEMPERATURE_TYPE_HOTSPOT, //!< Hottest temperature reported for entire die
    AMDSMI_TEMPERATURE_TYPE_JUNCTION = AMDSMI_TEMPERATURE_TYPE_HOTSPOT, //!< Synonymous with HOTSPOT
    AMDSMI_TEMPERATURE_TYPE_VRAM,    //!< VRAM temperature on graphics card
    AMDSMI_TEMPERATURE_TYPE_HBM_0,   //!< High Bandwidth 0 temperature per stack
    AMDSMI_TEMPERATURE_TYPE_HBM_1,   //!< High Bandwidth 1 temperature per stack
    AMDSMI_TEMPERATURE_TYPE_HBM_2,   //!< High Bandwidth 2 temperature per stack
    AMDSMI_TEMPERATURE_TYPE_HBM_3,   //!< High Bandwidth 3 temperature per stack
    AMDSMI_TEMPERATURE_TYPE_PLX,     //!< PCIe switch temperature

    // GPU Board Node temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_FIRST = 100,
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_RETIMER_X 
      = AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_FIRST,         //!< Retimer X temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC,         //!< OAM X IBC temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_IBC_2,       //!< OAM X IBC 2 temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR,    //!< OAM X VDD 1.8V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR, //!< OAM X 0.4V HBM B voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR, //!< OAM X 0.4V HBM D voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_NODE_LAST = 149,

    // GPU Board VR (Voltage Regulator) temperature 
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_FIRST = 150,
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD0
         = AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_FIRST,   //!< VDDCR VDD0 voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD1,        //!< VDDCR VDD1 voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD2,        //!< VDDCR VDD2 voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_VDD3,        //!< VDDCR VDD3 voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_A,       //!< VDDCR SOC A voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOC_C,       //!< VDDCR SOC C voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_A,     //!< VDDCR SOCIO A voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_SOCIO_C,     //!< VDDCR SOCIO C voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_085_HBM,       //!< VDD 0.85V HBM voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_B,    //!< VDDCR 1.1V HBM B voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDCR_11_HBM_D,    //!< VDDCR 1.1V HBM D voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDD_USR,           //!< VDD USR voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VDDIO_11_E32,      //!< VDDIO 1.1V E32 voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_GPUBOARD_VR_LAST = 199,

    // Baseboard System temperature 
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_FIRST = 200,
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA = AMDSMI_TEMPERATURE_TYPE_BASEBOARD_FIRST,  //!< UBB FPGA temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FRONT,          //!< UBB front temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_BACK,           //!< UBB back temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM7,           //!< UBB OAM7 temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_IBC,            //!< UBB IBC temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_UFPGA,          //!< UBB UFPGA temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_OAM1,           //!< UBB OAM1 temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_HSC,        //!< OAM 0-1 HSC temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_2_3_HSC,        //!< OAM 2-3 HSC temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_HSC,        //!< OAM 4-5 HSC temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_6_7_HSC,        //!< OAM 6-7 HSC temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_0V72_VR,   //!< UBB FPGA 0.72V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_UBB_FPGA_3V3_VR,    //!< UBB FPGA 3.3V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR,  //!< Retimer 0-1-2-3 1.2V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR,  //!< Retimer 4-5-6-7 1.2V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR, //!< Retimer 0-1 0.9V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR, //!< Retimer 4-5 0.9V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR, //!< Retimer 2-3 0.9V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR, //!< Retimer 6-7 0.9V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR, //!< OAM 0-1-2-3 3.3V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR, //!< OAM 4-5-6-7 3.3V voltage regulator temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC_HSC,            //!< IBC HSC temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC,                //!< IBC temperature
    AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST = 249,
    AMDSMI_TEMPERATURE_TYPE__MAX = AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST  //!< Maximum per GPU temperature type
} amdsmi_temperature_type_t;

/**
 * @brief The values of this enum are used to identify the various firmware
 * blocks.
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_FW_ID_SMU = 1,                   /**< System Management Unit (power management,
                                                 clock control, thermal monitoring, etc...) */
    AMDSMI_FW_ID_FIRST = AMDSMI_FW_ID_SMU,
    AMDSMI_FW_ID_CP_CE,                     //!< Compute Processor - Command_Engine (fetch, decode, dispatch)
    AMDSMI_FW_ID_CP_PFP,                    //!< Compute Processor - Pixel Front End Processor (pixelating process)
    AMDSMI_FW_ID_CP_ME,                     //!< Compute Processor - Micro Engine (specialize processing)
    AMDSMI_FW_ID_CP_MEC_JT1,                //!< Compute Processor - Micro Engine Controler Job Table 1 (queues, scheduling)
    AMDSMI_FW_ID_CP_MEC_JT2,                //!< Compute Processor - Micro Engine Controler Job Table 2 (queues, scheduling)
    AMDSMI_FW_ID_CP_MEC1,                   //!< Compute Processor - Micro Engine Controler 1 (scheduling, managing resources)
    AMDSMI_FW_ID_CP_MEC2,                   //!< Compute Processor - Micro Engine Controler 2 (scheduling, managing resources)
    AMDSMI_FW_ID_RLC,                       //!< Rasterizer and L2 Cache (rasterization processs)
    AMDSMI_FW_ID_SDMA0,                     //!< System Direct Memory Access 0 (high speed data transfers)
    AMDSMI_FW_ID_SDMA1,                     //!< System Direct Memory Access 1 (high speed data transfers)
    AMDSMI_FW_ID_SDMA2,                     //!< System Direct Memory Access 2 (high speed data transfers)
    AMDSMI_FW_ID_SDMA3,                     //!< System Direct Memory Access 3 (high speed data transfers)
    AMDSMI_FW_ID_SDMA4,                     //!< System Direct Memory Access 4 (high speed data transfers)
    AMDSMI_FW_ID_SDMA5,                     //!< System Direct Memory Access 5 (high speed data transfers)
    AMDSMI_FW_ID_SDMA6,                     //!< System Direct Memory Access 6 (high speed data transfers)
    AMDSMI_FW_ID_SDMA7,                     //!< System Direct Memory Access 7 (high speed data transfers)
    AMDSMI_FW_ID_VCN,                       //!< Video Core Next (encoding and decoding)
    AMDSMI_FW_ID_UVD,                       //!< Unified Video Decoder (decode specific video formats)
    AMDSMI_FW_ID_VCE,                       //!< Video Coding Engine (Encoding video)
    AMDSMI_FW_ID_ISP,                       //!< Image Signal Processor (processing raw image data from sensors)
    AMDSMI_FW_ID_DMCU_ERAM,                 //!< Digital Micro Controller Unit - Embedded RAM (memory used by DMU)
    AMDSMI_FW_ID_DMCU_ISR,                  //!< Digital Micro Controller Unit - Interrupt Service Routine (interrupt handlers)
    AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM,  //!< Rasterizier and L2 Cache Restore List Graphics Processor Memory
    AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM,  //!< Rasterizier and L2 Cache Restore List System RAM Memory
    AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL,     //!< Rasterizier and L2 Cache Restore List Control
    AMDSMI_FW_ID_RLC_V,                     //!< Rasterizier and L2 Cache Virtual memory
    AMDSMI_FW_ID_MMSCH,                     //!< Multi-Media Shader Hardware Scheduler
    AMDSMI_FW_ID_PSP_SYSDRV,                //!< Platform Security Processor System Driver
    AMDSMI_FW_ID_PSP_SOSDRV,                //!< Platform Security Processor Secure Operating System Driver
    AMDSMI_FW_ID_PSP_TOC,                   //!< Platform Security Processor Table of Contents
    AMDSMI_FW_ID_PSP_KEYDB,                 //!< Platform Security Processor Table of Contents
    AMDSMI_FW_ID_DFC,                       //!< Data Fabric Controler (bandwidth and coherency)
    AMDSMI_FW_ID_PSP_SPL,                   //!< Platform Security Processor Secure Program Loader
    AMDSMI_FW_ID_DRV_CAP,                   //!< Driver Capabilities (capabilities, features)
    AMDSMI_FW_ID_MC,                        //!< Memory Contoller (RAM and VRAM)
    AMDSMI_FW_ID_PSP_BL,                    //!< Platform Security Processor Bootloader (initial firmware)
    AMDSMI_FW_ID_CP_PM4,                    //!< Compute Processor Packet Processor 4 (processing command packets)
    AMDSMI_FW_ID_RLC_P,                     //!< Rasterizier and L2 Cache Partition
    AMDSMI_FW_ID_SEC_POLICY_STAGE2,         //!< Security Policy Stage 2 (security features)
    AMDSMI_FW_ID_REG_ACCESS_WHITELIST,      //!< Register Access Whitelist (Prevent unathorizied access)
    AMDSMI_FW_ID_IMU_DRAM,                  //!< Input/Output Memory Management Unit - Dynamic RAM
    AMDSMI_FW_ID_IMU_IRAM,                  //!< Input/Output Memory Management Unit - Instruction RAM
    AMDSMI_FW_ID_SDMA_TH0,                  //!< System Direct Memory Access - Thread Handler 0
    AMDSMI_FW_ID_SDMA_TH1,                  //!< System Direct Memory Access - Thread Handler 1
    AMDSMI_FW_ID_CP_MES,                    //!< Compute Processor - Micro Engine Scheduler
    AMDSMI_FW_ID_MES_KIQ,                   //!< Micro Engine Scheduler - Kernel Indirect Queue
    AMDSMI_FW_ID_MES_STACK,                 //!< Micro Engine Scheduler - Stack
    AMDSMI_FW_ID_MES_THREAD1,               //!< Micro Engine Scheduler - Thread 1
    AMDSMI_FW_ID_MES_THREAD1_STACK,         //!< Micro Engine Scheduler - Thread 1 Stack
    AMDSMI_FW_ID_RLX6,                      //!< Hardware Block RLX6
    AMDSMI_FW_ID_RLX6_DRAM_BOOT,            //!< Hardware Block RLX6 - Dynamic Ram Boot
    AMDSMI_FW_ID_RS64_ME,                   //!< Hardware Block RS64 - Micro Engine
    AMDSMI_FW_ID_RS64_ME_P0_DATA,           //!< Hardware Block RS64 - Micro Engine Partition 0 Data
    AMDSMI_FW_ID_RS64_ME_P1_DATA,           //!< Hardware Block RS64 - Micro Engine Partition 1 Data
    AMDSMI_FW_ID_RS64_PFP,                  //!< Hardware Block RS64 - Pixel Front End Processor
    AMDSMI_FW_ID_RS64_PFP_P0_DATA,          //!< Hardware Block RS64 - Pixel Front End Processor Partition 0 Data
    AMDSMI_FW_ID_RS64_PFP_P1_DATA,          //!< Hardware Block RS64 - Pixel Front End Processor Partition 1 Data
    AMDSMI_FW_ID_RS64_MEC,                  //!< Hardware Block RS64 - Micro Engine Controller
    AMDSMI_FW_ID_RS64_MEC_P0_DATA,          //!< Hardware Block RS64 - Micro Engine Controller Partition 0 Data
    AMDSMI_FW_ID_RS64_MEC_P1_DATA,          //!< Hardware Block RS64 - Micro Engine Controller Partition 1 Data
    AMDSMI_FW_ID_RS64_MEC_P2_DATA,          //!< Hardware Block RS64 - Micro Engine Controller Partition 2 Data
    AMDSMI_FW_ID_RS64_MEC_P3_DATA,          //!< Hardware Block RS64 - Micro Engine Controller Partition 3 Data
    AMDSMI_FW_ID_PPTABLE,                   //!< Power Policy Table (power management policies)
    AMDSMI_FW_ID_PSP_SOC,                   //!< Platform Security Processor - System On a Chip
    AMDSMI_FW_ID_PSP_DBG,                   //!< Platform Security Processor - Debug
    AMDSMI_FW_ID_PSP_INTF,                  //!< Platform Security Processor - Interface
    AMDSMI_FW_ID_RLX6_CORE1,                //!< Hardware Block RLX6 - Core 1
    AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1,      //!< Hardware Block RLX6 Core 1 - Dynamic RAM Boot
    AMDSMI_FW_ID_RLCV_LX7,                  //!< Hardware Block RLCV - Subsystem LX7
    AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST,     //!< Rasterizier and L2 Cache - Save Restore List
    AMDSMI_FW_ID_ASD,                       //!< Asynchronous Shader Dispatcher
    AMDSMI_FW_ID_TA_RAS,                    //!< Trusted Applications - Reliablity Availability and Serviceability
    AMDSMI_FW_ID_TA_XGMI,                   //!< Trusted Applications - Reliablity XGMI
    AMDSMI_FW_ID_RLC_SRLG,                  //!< Rasterizier and L2 Cache - Shared Resource Local Group
    AMDSMI_FW_ID_RLC_SRLS,                  //!< Rasterizier and L2 Cache - Shared Resource Local Segment
    AMDSMI_FW_ID_PM,                        //!< Power Management Firmware
    AMDSMI_FW_ID_DMCU,                      //!< Display Micro-Controller Unit
    AMDSMI_FW_ID_PLDM_BUNDLE,               //!< Platform Level Data Model Firmware Bundle
    AMDSMI_FW_ID__MAX
} amdsmi_fw_block_t;

/**
 * @brief vRam Types. This enum is used to identify various VRam types.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_VRAM_TYPE_UNKNOWN = 0,  //!< Unknown memory type
    // HBM
    AMDSMI_VRAM_TYPE_HBM   = 1,   //!< High Bandwidth Memory
    AMDSMI_VRAM_TYPE_HBM2  = 2,   //!< High Bandwidth Memory, Generation 2
    AMDSMI_VRAM_TYPE_HBM2E = 3,   //!< High Bandwidth Memory, Generation 2 Enhanced
    AMDSMI_VRAM_TYPE_HBM3  = 4,   //!< High Bandwidth Memory, Generation 3
    AMDSMI_VRAM_TYPE_HBM3E = 5,   //!< High Bandwidth Memory, Generation 3 Enhanced
    // DDR
    AMDSMI_VRAM_TYPE_DDR2  = 10,  //!< Double Data Rate, Generation 2
    AMDSMI_VRAM_TYPE_DDR3  = 11,  //!< Double Data Rate, Generation 3
    AMDSMI_VRAM_TYPE_DDR4  = 12,  //!< Double Data Rate, Generation 4
    AMDSMI_VRAM_TYPE_DDR5  = 13,  //!< Double Data Rate, Generation 5
    // GDDR
    AMDSMI_VRAM_TYPE_GDDR1 = 17,  //!< Graphics Double Data Rate, Generation 1
    AMDSMI_VRAM_TYPE_GDDR2 = 18,  //!< Graphics Double Data Rate, Generation 2
    AMDSMI_VRAM_TYPE_GDDR3 = 19,  //!< Graphics Double Data Rate, Generation 3
    AMDSMI_VRAM_TYPE_GDDR4 = 20,  //!< Graphics Double Data Rate, Generation 4
    AMDSMI_VRAM_TYPE_GDDR5 = 21,  //!< Graphics Double Data Rate, Generation 5
    AMDSMI_VRAM_TYPE_GDDR6 = 22,  //!< Graphics Double Data Rate, Generation 6
    AMDSMI_VRAM_TYPE_GDDR7 = 23,  //!< Graphics Double Data Rate, Generation 7
    // LPDDR
    AMDSMI_VRAM_TYPE_LPDDR4 = 30,  //!< Low Power Double Data Rate, Generation 4
    AMDSMI_VRAM_TYPE_LPDDR5 = 31,  //!< Low Power Double Data Rate, Generation 5
    AMDSMI_VRAM_TYPE__MAX = AMDSMI_VRAM_TYPE_LPDDR5
} amdsmi_vram_type_t;

/**
 * @brief This structure represents a range (e.g., frequencies or voltages).
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint64_t lower_bound;  //!< Lower bound of range
    uint64_t upper_bound;  //!< Upper bound of range
    uint64_t reserved[2];
} amdsmi_range_t;

/**
 * @brief XGMI Information
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint8_t xgmi_lanes;
    uint64_t xgmi_hive_id;
    uint64_t xgmi_node_id;
    uint32_t index;
    uint32_t reserved[9];
} amdsmi_xgmi_info_t;

/**
 * @brief VRam Usage
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint32_t vram_total;  //!< In MB
    uint32_t vram_used;   //!< In MB
    uint32_t reserved[2];
} amdsmi_vram_usage_t;

/**
 * @brief This structure hold violation status information.
 *        Note: for MI3x asics and higher, older ASICs will show unsupported.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint64_t reference_timestamp;  //!< Represents CPU timestamp in microseconds (uS)
    uint64_t violation_timestamp;  //!< Violation time.  Units in nanoseconds (ns) {@linux_bm}, in milliseconds (ms) {@host}
    uint64_t acc_counter;          //!< Current accumulated counter; Max uint64 means unsupported
    uint64_t acc_prochot_thrm;     //!< Current accumulated processor hot violation count; Max uint64 means unsupported
    uint64_t acc_ppt_pwr;          //!< PVIOL; Current accumulated Package Power Tracking (PPT) count; Max uint64 means unsupported
    uint64_t acc_socket_thrm;      //!< TVIOL; Current accumulated Socket thermal count; Max uint64 means unsupported
    uint64_t acc_vr_thrm;          //!< Current accumulated voltage regulator count; Max uint64 means unsupported
    uint64_t acc_hbm_thrm;         //!< Current accumulated High Bandwidth Memory (HBM) thermal count; Max uint64 means unsupported
    uint64_t acc_gfx_clk_below_host_limit; /**< UPDATED in new driver 1.8: use new *_gfx_clk_below_host_limit_pwr, *_gfx_clk_below_host_limit_thm, *_gfx_clk_below_host_limit_total values!
                                                Current gfx clock below host limit count; Max uint64 means unsupported.*/

    uint64_t per_prochot_thrm;     //!< Processor hot violation % (greater than 0% is a violation); Max uint64 means unsupported
    uint64_t per_ppt_pwr;          //!< PVIOL; Package Power Tracking (PPT) violation % (greater than 0% is a violation); Max uint64 means unsupported
    uint64_t per_socket_thrm;      //!< TVIOL; Socket thermal violation % (greater than 0% is a violation); Max uint64 means unsupported
    uint64_t per_vr_thrm;          //!< Voltage regulator violation % (greater than 0% is a violation); Max uint64 means unsupported
    uint64_t per_hbm_thrm;         //!< High Bandwidth Memory (HBM) thermal violation % (greater than 0% is a violation); Max uint64 means unsupported
    uint64_t per_gfx_clk_below_host_limit;  /**< UPDATED in new driver 1.8: use new *_gfx_clk_below_host_limit_pwr, *_gfx_clk_below_host_limit_thm, *_gfx_clk_below_host_limit_total values!
                                                 Gfx clock below host limit violation % (greater than 0% is a violation); Max uint64 means unsupported.*/

    uint8_t active_prochot_thrm;   //!< Processor hot violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint8_t active_ppt_pwr;        //!< Package Power Tracking (PPT) violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint8_t active_socket_thrm;    //!< Socket thermal violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint8_t active_vr_thrm;        //!< Voltage regulator violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint8_t active_hbm_thrm;       //!< High Bandwidth Memory (HBM) thermal violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint8_t active_gfx_clk_below_host_limit;  /**< UPDATED in new driver 1.8: use new *_gfx_clk_below_host_limit_total values!
                                                   Gfx clock below host limit violation; 1 = active 0 = not active; Max uint8 means unsupported.*/
    //GPU metrics 1.8 violations
    uint64_t acc_gfx_clk_below_host_limit_pwr[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];    //!< New Driver 1.8 fields: Current gfx clock below host limit power count; Max uint64 means unsupported
    uint64_t acc_gfx_clk_below_host_limit_thm[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];    //!< New Driver 1.8 fields: Current gfx clock below host limit thermal count; Max uint64 means unsupported
    uint64_t acc_low_utilization[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];                 //!< New Driver 1.8 fields: Current low utilization count; Max uint64 means unsupported
    uint64_t acc_gfx_clk_below_host_limit_total[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];  //!< New Driver 1.8 fields: Current gfx clock below host limit total count; Max uint64 means unsupported

    uint64_t per_gfx_clk_below_host_limit_pwr[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];    //!< New Driver 1.8 fields: Gfx clock below host limit power violation % (greater than 0% is a violation); Max uint64 means unsupported
    uint64_t per_gfx_clk_below_host_limit_thm[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];    //!< New Driver 1.8 fields: Gfx clock below host limit violation % (greater than 0% is a violation); Max uint64 means unsupported
    uint64_t per_low_utilization[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];                 //!< New Driver 1.8 fields: Low utilization violation % (greater than 0% is a violation); Max uint64 means unsupported
    uint64_t per_gfx_clk_below_host_limit_total[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];  //!< New Driver 1.8 fields: Any Gfx clock below host limit violation % (greater than 0% is a violation); Max uint64 means unsupported

    uint8_t active_gfx_clk_below_host_limit_pwr[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];  //!< New Driver 1.8 fields: Gfx clock below host limit power violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint8_t active_gfx_clk_below_host_limit_thm[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];  //!< New Driver 1.8 fields: Gfx clock below host limit thermal violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint8_t active_low_utilization[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];               //!< New Driver 1.8 fields: Low utilization violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint8_t active_gfx_clk_below_host_limit_total[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];//!< New Driver 1.8 fields: Any Gfx clock host limit violation; 1 = active 0 = not active; Max uint8 means unsupported
    uint64_t reserved[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];   // reserved for new violation info
    uint64_t reserved2[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];  // reserved for new violation info
    uint64_t reserved3[AMDSMI_MAX_NUM_XCP][AMDSMI_MAX_NUM_XCC];  // reserved for new violation info
} amdsmi_violation_status_t;

/**
 * @brief Frequency Range
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amdsmi_range_t supported_freq_range;  //!< In MHz
    amdsmi_range_t current_freq_range;    //!< In MHz
    uint32_t reserved[8];
} amdsmi_frequency_range_t;

/**
 * @brief bdf types
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef union {
    struct bdf_ {
        uint64_t function_number : 3;
        uint64_t device_number : 5;
        uint64_t bus_number : 8;
        uint64_t domain_number : 48;
    } bdf;
    struct {
        uint64_t function_number : 3;
        uint64_t device_number : 5;
        uint64_t bus_number : 8;
        uint64_t domain_number : 48;
    };
    uint64_t as_uint;
} amdsmi_bdf_t;

/**
 * @brief Structure holds enumeration information
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_1vf} @tag{guest_mvf} @endcond
 */
typedef struct {
    uint32_t drm_render; //!< the render node under /sys/class/drm/renderD*
    uint32_t drm_card;   //!< the graphic card device under /sys/class/drm/card*
    uint32_t hsa_id;     //!< the HSA enumeration ID
    uint32_t hip_id;     //!< the HIP enumeration ID
    char hip_uuid[AMDSMI_MAX_STRING_LENGTH];  //!< the HIP unique identifer
} amdsmi_enumeration_info_t;

/**
 * @brief Card Form Factor
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
typedef enum {
    AMDSMI_CARD_FORM_FACTOR_PCIE,    //!< PCIE card form factor
    AMDSMI_CARD_FORM_FACTOR_OAM,     //!< OAM form factor
    AMDSMI_CARD_FORM_FACTOR_CEM,     //!< CEM form factor
    AMDSMI_CARD_FORM_FACTOR_UNKNOWN  //!< Unknown Form factor
} amdsmi_card_form_factor_t;

/**
 * @brief pcie information
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
typedef struct {
    struct pcie_static_ {
        uint16_t max_pcie_width;              //!< maximum number of PCIe lanes
        uint32_t max_pcie_speed;              //!< maximum PCIe speed in GT/s
        uint32_t pcie_interface_version;      //!< PCIe interface version
        amdsmi_card_form_factor_t slot_type;  //!< card form factor
        uint32_t max_pcie_interface_version;  //!< maximum PCIe link generation
        uint64_t reserved[9];
    } pcie_static;
    struct pcie_metric_ {
        uint16_t pcie_width;                  //!< current PCIe width
        uint32_t pcie_speed;                  //!< current PCIe speed in MT/s
        uint32_t pcie_bandwidth;              //!< current instantaneous PCIe bandwidth in Mb/s
        uint64_t pcie_replay_count;           //!< total number of the replays issued on the PCIe link
        uint64_t pcie_l0_to_recovery_count;   //!< total number of times the PCIe link transitioned from L0 to the recovery state
        uint64_t pcie_replay_roll_over_count; //!< total number of replay rollovers issued on the PCIe link
        uint64_t pcie_nak_sent_count;         //!< total number of NAKs issued on the PCIe link by the device
        uint64_t pcie_nak_received_count;     //!< total number of NAKs issued on the PCIe link by the receiver
        uint32_t pcie_lc_perf_other_end_recovery_count;  //!< PCIe other end recovery counter
        uint64_t reserved[12];
    } pcie_metric;
    uint64_t reserved[32];
} amdsmi_pcie_info_t;

/**
 * @brief Power Cap Information
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    uint64_t power_cap;          //!< current power cap Units uW {@linux_bm} or W {@host}
    uint64_t default_power_cap;  //!< default power cap Units uW {@linux_bm} or W {@host}
    uint64_t dpm_cap;            //!< dpm power cap Units MHz {@linux_bm} or Hz {@host}
    uint64_t min_power_cap;      //!< minimum power cap Units uW {@linux_bm} or W {@host}
    uint64_t max_power_cap;      //!< maximum power cap Units uW {@linux_bm} or W {@host}
    uint64_t reserved[3];
} amdsmi_power_cap_info_t;

/**
 * @brief Power Cap Package Power Tracking (PPT) type
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_POWER_CAP_TYPE_PPT0,       //!< PPT0 power cap; lower limit, filtered input
    AMDSMI_POWER_CAP_TYPE_PPT1,       //!< PPT1 power cap; higher limit, raw input
} amdsmi_power_cap_type_t;

/**
 * @brief VBios Information
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @tag{host} @endcond
 */
typedef struct {
    char name[AMDSMI_MAX_STRING_LENGTH];
    char build_date[AMDSMI_MAX_STRING_LENGTH];
    char part_number[AMDSMI_MAX_STRING_LENGTH];
    char version[AMDSMI_MAX_STRING_LENGTH];
    char boot_firmware[AMDSMI_MAX_STRING_LENGTH]; // UBL (Unified BootLoader) Version information
    uint64_t reserved[36];
} amdsmi_vbios_info_t;

/**
 * @brief cache properties
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_CACHE_PROPERTY_ENABLED    = 0x00000001,  //!< Cache enabled
    AMDSMI_CACHE_PROPERTY_DATA_CACHE = 0x00000002,  //!< Data cache
    AMDSMI_CACHE_PROPERTY_INST_CACHE = 0x00000004,  //!< Instruction cache
    AMDSMI_CACHE_PROPERTY_CPU_CACHE  = 0x00000008,  //!< CPU cache
    AMDSMI_CACHE_PROPERTY_SIMD_CACHE = 0x00000010   //!< Single Instruction, Multiple Data Cache
} amdsmi_cache_property_type_t;

/**
 * @brief GPU Cache Information
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    uint32_t num_cache_types;
    struct cache_ {
        uint32_t cache_properties;    //!< amdsmi_cache_property_type_t which is a bitmask
        uint32_t cache_size;          //!< In KB
        uint32_t cache_level;
        uint32_t max_num_cu_shared;   //!< Indicates how many Compute Units share this cache instance
        uint32_t num_cache_instance;  //!< total number of instance of this cache type
        uint32_t reserved[3];
    } cache[AMDSMI_MAX_CACHE_TYPES];
    uint32_t reserved[15];
} amdsmi_gpu_cache_info_t;

/**
 * @brief Firmware Information
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
typedef struct {
    uint8_t num_fw_info;
    struct fw_info_list_ {
        amdsmi_fw_block_t fw_id;
        uint64_t fw_version;
        uint64_t reserved[2];
    } fw_info_list[AMDSMI_FW_ID__MAX];
    uint32_t reserved[7];
} amdsmi_fw_info_t;

/**
 * @brief ASIC Information
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @tag{host} @endcond
 */
typedef struct {
    char  market_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t vendor_id;    //!< Use 32 bit to be compatible with other platform.
    char vendor_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t subvendor_id; //!< The subsystem vendor ID
    uint64_t device_id;    //!< The device ID of a GPU
    uint32_t rev_id;       //!< The revision ID of a GPU
    char asic_serial[AMDSMI_MAX_STRING_LENGTH];
    uint32_t oam_id;       //!< 0xFFFFFFFF if not supported
    uint32_t num_of_compute_units;     //!< 0xFFFFFFFF if not supported
    uint64_t target_graphics_version;  //!< 0xFFFFFFFFFFFFFFFF if not supported
    uint32_t subsystem_id; //!> The subsystem ID
    uint32_t reserved[21];
} amdsmi_asic_info_t;

/**
 * @brief Structure holds kfd information
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint64_t kfd_id;   //!< 0xFFFFFFFFFFFFFFFF if not supported
    uint32_t node_id;  //!< 0xFFFFFFFF if not supported
    uint32_t current_partition_id;  //!< 0xFFFFFFFF if not supported
    uint32_t reserved[12];
} amdsmi_kfd_info_t;

/**
 * @brief This union holds memory partition bitmask.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef union {
    struct nps_flags_ {
        uint32_t nps1_cap :1;  //!< bool 1 = true; 0 = false
        uint32_t nps2_cap :1;  //!< bool 1 = true; 0 = false
        uint32_t nps4_cap :1;  //!< bool 1 = true; 0 = false
        uint32_t nps8_cap :1;  //!< bool 1 = true; 0 = false
        uint32_t reserved :28;
    } nps_flags;
    uint32_t nps_cap_mask;
} amdsmi_nps_caps_t;

/**
 * @brief Memory Partition Configuration.
 * This structure is used to identify various memory partition configurations.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    amdsmi_nps_caps_t partition_caps;
    amdsmi_memory_partition_type_t mp_mode;
    uint32_t num_numa_ranges;
    struct numa_range_ {
        amdsmi_vram_type_t memory_type;
        uint64_t start;
        uint64_t end;
    } numa_range[AMDSMI_MAX_NUM_NUMA_NODES];
    uint64_t reserved[11];
} amdsmi_memory_partition_config_t;

/**
 * @brief Accelerator Partition Resource Profile
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    amdsmi_accelerator_partition_type_t  profile_type;   //!< SPX, DPX, QPX, CPX and so on
    uint32_t num_partitions;        //!< On MI300X: SPX=>1, DPX=>2, QPX=>4, CPX=>8; length of resources
    amdsmi_nps_caps_t memory_caps;  //!< Possible memory partition capabilities
    uint32_t profile_index;         //!< Index in the profiles array in amdsmi_accelerator_partition_profile_t
    uint32_t num_resources;         //!< length of index_of_resources_profile
    uint32_t resources[AMDSMI_MAX_ACCELERATOR_PARTITIONS][AMDSMI_MAX_CP_PROFILE_RESOURCES];
    uint64_t reserved[13];
} amdsmi_accelerator_partition_profile_t;

/**
 * @brief  Accelerator Partition Resources.
 * This struct is used to identify various partition resource profiles.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    uint32_t profile_index;
    amdsmi_accelerator_partition_resource_type_t resource_type;
    uint32_t partition_resource;             //!< Resources a partition can use, which may be shared
    uint32_t num_partitions_share_resource;  //!< If it is greater than 1, then resource is shared.
    uint64_t reserved[6];
} amdsmi_accelerator_partition_resource_profile_t;

/**
 * @brief Accelerator Partition Profile Configurations
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
  uint32_t num_profiles;            //!< The length of profiles array
  uint32_t num_resource_profiles;
  amdsmi_accelerator_partition_resource_profile_t resource_profiles[AMDSMI_MAX_CP_PROFILE_RESOURCES];
  uint32_t default_profile_index;  //!< The index of the default profile in the profiles array
  amdsmi_accelerator_partition_profile_t profiles[AMDSMI_MAX_ACCELERATOR_PROFILE];
  uint64_t reserved[30];
} amdsmi_accelerator_partition_profile_config_t;

/**
 * @brief Link type
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_LINK_TYPE_INTERNAL = 0,        //!< Internal Link Type, within chip
    AMDSMI_LINK_TYPE_PCIE = 1,            //!< Peripheral Component Interconnect Express Link Type
    AMDSMI_LINK_TYPE_XGMI = 2,            //!< GPU Memory Interconnect (multi GPU communication)
    AMDSMI_LINK_TYPE_NOT_APPLICABLE = 3,  //!< Not Applicable Link Type
    AMDSMI_LINK_TYPE_UNKNOWN = 4          //!< Unknown Link Type
} amdsmi_link_type_t;

/**
 * @brief This structure holds CPU utilization information.
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
  uint32_t cpu_util_total;
  uint32_t cpu_util_user;
  uint32_t cpu_util_nice;
  uint32_t cpu_util_sys;
  uint32_t cpu_util_irq;
} amdsmi_cpu_util_t;

/**
 * @brief Link Metrics
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    uint32_t num_links;     //!< number of links
    struct _links {
        amdsmi_bdf_t bdf;               //!< bdf of the destination gpu
        uint32_t bit_rate;              //!< current link speed in Gb/s
        uint32_t max_bandwidth;         //!< max bandwidth of the link in Gb/s
        amdsmi_link_type_t link_type;   //!< type of the link
        uint64_t read;                  //!< total data received for each link in KB
        uint64_t write;                 //!< total data transfered for each link in KB
        uint64_t reserved[1];
    } links[AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK];
    uint64_t reserved[7];
} amdsmi_link_metrics_t;

/**
 * @brief VRam Information
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    amdsmi_vram_type_t vram_type;
    char  vram_vendor[AMDSMI_MAX_STRING_LENGTH];
    uint64_t vram_size;           //!< vram size in MB
    uint32_t vram_bit_width;      //!< In bits
    uint64_t vram_max_bandwidth;  //!< The VRAM max bandwidth at current memory clock (GB/s)
  uint64_t reserved[37];
} amdsmi_vram_info_t;

/**
 * @brief Driver Information
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @tag{host} @endcond
 */
typedef struct {
    char driver_version[AMDSMI_MAX_STRING_LENGTH];
    char driver_date[AMDSMI_MAX_STRING_LENGTH];
    char driver_name[AMDSMI_MAX_STRING_LENGTH];
} amdsmi_driver_info_t;

/**
 * @brief Board Information
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    char model_number[AMDSMI_MAX_STRING_LENGTH];
    char product_serial[AMDSMI_MAX_STRING_LENGTH];
    char fru_id[AMDSMI_MAX_STRING_LENGTH];
    char product_name[AMDSMI_MAX_STRING_LENGTH];
    char manufacturer_name[AMDSMI_MAX_STRING_LENGTH];
    uint64_t reserved[64];
} amdsmi_board_info_t;

/**
 * @brief Power Information
 *
 * Unsupported struct members are set to UINT32_MAX
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint64_t socket_power;          //!< Socket power in W
    uint32_t current_socket_power;  //!< Current socket power in W, Mi 300+ Series cards
    uint32_t average_socket_power;  //!< Average socket power in W, Navi + Mi 200 and earlier Series cards
    uint64_t gfx_voltage;           //!< GFX voltage measurement in mV
    uint64_t soc_voltage;           //!< SOC voltage measurement in mV
    uint64_t mem_voltage;           //!< MEM voltage measurement in mV
    uint32_t power_limit;           //!< The power limit in W
    uint64_t reserved[18];
} amdsmi_power_info_t;

/**
 * @brief Clock Information
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @tag{host} @endcond
 */
typedef struct {
    uint32_t clk;            //!< In MHz
    uint32_t min_clk;        //!< In MHz
    uint32_t max_clk;        //!< In MHz
    uint8_t clk_locked;      //!< True/False
    uint8_t clk_deep_sleep;  //!< True/False
    uint32_t reserved[4];
} amdsmi_clk_info_t;

/**
 * @brief Engine Usage
 * amdsmi_engine_usage_t:
 * This structure holds common
 * GPU activity values seen in both BM or
 * SRIOV
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @tag{host} @endcond
 **/
typedef struct {
    uint32_t gfx_activity;  //!< In %
    uint32_t umc_activity;  //!< In %
    uint32_t mm_activity;   //!< In %
    uint32_t reserved[13];
} amdsmi_engine_usage_t;

/**
 * @brief Process Handle
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @endcond
 */
typedef uint32_t amdsmi_process_handle_t;

/**
 * @brief Process Information
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @endcond
 */
typedef struct {
    char name[AMDSMI_MAX_STRING_LENGTH];
    amdsmi_process_handle_t pid;
    uint64_t mem;  //!< In Bytes
    struct engine_usage_ {
        uint64_t gfx;  //!< In nano-secs
        uint64_t enc;  //!< In nano-secs
        uint32_t reserved[12];
    } engine_usage; //!< time the process spends using these engines in ns
    struct memory_usage_ {
        uint64_t gtt_mem;   //!< In Bytes
        uint64_t cpu_mem;   //!< In Bytes
        uint64_t vram_mem;  //!< In Bytes
        uint32_t reserved[10];
    } memory_usage;  //!< In Bytes
    char container_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t cu_occupancy;  //!< Num CUs utilized
    uint32_t evicted_time;    //!< Time that queues are evicted on a GPU in milliseconds
    uint32_t reserved[10];
} amdsmi_proc_info_t;

/**
 * @brief IO Link P2P Capability
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    uint8_t is_iolink_coherent;       //!< 1 = true, 0 = false, UINT8_MAX = Not defined
    uint8_t is_iolink_atomics_32bit;  //!< 1 = true, 0 = false, UINT8_MAX = Not defined
    uint8_t is_iolink_atomics_64bit;  //!< 1 = true, 0 = false, UINT8_MAX = Not defined
    uint8_t is_iolink_dma;            //!< 1 = true, 0 = false, UINT8_MAX = Not defined
    uint8_t is_iolink_bi_directional; //!< 1 = true, 0 = false, UINT8_MAX = Not defined
} amdsmi_p2p_capability_t;

//! Guaranteed maximum possible number of supported frequencies
//! @cond @tag{gpu_bm_linux} @endcond
#define AMDSMI_MAX_NUM_FREQUENCIES 33

//! Maximum possible value for fan speed. Should be used as the denominator
//! when determining fan speed percentage.
//! @cond @tag{gpu_bm_linux} @endcond
#define AMDSMI_MAX_FAN_SPEED 255

//! The number of points that make up a voltage-frequency curve definition
//! @cond @tag{gpu_bm_linux} @endcond
#define AMDSMI_NUM_VOLTAGE_CURVE_POINTS 3

/**
 * @brief PowerPlay performance levels
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_DEV_PERF_LEVEL_AUTO = 0,         //!< Performance level is "auto"
    AMDSMI_DEV_PERF_LEVEL_FIRST = AMDSMI_DEV_PERF_LEVEL_AUTO,
    AMDSMI_DEV_PERF_LEVEL_LOW,              //!< Keep PowerPlay levels "low", regardless of workload
    AMDSMI_DEV_PERF_LEVEL_HIGH,             //!< Keep PowerPlay levels "high", regardless of workload
    AMDSMI_DEV_PERF_LEVEL_MANUAL,           /**< Only use values defined by manually
                                                 setting the AMDSMI_CLK_TYPE_SYS speed */
    AMDSMI_DEV_PERF_LEVEL_STABLE_STD,       //!< Stable power state with profiling clocks
    AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK,      //!< Stable power state with peak clocks
    AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK,  //!< Stable power state with minimum memory clock
    AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK,  //!< Stable power state with minimum system clock
    AMDSMI_DEV_PERF_LEVEL_DETERMINISM,      //!< Performance determinism state
    AMDSMI_DEV_PERF_LEVEL_LAST = AMDSMI_DEV_PERF_LEVEL_DETERMINISM,
    AMDSMI_DEV_PERF_LEVEL_UNKNOWN = 0x100   //!< Unknown performance level
} amdsmi_dev_perf_level_t;

/**
 * @brief Handle to performance event counter
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef uintptr_t amdsmi_event_handle_t;

/**
 * @brief Event Groups
 * Enum denoting an event group. The value of the enum is the
 * base value for all the event enums in the group.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_EVNT_GRP_XGMI = 0,            //!< Data Fabric (XGMI) related events
    AMDSMI_EVNT_GRP_XGMI_DATA_OUT = 10,  //!< XGMI Outbound data
    AMDSMI_EVNT_GRP_INVALID = 0xFFFFFFFF //!< Unknown Event Group
} amdsmi_event_group_t;

/**
 * @brief Event types
 * Event type enum. Events belonging to a particular event group
 * ::amdsmi_event_group_t should begin enumerating at the ::amdsmi_event_group_t
 * value for that group.
 *
 * Data beats sent to neighbor 0; Each beat represents 32 bytes.<br><br>
 *
 * XGMI throughput can be calculated by multiplying a BEATs event
 * such as ::AMDSMI_EVNT_XGMI_0_BEATS_TX by 32 and dividing by
 * the time for which event collection occurred,
 * ::amdsmi_counter_value_t.time_running (which is in nanoseconds). To get
 * bytes per second, multiply this value by 10<sup>9</sup>.<br>
 * <br>
 * Throughput = BEATS/time_running * 10<sup>9</sup>  (bytes/second)<br>
 *
 * Events in the AMDSMI_EVNT_GRP_XGMI_DATA_OUT group measure
 * the number of beats sent on an XGMI link. Each beat represents
 * 32 bytes. AMDSMI_EVNT_XGMI_DATA_OUT_n represents the number of
 * outbound beats (each representing 32 bytes) on link n.<br><br>
 *
 * XGMI throughput can be calculated by multiplying a event
 * such as ::AMDSMI_EVNT_XGMI_DATA_OUT_n by 32 and dividing by
 * the time for which event collection occurred,
 * ::amdsmi_counter_value_t.time_running (which is in nanoseconds). To get
 * bytes per second, multiply this value by 10<sup>9</sup>.<br>
 * <br>
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_EVNT_FIRST = AMDSMI_EVNT_GRP_XGMI,

    AMDSMI_EVNT_XGMI_FIRST = AMDSMI_EVNT_GRP_XGMI,
    AMDSMI_EVNT_XGMI_0_NOP_TX = AMDSMI_EVNT_XGMI_FIRST,  //!< NOPs sent to neighbor 0
    AMDSMI_EVNT_XGMI_0_REQUEST_TX,  //!< Outgoing requests to neighbor 0
    AMDSMI_EVNT_XGMI_0_RESPONSE_TX, //!< Outgoing responses to neighbor 0
    AMDSMI_EVNT_XGMI_0_BEATS_TX,    //!< Throughput = BEATS/time_running 10^9 bytes/sec
    AMDSMI_EVNT_XGMI_1_NOP_TX,      //!< NOPs sent to neighbor 1
    AMDSMI_EVNT_XGMI_1_REQUEST_TX,  //!< Outgoing requests to neighbor 1
    AMDSMI_EVNT_XGMI_1_RESPONSE_TX, //!< Outgoing responses to neighbor 1
    AMDSMI_EVNT_XGMI_1_BEATS_TX,    //!< Data beats sent to neighbor 1; Each beat represents 32 bytes
    AMDSMI_EVNT_XGMI_LAST = AMDSMI_EVNT_XGMI_1_BEATS_TX,
    AMDSMI_EVNT_XGMI_DATA_OUT_FIRST = AMDSMI_EVNT_GRP_XGMI_DATA_OUT,
    AMDSMI_EVNT_XGMI_DATA_OUT_0 = AMDSMI_EVNT_XGMI_DATA_OUT_FIRST,  //!< Outbound beats to neighbor 0
    AMDSMI_EVNT_XGMI_DATA_OUT_1,    //!< Outbound beats to neighbor 1
    AMDSMI_EVNT_XGMI_DATA_OUT_2,    //!< Outbound beats to neighbor 2
    AMDSMI_EVNT_XGMI_DATA_OUT_3,    //!< Outbound beats to neighbor 3
    AMDSMI_EVNT_XGMI_DATA_OUT_4,    //!< Outbound beats to neighbor 4
    AMDSMI_EVNT_XGMI_DATA_OUT_5,    //!< Outbound beats to neighbor 5
    AMDSMI_EVNT_XGMI_DATA_OUT_LAST = AMDSMI_EVNT_XGMI_DATA_OUT_5,
    AMDSMI_EVNT_LAST = AMDSMI_EVNT_XGMI_DATA_OUT_LAST
} amdsmi_event_type_t;

/**
 * @brief Event counter commands
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_CNTR_CMD_START = 0,  //!< Start the counter
    AMDSMI_CNTR_CMD_STOP        /**< Stop the counter; note that this should not
                                     be used before reading */
} amdsmi_counter_command_t;

/**
 * @brief Counter value
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint64_t value;         //!< Counter value
    uint64_t time_enabled;  //!< Time that the counter was enabled in nanoseconds
    uint64_t time_running;  //!< Time that the counter was running in nanoseconds
} amdsmi_counter_value_t;

/**
 * @brief Event notification event types
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_EVT_NOTIF_NONE = 0,                          //!< No events
    AMDSMI_EVT_NOTIF_VMFAULT = 1,                       //!< Virtual Memory Page Fault Event
    AMDSMI_EVT_NOTIF_FIRST = AMDSMI_EVT_NOTIF_VMFAULT,
    AMDSMI_EVT_NOTIF_THERMAL_THROTTLE = 2,              //!< thermal throttle
    AMDSMI_EVT_NOTIF_GPU_PRE_RESET = 3,                 //!< pre-reset
    AMDSMI_EVT_NOTIF_GPU_POST_RESET = 4,                //!< post-reset
    AMDSMI_EVT_NOTIF_MIGRATE_START = 5,                 //!< migrate start
    AMDSMI_EVT_NOTIF_MIGRATE_END = 6,                   //!< migrate end
    AMDSMI_EVT_NOTIF_PAGE_FAULT_START = 7,              //!< page fault start
    AMDSMI_EVT_NOTIF_PAGE_FAULT_END = 8,                //!< page fault end
    AMDSMI_EVT_NOTIF_QUEUE_EVICTION = 9,                //!< queue eviction
    AMDSMI_EVT_NOTIF_QUEUE_RESTORE = 10,                //!< queue restore
    AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU = 11,               //!< unmap from GPU
    AMDSMI_EVT_NOTIF_PROCESS_START = 12,                //!< KFD process start
    AMDSMI_EVT_NOTIF_PROCESS_END = 13,                  //!< KFD process end
    AMDSMI_EVT_NOTIF_LAST = AMDSMI_EVT_NOTIF_PROCESS_END
} amdsmi_evt_notification_type_t;

/**
 * @brief Macro to generate event bitmask from event id
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
#define AMDSMI_EVENT_MASK_FROM_INDEX(i) (1ULL << ((i) - 1))

/**
 * @brief Event notification data returned from event notification API
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amdsmi_processor_handle processor_handle;  //!< Handler of device that corresponds to the event
    amdsmi_evt_notification_type_t event;      //!< Event type
    char message[AMDSMI_MAX_STRING_LENGTH];    //!< Event message
} amdsmi_evt_notification_data_t;

/**
 * @brief Temperature Metrics. This enum is used to identify various
 * temperature metrics. Corresponding values will be in Celcius
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @tag{guest_windows} @endcond
 */
typedef enum {
    AMDSMI_TEMP_CURRENT = 0x0,   //!< Current temperature
    AMDSMI_TEMP_FIRST = AMDSMI_TEMP_CURRENT,
    AMDSMI_TEMP_MAX,             //!< Max temperature
    AMDSMI_TEMP_MIN,             //!< Min temperature
    AMDSMI_TEMP_MAX_HYST,        /**< Max limit hysteresis temperature
                                      (Absolute temperature, not a delta) */
    AMDSMI_TEMP_MIN_HYST,        /**< Min limit hysteresis temperature
                                      (Absolute temperature, not a delta) */
    AMDSMI_TEMP_CRITICAL,        /**< Critical max limit temperature, typically
                                      greater than max temperatures */
    AMDSMI_TEMP_CRITICAL_HYST,   /**< Critical hysteresis limit temperature
                                      (Absolute temperature, not a delta) */
    AMDSMI_TEMP_EMERGENCY,       /**< Emergency max temperature, for chips
                                      supporting more than two upper temperature
                                      limits. Must be equal or greater than
                                      corresponding temp_crit values */
    AMDSMI_TEMP_EMERGENCY_HYST,  /**< Emergency hysteresis limit temperature
                                      (Absolute temperature, not a delta) */
    AMDSMI_TEMP_CRIT_MIN,        /**< Critical min temperature, typically
                                      lower than minimum temperatures */
    AMDSMI_TEMP_CRIT_MIN_HYST,   /**< Min Hysteresis critical limit temperature
                                      (Absolute temperature, not a delta) */
    AMDSMI_TEMP_OFFSET,          /**< Temperature offset which is added to the
                                      temperature reading by the chip */
    AMDSMI_TEMP_LOWEST,          //!< Historical min temperature
    AMDSMI_TEMP_HIGHEST,         //!< Historical max temperature
    AMDSMI_TEMP_SHUTDOWN,        //!< Shutdown temperature
    AMDSMI_TEMP_LAST = AMDSMI_TEMP_SHUTDOWN
} amdsmi_temperature_metric_t;

/**
 * @brief Voltage Metrics.  This enum is used to identify various
 * Volatge metrics. Corresponding values will be in millivolt.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_VOLT_CURRENT = 0x0,  //!< Voltage current value.

    AMDSMI_VOLT_FIRST = AMDSMI_VOLT_CURRENT,
    AMDSMI_VOLT_MAX,       //!< Voltage max value.
    AMDSMI_VOLT_MIN_CRIT,  //!< Voltage critical min value.
    AMDSMI_VOLT_MIN,       //!< Voltage min value.
    AMDSMI_VOLT_MAX_CRIT,  //!< Voltage critical max value.
    AMDSMI_VOLT_AVERAGE,   //!< Average voltage.
    AMDSMI_VOLT_LOWEST,    //!< Historical minimum voltage.
    AMDSMI_VOLT_HIGHEST,   //!< Historical maximum voltage.

    AMDSMI_VOLT_LAST = AMDSMI_VOLT_HIGHEST
} amdsmi_voltage_metric_t;

/**
 * @brief This ennumeration is used to indicate which type of
 * voltage reading should be obtained.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_VOLT_TYPE_FIRST = 0,

    AMDSMI_VOLT_TYPE_VDDGFX = AMDSMI_VOLT_TYPE_FIRST,  //!< Vddgfx GPU voltage
    AMDSMI_VOLT_TYPE_VDDBOARD,                         //!< Voltage for VDDBOARD
    AMDSMI_VOLT_TYPE_LAST = AMDSMI_VOLT_TYPE_VDDBOARD,
    AMDSMI_VOLT_TYPE_INVALID = 0xFFFFFFFF              //!< Invalid type
} amdsmi_voltage_type_t;

/**
 * @brief Pre-set Profile Selections. These bitmasks can be AND'd with the
 * ::amdsmi_power_profile_status_t.available_profiles returned from
 * :: amdsmi_get_gpu_power_profile_presets to determine which power profiles
 * are supported by the system.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_PWR_PROF_PRST_CUSTOM_MASK = 0x1,        //!< Custom Power Profile
    AMDSMI_PWR_PROF_PRST_VIDEO_MASK = 0x2,         //!< Video Power Profile
    AMDSMI_PWR_PROF_PRST_POWER_SAVING_MASK = 0x4,  //!< Power Saving Profile
    AMDSMI_PWR_PROF_PRST_COMPUTE_MASK = 0x8,       //!< Compute Saving Profile
    AMDSMI_PWR_PROF_PRST_VR_MASK = 0x10,           //!< VR Power Profile

    // 3D Full Screen Power Profile
    AMDSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK = 0x20,  //!< 3D Full Screen Profile
    AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT = 0x40,    //!< Default Boot Up Profile
    AMDSMI_PWR_PROF_PRST_LAST = AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT,

    // Invalid power profile
    AMDSMI_PWR_PROF_PRST_INVALID = 0xFFFFFFFFFFFFFFFF  //!< Invalid Power Profile
} amdsmi_power_profile_preset_masks_t;

/**
 * @brief This enum is used to identify different GPU blocks.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_GPU_BLOCK_INVALID =   0,          //!< Invalid block
    AMDSMI_GPU_BLOCK_FIRST =     (1ULL << 0),
    AMDSMI_GPU_BLOCK_UMC =       AMDSMI_GPU_BLOCK_FIRST,  //!< UMC block
    AMDSMI_GPU_BLOCK_SDMA =      (1ULL << 1),   //!< SDMA block
    AMDSMI_GPU_BLOCK_GFX =       (1ULL << 2),   //!< GFX block
    AMDSMI_GPU_BLOCK_MMHUB =     (1ULL << 3),   //!< MMHUB block
    AMDSMI_GPU_BLOCK_ATHUB =     (1ULL << 4),   //!< ATHUB block
    AMDSMI_GPU_BLOCK_PCIE_BIF =  (1ULL << 5),   //!< PCIE_BIF block
    AMDSMI_GPU_BLOCK_HDP =       (1ULL << 6),   //!< HDP block
    AMDSMI_GPU_BLOCK_XGMI_WAFL = (1ULL << 7),   //!< XGMI block
    AMDSMI_GPU_BLOCK_DF =        (1ULL << 8),   //!< DF block
    AMDSMI_GPU_BLOCK_SMN =       (1ULL << 9),   //!< SMN block
    AMDSMI_GPU_BLOCK_SEM =       (1ULL << 10),  //!< SEM block
    AMDSMI_GPU_BLOCK_MP0 =       (1ULL << 11),  //!< MP0 block
    AMDSMI_GPU_BLOCK_MP1 =       (1ULL << 12),  //!< MP1 block
    AMDSMI_GPU_BLOCK_FUSE =      (1ULL << 13),  //!< Fuse block
    AMDSMI_GPU_BLOCK_MCA =       (1ULL << 14),  //!< MCA block
    AMDSMI_GPU_BLOCK_VCN =       (1ULL << 15),  //!< VCN block
    AMDSMI_GPU_BLOCK_JPEG =      (1ULL << 16),  //!< JPEG block
    AMDSMI_GPU_BLOCK_IH =        (1ULL << 17),  //!< IH block
    AMDSMI_GPU_BLOCK_MPIO =      (1ULL << 18),  //!< MPIO block
    AMDSMI_GPU_BLOCK_LAST =      AMDSMI_GPU_BLOCK_MPIO,
    AMDSMI_GPU_BLOCK_RESERVED =  (1ULL << 63)
} amdsmi_gpu_block_t;

/**
 * @brief The clk limit type
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum  {
    CLK_LIMIT_MIN,  //!< Min Clock value in MHz
    CLK_LIMIT_MAX   //!< Max Clock value in MHz
} amdsmi_clk_limit_type_t;

/**
 * @brief Cper sev
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED = 0,  //!< CPER Non-Fatal Uncorrected severity
    AMDSMI_CPER_SEV_FATAL                 = 1,  //!< CPER Fatal severity
    AMDSMI_CPER_SEV_NON_FATAL_CORRECTED   = 2,  //!< CPER Non-Fatal Corrected severity
    AMDSMI_CPER_SEV_NUM                   = 3,  //!< CPER severity Number
    AMDSMI_CPER_SEV_UNUSED                = 10  //!< CPER Unused severity
} amdsmi_cper_sev_t;

/**
 * @brief Cper notify
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_CPER_NOTIFY_TYPE_CMC  = 0x450eBDD72DCE8BB1,  //!< Corrected Memory Check
    AMDSMI_CPER_NOTIFY_TYPE_CPE  = 0x4a55D8434E292F96,  //!< Corrected Platform Error
    AMDSMI_CPER_NOTIFY_TYPE_MCE  = 0x4cc5919CE8F56FFE,  //!< Machine Check Exception
    AMDSMI_CPER_NOTIFY_TYPE_PCIE = 0x4dfc1A16CF93C01F,  //!< PCI Express Error
    AMDSMI_CPER_NOTIFY_TYPE_INIT = 0x454a9308CC5263E8,  //!< Initialization Error
    AMDSMI_CPER_NOTIFY_TYPE_NMI  = 0x42c9B7E65BAD89FF,  //!< Non_Maskable Interrupt
    AMDSMI_CPER_NOTIFY_TYPE_BOOT = 0x409aAB403D61A466,  //!< Boot Error
    AMDSMI_CPER_NOTIFY_TYPE_DMAR = 0x4c27C6B3667DD791,  //!< Direct Memory Access Remapping Error
    AMDSMI_CPER_NOTIFY_TYPE_SEA  = 0x11E4BBE89A78788A,  //!< System Error Architecture
    AMDSMI_CPER_NOTIFY_TYPE_SEI  = 0x4E87B0AE5C284C81,  //!< System Error Interface
    AMDSMI_CPER_NOTIFY_TYPE_PEI  = 0x4214520409A9D5AC,  //!< Platform Error Interface
    AMDSMI_CPER_NOTIFY_TYPE_CXL_COMPONENT = 0x49A341DF69293BC9  //!< Compute Express Link Component Error
} amdsmi_cper_notify_type_t;

/**
 * @brief The current ECC state
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_RAS_ERR_STATE_NONE = 0,   //!< No current errors
    AMDSMI_RAS_ERR_STATE_DISABLED,   //!< ECC is disabled
    AMDSMI_RAS_ERR_STATE_PARITY,     //!< ECC errors present, but type unknown
    AMDSMI_RAS_ERR_STATE_SING_C,     //!< Single correctable error
    AMDSMI_RAS_ERR_STATE_MULT_UC,    //!< Multiple uncorrectable errors
    AMDSMI_RAS_ERR_STATE_POISON,     /**< Firmware detected error and isolated
                                          page. Treat as uncorrectable */
    AMDSMI_RAS_ERR_STATE_ENABLED,    //!< ECC is enabled

    AMDSMI_RAS_ERR_STATE_LAST = AMDSMI_RAS_ERR_STATE_ENABLED,
    AMDSMI_RAS_ERR_STATE_INVALID = 0xFFFFFFFF
} amdsmi_ras_err_state_t;

/**
 * @brief Types of memory
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_MEM_TYPE_FIRST = 0,

    AMDSMI_MEM_TYPE_VRAM = AMDSMI_MEM_TYPE_FIRST,  //!< VRAM memory
    AMDSMI_MEM_TYPE_VIS_VRAM,                      //!< VRAM memory that is visible
    AMDSMI_MEM_TYPE_GTT,                           //!< GTT memory

    AMDSMI_MEM_TYPE_LAST = AMDSMI_MEM_TYPE_GTT
} amdsmi_memory_type_t;

/**
 * @brief The values of this enum are used as frequency identifiers.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_FREQ_IND_MIN = 0,  //!< Index used for the minimum frequency value
    AMDSMI_FREQ_IND_MAX = 1,  //!< Index used for the maximum frequency value
    AMDSMI_FREQ_IND_INVALID = 0xFFFFFFFF  //!< An invalid frequency index
} amdsmi_freq_ind_t;

/**
 * @brief XGMI Status
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_XGMI_STATUS_NO_ERRORS = 0,   //!< XGMI No Errors
    AMDSMI_XGMI_STATUS_ERROR,           //!< XGMI Errors
    AMDSMI_XGMI_STATUS_MULTIPLE_ERRORS  //!< XGMI Multiple Errors
} amdsmi_xgmi_status_t;

/**
 * @brief Bitfield used in various AMDSMI calls
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef uint64_t amdsmi_bit_field_t;

/**
 * @brief Reserved Memory Page States
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_MEM_PAGE_STATUS_RESERVED = 0,  //!< Reserved. This gpu page is reserved and not available for use
    AMDSMI_MEM_PAGE_STATUS_PENDING,       /**< Pending. This gpu page is marked as bad and will be marked
                                               reserved at the next window */
    AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE   //!< Unable to reserve this page
} amdsmi_memory_page_status_t;

/**
 * @brief The utilization counter type
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_UTILIZATION_COUNTER_FIRST = 0,
    // Course grain activity counters
    AMDSMI_COARSE_GRAIN_GFX_ACTIVITY = AMDSMI_UTILIZATION_COUNTER_FIRST, //!< Course Grain Graphic Activity
    AMDSMI_COARSE_GRAIN_MEM_ACTIVITY,  //!< Course Grain Memory Activity
    AMDSMI_COARSE_DECODER_ACTIVITY,    //!< Course Grain Decoder Activity
    // Fine grain activity counters
    AMDSMI_FINE_GRAIN_GFX_ACTIVITY = 100,  //!< Fine Grain Graphic Activity
    AMDSMI_FINE_GRAIN_MEM_ACTIVITY = 101,  //!< Fine Grain Memory Activity
    AMDSMI_FINE_DECODER_ACTIVITY   = 102,  //!< Fine Grain Decoder Activity
    AMDSMI_UTILIZATION_COUNTER_LAST = AMDSMI_FINE_DECODER_ACTIVITY
} amdsmi_utilization_counter_type_t;

#define AMDSMI_MAX_UTILIZATION_VALUES 4  //!< The max number of values per counter type

/**
 * @brief The utilization counter data
 *
 * The max number of values per counter type
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amdsmi_utilization_counter_type_t type;              //!< Utilization counter type
    uint64_t value;                                      //!< Coarse grain activity counter value (average)
    uint64_t fine_value[AMDSMI_MAX_UTILIZATION_VALUES];  //!< Utilization counter value
    uint16_t fine_value_count;
} amdsmi_utilization_counter_t;

/**
 * @brief Reserved Memory Page Record
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint64_t page_address;               //!< Start address of page
    uint64_t page_size;                  //!< Page size
    amdsmi_memory_page_status_t status;  //!< Page "reserved" status
} amdsmi_retired_page_record_t;

/**
 * @brief This structure contains information about which power profiles are
 * supported by the system for a given device, and which power profile is
 * currently active.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amdsmi_bit_field_t available_profiles;        //!< Which profiles are supported by this system
    amdsmi_power_profile_preset_masks_t current;  //!< Which power profile is currently active
    uint32_t num_profiles;                        //!< How many power profiles are available
} amdsmi_power_profile_status_t;

/**
 * @brief This structure holds information about clock frequencies.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    bool has_deep_sleep;     //!< Deep Sleep frequency is only supported by some GPUs
    uint32_t num_supported;  //!< The number of supported frequencies
    uint32_t current;        //!< The current frequency index in MHz
    uint64_t frequency[AMDSMI_MAX_NUM_FREQUENCIES]; /**< List of frequencies in MHz.
                                                         Only the first num_supported frequencies are valid */
} amdsmi_frequencies_t;

/**
 * @brief The dpm policy.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    uint32_t policy_id;
    char policy_description[AMDSMI_MAX_STRING_LENGTH];
} amdsmi_dpm_policy_entry_t;

#define AMDSMI_MAX_NUM_PM_POLICIES 32  //!< Maximum number of power management policies

/**
 * @brief DPM Policy
 *
 * Only the first num_supported policies are valid.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    uint32_t num_supported;  //!< The number of supported policies
    uint32_t current;        //!< The current policy index
    amdsmi_dpm_policy_entry_t policies[AMDSMI_MAX_NUM_PM_POLICIES]; //!< List of policies.
} amdsmi_dpm_policy_t;

/**
 * @brief This structure holds information about the possible PCIe
 * bandwidths. Specifically, the possible transfer rates and their
 * associated numbers of lanes are stored here.
 *
 * Only the first num_supported bandwidths are valid.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amdsmi_frequencies_t transfer_rate;          //!< Transfer rates (T/s) that are possible
    uint32_t lanes[AMDSMI_MAX_NUM_FREQUENCIES];  //!< List of lanes for corresponding transfer rate.
} amdsmi_pcie_bandwidth_t;

/**
 * @brief This structure holds version information.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint32_t major;     //!< Major version
    uint32_t minor;     //!< Minor version
    uint32_t release;   //!< Patch, build or stepping version
    const char *build;  //!< Full Build version string
} amdsmi_version_t;

/**
 * @brief This structure represents a point on the frequency-voltage plane.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint64_t frequency;  //!< Frequency coordinate (in Hz)
    uint64_t voltage;    //!< Voltage coordinate (in mV)
} amdsmi_od_vddc_point_t;

/**
 * @brief This structure holds 2 ::amdsmi_range_t's, one for frequency and one for
 * voltage. These 2 ranges indicate the range of possible values for the
 * corresponding ::amdsmi_od_vddc_point_t.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amdsmi_range_t freq_range;  //!< The frequency range for this VDDC Curve point
    amdsmi_range_t volt_range;  //!< The voltage range for this VDDC Curve point
} amdsmi_freq_volt_region_t;

/**
 * @brief OD Vold Curve
 * ::AMDSMI_NUM_VOLTAGE_CURVE_POINTS number of ::amdsmi_od_vddc_point_t's
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amdsmi_od_vddc_point_t vc_points[AMDSMI_NUM_VOLTAGE_CURVE_POINTS]; //!< make up the voltage frequency curve points.
} amdsmi_od_volt_curve_t;

/**
 * @brief This structure holds the frequency-voltage values for a device.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amdsmi_range_t curr_sclk_range;   //!< The current SCLK frequency range in MHz
    amdsmi_range_t curr_mclk_range;   //!< The current MCLK frequency range, upper bound only in MHz
    amdsmi_range_t sclk_freq_limits;  //!< The range possible of SCLK values in MHz
    amdsmi_range_t mclk_freq_limits;  //!< The range possible of MCLK values in MHz
    amdsmi_od_volt_curve_t curve;     //!< The current voltage curve
    uint32_t num_regions;             //!< The number of voltage curve regions
} amdsmi_od_volt_freq_data_t;

/**
 * @brief Structure holds the gpu metrics table header for a device
 *
 * Size and version information of metrics data
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    // TODO(amd) Doxygen documents
    // Note: This should match: AMDGpuMetricsHeader_v1_t
    /// \cond Ignore in docs.
    uint16_t      structure_size;
    uint8_t       format_revision;
    uint8_t       content_revision;
    /// \endcond
} amd_metrics_table_header_t;

/**
 * @brief The following structures hold the gpu statistics for a device.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    /**
     * @brief v1.6 additions
     * The max uint32_t will be used if that information is N/A
     */
    uint32_t gfx_busy_inst[AMDSMI_MAX_NUM_XCC];      //!< Utilization Instantaneous in %
    uint16_t jpeg_busy[AMDSMI_MAX_NUM_JPEG_ENG_V1];  //!< Utilization Instantaneous in % (UPDATED: to 40 in v1.8)
    uint16_t vcn_busy[AMDSMI_MAX_NUM_VCN];           //!< Utilization Instantaneous in %
    uint64_t gfx_busy_acc[AMDSMI_MAX_NUM_XCC];       //!< Utilization Accumulated in %

    /**
     * @brief v1.7 additions
     */
    /* Total App Clock Counter Accumulated */
    uint64_t gfx_below_host_limit_acc[AMDSMI_MAX_NUM_XCC]; //!< Total App Clock Counter Accumulated

    /**
     * @brief v1.8 additions
     */
    /* Total App Clock Counter Accumulated */
    uint64_t gfx_below_host_limit_ppt_acc[AMDSMI_MAX_NUM_XCC];
    uint64_t gfx_below_host_limit_thm_acc[AMDSMI_MAX_NUM_XCC];
    uint64_t gfx_low_utilization_acc[AMDSMI_MAX_NUM_XCC];
    uint64_t gfx_below_host_limit_total_acc[AMDSMI_MAX_NUM_XCC];
} amdsmi_gpu_xcp_metrics_t;

/**
 * @brief Structure holds the gpu metrics values for a device
 *
 * This structure is extended to fit the needs of different GPU metric
 * versions when exposing data through the structure.
 * Depending on the version, some data members will hold data, and
 * some will not. A good example is the set of 'current clocks':
 * current_gfxclk, current_socclk, current_vclk0, current_dclk0.
 * These are single-valued data members, up to version 1.3.
 * For version 1.4 and up these are multi-valued data members (arrays)
 * and their counterparts; current_gfxclks[], current_socclks[],
 * current_vclk0s[], current_dclk0s[], will hold the data
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    amd_metrics_table_header_t common_header;

    /**
     * @brief v1.0 Base
     *
     * Temperature in C
     */
    uint16_t temperature_edge;
    uint16_t temperature_hotspot;
    uint16_t temperature_mem;
    uint16_t temperature_vrgfx;
    uint16_t temperature_vrsoc;
    uint16_t temperature_vrmem;

    /**
     * @brief Average Utilization (in %)
     */
    uint16_t average_gfx_activity;  //!< gfx
    uint16_t average_umc_activity;  //!< memory controller
    uint16_t average_mm_activity;   //!< UVD or VCN

    /**
     * @brief Power (W) /Energy (15.259uJ per 1ns)
     */
    uint16_t average_socket_power;
    uint64_t energy_accumulator;    //!< v1 mod. (32->64)

    //! Driver attached timestamp (in ns)
    uint64_t system_clock_counter;  //!< v1 mod. (moved from top of struct)

    /**
     * @brief Average clocks (MHz)
     */
    uint16_t average_gfxclk_frequency;
    uint16_t average_socclk_frequency;
    uint16_t average_uclk_frequency;
    uint16_t average_vclk0_frequency;
    uint16_t average_dclk0_frequency;
    uint16_t average_vclk1_frequency;
    uint16_t average_dclk1_frequency;

    /**
     * @brief Current clocks (MHz)
     */
    uint16_t current_gfxclk;
    uint16_t current_socclk;
    uint16_t current_uclk;
    uint16_t current_vclk0;
    uint16_t current_dclk0;
    uint16_t current_vclk1;
    uint16_t current_dclk1;

    uint32_t throttle_status;  //!< Throttle status

    uint16_t current_fan_speed;  //!< Fans (RPM)

    /**
     * @brief Link width (number of lanes) /speed (0.1 GT/s)
     */
    uint16_t pcie_link_width;  //!< v1 mod.(8->16)
    uint16_t pcie_link_speed;  //!< in 0.1 GT/s; v1 mod. (8->16)

    /*
     * v1.1 additions
     */
    uint32_t gfx_activity_acc;  //!< new in v1
    uint32_t mem_activity_acc;  //!< new in v1
    uint16_t temperature_hbm[AMDSMI_NUM_HBM_INSTANCES];  //!< new in v1

    /*
     * v1.2 additions
     */
    uint64_t firmware_timestamp;  //!< PMFW attached timestamp (10ns resolution)

    /*
     * v1.3 additions
     */
    uint16_t voltage_soc;  //!< In mV
    uint16_t voltage_gfx;  //!< In mV
    uint16_t voltage_mem;  //!< In mV

    uint64_t indep_throttle_status;  //!< Throttle status

    /*
     * v1.4 additions
     */
    uint16_t current_socket_power;  //!< In Watts

    uint16_t vcn_activity[AMDSMI_MAX_NUM_VCN];  //!< Utilization (%)

    uint32_t gfxclk_lock_status;  //!< Clock Lock Status. Each bit corresponds to clock instance

    uint16_t xgmi_link_width;  //!< XGMI bus width in GB/s
    uint16_t xgmi_link_speed;  //!< XGMI bus bitrate in GB/s

    uint64_t pcie_bandwidth_acc; //!< PCIE accumulated bandwidth (GB/sec)
    uint64_t pcie_bandwidth_inst; //!< PCIE instantaneous bandwidth (GB/sec)
    uint64_t pcie_l0_to_recov_count_acc; //!< PCIE L0 to recovery state transition accumulated count
    uint64_t pcie_replay_count_acc; //!< PCIE replay accumulated count
    uint64_t pcie_replay_rover_count_acc; //!< PCIE replay rollover accumulated count

    /**
     * @brief XGMI accumulated data transfer size(KiloBytes)
     */
    uint64_t xgmi_read_data_acc[AMDSMI_MAX_NUM_XGMI_LINKS];   //!< In KB
    uint64_t xgmi_write_data_acc[AMDSMI_MAX_NUM_XGMI_LINKS];  //!< In KB

    /**
     * @brief XGMI current data transfer size(KiloBytes)
     */
    uint16_t current_gfxclks[AMDSMI_MAX_NUM_GFX_CLKS];  //!< In KB
    uint16_t current_socclks[AMDSMI_MAX_NUM_CLKS];      //!< In KB
    uint16_t current_vclk0s[AMDSMI_MAX_NUM_CLKS];       //!< In KB
    uint16_t current_dclk0s[AMDSMI_MAX_NUM_CLKS];       //!< In KB

    /**
     * @brief v1.5 additions
     */
    uint16_t jpeg_activity[AMDSMI_MAX_NUM_JPEG];  //!< JPEG activity percent (encode/decode)
    uint32_t pcie_nak_sent_count_acc;  //!< PCIE NAK sent accumulated count
    uint32_t pcie_nak_rcvd_count_acc;  //!< PCIE NAK received accumulated count

    /**
     * @brief v1.6 additions
     */
    uint64_t accumulation_counter;  //!< Accumulation cycle counter

    /**
     * @brief Accumulated throttler residencies
     */
    uint64_t prochot_residency_acc;

    /**
     * @brief Accumulated throttler residencies
     *
     * Prochot (thermal) - PPT (power)
     * Package Power Tracking (PPT) violation % (greater than 0% is a violation);
     * aka PVIOL
     *
     * Ex. PVIOL/TVIOL calculations
     * Where A and B are measurments recorded at prior points in time.
     * Typically A is the earlier measured value and B is the latest measured value.
     *
     * PVIOL % = (PptResidencyAcc (B) - PptResidencyAcc (A)) * 100/ (AccumulationCounter (B) - AccumulationCounter (A))
     * TVIOL % = (SocketThmResidencyAcc (B) -  SocketThmResidencyAcc (A)) * 100 / (AccumulationCounter (B) - AccumulationCounter (A))
    */
    uint64_t ppt_residency_acc;

    /**
     * @brief Accumulated throttler residencies
     *
     * Socket (thermal) -
     * Socket thermal violation % (greater than 0% is a violation);
     * aka TVIOL
     *
     * Ex. PVIOL/TVIOL calculations
     * Where A and B are measurments recorded at prior points in time.
     * Typically A is the earlier measured value and B is the latest measured value.
     *
     * PVIOL % = (PptResidencyAcc (B) - PptResidencyAcc (A)) * 100/ (AccumulationCounter (B) - AccumulationCounter (A))
     * TVIOL % = (SocketThmResidencyAcc (B) -  SocketThmResidencyAcc (A)) * 100 / (AccumulationCounter (B) - AccumulationCounter (A))
    */
    uint64_t socket_thm_residency_acc;
    uint64_t vr_thm_residency_acc;
    uint64_t hbm_thm_residency_acc;

    uint16_t num_partition; //!< Number of current partition

    amdsmi_gpu_xcp_metrics_t xcp_stats[AMDSMI_MAX_NUM_XCP]; //!< XCP (Graphic Cluster Partitions) metrics stats

    uint32_t pcie_lc_perf_other_end_recovery; //!< PCIE other end recovery counter

    /**
    * @brief v1.7 additions
    */
    uint64_t vram_max_bandwidth; //!< VRAM max bandwidth at max memory clock (GB/s)

    uint16_t xgmi_link_status[AMDSMI_MAX_NUM_XGMI_LINKS]; //!< XGMI link status(up/down)
} amdsmi_gpu_metrics_t;

/**
 * @brief XGMI Link Status Type
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_XGMI_LINK_DOWN,     //!< XGMI link status is down
    AMDSMI_XGMI_LINK_UP,       //!< XGMI link status is up
    AMDSMI_XGMI_LINK_DISABLE   //!< XGMI link status is disabled
} amdsmi_xgmi_link_status_type_t;

/**
 * @brief XGMI Link Status
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint32_t total_links;   //!< The total links in the status array
    amdsmi_xgmi_link_status_type_t status[AMDSMI_MAX_NUM_XGMI_LINKS];
    uint64_t reserved[7];
} amdsmi_xgmi_link_status_t;

/**
 * @brief This structure holds the name value pairs
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    char name[AMDSMI_MAX_STRING_LENGTH];  //!< Name
    uint64_t value;                     //!< Use uint64_t to make it universal
} amdsmi_name_value_t;

/**
 * @brief This register type for register table
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef enum {
    AMDSMI_REG_XGMI,  //!< XGMI registers
    AMDSMI_REG_WAFL,  //!< WAFL registers
    AMDSMI_REG_PCIE,  //!< PCIe registers
    AMDSMI_REG_USR,   //!< Usr registers
    AMDSMI_REG_USR1   //!< Usr1 registers
} amdsmi_reg_type_t;

/**
 * @brief This structure holds ras feature
 *
 * @cond @tag{gpu_bm_linux} @platform{guest_windows} @tag{host} @endcond
 */
typedef struct {
    uint32_t ras_eeprom_version;          /**< PARITY error(bit 0), Single Bit correctable (bit1),
                                               Double bit error detection (bit2), Poison (bit 3). */
    uint32_t ecc_correction_schema_flag;  /**< ecc_correction_schema mask.
                                               PARITY error(bit 0), Single Bit correctable (bit1),
                                               Double bit error detection (bit2), Poison (bit 3) */
} amdsmi_ras_feature_t;

/**
 * @brief This structure holds error counts.
 *
 * @cond @tag{gpu_bm_linux} @tag{guest_windows} @tag{host} @endcond
 */
typedef struct {
    uint64_t correctable_count;    //!< Accumulated correctable errors
    uint64_t uncorrectable_count;  //!< Accumulated uncorrectable errors
    uint64_t deferred_count;       //!< Accumulated deferred errors
    uint64_t reserved[5];
} amdsmi_error_count_t;

/**
 * @brief This structure contains information specific to a process.
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint32_t process_id;    //!< Process ID
    uint64_t vram_usage;    //!< VRAM usage in MB
    uint64_t sdma_usage;    //!< SDMA usage in microseconds
    uint32_t cu_occupancy;  //!< Compute Unit usage in percent
    uint32_t evicted_time;    //!< Time that queues are evicted on a GPU in milliseconds
} amdsmi_process_info_t;

/**
 * @brief Topology Nearest
 *
 * @cond @tag{gpu_bm_linux} @endcond
 */
typedef struct {
    uint32_t count;
    amdsmi_processor_handle processor_list[AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP];
    uint64_t reserved[15];
} amdsmi_topology_nearest_t;

/**
 * @brief Variant placeholder
 *
 * Place-holder "variant" for functions that have don't have any variants,
 * but do have monitors or sensors.
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_VIRTUALIZATION_MODE_UNKNOWN = 0,  //!< Unknown Virtualization Mode
    AMDSMI_VIRTUALIZATION_MODE_BAREMETAL,    //!< Baremetal Virtualization Mode
    AMDSMI_VIRTUALIZATION_MODE_HOST,         //!< Host Virtualization Mode
    AMDSMI_VIRTUALIZATION_MODE_GUEST,        //!< Guest Virtualization Mode
    AMDSMI_VIRTUALIZATION_MODE_PASSTHROUGH   //!< Passthrough Virtualization Mode
} amdsmi_virtualization_mode_t;


/**
 * @brief Scope for Numa affinity or Socket affinity
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum {
    AMDSMI_AFFINITY_SCOPE_NODE,    //!< Memory affinity as numa node
    AMDSMI_AFFINITY_SCOPE_SOCKET   //!< socket affinity
} amdsmi_affinity_scope_t;

/**
 * @brief NPM status
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef enum  {
    AMDSMI_NPM_STATUS_DISABLED,
    AMDSMI_NPM_STATUS_ENABLED
} amdsmi_npm_status_t;

/**
 * @brief NPM info
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    amdsmi_npm_status_t status; //!< NPM status (enabled/disabled).
    uint64_t            limit;  //!< Node-level power limit in Watts.
    uint64_t            reserved[6];
} amdsmi_npm_info_t;

#ifdef ENABLE_ESMI_LIB

/**
 * @brief This structure holds SMU Firmware version information.
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
    uint8_t debug;   //!< SMU fw Debug version number
    uint8_t minor;   //!< SMU fw Minor version number
    uint8_t major;   //!< SMU fw Major version number
    uint8_t unused;  //!< reserved fields
} amdsmi_smu_fw_version_t;

/**
 * @brief DDR bandwidth metrics.
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
    uint32_t max_bw;        //!< DDR Maximum theoritical bandwidth in GB/s
    uint32_t utilized_bw;   //!< DDR bandwidth utilization in GB/s
    uint32_t utilized_pct;  //!< DDR bandwidth utilization in % of theoritical max
} amdsmi_ddr_bw_metrics_t;

/**
 * @brief temperature range and refresh rate metrics of a DIMM
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
    uint8_t range : 3;     //!< temp range[2:0](3 bit data)
    uint8_t ref_rate : 1;  //!< DDR refresh rate mode[3](1 bit data)
} amdsmi_temp_range_refresh_rate_t;

/**
 * @brief DIMM Power(mW), power update rate(ms) and dimm address
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
    uint16_t power : 15;       //!< Dimm power consumption[31:17](15 bits data)
    uint16_t update_rate : 9;  //!< Time since last update[16:8](9 bit data)
    uint8_t dimm_addr;         //!< Dimm address[7:0](8 bit data)
} amdsmi_dimm_power_t;

/**
 * @brief DIMM temperature(°C) and update rate(ms) and dimm address
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
    uint16_t sensor : 11;      //!< Dimm thermal sensor[31:21](11 bit data)
    uint16_t update_rate : 9;  //!< Time since last update[16:8](9 bit data)
    uint8_t dimm_addr;         //!< Dimm address[7:0](8 bit data)
    float temp;                //!< temperature in degree celcius
} amdsmi_dimm_thermal_t;

/**
 * @brief xGMI Bandwidth Encoding types
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef enum {
    AGG_BW0 = 1,  //!< Aggregate Bandwidth
    RD_BW0 = 2,   //!< Read Bandwidth
    WR_BW0 = 4    //!< Write Bandwdith
} amdsmi_io_bw_encoding_t;

/**
 * @brief LINK name and Bandwidth type Information.It contains
 * link names i.e valid link names are
 * "P0", "P1", "P2", "P3", "P4", "G0", "G1", "G2", "G3", "G4"
 * "G5", "G6", "G7"
 * Valid bandwidth types 1(Aggregate_BW), 2 (Read BW), 4 (Write BW).
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
    amdsmi_io_bw_encoding_t bw_type;  //!< Bandwidth Type Information [1, 2, 4]
    char *link_name;                  //!< Link name [P0, P1, G0, G1 etc]
} amdsmi_link_id_bw_type_t;

/**
 * @brief max and min LCLK DPM level on a given NBIO ID.
 * Valid max and min DPM level values are 0 - 1.
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
    uint8_t max_dpm_level;  //!< Max LCLK DPM level[15:8](8 bit data)
    uint8_t min_dpm_level;  //!< Min LCLK DPM level[7:0](8 bit data)
} amdsmi_dpm_level_t;

/**
 * @brief HSMP Metrics table (supported only with hsmp proto version 6).
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct __attribute__((__packed__)) {
    uint32_t accumulation_counter;  //!< Incremented every time the accumulator values are updated in this table

    uint32_t max_socket_temperature;    //!< Maximum temperature reported by all on-die thermal sensors on all AIDs, CCDs, and XCDs in the socket
    uint32_t max_vr_temperature;        //!< Maximum temperature reported by SVI3 telemetry for all slave addresses
    uint32_t max_hbm_temperature;       //!< Maximum temperature reported by all HBM stacks in the socket
    uint64_t max_socket_temperature_acc;//!< Accumulated version of "max_socket_temperature"
    uint64_t max_vr_temperature_acc;    //!< Accumulated version of "max_vr_temperature"
    uint64_t max_hbm_temperature_acc;   //!< Accumulated version of "max_hbm_temperature"

    uint32_t socket_power_limit;      //!< Power limit currently being enforced by the power throttling controller
    uint32_t max_socket_power_limit;  //!< Maximum power limit the power throttling controller is allowed to be configured to
    uint32_t socket_power;            //!< Power consumption of all die in the socket (AID+CCD+XCD+HBM)

    uint64_t timestamp;               //!< Timestamp corresponding to the energy accumulators in 10ns units
    uint64_t socket_energy_acc;       //!< Energy accumulator of all die in the socket (AID+CCD+XCD+HBM)
    uint64_t ccd_energy_acc;          //!< Energy accumulator of VDDCR_VDD rails powering CCDs
    uint64_t xcd_energy_acc;          //!< Energy accumulator of VDDCR_VDD rails powering XCDs
    uint64_t aid_energy_acc;          //!< Energy accumulator of rails powering the AIDs
    uint64_t hbm_energy_acc;          //!< Energy accumulator of all HBM stacks in the socket

    uint32_t cclk_frequency_limit;     //!< Minimum CCLK frequency limit enforced from the infrastructure controllers
    uint32_t gfxclk_frequency_limit;   //!< Minimum GFXCLK frequency limit enforced from the infrastructure controllers
    uint32_t fclk_frequency;           //!< Effective FCLK frequency
    uint32_t uclk_frequency;           //!< Effective UCLK frequency
    uint32_t socclk_frequency[4];      //!< Effective SOCCLK frequency per AID
    uint32_t vclk_frequency[4];        //!< Effective VCLK frequency per AID
    uint32_t dclk_frequency[4];        //!< Effective DCLK frequency per AID
    uint32_t lclk_frequency[4];        //!< Effective LCLK frequency per AID
    uint64_t gfxclk_frequency_acc[8];  //!< GFXCLK frequency for the target XCC
    uint64_t cclk_frequency_acc[96];   //!< CCLK frequency for the target CPU

    uint32_t max_cclk_frequency;         //!< Maximum CCLK frequency supported by the CPU
    uint32_t min_cclk_frequency;         //!< Minimum CCLK frequency supported by the CPU
    uint32_t max_gfxclk_frequency;       //!< Maximum GFXCLK frequency supported by the accelerator
    uint32_t min_gfxclk_frequency;       //!< Minimum GFXCLK frequency supported by the accelerator
    uint32_t fclk_frequency_table[4];    //!< List of supported FCLK frequencies (0 means that state is not supported)
    uint32_t uclk_frequency_table[4];    //!< List of supported UCLK frequencies (0 means that state is not supported)
    uint32_t socclk_frequency_table[4];  //!< List of supported SOCCLK frequencies (0 means that state is not supported)
    uint32_t vclk_frequency_table[4];    //!< List of supported VCLK frequencies (0 means that state is not supported)
    uint32_t dclk_frequency_table[4];    //!< List of supported DCLK frequencies (0 means that state is not supported)
    uint32_t lclk_frequency_table[4];    //!< List of supported LCLK frequencies (0 means that state is not supported)
    uint32_t max_lclk_dpm_range;         //!< Maximum LCLK DPM state constraint defined by the user
    uint32_t min_lclk_dpm_range;         //!< Minimum LCLK DPM state constraint defined by the user

    uint32_t xgmi_width;                 //!< Current operating XGMI link width
    uint32_t xgmi_bitrate;               //!< Current operating XGMI link bitrate
    uint64_t xgmi_read_bandwidth_acc[8]; //!< XGMI read bandwidth for the target XGMI link in the local socket
    uint64_t xgmi_write_bandwidth_acc[8];//!< XGMI write bandwidth for the target XGMI link in the local socket

    uint32_t socket_c0_residency;           //!< Average CPU C0 residency of all enabled cores in the socket
    uint32_t socket_gfx_busy;               //!< Average XCC busy for all enabled XCCs in the socket
    uint32_t dram_bandwidth_utilization;    //!< HBM bandwidth utilization for all HBM stacks in the socket
    uint64_t socket_c0_residency_acc;       //!< Accumulated value of "socket_c0_residency"
    uint64_t socket_gfx_busy_acc;           //!< Accumulated value of "socket_gfx_busy"
    uint64_t dram_bandwidth_acc;            //!< HBM bandwidth for all HBM stacks in the socket
    uint32_t max_dram_bandwidth;            //!< Maximum supported HBM bandwidth for all HBM stacks running at the maximum supported UCLK frequency
    uint64_t dram_bandwidth_utilization_acc;//!< Accumulated value of "dram_bandwidth_utilization"
    uint64_t pcie_bandwidth_acc[4];         //!< PCIe bandwidth for all PCIe devices connected to the target AID

    uint32_t prochot_residency_acc;    //!< Incremented every iteration PROCHOT is active
    uint32_t ppt_residency_acc;        //!< Incremented every iteration the PPT controller is active
    uint32_t socket_thm_residency_acc; //!< Incremented every iteration the socket thermal throttling controller is active
    uint32_t vr_thm_residency_acc;     //!< Incremented every iteration the VR thermal throttling controller is active
    uint32_t hbm_thm_residency_acc;    //!< Incremented every iteration the HBM thermal throttling controller is active
    uint32_t spare;                    //!< spare

    uint32_t gfxclk_frequency[8];  //!< Effective GFXCLK frequency per XCD
} amdsmi_hsmp_metrics_table_t;

/**
 * @brief hsmp frequency limit source names
 *
 * @cond @tag{cpu_bm} @endcond
 */
static char* const amdsmi_hsmp_freqlimit_src_names[] = {
    "cHTC-Active",
    "PROCHOT",
    "TDC limit",
    "PPT Limit",
    "OPN Max",
    "Reliability Limit",
    "APML Agent",
    "HSMP Agent"
};

/**
 * @brief cpu info data
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
  char  model_name[AMDSMI_MAX_STRING_LENGTH]; //!< cpu model name
  uint32_t cpu_family_id;     //!< cpu family id
  uint32_t model_id;          //!< cpu model id
  uint32_t threads_per_core;  //!< vitual processing units per cpu core
  uint32_t cores_per_socket;  //!< cpu cores per socket
  bool frequency_boost;       //!< boost frequency
  uint32_t vendor_id;         //!< Use 32 bit to be compatible with other platform.
  char vendor_name[AMDSMI_MAX_STRING_LENGTH]; //!< vendor name
  uint32_t subvendor_id;      //!< The subsystem vendor id
  uint64_t device_id;         //!< The device id of a GPU
  uint32_t rev_id;            //!< Revision
  char asic_serial[AMDSMI_MAX_STRING_LENGTH]; //!< Asic serial id
  uint32_t socket_id;         //!< 0xFFFF if not supported
  uint32_t core_id;           //!< cpu core id
  uint32_t num_of_cpu_cores;  //!< 0xFFFFFFFF if not supported
  uint32_t socket_count;      //!< count of cpu sockets
  uint32_t core_count;        //!< cpu core count
  uint32_t reserved[17];      //!< reserved
} amdsmi_cpu_info_t;

#endif

/**
 * @brief cpu socket info data
 *
 * @cond @tag{cpu_bm} @endcond
 */
typedef struct {
  uint32_t socket_id;
  uint32_t cores_per_socket;
} amdsmi_sock_info_t;

/*****************************************************************************/
/** @defgroup tagInitShutdown Initialization and Shutdown
 *  @{
 */

/**
 *  @brief Initialize the AMD SMI library
 *
 *  @ingroup tagInitShutdown
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function initializes the library and the internal data structures,
 *  including those corresponding to sources of information that SMI provides. 
 *  Singleton Design, requires the same number of inits as shutdowns.
 *
 *  The @p init_flags decides which type of processor
 *  can be discovered by ::amdsmi_get_socket_handles(). AMDSMI_INIT_AMD_GPUS returns
 *  sockets with AMD GPUS, and AMDSMI_INIT_AMD_GPUS | AMDSMI_INIT_AMD_CPUS returns
 *  sockets with either AMD GPUS or CPUS.
 *  Currently, only AMDSMI_INIT_AMD_GPUS is supported.
 *
 *  @param[in] init_flags Bit flags that tell SMI how to initialze. Values of
 *  ::amdsmi_init_flags_t may be OR'd together and passed through @p init_flags
 *  to modify how AMDSMI initializes.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_init(uint64_t init_flags);

/**
 *  @brief Shutdown the AMD SMI library
 *
 *  @ingroup tagInitShutdown
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function shuts down the library and internal data structures and
 *  performs any necessary clean ups. Singleton Design, requires the same number
 *  of inits as shutdowns.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_shut_down(void);

/** @} End tagInitShutdown */

/*****************************************************************************/
/** @defgroup tagProcDiscovery Discovery Queries
 *  These functions provide discovery of the sockets.
 *  @{
 */

/**
 *  @brief Get the list of socket handles in the system.
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details Depends on what flag is passed to ::amdsmi_init.  AMDSMI_INIT_AMD_GPUS
 *  returns sockets with AMD GPUS, and AMDSMI_INIT_AMD_GPUS | AMDSMI_INIT_AMD_CPUS returns
 *  sockets with either AMD GPUS or CPUS.
 *  The socket handles can be used to query the processor handles in that socket, which
 *  will be used in other APIs to get processor detail information or telemtries.
 *
 *  @param[in,out] socket_count As input, the value passed
 *  through this parameter is the number of ::amdsmi_socket_handle that
 *  may be safely written to the memory pointed to by @p socket_handles. This is the
 *  limit on how many socket handles will be written to @p socket_handles. On return, @p
 *  socket_count will contain the number of socket handles written to @p socket_handles,
 *  or the number of socket handles that could have been written if enough memory had been
 *  provided.
 *  If @p socket_handles is NULL, as output, @p socket_count will contain
 *  how many sockets are available to read in the system.
 *
 *  @param[in,out] socket_handles A pointer to a block of memory to which the
 *  ::amdsmi_socket_handle values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many sockets are
 *  available to read in the system.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_socket_handles(uint32_t *socket_count, amdsmi_socket_handle* socket_handles);

#ifdef ENABLE_ESMI_LIB

/**
 *  @brief Get the list of cpu handles in the system.
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{cpu_bm}
 *
 *  @details Depends on AMDSMI_INIT_AMD_CPUS flag passed to ::amdsmi_init.
 *  The processor handles can be used in other APIs to get processor detail information.
 *
 *  @param[in,out] cpu_count As input, the value passed
 *  through this parameter is the number of ::amdsmi_processor_handle that
 *  may be safely written to the memory pointed to by @p processor_handles. This is the
 *  limit on how many processor handles will be written to @p processor_handles. On return, @p
 *  socket_count will contain the number of processor handles written to @p processor_handles,
 *  or the number of processor handles that could have been written if enough memory had been
 *  provided.
 *  If @p processor_handles is NULL, as output, @p cpu_count will contain
 *  how many processors are available to read in the system.
 *
 *  @param[in,out] processor_handles A pointer to a block of memory to which the
 *  ::amdsmi_processor_handle values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many processors are
 *  available to read in the system.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_handles(uint32_t *cpu_count, amdsmi_processor_handle *processor_handles);

#endif

/**
 *  @brief Get information about the given socket
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function retrieves socket information. The @p socket_handle must
 *  be provided to retrieve the Socket ID.
 *
 *  @param[in] socket_handle a socket handle
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @param[out] name The id of the socket.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_socket_info(amdsmi_socket_handle socket_handle, size_t len, char *name);

#ifdef ENABLE_ESMI_LIB

/**
 *  @brief Get information about the given processor
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{cpu_bm}
 *
 *  @details This function retrieves processor information. The @p processor_handle must
 *  be provided to retrieve the processor ID.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @param[out] name The id of the processor.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_info(amdsmi_processor_handle processor_handle, size_t len, char *name);

/**
 *  @brief Get respective processor counts from the processor handles
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{cpu_bm}
 *
 *  @details This function retrieves respective processor counts information.
 *  The @p processor_handle must be provided to retrieve the processor ID.
 *
 *  @param[in] processor_handles A pointer to a block of memory to which the
 *  ::amdsmi_processor_handle values will be written. This value may be NULL.
 *
 *  @param[in] processor_count total processor count per socket
 *
 *  @param[out] nr_cpusockets Total number of cpu sockets
 *
 *  @param[out] nr_cpucores Total number of cpu cores
 *
 *  @param[out] nr_gpus Total number of gpu devices
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_count_from_handles(amdsmi_processor_handle* processor_handles,
                                                        uint32_t* processor_count, uint32_t* nr_cpusockets,
                                                        uint32_t* nr_cpucores, uint32_t* nr_gpus);

/**
 *  @brief Get processor list as per processor type
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{cpu_bm}
 *
 *  @details This function retrieves processor list as per the processor type
 *  from the total processor handles list.
 *  The @p list of processor_handles and processor type must be provided.
 *
 *  @param[in] socket_handle socket handle
 *
 *  @param[in] processor_type processor type
 *
 *  @param[out] processor_handles list of processor handles as per processor type
 *
 *  @param[out] processor_count processor count as per processor type selected
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_handles_by_type(amdsmi_socket_handle socket_handle,
                                                     processor_type_t processor_type,
                                                     amdsmi_processor_handle* processor_handles,
                                                     uint32_t* processor_count);

#endif

/**
 *  @brief Get the list of the processor handles associated to a socket.
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function retrieves the processor handles of a socket. The
 *  @p socket_handle must be provided for the processor. A socket may have mulitple different
 *  type processors: An APU on a socket have both CPUs and GPUs.
 *  Currently, only AMD GPUs are supported.
 *
 *  @note Sockets are not supported on the @platform{host}.
 *
 *  The number of processor count is returned through @p processor_count
 *  if @p processor_handles is NULL. Then the number of @p processor_count can be pass
 *  as input to retrieval all processors on the socket to @p processor_handles.
 *
 *  @param[in] socket_handle The socket to query
 *
 *  @param[in,out] processor_count As input, the value passed
 *  through this parameter is the number of ::amdsmi_processor_handle's that
 *  may be safely written to the memory pointed to by @p processor_handles. This is the
 *  limit on how many processor handles will be written to @p processor_handles. On return, @p
 *  processor_count will contain the number of processor handles written to @p processor_handles,
 *  or the number of processor handles that could have been written if enough memory had been
 *  provided.
 *  If @p processor_handles is NULL, as output, @p processor_count will contain
 *  how many processors are available to read for the socket.
 *
 *  @param[in,out] processor_handles A pointer to a block of memory to which the
 *  ::amdsmi_processor_handle values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many processors are
 *  available to read.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_handles(amdsmi_socket_handle socket_handle,
                                    uint32_t *processor_count,
                                    amdsmi_processor_handle* processor_handles);

/**
 *  @brief Get the node handle associated with processor handle. 
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details This function retrieves the node handle of a processor handler. The
 *  @p processor_handle must be provided for the processor.
 *  Currently, only AMD GPUs are supported.
 *
 *  @param[in] processor_handle A pointer to a ::amdsmi_processor_handle, this 
 *  is required to be OAM ID 0 otherwise the API will fail. OAM ID is sourced
 *  from amdsmi_get_gpu_asic_info API.
 *
 *  @param[out] amdsmi_node_handle* A pointer to a block of memory where amdsmi_node_handle
 *  will be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_node_handle(amdsmi_processor_handle processor_handle, amdsmi_node_handle *node_handle);


#ifdef ENABLE_ESMI_LIB
/**
 *  @brief Get the list of the cpu core handles in a system.
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{cpu_bm}
 *
 *  @details This function retrieves the cpu core handles of a system.
 *
 *  @param[in,out] cores_count As input, the value passed
 *  through this parameter is the number of ::amdsmi_processor_handle's that
 *  may be safely written to the memory pointed to by @p processor_handles. This is the
 *  limit on how many core handles will be written to @p processor_handles. On return, @p
 *  cores_count will contain the number of core processor handles written to @p processor_handles,
 *  or the number of core processor handles that could have been written if enough memory had been
 *  provided.
 *  If @p processor_handles is NULL, as output, @p processor_count will contain
 *  how many cpu cores are available to read in the system.
 *
 *  @param[in,out] processor_handles A pointer to a block of memory to which the
 *  ::amdsmi_processor_handle values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many processors are
 *  available to read.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
*/
amdsmi_status_t amdsmi_get_cpucore_handles(uint32_t *cores_count,
                                           amdsmi_processor_handle* processor_handles);
#endif

/**
 *  @brief Get the processor type of the processor_handle
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details This function retrieves the processor type. A processor_handle must be provided
 *  for that processor.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[out] processor_type a pointer to processor_type_t to which the processor type
 *  will be written. If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_type(amdsmi_processor_handle processor_handle, processor_type_t* processor_type);

/**
 *  @brief Get processor handle with the matching bdf.
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
 *  @platform{guest_mvf} @platform{guest_windows}
 *
 *  @details Given bdf info @p bdf, this function will get
 *  the processor handle with the matching bdf.
 *
 *  @param[in] bdf The bdf to match with corresponding processor handle.
 *
 *  @param[out] processor_handle processor handle with the matching bdf.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_processor_handle_from_bdf(amdsmi_bdf_t bdf, amdsmi_processor_handle* processor_handle);

/**
 *  @brief Returns BDF of the given device
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     bdf Reference to BDF. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_device_bdf(amdsmi_processor_handle processor_handle, amdsmi_bdf_t *bdf);

/**
 *  @brief Returns the UUID of the device
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[in,out] uuid_length Length of the uuid string. As input, must be
 *                 equal or greater than AMDSMI_GPU_UUID_SIZE and be allocated by
 *                 user. As output it is the length of the uuid string.
 *
 *  @param[out] uuid Pointer to string to store the UUID. Must be
 *              allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_device_uuid(amdsmi_processor_handle processor_handle, unsigned int *uuid_length, char *uuid);

/**
 *  @brief          Returns the Enumeration information for the device
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @details        This function returns Enumeration information of the corresponding
 *                  processor_handle. It will return the render number, card number,
 *                  HSA ID, HIP ID, and the HIP UUID.
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to Enumeration information structure.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_enumeration_info(amdsmi_processor_handle processor_handle, amdsmi_enumeration_info_t *info);

/**
 *  @brief Retrieves an array of uint64_t (sized to cpu_set_size) of bitmasks with the
 *  affinity within numa node or socket for the device.
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, the size of the cpu_set array @p cpu_set_size,
 *  and a pointer to an array of int64_t @p cpu_set, and @p scope, this function will write the CPU affinity bitmask
 *  to the array pointed to by @p cpu_set.
 *
 * User must allocate the enough memory for the cpu_set array. The size of the array is determined by the
 * number of CPU cores in the system. As an example, if there are 2 CPUs and each has 112 cores, the size
 * should be ceiling(2*112/64) = 4, where 64 is the bits of uint64_t. The function will write the CPU affinity bitmask
 * to the array. For example, to describe the CPU cores 0-55,112-167, it will set the 0-55 and 112-167 bits
 * to 1 and the reset of bits to 0 in the cpu_set array.
 *
 *  @param[in] processor_handle a processor handle
 *  @param[in] cpu_set_size The size of the cpu_set array that is safe to access
 *  @param[in,out] cpu_set Array reference in which to return a bitmask of CPU cores that this processor affinities with.
 *  @param[in] scope Scope for socket or numa affinity.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_affinity_with_scope(amdsmi_processor_handle processor_handle,
            uint32_t cpu_set_size, uint64_t *cpu_set, amdsmi_affinity_scope_t scope);

/**
 *  @brief Returns the virtualization mode for the target device.
 *
 *  @ingroup tagProcDiscovery
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{host}
 *
 *  @details The virtualization mode is detected and returned as an enum.
 *
 *  @param[in] processor_handle The identifier of the given device.
 *
 *  @param[in,out] mode Reference to the enum representing virtualization mode.
 *                  - When zero, the virtualization mode is unknown
 *                  - When non-zero, the virtualization mode is detected
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail.
 */
amdsmi_status_t
amdsmi_get_gpu_virtualization_mode(amdsmi_processor_handle processor_handle,
                                   amdsmi_virtualization_mode_t* mode);

/** @} End tagProcDiscovery */

/*****************************************************************************/
/** @defgroup tagIdentQuery Identifier Queries
 *  These functions provide identification information.
 *  @{
 */

/**
 *  @brief Get the device id associated with the device with provided device
 *  handler.
 *
 *  @ingroup tagIdentQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t @p id,
 *  this function will write the device id value to the uint64_t pointed to by
 *  @p id. This ID is an identification of the type of device, so calling this
 *  function for different devices will give the same value if they are kind
 *  of device. Consequently, this function should not be used to distinguish
 *  one device from another. amdsmi_get_gpu_bdf_id() should be used to get a
 *  unique identifier.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] id a pointer to uint64_t to which the device id will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_id(amdsmi_processor_handle processor_handle, uint16_t *id);

/**
 *  @brief Get the device revision associated with the device
 *
 *  @ingroup tagIdentQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  uint16_t @p revision to which the revision id will be written
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[out] revision a pointer to uint16_t to which the device revision
 *  will be written
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_revision(amdsmi_processor_handle processor_handle, uint16_t *revision);

/**
 *  @brief Get the name string for a give vendor ID
 *
 *  @ingroup tagIdentQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a pointer to a caller provided
 *  char buffer @p name, and a length of this buffer @p len, this function will
 *  write the name of the vendor (up to @p len characters) buffer @p name. The
 *  @p id may be a device vendor or subsystem vendor ID.
 *
 *  If the integer ID associated with the vendor is not found in one of the
 *  system files containing device name information (e.g.
 *  /usr/share/misc/pci.ids), then this function will return the hex vendor ID
 *  as a string. Updating the system name files can be accompplished with
 *  "sudo update-pciids".
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] name a pointer to a caller provided char buffer to which the
 *  name will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @note ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_vendor_name(amdsmi_processor_handle processor_handle, char *name,
                                                                  size_t len);

/**
 *  @brief Get the vram vendor string of a device.
 *
 *  @ingroup tagIdentQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function retrieves the vram vendor name given a processor handle
 *  @p processor_handle, a pointer to a caller provided
 *  char buffer @p brand, and a length of this buffer @p len, this function
 *  will write the vram vendor of the device (up to @p len characters) to the
 *  buffer @p brand.
 *
 *  If the vram vendor for the device is not found as one of the values
 *  contained within amdsmi_get_gpu_vram_vendor, then this function will return
 *  the string 'unknown' instead of the vram vendor.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] brand a pointer to a caller provided char buffer to which the
 *  vram vendor will be written
 *
 *  @param[in] len the length of the caller provided buffer @p brand.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_vram_vendor(amdsmi_processor_handle processor_handle, char *brand, uint32_t len);

/**
 *  @brief Get the subsystem device id associated with the device with
 *  provided processor handle.
 *
 *  @ingroup tagIdentQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t @p id,
 *  this function will write the subsystem device id value to the uint64_t
 *  pointed to by @p id.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] id a pointer to uint64_t to which the subsystem device id
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_subsystem_id(amdsmi_processor_handle processor_handle, uint16_t *id);

/**
 *  @brief Get the name string for the device subsytem
 *
 *  @ingroup tagIdentQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a pointer to a caller provided
 *  char buffer @p name, and a length of this buffer @p len, this function
 *  will write the name of the device subsystem (up to @p len characters)
 *  to the buffer @p name.
 *
 *  If the integer ID associated with the sub-system is not found in one of the
 *  system files containing device name information (e.g.
 *  /usr/share/misc/pci.ids), then this function will return the hex sub-system
 *  ID as a string. Updating the system name files can be accompplished with
 *  "sudo update-pciids".
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] name a pointer to a caller provided char buffer to which the
 *  name will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.

 *  @param[in] len the length of the caller provided buffer @p name.
 *
 *  @note ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire name. In this case, only @p len bytes will
 *  be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_subsystem_name(amdsmi_processor_handle processor_handle, char *name, size_t len);

/** @} End tagIdentQuery */

/*****************************************************************************/
/** @defgroup tagPCIeQuery PCIe Queries
 *  These functions provide information about PCIe.
 *  @{
 */

/**
 *  @brief Get the list of possible PCIe bandwidths that are available. It is not
 *  supported on virtual machine guest
 *
 *  @ingroup tagPCIeQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a to an
 *  ::amdsmi_pcie_bandwidth_t structure @p bandwidth, this function will fill in
 *  @p bandwidth with the possible T/s values and associated number of lanes,
 *  and indication of the current selection.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] bandwidth a pointer to a caller provided
 *  ::amdsmi_pcie_bandwidth_t structure to which the frequency information will be
 *  written
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle,
                             amdsmi_pcie_bandwidth_t *bandwidth);

/**
 *  @brief Get the unique PCI device identifier associated for a device
 *
 *  @ingroup tagPCIeQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Give a processor handle @p processor_handle and a pointer to a uint64_t @p
 *  bdfid, this function will write the Bus/Device/Function PCI identifier
 *  (BDFID) associated with device @p processor_handle to the value pointed to by
 *  @p bdfid.
 *
 *  The format of @p bdfid will be as follows:
 *
 *      BDFID = ((DOMAIN & 0xFFFFFFFF) << 32) | ((Partition & 0xF) << 28)
 *              | ((BUS & 0xFF) << 8) | ((DEVICE & 0x1F) <<3 )
 *              | (FUNCTION & 0x7)
 *
 *  | Name         | Field   | KFD property     | KFD -> PCIe ID (uint64_t)    |
 *  -------------- | ------- | ---------------- | ---------------------------- |
 *  | Domain       | [63:32] | "domain"         | (DOMAIN & 0xFFFFFFFF) << 32  |
 *  | Partition id | [31:28] | "location id"    | (LOCATION & 0xF0000000)      |
 *  | Reserved     | [27:16] | "location id"    | N/A                          |
 *  | Bus          | [15: 8] | "location id"    | (LOCATION & 0xFF00)          |
 *  | Device       | [ 7: 3] | "location id"    | (LOCATION & 0xF8)            |
 *  | Function     | [ 2: 0] | "location id"    | (LOCATION & 0x7)             |
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] bdfid a pointer to uint64_t to which the device bdfid value
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_bdf_id(amdsmi_processor_handle processor_handle, uint64_t *bdfid);

/**
 *  @brief Get the NUMA node associated with a device
 *
 *  @ingroup tagPCIeQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a int32_t @p
 *  numa_node, this function will retrieve the NUMA node value associated
 *  with device @p processor_handle and store the value at location pointed to by
 *  @p numa_node.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] numa_node pointer to location where NUMA node value will
 *  be written.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_topo_numa_affinity(amdsmi_processor_handle processor_handle, int32_t *numa_node);

/**
 *  @brief Get PCIe traffic information. It is not supported on virtual machine guest
 *
 *  @ingroup tagPCIeQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Give a processor handle @p processor_handle and pointers to a uint64_t's, @p
 *  sent, @p received and @p max_pkt_sz, this function will write the number
 *  of bytes sent and received in 1 second to @p sent and @p received,
 *  respectively. The maximum possible packet size will be written to
 *  @p max_pkt_sz.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] sent a pointer to uint64_t to which the number of bytes sent
 *  will be written in 1 second. If pointer is NULL, it will be ignored.
 *
 *  @param[in,out] received a pointer to uint64_t to which the number of bytes
 *  received will be written. If pointer is NULL, it will be ignored.
 *
 *  @param[in,out] max_pkt_sz a pointer to uint64_t to which the maximum packet
 *  size will be written. If pointer is NULL, it will be ignored.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_pci_throughput(amdsmi_processor_handle processor_handle, uint64_t *sent,
                                              uint64_t *received, uint64_t *max_pkt_sz);

/**
 *  @brief Get PCIe replay counter
 *
 *  @ingroup tagPCIeQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint64_t @p
 *  counter, this function will write the sum of the number of NAK's received
 *  by the GPU and the NAK's generated by the GPU to memory pointed to by @p
 *  counter.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] counter a pointer to uint64_t to which the sum of the NAK's
 *  received and generated by the GPU is written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_pci_replay_counter(amdsmi_processor_handle processor_handle, uint64_t *counter);

/** @} End tagPCIeQuery */

/*****************************************************************************/
/** @defgroup tagPCIeControl PCIe Control
 *  These functions provide some control over PCIe.
 *  @{
 */

/**
 *  @brief Control the set of allowed PCIe bandwidths that can be used. It is not
 *  supported on virtual machine guest
 *
 *  @ingroup tagPCIeControl
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a 64 bit bitmask @p bw_bitmask,
 *  this function will limit the set of allowable bandwidths. If a bit in @p
 *  bw_bitmask has a value of 1, then the frequency (as ordered in an
 *  ::amdsmi_frequencies_t returned by :: amdsmi_get_clk_freq()) corresponding
 *  to that bit index will be allowed.
 *
 *  This function will change the performance level to
 *  ::AMDSMI_DEV_PERF_LEVEL_MANUAL in order to modify the set of allowable
 *  band_widths. Caller will need to set to ::AMDSMI_DEV_PERF_LEVEL_AUTO in order
 *  to get back to default state.
 *
 *  All bits with indices greater than or equal to the value of the
 *  ::amdsmi_frequencies_t::num_supported field of ::amdsmi_pcie_bandwidth_t will be
 *  ignored.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] bw_bitmask A bitmask indicating the indices of the
 *  bandwidths that are to be enabled (1) and disabled (0). Only the lowest
 *  ::amdsmi_frequencies_t::num_supported (of ::amdsmi_pcie_bandwidth_t) bits of
 *  this mask are relevant.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_pci_bandwidth(amdsmi_processor_handle processor_handle, uint64_t bw_bitmask);

/** @} End tagPCIeControl */

/*****************************************************************************/
/** @defgroup tagPowerQuery Power Queries
 *  These functions provide information about power usage.
 *  @{
 */

/**
 *  @brief Get the energy accumulator counter of the processor with provided
 *  processor handle. It is not supported on virtual machine guest
 *
 *  @ingroup tagPowerQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a pointer to a uint64_t
 *  @p energy_accumulator, and a pointer to a uint64_t @p timestamp, this function
 *  will write amount of energy consumed to the uint64_t pointed to by
 *  @p energy_accumulator, and the timestamp to the uint64_t pointed to by @p timestamp.
 *  This function accumulates all energy consumed.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] energy_accumulator a pointer to uint64_t to which the energy
 *  counter will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in,out] counter_resolution resolution of the counter @p energy_accumulator in
 *  micro Joules
 *
 *  @param[in,out] timestamp a pointer to uint64_t to which the timestamp
 *  will be written. Resolution: 1 ns.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_energy_count(amdsmi_processor_handle processor_handle, uint64_t *energy_accumulator,
                        float *counter_resolution, uint64_t *timestamp);

/** @} End tagPowerQuery */

/*****************************************************************************/
/** @defgroup tagPowerControl Power Control
 *  These functions provide ways to control power usage.
 *  @{
 */

/**
 *  @brief Set the maximum gpu power cap value. It is not supported on virtual
 *  machine guest
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{host} @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 *  @details Set the power cap to the provided value @p cap.
 *  @p cap must be between the minimum and maximum power cap values set by the
 *  system, which can be obtained from ::amdsmi_dev_power_cap_range_get.
 *
 *  @param[in] processor_handle A processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a processor has more than one sensor, it could be greater than 0.
 *
 *  @param[in] cap a uint64_t that indicates the desired power cap.
 *  The @p cap value must be greater than 0.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_power_cap(amdsmi_processor_handle processor_handle,
                                     uint32_t sensor_ind, uint64_t cap);

/**
 *  @brief Set the power performance profile. It is not supported on virtual machine guest
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function will attempt to set the current profile to the provided
 *  profile, given a processor handle @p processor_handle and a @p profile. The provided
 *  profile must be one of the currently supported profiles, as indicated by a
 *  call to :: amdsmi_get_gpu_power_profile_presets()
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] reserved Not currently used. Set to 0.
 *
 *  @param[in] profile a ::amdsmi_power_profile_preset_masks_t that hold the mask
 *  of the desired new power profile
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_set_gpu_power_profile(amdsmi_processor_handle processor_handle, uint32_t reserved,
                             amdsmi_power_profile_preset_masks_t profile);

/**
 *  @brief Query the supported power cap sensors and their types for a device.
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details This function returns the list of supported power cap sensors for the given device,
 *  including their sensor indices and types (e.g., PPT0, PPT1).
 *
 *  @param[in]  processor_handle A processor handle.
 *  @param[out] sensor_count Pointer to a uint32_t that will be set to the number of supported sensors.
 *  @param[out] sensor_inds Pointer to an array of uint32_t to be filled with sensor indices.
 *                          The array must be allocated by the caller with enough space.
 *  @param[out] sensor_types Pointer to an array of amdsmi_power_cap_type_t to be filled with sensor types.
 *                          The array must be allocated by the caller with enough space.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail.
 */
amdsmi_status_t 
amdsmi_get_supported_power_cap(amdsmi_processor_handle processor_handle, uint32_t *sensor_count,
                                 uint32_t *sensor_inds, amdsmi_power_cap_type_t *sensor_types);

/**
 *  @brief Get the socket power.
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    ppower - Input buffer to return socket power
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_power(amdsmi_processor_handle processor_handle,
                                            uint32_t *ppower);

/**
 *  @brief Get the socket power cap.
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    pcap - Input buffer to return power cap.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_power_cap(amdsmi_processor_handle processor_handle,
                                                uint32_t *pcap);

/**
 *  @brief Get the maximum power cap value for a given socket.
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    pmax - Input buffer to return maximum power limit value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_power_cap_max(amdsmi_processor_handle processor_handle,
                                                    uint32_t *pmax);

/**
 *  @brief Get the SVI based power telemetry for all rails.
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    power - Input buffer to return svi based power value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_pwr_svi_telemetry_all_rails(amdsmi_processor_handle processor_handle,
                                                           uint32_t *power);

/**
 *  @brief Set the power cap value for a given socket.
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]  processor_handle Cpu socket which to query
 *
 *  @param[in]  pcap - Input power limit value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_socket_power_cap(amdsmi_processor_handle processor_handle,
                                                uint32_t pcap);

/**
 *  @brief Set the power efficiency profile policy.
 *
 *  @ingroup tagPowerControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in] processor_handle Cpu socket which to query
 *
 *  @param[in] mode - mode to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_pwr_efficiency_mode(amdsmi_processor_handle processor_handle,
                                                   uint8_t mode);

/** @} End tagPowerControl */

/*****************************************************************************/
/** @defgroup tagMemoryQuery Memory Queries
 *  These functions provide information about memory systems.
 *  @{
 */

/**
 *  @brief Get the total amount of memory that exists
 *
 *  @ingroup tagMemoryQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a type of memory @p mem_type, and
 *  a pointer to a uint64_t @p total, this function will write the total amount
 *  of @p mem_type memory that exists to the location pointed to by @p total.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] mem_type The type of memory for which the total amount will be
 *  found
 *
 *  @param[in,out] total a pointer to uint64_t to which the total amount of
 *  memory will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_memory_total(amdsmi_processor_handle processor_handle, amdsmi_memory_type_t mem_type,
                            uint64_t *total);

/**
 *  @brief Get the current memory usage
 *
 *  @ingroup tagMemoryQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function will write the amount of @p mem_type memory that
 *  that is currently being used to the location pointed to by @p used.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] mem_type The type of memory for which the amount being used will
 *  be found
 *
 *  @param[in,out] used a pointer to uint64_t to which the amount of memory
 *  currently being used will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_memory_usage(amdsmi_processor_handle processor_handle, amdsmi_memory_type_t mem_type,
                            uint64_t *used);

/**
 *  @brief Get the bad pages of a processor. It is not supported on virtual
 *  machine guest
 *
 *  @ingroup tagMemoryQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This call will query the device @p processor_handle for the
 *  number of bad pages (written to @p num_pages address). The results are
 *  written to address held by the @p info pointer.
 *  The first call to this API returns the number of bad pages which
 *  should be used to allocate the buffer that should contain the bad page
 *  records.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[out] num_pages Number of bad page records.
 *
 *  @param[out] info The results will be written to the
 *  amdsmi_retired_page_record_t pointer.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_bad_page_info(amdsmi_processor_handle processor_handle, uint32_t *num_pages,
                             amdsmi_retired_page_record_t *info);

/**
 *  @brief Get the bad pages threshold of a processor. It is not supported on virtual
 *  machine guest
 *
 *  @ingroup tagMemoryQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This call will query the device @p processor_handle for the
 *  threshold of bad pages (written to @p threshold address).
 *
 *  @param[in] processor_handle a processor handle
 *  @param[out] threshold of bad page count.
 *
 *  @note This function requires root access
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_bad_page_threshold(amdsmi_processor_handle processor_handle, uint32_t *threshold);


/**
 *  @brief Verify the checksum of RAS EEPROM. It is not supported on virtual
 *  machine guest
 *
 *  @ingroup tagMemoryQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This call will verify the device @p processor_handle for the
 *  checksum of RAS EEPROM.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @note This function requires root access
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success
 *          AMDSMI_STATUS_CORRUPTED_EEPROM on the device's EEPROM corruption
 *          others on fail
 */
amdsmi_status_t amdsmi_gpu_validate_ras_eeprom(amdsmi_processor_handle processor_handle);

/**
 *  @brief Returns if RAS features are enabled or disabled for given block. It is not
 *  supported on virtual machine guest
 *
 *  @ingroup tagMemoryQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, this function queries the
 *  state of RAS features for a specific block @p block. Result will be written
 *  to address held by pointer @p state.
 *
 *  @param[in] processor_handle Device handle which to query
 *
 *  @param[in] block Block which to query
 *
 *  @param[in,out] state A pointer to amdsmi_ras_err_state_t to which the state
 *  of block will be written.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_ras_block_features_enabled(amdsmi_processor_handle processor_handle,
                                          amdsmi_gpu_block_t block,
                                          amdsmi_ras_err_state_t *state);

/**
 *  @brief Get information about reserved ("retired") memory pages. It is not supported on
 *  virtual machine guest
 *
 *  @ingroup tagMemoryQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, this function returns retired page
 *  information @p records corresponding to the device with the provided processor
 *  handle @p processor_handle. The number of retired page records is returned through @p
 *  num_pages. @p records may be NULL on input. In this case, the number of
 *  records available for retrieval will be returned through @p num_pages.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] num_pages a pointer to a uint32. As input, the value passed
 *  through this parameter is the number of ::amdsmi_retired_page_record_t's that
 *  may be safely written to the memory pointed to by @p records. This is the
 *  limit on how many records will be written to @p records. On return, @p
 *  num_pages will contain the number of records written to @p records, or the
 *  number of records that could have been written if enough memory had been
 *  provided.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in,out] records A pointer to a block of memory to which the
 *  ::amdsmi_retired_page_record_t values will be written. This value may be NULL.
 *  In this case, this function can be used to query how many records are
 *  available to read.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_memory_reserved_pages(amdsmi_processor_handle processor_handle,
                                      uint32_t *num_pages,
                                      amdsmi_retired_page_record_t *records);

/** @} End tagMemoryQuery */

/*****************************************************************************/
/** @defgroup tagPhysicalStateQuery Physical State Queries
 *  These functions provide information about the physical characteristics of
 *  the device.
 *  @{
 */

/**
 *  @brief Get the fan speed in RPMs of the device with the specified processor
 *  handle and 0-based sensor index. It is not supported on virtual machine guest
 *
 *  @ingroup tagPhysicalStateQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t
 *  @p speed, this function will write the current fan speed in RPMs to the
 *  uint32_t pointed to by @p speed
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in,out] speed a pointer to uint32_t to which the speed will be
 *  written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_fan_rpms(amdsmi_processor_handle processor_handle,
                                        uint32_t sensor_ind, int64_t *speed);

/**
 *  @brief Get the fan speed for the specified device as a value relative to
 *  ::AMDSMI_MAX_FAN_SPEED. It is not supported on virtual machine guest
 *
 *  @ingroup tagPhysicalStateQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t
 *  @p speed, this function will write the current fan speed (a value
 *  between 0 and the maximum fan speed, ::AMDSMI_MAX_FAN_SPEED) to the uint32_t
 *  pointed to by @p speed
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in,out] speed a pointer to uint32_t to which the speed will be
 *  written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_fan_speed(amdsmi_processor_handle processor_handle,
                                         uint32_t sensor_ind, int64_t *speed);

/**
 *  @brief Get the max. fan speed of the device with provided processor handle. It is
 *  not supported on virtual machine guest
 *
 *  @ingroup tagPhysicalStateQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t
 *  @p max_speed, this function will write the maximum fan speed possible to
 *  the uint32_t pointed to by @p max_speed
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in,out] max_speed a pointer to uint32_t to which the maximum speed
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_fan_speed_max(amdsmi_processor_handle processor_handle,
                                             uint32_t sensor_ind, uint64_t *max_speed);

/**
 *  @brief Returns gpu cache info.
 *
 *  @ingroup tagPhysicalStateQuery
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @param[in] processor_handle PF of a processor for which to query
 *
 *  @param[out] info reference to the cache info struct.
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_cache_info(amdsmi_processor_handle processor_handle, amdsmi_gpu_cache_info_t *info);

/**
 *  @brief Get the voltage metric value for the specified metric, from the
 *  specified voltage sensor on the specified device. It is not supported on
 *  virtual machine guest
 *
 *  @ingroup tagPhysicalStateQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a sensor type @p sensor_type, a
 *  ::amdsmi_voltage_metric_t @p metric and a pointer to an int64_t @p
 *  voltage, this function will write the value of the metric indicated by
 *  @p metric and @p sensor_type to the memory location @p voltage.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_type part of device from which voltage should be
 *  obtained. This should come from the enum ::amdsmi_voltage_type_t
 *
 *  @param[in] metric enum indicated which voltage value should be
 *  retrieved
 *
 *  @param[in,out] voltage a pointer to int64_t to which the voltage
 *  will be written, in millivolts.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_volt_metric(amdsmi_processor_handle processor_handle,
                                           amdsmi_voltage_type_t sensor_type,
                                           amdsmi_voltage_metric_t metric, int64_t *voltage);

/** @} End tagPhysicalStateQuery */

/*****************************************************************************/
/** @defgroup tagPhysicalStateControl Physical State Control
 *  These functions provide control over the physical state of a device.
 *  @{
 */

/**
 *  @brief Reset the fan to automatic driver control. It is not supported on virtual
 *  machine guest
 *
 *  @ingroup tagPhysicalStateControl
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function returns control of the fan to the system
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_reset_gpu_fan(amdsmi_processor_handle processor_handle, uint32_t sensor_ind);

/**
 *  @brief Set the fan speed for the specified device with the provided speed,
 *  in RPMs. It is not supported on virtual machine guest
 *
 *  @ingroup tagPhysicalStateControl
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a integer value indicating
 *  speed @p speed, this function will attempt to set the fan speed to @p speed.
 *  An error will be returned if the specified speed is outside the allowable
 *  range for the device. The maximum value is 255 and the minimum is 0.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in] speed the speed to which the function will attempt to set the fan
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_fan_speed(amdsmi_processor_handle processor_handle,
      uint32_t sensor_ind, uint64_t speed);

/** @} End tagPhysicalStateControl */

/*****************************************************************************/
/** @defgroup tagClkPowerPerfQuery Clock, Power and Performance Queries
 *  These functions provide information about clock frequencies and
 *  performance.
 *  @{
 */

/**
 *  @brief Get GPU busy percent from gpu_busy_percent sysfs file
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, this function returns GPU busy
 *  percentage.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] gpu_busy_percent Direct output from the gpu_busy_percent sysfs file
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_busy_percent(amdsmi_processor_handle processor_handle,
                                            uint32_t *gpu_busy_percent);

/**
 *  @brief Get coarse grain utilization counter of the specified device
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, the array of the utilization counters,
 *  the size of the array, this function returns the coarse grain utilization counters
 *  and timestamp.
 *  The counter is the accumulated percentages. Every milliseconds the firmware calculates
 *  % busy count and then accumulates that value in the counter. This provides minimally
 *  invasive coarse grain GPU usage information.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] utilization_counters Multiple utilization counters can be retreived with a single
 *  call. The caller must allocate enough space to the utilization_counters array. The caller also
 *  needs to set valid AMDSMI_UTILIZATION_COUNTER_TYPE type for each element of the array.
 *  ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the provided arguments.
 *
 *  If the function reutrns AMDSMI_STATUS_SUCCESS, the counter will be set in the value field of
 *  the amdsmi_utilization_counter_t.
 *
 *  @param[in] count The size of @p utilization_counters array.
 *
 *  @param[in,out] timestamp The timestamp when the counter is retrieved. Resolution: 1 ns.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_utilization_count(amdsmi_processor_handle processor_handle,
                             amdsmi_utilization_counter_t utilization_counters[],
                             uint32_t count, uint64_t *timestamp);

/**
 *  @brief Get the performance level of the device. It is not supported on virtual
 *  machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function will write the ::amdsmi_dev_perf_level_t to the uint32_t
 *  pointed to by @p perf, for a given processor handle @p processor_handle and a pointer
 *  to a uint32_t @p perf.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] perf a pointer to ::amdsmi_dev_perf_level_t to which the
 *  performance level will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_perf_level(amdsmi_processor_handle processor_handle,
                                          amdsmi_dev_perf_level_t *perf);

/**
 *  @brief Enter performance determinism mode with provided processor handle. It is
 *  not supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and @p clkvalue this function
 *  will enable performance determinism mode, which enforces a GFXCLK frequency
 *  SoftMax limit per GPU set by the user. This prevents the GFXCLK PLL from
 *  stretching when running the same workload on different GPUS, making
 *  performance variation minimal. This call will result in the performance
 *  level ::amdsmi_dev_perf_level_t of the device being
 *  ::AMDSMI_DEV_PERF_LEVEL_DETERMINISM.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] clkvalue Softmax value for GFXCLK in MHz.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_set_gpu_perf_determinism_mode(amdsmi_processor_handle processor_handle, uint64_t clkvalue);

/**
 *  @brief Get the overdrive percent associated with the device with provided
 *  processor handle. It is not supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t @p od,
 *  this function will write the overdrive percentage to the uint32_t pointed
 *  to by @p od
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] od a pointer to uint32_t to which the overdrive percentage
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t *od);

/**
 *  @brief Get the GPU memory clock overdrive percent associated with the device with provided
 *  processor handle. It is not supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a uint32_t @p od,
 *  this function will write the overdrive percentage to the uint32_t pointed
 *  to by @p od
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] od a pointer to uint32_t to which the GPU memory clock overdrive percentage
 *  will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_mem_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t *od);

/**
 *  @brief Get the list of possible system clock speeds of device for a
 *  specified clock type. It is not supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a clock type @p clk_type, and a
 *  pointer to a to an ::amdsmi_frequencies_t structure @p f, this function will
 *  fill in @p f with the possible clock speeds, and indication of the current
 *  clock speed selection.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] clk_type the type of clock for which the frequency is desired
 *
 *  @param[in,out] f a pointer to a caller provided ::amdsmi_frequencies_t structure
 *  to which the frequency information will be written. Frequency values are in
 *  Hz.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_clk_freq(amdsmi_processor_handle processor_handle,
                             amdsmi_clk_type_t clk_type, amdsmi_frequencies_t *f);

/**
 *  @brief Reset the gpu associated with the device with provided processor handle. It is not
 *  supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, this function will reset the GPU
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_reset_gpu(amdsmi_processor_handle processor_handle);

/**
 *  @brief This function retrieves the overdrive GFX & MCLK information. If valid
 *  for the GPU it will also populate the voltage curve data. It is not supported
 *  on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  ::amdsmi_od_volt_freq_data_t structure @p odv, this function will populate @p
 *  odv. See ::amdsmi_od_volt_freq_data_t for more details.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] odv a pointer to an ::amdsmi_od_volt_freq_data_t structure
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_od_volt_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_od_volt_freq_data_t *odv);

/**
 *  @brief Get the 'metrics_header_info' from the GPU metrics associated with the device
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a amd_metrics_table_header_t in which
 *  the 'metrics_header_info' will stored
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] header_value a pointer to amd_metrics_table_header_t to which the device gpu
 *  metric unit will be stored
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_NOT_SUPPORTED is returned in case the metric unit
 *            does not exist for the given device
 *  @return ::amdsmi_status_t
 */
amdsmi_status_t
amdsmi_get_gpu_metrics_header_info(amdsmi_processor_handle processor_handle, amd_metrics_table_header_t* header_value);

/**
 *  @brief This function retrieves the gpu metrics information. It is not supported
 *  on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  ::amdsmi_gpu_metrics_t structure @p pgpu_metrics, this function will populate
 *  @p pgpu_metrics. See ::amdsmi_gpu_metrics_t for more details.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] pgpu_metrics a pointer to an ::amdsmi_gpu_metrics_t structure
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_metrics_info(amdsmi_processor_handle processor_handle,
                                            amdsmi_gpu_metrics_t *pgpu_metrics);

/**
 *  @brief This function retrieves the partition metrics information.
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  ::amdsmi_gpu_metrics_t structure @p pgpu_metrics, this function will populate
 *  @p pgpu_metrics. See ::amdsmi_gpu_metrics_t for more details.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] pgpu_metrics a pointer to an ::amdsmi_gpu_metrics_t structure
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_partition_metrics_info(amdsmi_processor_handle processor_handle,
                                                      amdsmi_gpu_metrics_t *pgpu_metrics);

/**
 *  @brief Get the pm metrics table with provided device index.
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a device handle @p processor_handle, @p pm_metrics pointer,
 *  and @p num_of_metrics pointer,
 *  this function will write the pm metrics name value pair
 *  to the array at @p pm_metrics and the number of metrics retreived to @p num_of_metrics
 *  Note: the library allocated memory for pm_metrics, and user must call
 *  free(pm_metrics) to free it after use.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[inout] pm_metrics A pointerto an array to hold multiple PM metrics. On successs,
 *  the library will allocate memory of pm_metrics and write metrics to this array.
 *  The caller must free this memory after usage to avoid memory leak.
 *
 *  @param[inout] num_of_metrics a pointer to uint32_t to which the number of
 *  metrics is allocated for pm_metrics array as input, and the number of metrics retreived
 *  as output. If this parameter is NULL, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @return ::amdsmi_status_t
 *
 */
amdsmi_status_t amdsmi_get_gpu_pm_metrics_info(amdsmi_processor_handle processor_handle,
                                               amdsmi_name_value_t** pm_metrics,
                                               uint32_t *num_of_metrics);

/**
 *  @brief Get the register metrics table with provided device index and register type.
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a device handle @p processor_handle, @p reg_type, @p reg_metrics pointer,
 *  and @p num_of_metrics pointer,
 *  this function will write the register metrics name value pair
 *  to the array at @p reg_metrics and the number of metrics retreived to @p num_of_metrics
 *  Note: the library allocated memory for reg_metrics, and user must call
 *  free(reg_metrics) to free it after use.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] reg_type The register type
 *
 *  @param[inout] reg_metrics A pointerto an array to hold multiple register metrics. On successs,
 *  the library will allocate memory of reg_metrics and write metrics to this array.
 *  The caller must free this memory after usage to avoid memory leak.
 *
 *  @param[inout] num_of_metrics a pointer to uint32_t to which the number of
 *  metrics is allocated for reg_metrics array as input, and the number of metrics retreived
 *  as output. If this parameter is NULL, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function with the given arguments
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @return ::amdsmi_status_t
 *
 */
amdsmi_status_t amdsmi_get_gpu_reg_table_info(amdsmi_processor_handle processor_handle,
                                              amdsmi_reg_type_t reg_type,
                                              amdsmi_name_value_t** reg_metrics,
                                              uint32_t *num_of_metrics);

/**
 *  @brief This function sets the clock range information. It is not supported on virtual
 *  machine guest
 *
 *  @deprecated ::amdsmi_set_gpu_clk_limit() should be used, with an
 *  interface that set the min_value and then max_value.
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a minimum clock value @p minclkvalue,
 *  a maximum clock value @p maxclkvalue and a clock type @p clkType this function
 *  will set the sclk|mclk range
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] minclkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] maxclkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] clkType AMDSMI_CLK_TYPE_SYS | AMDSMI_CLK_TYPE_MEM range type
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_clk_range(amdsmi_processor_handle processor_handle,
                                         uint64_t minclkvalue,
                                         uint64_t maxclkvalue,
                                         amdsmi_clk_type_t clkType);

/**
 *  @brief This function sets the clock sets the clock min/max level
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a clock type @p clk_type,
 *  a value @p clk_value needs to be set, and the @p level indicates min or max
 *  clock you want to set, this function the clock limit.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] clk_type AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_TYPE_MEM and so on
 *
 *  @param[in] limit_type AMDSMI_FREQ_IND_MIN|AMDSMI_FREQ_IND_MAX to set the
 *  minimum (0) or maximum (1) speed.
 *
 *  @param[in] clk_value value to apply to. Frequency values are in MHz.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_clk_limit(amdsmi_processor_handle processor_handle,
                                         amdsmi_clk_type_t clk_type,
                                         amdsmi_clk_limit_type_t limit_type,
                                         uint64_t clk_value);

/**
 *  @brief This function sets the clock frequency information. It is not supported on
 *  virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a frequency level @p level,
 *  a clock value @p clkvalue and a clock type @p clkType this function
 *  will set the sclk|mclk range
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] level AMDSMI_FREQ_IND_MIN|AMDSMI_FREQ_IND_MAX to set the
 *  minimum (0) or maximum (1) speed.
 *
 *  @param[in] clkvalue value to apply to the clock range. Frequency values
 *  are in MHz.
 *
 *  @param[in] clkType AMDSMI_CLK_TYPE_SYS | AMDSMI_CLK_TYPE_MEM range type
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_od_clk_info(amdsmi_processor_handle processor_handle,
                                        amdsmi_freq_ind_t level,
                                        uint64_t clkvalue,
                                        amdsmi_clk_type_t clkType);

/**
 *  @brief This function sets  1 of the 3 voltage curve points. It is not supported
 *  on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a voltage point @p vpoint
 *  and a voltage value @p voltvalue this function will set voltage curve point
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] vpoint voltage point [0|1|2] on the voltage curve
 *
 *  @param[in] clkvalue clock value component of voltage curve point.
 *  Frequency values are in MHz.
 *
 *  @param[in] voltvalue voltage value component of voltage curve point.
 *  Voltage is in mV.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_od_volt_info(amdsmi_processor_handle processor_handle,
                                            uint32_t vpoint, uint64_t clkvalue, uint64_t voltvalue);

/**
 *  @brief This function will retrieve the current valid regions in the
 *  frequency/voltage space. It is not supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a pointer to an unsigned integer
 *  @p num_regions and a buffer of ::amdsmi_freq_volt_region_t structures, @p
 *  buffer, this function will populate @p buffer with the current
 *  frequency-volt space regions. The caller should assign @p buffer to memory
 *  that can be written to by this function. The caller should also
 *  indicate the number of ::amdsmi_freq_volt_region_t structures that can safely
 *  be written to @p buffer in @p num_regions.
 *
 *  The number of regions to expect this function provide (@p num_regions) can
 *  be obtained by calling :: amdsmi_get_gpu_od_volt_info().
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] num_regions As input, this is the number of
 *  ::amdsmi_freq_volt_region_t structures that can be written to @p buffer. As
 *  output, this is the number of ::amdsmi_freq_volt_region_t structures that were
 *  actually written.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @param[in,out] buffer a caller provided buffer to which
 *  ::amdsmi_freq_volt_region_t structures will be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_od_volt_curve_regions(amdsmi_processor_handle processor_handle,
                                                     uint32_t *num_regions, amdsmi_freq_volt_region_t *buffer);

/**
 *  @brief Get the list of available preset power profiles and an indication of
 *  which profile is currently active. It is not supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a pointer to a
 *  ::amdsmi_power_profile_status_t @p status, this function will set the bits of
 *  the ::amdsmi_power_profile_status_t.available_profiles bit field of @p status to
 *  1 if the profile corresponding to the respective
 *  ::amdsmi_power_profile_preset_masks_t profiles are enabled. For example, if both
 *  the VIDEO and VR power profiles are available selections, then
 *  ::AMDSMI_PWR_PROF_PRST_VIDEO_MASK AND'ed with
 *  ::amdsmi_power_profile_status_t.available_profiles will be non-zero as will
 *  ::AMDSMI_PWR_PROF_PRST_VR_MASK AND'ed with
 *  ::amdsmi_power_profile_status_t.available_profiles. Additionally,
 *  ::amdsmi_power_profile_status_t.current will be set to the
 *  ::amdsmi_power_profile_preset_masks_t of the profile that is currently active.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_ind a 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *
 *  @param[in,out] status a pointer to ::amdsmi_power_profile_status_t that will be
 *  populated by a call to this function
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_power_profile_presets(amdsmi_processor_handle processor_handle, uint32_t sensor_ind,
                                         amdsmi_power_profile_status_t *status);

/** @} End tagClkPowerPerfQuery */

/*****************************************************************************/
/** @defgroup tagClkPowerPerfControl Clock, Power and Performance Control
 *  These functions provide control over clock frequencies, power and
 *  performance.
 *  @{
 */

/**
 *  @brief Set the PowerPlay performance level associated with the device with
 *  provided processor handle with the provided value. It is not supported
 *  on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and an ::amdsmi_dev_perf_level_t @p
 *  perf_level, this function will set the PowerPlay performance level for the
 *  device to the value @p perf_lvl.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] perf_lvl the value to which the performance level should be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_set_gpu_perf_level(amdsmi_processor_handle processor_handle, amdsmi_dev_perf_level_t perf_lvl);

/**
 *  @brief Set the overdrive percent associated with the device with provided
 *  processor handle with the provided value. See details for WARNING. It is
 *  not supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and an overdrive level @p od,
 *  this function will set the overdrive level for the device to the value
 *  @p od. The overdrive level is an integer value between 0 and 20, inclusive,
 *  which represents the overdrive percentage; e.g., a value of 5 specifies
 *  an overclocking of 5%.
 *
 *  The overdrive level is specific to the gpu system clock.
 *
 *  The overdrive level is the percentage above the maximum Performance Level
 *  to which overclocking will be limited. The overclocking percentage does
 *  not apply to clock speeds other than the maximum. This percentage is
 *  limited to 20%.
 *
 *   ******WARNING******
 *  Operating your AMD GPU outside of official AMD specifications or outside of
 *  factory settings, including but not limited to the conducting of
 *  overclocking (including use of this overclocking software, even if such
 *  software has been directly or indirectly provided by AMD or otherwise
 *  affiliated in any way with AMD), may cause damage to your AMD GPU, system
 *  components and/or result in system failure, as well as cause other problems.
 *  DAMAGES CAUSED BY USE OF YOUR AMD GPU OUTSIDE OF OFFICIAL AMD SPECIFICATIONS
 *  OR OUTSIDE OF FACTORY SETTINGS ARE NOT COVERED UNDER ANY AMD PRODUCT
 *  WARRANTY AND MAY NOT BE COVERED BY YOUR BOARD OR SYSTEM MANUFACTURER'S
 *  WARRANTY. Please use this utility with caution.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] od the value to which the overdrive level should be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_overdrive_level(amdsmi_processor_handle processor_handle, uint32_t od);

/**
 *  @brief Control the set of allowed frequencies that can be used for the
 *  specified clock. It is not supported on virtual machine guest
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a clock type @p clk_type, and a
 *  64 bit bitmask @p freq_bitmask, this function will limit the set of
 *  allowable frequencies. If a bit in @p freq_bitmask has a value of 1, then
 *  the frequency (as ordered in an ::amdsmi_frequencies_t returned by
 *  amdsmi_get_clk_freq()) corresponding to that bit index will be
 *  allowed.
 *
 *  This function will change the performance level to
 *  ::AMDSMI_DEV_PERF_LEVEL_MANUAL in order to modify the set of allowable
 *  frequencies. Caller will need to set to ::AMDSMI_DEV_PERF_LEVEL_AUTO in order
 *  to get back to default state.
 *
 *  All bits with indices greater than or equal to
 *  ::amdsmi_frequencies_t::num_supported will be ignored.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] clk_type the type of clock for which the set of frequencies
 *  will be modified
 *
 *  @param[in] freq_bitmask A bitmask indicating the indices of the
 *  frequencies that are to be enabled (1) and disabled (0). Only the lowest
 *  ::amdsmi_frequencies_t.num_supported bits of this mask are relevant.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_clk_freq(amdsmi_processor_handle processor_handle,
                                    amdsmi_clk_type_t clk_type, uint64_t freq_bitmask);

/**
 *  @brief Get the soc pstate policy for the processor
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, this function will write
 *  current soc pstate  policy settings to @p policy. All the processors at the same socket
 *  will have the same policy.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] policy the soc pstate policy for this processor.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_soc_pstate(amdsmi_processor_handle processor_handle,
                                      amdsmi_dpm_policy_t* policy);

/**
 *  @brief Set the soc pstate policy for the processor
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle and a soc pstate  policy @p policy_id,
 *  this function will set the soc pstate  policy for this processor. All the processors at
 *  the same socket will be set to the same policy.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] policy_id the soc pstate  policy id to set. The id is the id in
 *  amdsmi_dpm_policy_entry_t, which can be obtained by calling
 *  amdsmi_get_soc_pstate()
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_soc_pstate(amdsmi_processor_handle processor_handle,
                                      uint32_t policy_id);

/**
 *  @brief Get the xgmi per-link power down policy parameter for the processor
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, this function will write
 *  current xgmi plpd settings to @p policy. All the processors at the same socket
 *  will have the same policy.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] xgmi_plpd the xgmi plpd for this processor.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_xgmi_plpd(amdsmi_processor_handle processor_handle,
                                     amdsmi_dpm_policy_t* xgmi_plpd);

/**
 *  @brief Set the xgmi per-link power down policy parameter for the processor
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf}
 *
 *  @details Given a processor handle @p processor_handle and a dpm policy @p policy_id,
 *  this function will set the xgmi plpd for this processor. All the processors at
 *  the same socket will be set to the same policy.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] policy_id the xgmi plpd id to set. The id is the id in
 *  amdsmi_dpm_policy_entry_t, which can be obtained by calling
 *  amdsmi_get_xgmi_plpd()
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_xgmi_plpd(amdsmi_processor_handle processor_handle, uint32_t policy_id);

/**
 *  @brief Get the status of the Process Isolation
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_windows}
 *
 *  @details Given a processor handle @p processor_handle, this function will write
 *  current process isolation status to @p pisolate. The 0 is the process isolation
 *  disabled, and the 1 is the process isolation enabled.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] pisolate the process isolation status.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_process_isolation(amdsmi_processor_handle processor_handle,
                                                 uint32_t* pisolate);

/**
 *  @brief Enable/disable the system Process Isolation
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_windows}
 *
 *  @details Given a processor handle @p processor_handle and a process isolation @p pisolate,
 *  flag, this function will set the Process Isolation for this processor. The 0 is the process
 *  isolation disabled, and the 1 is the process isolation enabled.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] pisolate the process isolation status to set.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_gpu_process_isolation(amdsmi_processor_handle processor_handle,
                                                 uint32_t pisolate);

/**
 *  @brief Run the cleaner shader to clean up data in LDS/GPRs
 *
 *  @ingroup tagClkPowerPerfControl
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_windows}
 *
 *  @details Given a processor handle @p processor_handle,
 *  this function will clean the local data of this processor. This can be called between
 *  user logins to prevent information leak.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_clean_gpu_local_data(amdsmi_processor_handle processor_handle);

/** @} End tagClkPowerPerfControl */

/*****************************************************************************/
/** @defgroup tagVersionQuery Version Queries
 *  These functions provide version information about various subsystems.
 *  @{
 */

/**
 *  @brief Get the build version information for the currently running build of AMDSMI
 *
 *  @ingroup tagVersionQuery
 *
 *  @platform{gpu_bm_linux} @platform{cpu_bm} @platform{guest_1vf} @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @details  Get the major, minor, patch and build string for AMDSMI build
 *  currently in use through @p version
 *
 *  @param[in,out] version A pointer to an ::amdsmi_version_t structure that will
 *  be updated with the version information upon return.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_lib_version(amdsmi_version_t *version);

/** @} End tagVersionQuery */

/*****************************************************************************/
/** @defgroup tagECCInfo ECC Information
 *  @{
 */

/**
 *  @brief Retrieve the error counts for a GPU block. It is not supported on virtual
 *  machine guest
 *
 *  See [RAS Error Count sysfs Interface (AMDGPU RAS Support - Linux Kernel
 *  documentation)](https://docs.kernel.org/gpu/amdgpu/ras.html#ras-error-count-sysfs-interface)
 *  to learn how these error counts are accessed.
 *
 *  @ingroup tagECCInfo
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, an ::amdsmi_gpu_block_t @p block and a
 *  pointer to an ::amdsmi_error_count_t @p ec, this function will write the error
 *  count values for the GPU block indicated by @p block to memory pointed to by
 *  @p ec.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] block The block for which error counts should be retrieved
 *
 *  @param[in,out] ec A pointer to an ::amdsmi_error_count_t to which the error
 *  counts should be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_ecc_count(amdsmi_processor_handle processor_handle,
                                         amdsmi_gpu_block_t block, amdsmi_error_count_t *ec);

/**
 *  @brief Retrieve the enabled ECC bit-mask. It is not supported on virtual machine guest
 *
 *  See [RAS Error Count sysfs Interface (AMDGPU RAS Support - Linux Kernel
 *  documentation)](https://docs.kernel.org/gpu/amdgpu/ras.html#ras-error-count-sysfs-interface)
 *  to learn how these error counts are accessed.
 *
 *  @ingroup tagECCInfo
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, and a pointer to a uint64_t @p
 *  enabled_mask, this function will write bits to memory pointed to by
 *  @p enabled_blocks. Upon a successful call, @p enabled_blocks can then be
 *  AND'd with elements of the ::amdsmi_gpu_block_t ennumeration to determine if
 *  the corresponding block has ECC enabled. Note that whether a block has ECC
 *  enabled or not in the device is independent of whether there is kernel
 *  support for error counting for that block. Although a block may be enabled,
 *  but there may not be kernel support for reading error counters for that
 *  block.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] enabled_blocks A pointer to a uint64_t to which the enabled
 *  blocks bits will be written.
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_ecc_enabled(amdsmi_processor_handle processor_handle,
                                           uint64_t *enabled_blocks);

/**
 *  @brief Returns the total number of ECC errors (correctable,
 *         uncorrectable and deferred) in the given GPU. It is not supported on
 *         virtual machine guest
 *
 *  See [RAS Error Count sysfs Interface (AMDGPU RAS Support - Linux Kernel
 *  documentation)](https://docs.kernel.org/gpu/amdgpu/ras.html#ras-error-count-sysfs-interface)
 *  to learn how these error counts are accessed.
 *
 *  @ingroup tagECCInfo
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] ec Reference to ecc error count structure.
 *              Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_total_ecc_count(amdsmi_processor_handle processor_handle, amdsmi_error_count_t *ec);

#pragma pack(push, 1)

/**
 * @brief Cper
 *
 * @cond @tag{gpu_bm_linux} @tag{host} @endcond
 */
typedef struct {
    unsigned char b[16];
} amdsmi_cper_guid_t;

typedef struct {
    uint8_t seconds;
    uint8_t minutes;
    uint8_t hours;
    uint8_t flag;
    uint8_t day;
    uint8_t month;
    uint8_t year;
    uint8_t century;
} amdsmi_cper_timestamp_t;

typedef union {
    struct valid_bits_ {
        uint32_t platform_id  : 1;
        uint32_t timestamp    : 1;
        uint32_t partition_id : 1;
        uint32_t reserved     : 29;
    } valid_bits;
    uint32_t valid_mask;
} amdsmi_cper_valid_bits_t;

typedef struct {
    char                     signature[4];      //!< "CPER"
    uint16_t                 revision;
    uint32_t                 signature_end;     //!< 0xFFFFFFFF
    uint16_t                 sec_cnt;
    amdsmi_cper_sev_t        error_severity;
    amdsmi_cper_valid_bits_t cper_valid_bits;
    uint32_t                 record_length;     //!< Total size of CPER Entry
    amdsmi_cper_timestamp_t  timestamp;
    char                     platform_id[16];
    amdsmi_cper_guid_t       partition_id;      //!< Reserved
    char                     creator_id[16];
    amdsmi_cper_guid_t       notify_type;       //!< CMC, MCE, can use amdsmi_cper_notifiy_type_t to decode
    char                     record_id[8];      //!< Unique CPER Entry ID
    uint32_t                 flags;             //!< Reserved
    uint64_t                 persistence_info;  //!< Reserved
    uint8_t                  reserved[12];      //!< Reserved
} amdsmi_cper_hdr_t;

#pragma pack(pop)

/**
 * @brief Retrieve CPER entries cached in the driver.
 *
 * The user will pass buffers to hold the CPER data and CPER headers. The library will
 * fill the buffer based on the severity_mask user passed. It will also parse the CPER header
 * and stored in the cper_hdrs array. The user can use the cper_hdrs to get the timestamp and other header information.
 * A cursor is also returned to the user, which can be used to get the next set of CPER entries.
 *
 * If there are more data than any of the buffers user pass, the library will return AMDSMI_STATUS_MORE_DATA.
 * User can call the API again with the cursor returned at previous call to get more data.
 * If the buffer size is too small to even hold one entry, the library
 * will return AMDSMI_STATUS_OUT_OF_RESOURCES.
 *
 * Even if the API returns AMDSMI_STATUS_MORE_DATA, the 2nd call may still get the entry_count == 0 as the driver
 * cache may not contain the serverity user is interested in. The API should return AMDSMI_STATUS_SUCCESS in this case
 * so that user can ignore that call.
 *
 * @ingroup tagECCInfo
 *
 * @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
 *
 * @param[in] processor_handle Handle to the processor for which CPER entries are to be retrieved.
 * @param[in] severity_mask The severity mask of the entries to be retrieved.
 * @param[in,out] cper_data Pointer to a buffer where the CPER data will be stored. User must allocate the buffer
 *                and set the buf_size correctly.
 * @param[in,out] buf_size Pointer to a variable that specifies the size of the cper_data.
 *                On return, it will contain the actual size of the data written to the cper_data.
 * @param[in,out] cper_hdrs Array of the parsed headers of the cper_data. The user must allocate
 *                the array of pointers to cper_hdr. The library will fill the array with the pointers to the parsed
 *                headers. The underlying data is in the cper_data buffer and only pointer is stored in this array.
 * @param[in,out] entry_count Pointer to a variable that specifies the array length of the cper_hdrs user allocated.
 *                On return, it will contain the actual entries written to the cper_hdrs.
 * @param[in,out] cursor Pointer to a variable that will contain the  cursor  for the next call.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_cper_entries(amdsmi_processor_handle processor_handle, uint32_t severity_mask, char *cper_data,
    uint64_t *buf_size, amdsmi_cper_hdr_t** cper_hdrs, uint64_t *entry_count, uint64_t *cursor);

/** @} End tagECCInfo */

/*****************************************************************************/
/** @defgroup tagRasInfo     RAS information
 *  @{
 */

/**
 *  @brief Get the AFIDs from CPER buffer
 *
 *  @ingroup tagRasInfo
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf}
 *  @platform{guest_mvf}
 *
 *  @details A utility function which retrieves the AFIDs from the CPER record.
 *
 *  @param[in] cper_buffer a pointer to the buffer with one CPER record.
 *  The caller must make sure the whole CPER record is loaded into the buffer.
 *
 *  @param[in] buf_size is the size of the cper_buffer.
 *
 *  @param[out] afids a pointer to an array of uint64_t to which the AF IDs will be written
 *
 *  @param[in,out] num_afids As input, the value passed through this parameter is the number of
 *  uint64_t that may be safely written to the memory pointed to by @p afids. This is the limit
 *  on how many AF IDs will be written to @p afids. On return, @p num_afids will contain the
 *  number of AF IDs written to @p afids, or the number of AF IDs that could have been written
 *  if enough memory had been provided. It is suggest to pass MAX_NUMBER_OF_AFIDS_PER_RECORD for all
 *  AF Ids.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_afids_from_cper(char* cper_buffer, uint32_t buf_size, uint64_t* afids, uint32_t* num_afids);

/**
 *  @brief Returns RAS features info.
 *
 *  @ingroup tagRasInfo
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @param[in] processor_handle Device handle which to query
 *
 *  @param[out] ras_feature RAS features that are currently enabled and supported on
 *  the processor. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_ras_feature_info(amdsmi_processor_handle processor_handle, amdsmi_ras_feature_t *ras_feature);

/** @} End tagRasInfo */

/*****************************************************************************/
/** @defgroup tagErrorQuery Error Queries
 *  These functions provide error information about AMDSMI calls as well as
 *  device errors.
 *  @{
 */

/**
 *  @brief Retrieve the ECC status for a GPU block. It is not supported on virtual machine
 *  guest
 *
 *  See [RAS Error Count sysfs Interface (AMDGPU RAS Support - Linux Kernel
 *  documentation)](https://docs.kernel.org/gpu/amdgpu/ras.html#ras-error-count-sysfs-interface)
 *  to learn how these error counts are accessed.
 *
 *  @ingroup tagErrorQuery
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, an ::amdsmi_gpu_block_t @p block and
 *  a pointer to an ::amdsmi_ras_err_state_t @p state, this function will write
 *  the current state for the GPU block indicated by @p block to memory pointed
 *  to by @p state.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] block The block for which error counts should be retrieved
 *
 *  @param[in,out] state A pointer to an ::amdsmi_ras_err_state_t to which the
 *  ECC state should be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_ecc_status(amdsmi_processor_handle processor_handle,
                                          amdsmi_gpu_block_t block,
                                          amdsmi_ras_err_state_t *state);

/**
 *  @brief Get a description of a provided AMDSMI error status
 *
 *  @ingroup tagErrorQuery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{cpu_bm} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @details Set the provided pointer to a const char *, @p status_string, to
 *  a string containing a description of the provided error code @p status.
 *
 *  @param[in] status The error status for which a description is desired
 *
 *  @param[in,out] status_string A pointer to a const char * which will be made
 *  to point to a description of the provided error code
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_status_code_to_string(amdsmi_status_t status, const char **status_string);

/** @} End tagErrorQuery */

/*****************************************************************************/
/** @defgroup tagPerfCounter Performance Counter Functions
 *  These functions are used to configure, query and control performance
 *  counting.
 *
 *  These functions use the same mechanisms as the "perf" command line
 *  utility. They share the same underlying resources and have some similarities
 *  in how they are used. The events supported by this API should have
 *  corresponding perf events that can be seen with "perf stat ...". The events
 *  supported by perf can be seen with "perf list"
 *
 *  The types of events available and the ability to count those
 *  events are dependent on which device is being targeted and if counters are
 *  still available for that device, respectively.
 *  ::amdsmi_gpu_counter_group_supported() can be used to see which event types
 *  (::amdsmi_event_group_t) are supported for a given device. Assuming a device
 *  supports a given event type, we can then check to see if there are counters
 *  available to count a specific event with
 *  :: amdsmi_get_gpu_available_counters(). Counters may be occupied by other
 *  perf based programs.
 *
 *  Once it is determined that events are supported and counters are available,
 *  an event counter can be created/destroyed and controlled.
 *
 *  ::amdsmi_gpu_create_counter() allocates internal data structures that will be
 *  used to used to control the event counter, and return a handle to this data
 *  structure.
 *
 *  Once an event counter handle is obtained, the event counter can be
 *  controlled (i.e., started, stopped,...) with ::amdsmi_gpu_control_counter() by
 *  passing ::amdsmi_counter_command_t commands. ::AMDSMI_CNTR_CMD_START starts an
 *  event counter and ::AMDSMI_CNTR_CMD_STOP stops a counter.
 *  ::amdsmi_gpu_read_counter() reads an event counter.
 *
 *  Once the counter is no longer needed, the resources it uses should be freed
 *  by calling ::amdsmi_gpu_destroy_counter().
 *
 *  Important Notes about Counter Values
 *  ====================================
 *  - A running "absolute" counter is kept internally. For the discussion that
 *  follows, we will call the internal counter value at time @a t @a
 *  val<sub>t</sub>
 *  - Issuing ::AMDSMI_CNTR_CMD_START or calling ::amdsmi_gpu_read_counter(), causes
 *  AMDSMI (in kernel) to internally record the current absolute counter value
 *  - ::amdsmi_gpu_read_counter() returns the number of events that have occurred
 *  since the previously recorded value (ie, a relative value,
 *  @a val<sub>t</sub> - val<sub>t-1</sub>) from the issuing of
 *  ::AMDSMI_CNTR_CMD_START or calling ::amdsmi_gpu_read_counter()
 *
 *  Example of event counting sequence:
 *
 *  @latexonly
 *  \pagebreak
 *  @endlatexonly
 *  @code{.cpp}
 *
 *    amdsmi_counter_value_t value;
 *
 *    // Determine if AMDSMI_EVNT_GRP_XGMI is supported for device dv_ind
 *    ret = amdsmi_gpu_counter_group_supported(dv_ind, AMDSMI_EVNT_GRP_XGMI);
 *
 *    // See if there are counters available for device dv_ind for event
 *    // AMDSMI_EVNT_GRP_XGMI
 *
 *    ret =  amdsmi_get_gpu_available_counters(dv_ind,
 *                                 AMDSMI_EVNT_GRP_XGMI, &counters_available);
 *
 *    // Assuming AMDSMI_EVNT_GRP_XGMI is supported and there is at least 1
 *    // counter available for AMDSMI_EVNT_GRP_XGMI on device dv_ind, create
 *    // an event object for an event of group AMDSMI_EVNT_GRP_XGMI (e.g.,
 *    // AMDSMI_EVNT_XGMI_0_BEATS_TX) and get the handle
 *    // (amdsmi_event_handle_t).
 *
 *    ret = amdsmi_gpu_create_counter(dv_ind, AMDSMI_EVNT_XGMI_0_BEATS_TX,
 *                                                          &evnt_handle);
 *
 *    // A program that generates the events of interest can be started
 *    // immediately before or after starting the counters.
 *    // Start counting:
 *    ret = amdsmi_gpu_control_counter(evnt_handle, AMDSMI_CNTR_CMD_START, NULL);
 *
 *    // Wait...
 *
 *    // Get the number of events since AMDSMI_CNTR_CMD_START was issued:
 *    ret = amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle, &value)
 *
 *    // Wait...
 *
 *    // Get the number of events since amdsmi_gpu_read_counter() was last called:
 *    ret = amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle, &value)
 *
 *    // Stop counting.
 *    ret = amdsmi_gpu_control_counter(evnt_handle, AMDSMI_CNTR_CMD_STOP, NULL);
 *
 *    // Release all resources (e.g., counter and memory resources) associated
 *    with evnt_handle.
 *    ret = amdsmi_gpu_destroy_counter(evnt_handle);
 *  @endcode
 *  @{
 */

/**
 *  @brief Tell if an event group is supported by a given device. It is not supported
 *  on virtual machine guest
 *
 *  @ingroup tagPerfCounter
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and an event group specifier @p
 *  group, tell if @p group type events are supported by the device associated
 *  with @p processor_handle
 *
 *  @param[in] processor_handle processor handle of device being queried
 *
 *  @param[in] group ::amdsmi_event_group_t identifier of group for which support
 *  is being queried
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_counter_group_supported(amdsmi_processor_handle processor_handle, amdsmi_event_group_t group);

/**
 *  @brief Create a performance counter object
 *
 *  @ingroup tagPerfCounter
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Create a performance counter object of type @p type for the device
 *  with a processor handle of @p processor_handle, and write a handle to the object to the
 *  memory location pointed to by @p evnt_handle. @p evnt_handle can be used
 *  with other performance event operations. The handle should be deallocated
 *  with ::amdsmi_gpu_destroy_counter() when no longer needed.
 *
 *  @note This function requires root access
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] type the ::amdsmi_event_type_t of performance event to create
 *
 *  @param[in,out] evnt_handle A pointer to a ::amdsmi_event_handle_t which will be
 *  associated with a newly allocated counter
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_create_counter(amdsmi_processor_handle processor_handle, amdsmi_event_type_t type,
                                            amdsmi_event_handle_t *evnt_handle);

/**
 *  @brief Deallocate a performance counter object
 *
 *  @ingroup tagPerfCounter
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Deallocate the performance counter object with the provided
 *  ::amdsmi_event_handle_t @p evnt_handle
 *
 *  @note This function requires root access
 *
 *  @param[in] evnt_handle handle to event object to be deallocated
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_destroy_counter(amdsmi_event_handle_t evnt_handle);

/**
 *  @brief Issue performance counter control commands. It is not supported on
 *  virtual machine guest
 *
 *  @ingroup tagPerfCounter
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Issue a command @p cmd on the event counter associated with the
 *  provided handle @p evt_handle.
 *
 *  @note This function requires root access
 *
 *  @param[in] evt_handle an event handle
 *
 *  @param[in] cmd The event counter command to be issued
 *
 *  @param[in,out] cmd_args Currently not used. Should be set to NULL.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_control_counter(amdsmi_event_handle_t evt_handle,
                           amdsmi_counter_command_t cmd, void *cmd_args);

/**
 *  @brief Read the current value of a performance counter
 *
 *  @ingroup tagPerfCounter
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Read the current counter value of the counter associated with the
 *  provided handle @p evt_handle and write the value to the location pointed
 *  to by @p value.
 *
 *  @note This function requires root access
 *
 *  @param[in] evt_handle an event handle
 *
 *  @param[in,out] value pointer to memory of size of ::amdsmi_counter_value_t to
 *  which the counter value will be written
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_read_counter(amdsmi_event_handle_t evt_handle,
                        amdsmi_counter_value_t *value);

/**
 *  @brief Get the number of currently available counters. It is not supported on
 *  virtual machine guest
 *
 *  @ingroup tagPerfCounter
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a performance event group @p grp,
 *  and a pointer to a uint32_t @p available, this function will write the
 *  number of @p grp type counters that are available on the device with handle
 *  @p processor_handle to the memory that @p available points to.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] grp an event device group
 *
 *  @param[in,out] available A pointer to a uint32_t to which the number of
 *  available counters will be written
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_available_counters(amdsmi_processor_handle processor_handle,
                                   amdsmi_event_group_t grp, uint32_t *available);

/** @} End tagPerfCounter */

/*****************************************************************************/
/** @defgroup tagSystemInfo System Information Functions
 *  These functions are used to configure, query and control performance
 *  counting.
 *  @{
 */

/**
 *  @brief Get process information about processes currently using GPU
 *
 *  @ingroup tagSystemInfo
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a non-NULL pointer to an array @p procs of
 *  ::amdsmi_process_info_t's, of length *@p num_items, this function will write
 *  up to *@p num_items instances of ::amdsmi_process_info_t to the memory pointed
 *  to by @p procs. These instances contain information about each process
 *  utilizing a GPU. If @p procs is not NULL, @p num_items will be updated with
 *  the number of processes actually written. If @p procs is NULL, @p num_items
 *  will be updated with the number of processes for which there is current
 *  process information. Calling this function with @p procs being NULL is a way
 *  to determine how much memory should be allocated for when @p procs is not
 *  NULL.
 *
 *  @param[in,out] procs a pointer to memory provided by the caller to which
 *  process information will be written. This may be NULL in which case only @p
 *  num_items will be updated with the number of processes found.
 *
 *  @param[in,out] num_items A pointer to a uint32_t, which on input, should
 *  contain the amount of memory in ::amdsmi_process_info_t's which have been
 *  provided by the @p procs argument. On output, if @p procs is non-NULL, this
 *  will be updated with the number ::amdsmi_process_info_t structs actually
 *  written. If @p procs is NULL, this argument will be updated with the number
 *  processes for which there is information.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_compute_process_info(amdsmi_process_info_t *procs, uint32_t *num_items);

/**
 *  @brief Get process information about a specific process
 *
 *  @ingroup tagSystemInfo
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a pointer to an ::amdsmi_process_info_t @p proc and a process
 *  id
 *  @p pid, this function will write the process information for @p pid, if
 *  available, to the memory pointed to by @p proc.
 *
 *  @param[in] pid The process ID for which process information is being
 *  requested
 *
 *  @param[in,out] proc a pointer to a ::amdsmi_process_info_t to which
 *  process information for @p pid will be written if it is found.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_compute_process_info_by_pid(uint32_t pid, amdsmi_process_info_t *proc);

/**
 *  @brief Get the device indices currently being used by a process
 *
 *  @ingroup tagSystemInfo
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a process id @p pid, a non-NULL pointer to an array of
 *  uint32_t's @p processor_handleices of length *@p num_devices, this function will
 *  write up to @p num_devices device indices to the memory pointed to by
 *  @p processor_handleices. If @p processor_handleices is not NULL, @p num_devices will be
 *  updated with the number of gpu's currently being used by process @p pid.
 *  If @p processor_handleices is NULL, @p processor_handleices will be updated with the number of
 *  gpus currently being used by @p pid. Calling this function with @p
 *  dv_indices being NULL is a way to determine how much memory is required
 *  for when @p processor_handleices is not NULL.
 *
 *  @param[in] pid The process id of the process for which the number of gpus
 *  currently being used is requested
 *
 *  @param[in,out] dv_indices a pointer to memory provided by the caller to
 *  which indices of devices currently being used by the process will be
 *  written. This may be NULL in which case only @p num_devices will be
 *  updated with the number of devices being used.
 *
 *  @param[in,out] num_devices A pointer to a uint32_t, which on input, should
 *  contain the amount of memory in uint32_t's which have been provided by the
 *  @p processor_handleices argument. On output, if @p processor_handleices is non-NULL, this will
 *  be updated with the number uint32_t's actually written. If @p processor_handleices is
 *  NULL, this argument will be updated with the number devices being used.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_compute_process_gpus(uint32_t pid, uint32_t *dv_indices, uint32_t *num_devices);

/** @} End tagSystemInfo */

/*****************************************************************************/
/** @defgroup tagXGMI XGMI Functions
 *  These functions are used to configure, query and control XGMI.
 *  @{
 */

/**
 *  @brief Retrieve the XGMI error status for a device. It is not supported on
 *  virtual machine guest
 *
 *  @ingroup tagXGMI
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, and a pointer to an
 *  ::amdsmi_xgmi_status_t @p status, this function will write the current XGMI
 *  error state ::amdsmi_xgmi_status_t for the device @p processor_handle to the memory
 *  pointed to by @p status.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] status A pointer to an ::amdsmi_xgmi_status_t to which the
 *  XGMI error state should be written
 *  If this parameter is nullptr, this function will return
 *  ::AMDSMI_STATUS_INVAL if the function is supported with the provided,
 *  arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not supported with the
 *  provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_gpu_xgmi_error_status(amdsmi_processor_handle processor_handle, amdsmi_xgmi_status_t *status);

/**
 *  @brief Reset the XGMI error status for a device. It is not supported on virtual
 *  machine guest
 *
 *  @ingroup tagXGMI
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, this function will reset the
 *  current XGMI error state ::amdsmi_xgmi_status_t for the device @p processor_handle to
 *  amdsmi_xgmi_status_t::AMDSMI_XGMI_STATUS_NO_ERRORS
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_reset_gpu_xgmi_error(amdsmi_processor_handle processor_handle);

/**
 *  @brief          Returns XGMI information for the GPU.
 *
 *  @ingroup tagXGMI
 *
 *  @platform{gpu_bm_linux}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to xgmi information structure. Must be
 *                  allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_xgmi_info(amdsmi_processor_handle processor_handle, amdsmi_xgmi_info_t *info);

/**
 *  @brief Get the XGMI link status
 *
 *  @ingroup tagXGMI
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle,  this function
 *  will return the link status for each XGMI link connect to this processor.
 *  If the processor link type is not XGMI, it should return AMDSMI_STATUS_NOT_SUPPORTED.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[out] link_status The link status of the XGMI connect to this processor.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle processor_handle,
                                                amdsmi_xgmi_link_status_t* link_status);

/** @} End tagXGMI */

/*****************************************************************************/
/** @defgroup tagHWTopology Hardware Topology Functions
 *  These functions are used to query Hardware topology.
 *  @{
 */

/**
 *  @brief Return link metric information
 *
 *  @ingroup tagHWTopology
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @param[in] processor_handle PF of a processor for which to query
 *
 *  @param[out] link_metrics reference to the link metrics struct.
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_link_metrics(amdsmi_processor_handle processor_handle,
                                        amdsmi_link_metrics_t *link_metrics);

/**
 *  @brief Retrieve the NUMA CPU node number for a device
 *
 *  @ingroup tagHWTopology
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle, and a pointer to an
 *  uint32_t @p numa_node, this function will write the
 *  node number of NUMA CPU for the device @p processor_handle to the memory
 *  pointed to by @p numa_node.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in,out] numa_node A pointer to an uint32_t to which the
 *  numa node number should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_topo_get_numa_node_number(amdsmi_processor_handle processor_handle, uint32_t *numa_node);

/**
 *  @brief Retrieve the weight for a connection between 2 GPUs
 *
 *  @ingroup tagHWTopology
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst, and a pointer to an
 *  uint64_t @p weight, this function will write the
 *  weight for the connection between the device @p processor_handle_src
 *  and @p processor_handle_dst to the memory pointed to by @p weight.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] weight A pointer to an uint64_t to which the
 *  weight for the connection should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_topo_get_link_weight(amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst,
                            uint64_t *weight);

/**
 *  @brief Retreive minimal and maximal io link bandwidth between 2 GPUs
 *
 *  @ingroup tagHWTopology
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst,  pointer to an
 *  uint64_t @p min_bandwidth, and a pointer to uint64_t @p max_bandiwidth,
 *  this function will write theoretical minimal and maximal bandwidth limits.
 *  API works if src and dst are connected via xgmi and have 1 hop distance.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] min_bandwidth A pointer to an uint64_t to which the
 *  minimal bandwidth for the connection should be written.
 *
 *  @param[in,out] max_bandwidth A pointer to an uint64_t to which the
 *  maximal bandwidth for the connection should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_minmax_bandwidth_between_processors(amdsmi_processor_handle processor_handle_src,
                                                amdsmi_processor_handle processor_handle_dst,
                                                uint64_t *min_bandwidth,
                                                uint64_t *max_bandwidth);

/**
 *  @brief Retrieve the hops and the connection type between 2 GPUs
 *
 *  @ingroup tagHWTopology
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst, and a pointer to an
 *  uint64_t @p hops and a pointer to an AMDSMI_INK_TYPE @p type,
 *  this function will write the number of hops and the connection type
 *  between the device @p processor_handle_src and @p processor_handle_dst to the memory
 *  pointed to by @p hops and @p type.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] hops A pointer to an uint64_t to which the
 *  hops for the connection should be written.
 *
 *  @param[in,out] type A pointer to an ::AMDSMI_LINK_TYPE to which the
 *  type for the connection should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_topo_get_link_type(amdsmi_processor_handle processor_handle_src,
                          amdsmi_processor_handle processor_handle_dst,
                          uint64_t *hops, amdsmi_link_type_t *type);

/**
 *  @brief Retrieve the set of GPUs that are nearest to a given device
 *         at a specific interconnectivity level.
 *
 *  @ingroup tagHWTopology
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Once called topology_nearest_info will get populated with a list of
 *           all nearest devices for a given link_type. The list has a count of
 *           the number of devices found and their respective handles/identifiers.
 *
 *  @param[in] processor_handle The identifier of the given device.
 *
 *  @param[in] link_type The amdsmi_link_type_t level to search for nearest GPUs.
 *
 *  @param[in,out] topology_nearest_info
 *                 .count;
 *                   - When zero, set to the number of matching GPUs such that .device_list can be malloc'd.
 *                   - When non-zero, .device_list will be filled with count number of processor_handle.
 *                 .device_list An array of processor_handle for GPUs found at level.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail.
 */
amdsmi_status_t
amdsmi_get_link_topology_nearest(amdsmi_processor_handle processor_handle,
                                 amdsmi_link_type_t link_type,
                                 amdsmi_topology_nearest_t* topology_nearest_info);

/**
 *  @brief Return P2P availability status between 2 GPUs
 *
 *  @ingroup tagHWTopology
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst, and a pointer to a
 *  bool @p accessible, this function will write the P2P connection status
 *  between the device @p processor_handle_src and @p processor_handle_dst to the memory
 *  pointed to by @p accessible.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] accessible A pointer to a bool to which the status for
 *  the P2P connection availablity should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_is_P2P_accessible(amdsmi_processor_handle processor_handle_src,
                         amdsmi_processor_handle processor_handle_dst,
                         bool *accessible);

/**
 *  @brief Retrieve connection type and P2P capabilities between 2 GPUs
 *
 *  @ingroup tagHWTopology
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @details Given a source processor handle @p processor_handle_src and
 *  a destination processor handle @p processor_handle_dst, a pointer to an amdsmi_link_type_t @p type,
 *  and a pointer to amdsmi_p2p_capability_t @p cap. This function will write the connection type,
 *  and io link capabilities between the device
 *  @p processor_handle_src and @p processor_handle_dst to the memory
 *  pointed to by @p cap and @p type.
 *
 *  @param[in] processor_handle_src the source processor handle
 *
 *  @param[in] processor_handle_dst the destination processor handle
 *
 *  @param[in,out] type A pointer to an ::amdsmi_link_type_t to which the
 *  type for the connection should be written.
 *
 *  @param[in,out] cap A pointer to an ::amdsmi_p2p_capability_t to which the
 *  io link capabilities should be written.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_topo_get_p2p_status(amdsmi_processor_handle processor_handle_src,
                           amdsmi_processor_handle processor_handle_dst,
                           amdsmi_link_type_t *type, amdsmi_p2p_capability_t *cap);

/** @} End tagHWTopology */

/*****************************************************************************/
/** @defgroup tagComputePartition Compute Partition Functions
 *  These functions are used to configure and query the device's
 *  compute parition setting.
 *  @{
 */

/**
 *  @brief Retrieves the current compute partitioning for a desired device
 *
 *  @ingroup tagComputePartition
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details
 *  Given a processor handle @p processor_handle and a string @p compute_partition ,
 *  and uint32 @p len , this function will attempt to obtain the device's
 *  current compute partition setting string. Upon successful retreival,
 *  the obtained device's compute partition settings string shall be stored in
 *  the passed @p compute_partition char string variable.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] compute_partition a pointer to a char string variable,
 *  which the device's current compute partition will be written to.
 *
 *  @param[in] len the length of the caller provided buffer @p compute_partition,
 *  suggested length is 4 or greater.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @retval ::AMDSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire compute partition value. In this case,
 *  only @p len bytes will be written.
 *  @return ::amdsmi_status_t
 */
amdsmi_status_t
amdsmi_get_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                 char *compute_partition, uint32_t len);

/**
 *  @brief Modifies a selected device's compute partition setting.
 *
 *  @ingroup tagComputePartition
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle, a type of compute partition
 *  @p compute_partition, this function will attempt to update the selected
 *  device's compute partition setting. This function does not allow any concurrent operations.
 *  Device must be idle and have no workloads when performing set partition operations.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[in] compute_partition using enum ::amdsmi_compute_partition_type_t,
 *  define what the selected device's compute partition setting should be
 *  updated to.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_PERMISSION function requires root access
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @retval ::AMDSMI_STATUS_SETTING_UNAVAILABLE the provided setting is
 *  unavailable for current device
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @return ::amdsmi_status_t
 */
amdsmi_status_t
amdsmi_set_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                 amdsmi_compute_partition_type_t compute_partition);

/** @} End tagComputePartition */

/*****************************************************************************/
/** @defgroup tagMemoryPartition Memory Partition Functions
 *  These functions are used to query and set the device's current memory
 *  partition.
 *  @{
 */

/**
 *  @brief Retrieves the current memory partition for a desired device
 *
 *  @ingroup tagMemoryPartition
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details
 *  Given a processor handle @p processor_handle and a string @p memory_partition ,
 *  and uint32 @p len , this function will attempt to obtain the device's
 *  memory partition string. Upon successful retreival, the obtained device's
 *  memory partition string shall be stored in the passed @p memory_partition
 *  char string variable.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] memory_partition a pointer to a char string variable,
 *  which the device's memory partition will be written to.
 *
 *  @param[in] len the length of the caller provided buffer @p memory_partition ,
 *  suggested length is 5 or greater.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @retval ::AMDSMI_STATUS_UNEXPECTED_DATA data provided to function is not valid
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::AMDSMI_STATUS_INSUFFICIENT_SIZE is returned if @p len bytes is not
 *  large enough to hold the entire memory partition value. In this case,
 *  only @p len bytes will be written.
 *  @return ::amdsmi_status_t
 */
amdsmi_status_t
amdsmi_get_gpu_memory_partition(amdsmi_processor_handle processor_handle, char *memory_partition, uint32_t len);

/**
 *  @brief Modifies a selected device's current memory partition setting.
 *
 *  @ingroup tagMemoryPartition
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a type of memory partition
 *  @p memory_partition, this function will attempt to update the selected
 *  device's memory partition setting. This function does not allow any concurrent operations.
 *  Device must be idle and have no workloads when performing set partition operations.
 *
 *  On @platform{gpu_bm_linux} AMDGPU driver restart is REQUIRED to complete updating to
 *  the new memory partition setting. Refer to `amdsmi_gpu_driver_reload()` for more details.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[in] memory_partition using enum ::amdsmi_memory_partition_type_t,
 *  define what the selected device's current mode setting should be updated to.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS call was successful
 *  @retval ::AMDSMI_STATUS_PERMISSION function requires root access
 *  @retval ::AMDSMI_STATUS_INVAL the provided arguments are not valid
 *  @retval ::AMDSMI_STATUS_NOT_SUPPORTED installed software or hardware does not
 *  support this function
 *  @retval ::AMDSMI_STATUS_AMDGPU_RESTART_ERR could not successfully restart the amdgpu driver
 *  @return ::amdsmi_status_t
 *
 */
amdsmi_status_t
amdsmi_set_gpu_memory_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_memory_partition_type_t memory_partition);
/**
 *  @brief Returns current gpu memory partition capabilities
 *
 *  @ingroup tagMemoryPartition
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[out] config reference to the memory partition config.
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_memory_partition_config(amdsmi_processor_handle processor_handle,
                                       amdsmi_memory_partition_config_t *config);

/**
 *  @brief Sets memory partition mode
 *  Set accelerator partition setting based on profile_index
 *  from amdsmi_get_gpu_accelerator_partition_profile_config
 *
 *  @ingroup tagMemoryPartition
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @details Given a processor handle @p processor_handle and a type of memory partition
 *  @p mode, this function will attempt to update the selected
 *  device's memory partition setting. This function does not allow any concurrent operations.
 *  Device must be idle and have no workloads when performing set partition operations.
 *
 *  On @platform{gpu_bm_linux} AMDGPU driver restart is REQUIRED to complete updating to
 *  the new memory partition setting. Refer to `amdsmi_gpu_driver_reload()` for more details.
 *
 *  @param[in] processor_handle A processor handle
 *
 *  @param[in] mode Enum representing memory partitioning mode to set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_set_gpu_memory_partition_mode(amdsmi_processor_handle processor_handle,
              amdsmi_memory_partition_type_t mode);

/** @} End tagMemoryPartition */

/*****************************************************************************/
/** @defgroup tagAcceleratorPartition Accelerator Partition Profile Functions
 *  These functions are used to configure and query the device's
 *  accelerator parition profile setting.
 *  @{
 */

/**
 *  @brief Returns gpu accelerator partition caps as currently configured in the system
 *
 *  @ingroup tagAcceleratorPartition
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @note User must use admin/elevated privledges to run this API, or API will not be able to read resources.
 *  Otherwise, API will fill in the structure with as much information as possible.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] profile_config reference to the accelerator partition config.
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_accelerator_partition_profile_config(amdsmi_processor_handle processor_handle,
                              amdsmi_accelerator_partition_profile_config_t *profile_config);

/**
 *  @brief Returns current gpu accelerator partition cap
 *
 *  @ingroup tagAcceleratorPartition
 *
 *  @note User must use admin/elevated privledges to run this API, or API will not be able to read resources.
 *  Otherwise, API will fill in the structure with as much information as possible.
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] profile reference to the accelerator partition profile.
 *  Must be allocated by user.
 *
 *  @param[in,out] partition_id array of ids for current accelerator profile.
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_accelerator_partition_profile(amdsmi_processor_handle processor_handle,
                                             amdsmi_accelerator_partition_profile_t *profile,
                                             uint32_t *partition_id);

/**
 *  @brief Set accelerator partition setting based on profile_index
 *  from amdsmi_get_gpu_accelerator_partition_profile_config
 *
 *  @ingroup tagAcceleratorPartition
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @note On @platform{gpu_bm_linux} User must use admin/elevated privledges
 *  to run this API, or API will not be able to read resources.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[in] profile_index Represents index of a partition user wants to set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_set_gpu_accelerator_partition_profile(amdsmi_processor_handle processor_handle,
                                             uint32_t profile_index);

/** @} End tagAcceleratorPartition */

/*****************************************************************************/
/** @defgroup tagEventNotification Event Notification Functions
 *  These functions are used to configure for and get asynchronous event
 *  notifications.
 *  @{
 */

/**
 *  @brief Prepare to collect event notifications for a GPU
 *
 *  @ingroup tagEventNotification
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details This function prepares to collect events for the GPU with device
 *  ID @p processor_handle, by initializing any required system parameters. This call
 *  may open files which will remain open until ::amdsmi_stop_gpu_event_notification()
 *  is called.
 *
 *  @param[in] processor_handle a processor handle corresponding to the device on which to
 *  listen for events
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_init_gpu_event_notification(amdsmi_processor_handle processor_handle);

/**
 *  @brief Specify which events to collect for a device
 *
 *  @ingroup tagEventNotification
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a processor handle @p processor_handle and a @p mask consisting of
 *  elements of ::amdsmi_evt_notification_type_t OR'd together, this function
 *  will listen for the events specified in @p mask on the device
 *  corresponding to @p processor_handle.
 *
 *  @param[in] processor_handle a processor handle corresponding to the device on which to
 *  listen for events
 *
 *  @param[in] mask Bitmask generated by OR'ing 1 or more elements of
 *  ::amdsmi_evt_notification_type_t indicating which event types to listen for,
 *  where the amdsmi_evt_notification_type_t value indicates the bit field, with
 *  bit position starting from 1.
 *  For example, if the mask field is 0x0000000000000003, which means first bit,
 *  bit 1 (bit position start from 1) and bit 2 are set, which indicate interest
 *  in receiving AMDSMI_EVT_NOTIF_VMFAULT (which has a value of 1) and
 *  AMDSMI_EVT_NOTIF_THERMAL_THROTTLE event (which has a value of 2).
 *
 *  @note ::AMDSMI_STATUS_INIT_ERROR is returned if
 *  ::amdsmi_init_gpu_event_notification() has not been called before a call to this
 *  function
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle processor_handle, uint64_t mask);

/**
 *  @brief Collect event notifications, waiting a specified amount of time
 *
 *  @ingroup tagEventNotification
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Given a time period @p timeout_ms in milliseconds and a caller-
 *  provided buffer of ::amdsmi_evt_notification_data_t's @p data with a length
 *  (in ::amdsmi_evt_notification_data_t's, also specified by the caller) in the
 *  memory location pointed to by @p num_elem, this function will collect
 *  ::amdsmi_evt_notification_type_t events for up to @p timeout_ms milliseconds,
 *  and write up to *@p num_elem event items to @p data. Upon return @p num_elem
 *  is updated with the number of events that were actually written. If events
 *  are already present when this function is called, it will write the events
 *  to the buffer then poll for new events if there is still caller-provided
 *  buffer available to write any new events that would be found.
 *
 *  This function requires prior calls to ::amdsmi_init_gpu_event_notification() and
 *  :: amdsmi_set_gpu_event_notification_mask(). This function polls for the
 *  occurrance of the events on the respective devices that were previously
 *  specified by :: amdsmi_set_gpu_event_notification_mask().
 *
 *  @param[in] timeout_ms number of milliseconds to wait for an event
 *  to occur
 *
 *  @param[in,out] num_elem pointer to uint32_t, provided by the caller. On
 *  input, this value tells how many ::amdsmi_evt_notification_data_t elements
 *  are being provided by the caller with @p data. On output, the location
 *  pointed to by @p num_elem will contain the number of items written to
 *  the provided buffer.
 *
 *  @param[out] data pointer to a caller-provided memory buffer of size
 *  @p num_elem ::amdsmi_evt_notification_data_t to which this function may safely
 *  write. If there are events found, up to @p num_elem event items will be
 *  written to @p data.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_event_notification(int timeout_ms, uint32_t *num_elem, amdsmi_evt_notification_data_t *data);

/**
 *  @brief Close any file handles and free any resources used by event
 *  notification for a GPU
 *
 *  @ingroup tagEventNotification
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details Any resources used by event notification for the GPU with
 *  processor handle @p processor_handle will be free with this
 *  function. This includes freeing any memory and closing file handles. This
 *  should be called for every call to ::amdsmi_init_gpu_event_notification()
 *
 *  @param[in] processor_handle The processor handle of the GPU for which event
 *  notification resources will be free
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_stop_gpu_event_notification(amdsmi_processor_handle processor_handle);

/** @} End tagEventNotification */

/*****************************************************************************/
/** @defgroup tagSoftwareVersion     Software Version Information
 *  @{
 */

/**
 *  @brief Returns the driver version information
 *
 *  @ingroup tagSoftwareVersion
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] info Reference to driver information structure. Must be
 *              allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_driver_info(amdsmi_processor_handle processor_handle, amdsmi_driver_info_t *info);

/** @} End tagSoftwareVersion */

/*****************************************************************************/
/** @defgroup tagAsicBoardInfo  ASIC & Board Static Information
 *  @{
 */

/**
 *  @brief Returns the ASIC information for the device
 *
 *  @ingroup tagAsicBoardInfo
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @details This function returns ASIC information such as the product name,
 *           the vendor ID, the subvendor ID, the device ID,
 *           the revision ID and the serial number.
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] info Reference to static asic information structure.
 *              Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_asic_info(amdsmi_processor_handle processor_handle, amdsmi_asic_info_t *info);


/**
 *  @brief          Returns the KFD (Kernel Fusion Driver) information for the device
 *
 *  @ingroup tagAsicBoardInfo
 *
 *  @platform{gpu_bm_linux}
 *
 *  @details        This function returns KFD information populated into the amdsmi_kfd_info_t.
 *                  This contains the kfd_id and node_id which allow for the ID and
 *                  index of this device in the KFD.
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to kfd information structure.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_kfd_info(amdsmi_processor_handle processor_handle, amdsmi_kfd_info_t *info);

/**
 *  @brief Returns vram info
 *
 *  @ingroup tagAsicBoardInfo
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @param[in] processor_handle PF of a processor for which to query
 *
 *  @param[out] info Reference to vram info structure
 *  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_gpu_vram_info(amdsmi_processor_handle processor_handle, amdsmi_vram_info_t *info);

/**
 *  @brief Returns the board part number and board information for the requested device
 *
 *  @ingroup tagAsicBoardInfo
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] info Reference to board info structure.
 *              Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_board_info(amdsmi_processor_handle processor_handle, amdsmi_board_info_t *info);

/**
 *  @brief Returns the power caps as currently configured in the system.
 *
 *  @ingroup tagAsicBoardInfo
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[in] sensor_ind A 0-based sensor index. Normally, this will be 0.
 *  If a device has more than one sensor, it could be greater than 0.
 *  Parameter @p sensor_ind is unused on @platform{host}.
 *
 *  @param[out] info Reference to power caps information structure. Must be
 *  allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_power_cap_info(amdsmi_processor_handle processor_handle, uint32_t sensor_ind,
                          amdsmi_power_cap_info_t *info);

/**
 *  @brief Returns the PCIe info for the GPU.
 *
 *  @ingroup tagAsicBoardInfo
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] info Reference to the PCIe information
 *  returned by the library. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_pcie_info(amdsmi_processor_handle processor_handle, amdsmi_pcie_info_t *info);

/**
 *  @brief Returns the 'xcd_counter' from the GPU metrics associated with the device
 *
 *  @ingroup tagAsicBoardInfo
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[inout] xcd_count a pointer to uint16_t to which the device gpu
 *  metric unit will be stored. Must be allocated by user.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS is returned upon successful call.
 *          ::AMDSMI_STATUS_NOT_SUPPORTED is returned in case the metric unit
 *            does not exist for the given device.
 */
amdsmi_status_t amdsmi_get_gpu_xcd_counter(amdsmi_processor_handle processor_handle,
                                           uint16_t *xcd_count);

/**
 * @brief Retrieves node power management (NPM) status and power limit for the specified node.
 *
 * @ingroup tagNodeInfo
 *
 * @platform{gpu_bm_linux} @platform{host}
 *
 * @details This function queries the NPM controller for the given node and returns whether NPM is enabled,
 * along with the current node-level power limit in Watts. The NPM status and limit are set out-of-band
 * and reported via this API.
 *
 * @param[in]  node_handle Handle to the Node to query.
 * @param[out] info Pointer to amdsmi_npm_info_t structure to receive NPM status and limit.
 *             Must be allocated by the user.
 *
 * @return ::AMDSMI_STATUS_SUCCESS on success, non-zero on failure.
 */
amdsmi_status_t amdsmi_get_npm_info(amdsmi_node_handle node_handle, amdsmi_npm_info_t *info);

/** @} End tagAsicBoardInfo */

/*****************************************************************************/
/** @defgroup tagFWVbiosQuery Firmware & VBIOS queries
 *  @{
 */

/**
 *  @brief Returns the firmware versions running on the device.
 *
 *  @ingroup tagFWVbiosQuery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] info Reference to the fw info. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_fw_info(amdsmi_processor_handle processor_handle, amdsmi_fw_info_t *info);

/**
 *  @brief Returns the static information for the vBIOS on the device.
 *
 *  @ingroup tagFWVbiosQuery
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_1vf} @platform{guest_mvf}
 *  @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] info Reference to static vBIOS information.
 *              Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_vbios_info(amdsmi_processor_handle processor_handle, amdsmi_vbios_info_t *info);

/** @} End tagFWVbiosQuery */

/*****************************************************************************/
/** @defgroup tagGPUMonitor GPU Monitoring
 *  @{
 */

/**
 *  @brief Get the temperature metric value for the specified metric, from the
 *  specified temperature sensor on the specified device. It is not supported on
 *  virtual machine guest
 *
 *  @ingroup tagGPUMonitor
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}
 *
 *  @details Given a processor handle @p processor_handle, a sensor type @p sensor_type, a
 *  ::amdsmi_temperature_metric_t @p metric and a pointer to an int64_t @p
 *  temperature, this function will write the value of the metric indicated by
 *  @p metric and @p sensor_type to the memory location @p temperature.
 *
 *  @param[in] processor_handle a processor handle
 *
 *  @param[in] sensor_type part of device from which temperature should be
 *  obtained. This should come from the enum ::amdsmi_temperature_type_t
 *
 *  @param[in] metric enum indicated which temperature value should be
 *  retrieved
 *
 *  @param[in,out] temperature a pointer to int64_t to which the temperature is in Celsius.
 *  If this parameter is nullptr, this function will return ::AMDSMI_STATUS_INVAL if the function
 *  is supported with the provided, arguments and ::AMDSMI_STATUS_NOT_SUPPORTED if it is not
 *  supported with the provided arguments.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_temp_metric(amdsmi_processor_handle processor_handle,
                                       amdsmi_temperature_type_t sensor_type,
                                       amdsmi_temperature_metric_t metric, int64_t *temperature);

/**
 *  @brief Returns the current usage of the GPU engines (GFX, MM and MEM).
 *  Each usage is reported as a percentage from 0-100%.
 *
 *  @ingroup tagGPUMonitor
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[out] info Reference to the gpu engine usage structure. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_activity(amdsmi_processor_handle processor_handle, amdsmi_engine_usage_t *info);

/**
 *  @brief Returns the current power and voltage of the GPU.
 *
 *  @ingroup tagGPUMonitor
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_windows} @platform{guest_1vf}
 *
 *  @note amdsmi_power_info_t::socket_power metric can rarely spike above the socket power limit in some cases
 *  @note unsupported struct members are set to UINT32_MAX
 *
 *  @param[in] processor_handle PF of a processor for which  to query
 *
 *  @param[out] info Reference to the gpu power structure. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_power_info(amdsmi_processor_handle processor_handle, amdsmi_power_info_t *info);

/**
 *  @brief Returns is power management enabled
 *
 *  @ingroup tagGPUMonitor
 *
 *  @platform{gpu_bm_linux} @platform{host}
 *
 *  @param[in] processor_handle PF of a processor for which to query
 *
 *  @param[out] enabled Reference to bool. Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_is_gpu_power_management_enabled(amdsmi_processor_handle processor_handle, bool *enabled);

/**
 *  @brief Returns the measurements of the clocks in the GPU
 *         for the GFX and multimedia engines and Memory. This call
 *         reports the averages over 1s in MHz. It is not supported
 *         on virtual machine guest
 *
 *  @ingroup tagGPUMonitor
 *
 *  @platform{gpu_bm_linux} @platform{host} @platform{guest_windows}
 *
 *  @param[in] processor_handle Device which to query
 *
 *  @param[in] clk_type Enum representing the clock type to query.
 *
 *  @param[out] info Reference to the gpu clock structure.
 *              Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_clock_info(amdsmi_processor_handle processor_handle, amdsmi_clk_type_t clk_type, amdsmi_clk_info_t *info);

/**
 *  @brief          Returns the VRAM usage (both total and used memory)
 *                  in MegaBytes.
 *
 *  @ingroup tagGPUMonitor
 *
 *  @platform{gpu_bm_linux}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to vram information.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_gpu_vram_usage(amdsmi_processor_handle processor_handle, amdsmi_vram_usage_t *info);

/**
 *  @brief          Returns the violations for a processor
 *
 *  Warning: API will be slow due to polling driver for 2 samples. Require
 *  a minimum wait of 100ms between the 2 samples in order to calculate. Otherwise
 *  users would need to use amdsmi_get_gpu_metrics_info for BM. See that API's struct
 *  for calculations.
 *
 *  @ingroup tagGPUMonitor
 *
 *  @platform{gpu_bm_linux}
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[out]     info Reference to all violation status details available.
 *                  Must be allocated by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t
amdsmi_get_violation_status(amdsmi_processor_handle processor_handle,
                            amdsmi_violation_status_t *info);

/** @} End tagGPUMonitor */

/*****************************************************************************/
/** @defgroup tagProcessInfo Process information
 *  @{
 */

/**
 *  @brief Returns the list of process information running on a given GPU.
 *  If pdh.dll is not present on the system, this API returns
 *  AMDSMI_STATUS_NOT_SUPPORTED.
 *
 *  @ingroup tagProcessInfo
 *
 *  @platform{gpu_bm_linux} @platform{guest_windows}
 *
 *  @warning IMPORTANT: To get valid return values, at least 1 second needs to pass
 *  from starting the program to the first call of this function,
 *  and before every following call of this function after that, to get correct values
 *
 *  @note The user provides a buffer to store the list and the maximum
 *        number of processes that can be returned. If the user sets
 *        max_processes to 0, the current total number of processes will
 *        replace max_processes param. After that, the function needs to be
 *        called again, with updated max_processes, to successfully fill the
 *        process list, which was previously allocated with max_processes
 *
 *  @note If the reserved size for processes is smaller than the number of
 *        actual processes running. The AMDSMI_STATUS_OUT_OF_RESOURCES is
 *        an indication the caller should handle the situation (resize).
 *        The max_processes is always changed to reflect the actual size of
 *        list of processes running, so the caller knows where it is at.
 *
 *  @param[in]      processor_handle Device which to query
 *
 *  @param[in,out]  max_processes Reference to the size of the list buffer in
 *                  number of elements. Returns the return number of elements
 *                  in list or the number of running processes if equal to 0,
 *                  and if given value in param max_processes is less than
 *                  number of processes currently running,
 *                  AMDSMI_STATUS_OUT_OF_RESOURCES will be returned.
 *
 *                  For cases where max_process is not zero (0), it specifies the list's size limit.
 *                  That is, the maximum size this list will be able to hold. After the list is built
 *                  internally, as a return status, we will have AMDSMI_STATUS_OUT_OF_RESOURCES when
 *                  the original size limit is smaller than the actual list of processes running.
 *                  Hence, the caller is aware the list size needs to be resized, or
 *                  AMDSMI_STATUS_SUCCESS otherwise.
 *                  Holding a copy of max_process before it is passed in will be helpful for monitoring
 *                  the allocations done upon each call since the max_process will permanently be changed
 *                  to reflect the actual number of processes running.
 *
 *  @param[out]     list Reference to a user-provided buffer where the process
 *                  list will be returned. This buffer must contain at least
 *                  max_processes entries of type amd_proc_info_list_t. Must be allocated
 *                  by user.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success,
 *                            | ::AMDSMI_STATUS_OUT_OF_RESOURCES, filled list buffer with data, but number of
 *                                actual running processes is larger than the size provided.
 */
amdsmi_status_t
amdsmi_get_gpu_process_list(amdsmi_processor_handle processor_handle, uint32_t *max_processes, amdsmi_proc_info_t *list);

/** @} End tagProcessInfo */


/*****************************************************************************/
/** @defgroup tagDriverControl Driver control mechanisms
 *  These functions provide control over the driver. Users should use with
 *  caution as they may cause the driver to become unstable.
 *  @{
 */
/**
 *  @brief Restart the device driver (kmod module) for all AMD GPUs on the
 *  system.
 *
 *  @ingroup tagDriverControl
 *
 *  @platform{gpu_bm_linux} @platform{guest_1vf} @platform{guest_mvf}
 *
 *  @details This function will reload the AMD GPU driver as described in
 *  the Linux kernel documentation -
 *  https://docs.kernel.org/admin-guide/sysctl/kernel.html#modprobe
 *  with no extra parameters as specified in
 *  https://docs.kernel.org/gpu/amdgpu/module-parameters.html.
 * 
 *  Use this function with caution, as it will unload and reload the AMD GPU
 *  driver: `modprobe -r amdgpu && modprobe amdgpu`. 
 *  
 *  Any process or workload using the AMD GPU driver is REQUIRED to be
 *  stopped before calling this function. Otherwise, function will return
 *  ::AMDSMI_STATUS_AMDGPU_RESTART_ERR could not successfully restart
 *  the amdgpu driver.
 * 
 *  User is REQUIRED to have root/admin privileges to call this function.
 *  Otherwise, this function will return ::AMDSMI_STATUS_NO_PERM.
 * 
 *  This API will take time to complete, as we are checking the driver's
 *  loading status to confirm it reloaded properly. If
 *  ::AMDSMI_STATUS_AMDGPU_RESTART_ERR is returned, it means the driver
 *  did not reload properly and the user should check dmesg logs.
 * 
 *  This function has been created in order to conviently reload the
 *  AMD GPU driver once `amdsmi_set_gpu_memory_partition()` or
 *  `amdsmi_set_gpu_memory_partition_mode()` successfully has been changed
 *  on Baremetal systems. Now users can control the reload once all GPU
 *  processes/workloads have been stopped on the AMD GPU driver.
 *  A (AMD GPU) driver reload is REQUIRED to complete changing
 *  to the new memory partition configuration
 *  (`amdsmi_set_gpu_memory_partition()`/`amdsmi_set_gpu_memory_partition_mode()`)
 *  operation MUST be successful. This function WILL EFFECT all GPUs in the
 *  hive to be reconfigured with the specified memory partition configuration.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success
 *  @return                   | ::AMDSMI_STATUS_NO_PERM function requires root access
 *  @return                   | ::AMDSMI_STATUS_AMDGPU_RESTART_ERR could not successfully restart
 *                                the amdgpu driver.
 */
amdsmi_status_t amdsmi_gpu_driver_reload(void);

/** @} End tagDriverControl */

#ifdef ENABLE_ESMI_LIB

/*****************************************************************************/
/** @defgroup tagEnergyInfo Energy information (RAPL MSR)
 *  @{
 */

/**
 *  @brief Get the core energy for a given core.
 *
 *  @ingroup tagEnergyInfo
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu core which to query
 *
 *  @param[in,out]    penergy - Input buffer to return the core energy
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_core_energy(amdsmi_processor_handle processor_handle,
                                           uint64_t *penergy);

/**
 *  @brief Get the socket energy for a given socket.
 *
 *  @ingroup tagEnergyInfo
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    penergy - Input buffer to return the socket energy
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_energy(amdsmi_processor_handle processor_handle,
                                             uint64_t *penergy);

/** @} End tagEnergyInfo */

/*****************************************************************************/
/** @defgroup tagHSMPSystemStats HSMP system statistics
 *  @{
 */

/**
 *  @brief Get Number of threads Per Core.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in,out]    threads_per_core - Input buffer to return the Number of threads Per Core
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_threads_per_core(uint32_t *threads_per_core);

/**
 *  @brief Get HSMP Driver Version.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in,out]  amdsmi_hsmp_driver_ver - Input buffer to return the HSMP Driver version
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_hsmp_driver_version(amdsmi_processor_handle processor_handle,
                                              amdsmi_hsmp_driver_version_t *amdsmi_hsmp_driver_ver);

/**
 *  @brief Get SMU Firmware Version.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in,out]    amdsmi_smu_fw - Input buffer to return the firmware version
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_smu_fw_version(amdsmi_processor_handle processor_handle,
                                              amdsmi_smu_fw_version_t *amdsmi_smu_fw);

/**
 *  @brief Get HSMP protocol Version.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in,out]    proto_ver - Input buffer to return the protocol version
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_hsmp_proto_ver(amdsmi_processor_handle processor_handle,
                                              uint32_t *proto_ver);

/**
 *  @brief Get normalized status of the processor's PROCHOT status.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    prochot - Input buffer to return the procohot status.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_prochot_status(amdsmi_processor_handle processor_handle,
                                              uint32_t *prochot);

/**
 *  @brief Get Data fabric clock and Memory clock in MHz.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    fclk - Input buffer to return fclk
 *
 *  @param[in,out]    mclk - Input buffer to return mclk
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_fclk_mclk(amdsmi_processor_handle processor_handle,
                                         uint32_t *fclk, uint32_t *mclk);

/**
 *  @brief Get core clock in MHz.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    cclk - Input buffer to return core clock
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_cclk_limit(amdsmi_processor_handle processor_handle,
                                          uint32_t *cclk);

/**
 *  @brief Get current active frequency limit of the socket.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    freq - Input buffer to return frequency value in MHz
 *
 *  @param[in,out]    src_type - Input buffer to return frequency source name
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_current_active_freq_limit(amdsmi_processor_handle processor_handle,
                                                                uint16_t *freq, char **src_type);

/**
 *  @brief Get socket frequency range.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]    fmax - Input buffer to return maximum frequency
 *
 *  @param[in,out]    fmin - Input buffer to return minimum frequency
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_freq_range(amdsmi_processor_handle processor_handle,
                                                 uint16_t *fmax, uint16_t *fmin);

/**
 *  @brief Get socket frequency limit of the core.
 *
 *  @ingroup tagHSMPSystemStats
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu core which to query
 *
 *  @param[in,out]    freq - Input buffer to return frequency.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_core_current_freq_limit(amdsmi_processor_handle processor_handle,
                                                       uint32_t *freq);

/** @} End tagHSMPSystemStats */

/*****************************************************************************/
/** @defgroup tagPerfBoostControl Performance (Boost limit) Control
 *  @{
 */

/**
 *  @brief Get the core boost limit.
 *
 *  @ingroup tagPerfBoostControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]     processor_handle Cpu core which to query
 *
 *  @param[in,out] pboostlimit - Input buffer to fill the boostlimit value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_core_boostlimit(amdsmi_processor_handle processor_handle,
                                               uint32_t *pboostlimit);

/**
 *  @brief Get the socket c0 residency.
 *
 *  @ingroup tagPerfBoostControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]     processor_handle Cpu socket which to query
 *
 *  @param[in,out] pc0_residency - Input buffer to fill the c0 residency value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_c0_residency(amdsmi_processor_handle processor_handle,
                                                   uint32_t *pc0_residency);

/**
 *  @brief Set the core boostlimit value.
 *
 *  @ingroup tagPerfBoostControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in] processor_handle Cpu core which to query
 *
 *  @param[in] boostlimit - boostlimit value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_core_boostlimit(amdsmi_processor_handle processor_handle,
                                               uint32_t boostlimit);

/**
 *  @brief Set the socket boostlimit value.
 *
 *  @ingroup tagPerfBoostControl
 *
 *  @platform{cpu_bm}
 *
 *  @param[in] processor_handle Cpu socket which to query
 *
 *  @param[in] boostlimit - boostlimit value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_socket_boostlimit(amdsmi_processor_handle processor_handle,
                                                 uint32_t boostlimit);

/** @} End tagPerfBoostControl */

/*****************************************************************************/
/** @defgroup tagDDRBandwidthMonitor DDR bandwidth monitor
 *  @{
 */

/**
 *  @brief Get the DDR bandwidth data.
 *
 *  @ingroup tagDDRBandwidthMonitor
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]     processor_handle Cpu socket which to query
 *
 *  @param[in,out] ddr_bw - Input buffer to fill ddr bandwidth data
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_ddr_bw(amdsmi_processor_handle processor_handle,
                                      amdsmi_ddr_bw_metrics_t *ddr_bw);

/** @} End tagDDRBandwidthMonitor */

/*****************************************************************************/
/** @defgroup  tagTempQuery Temperature Query
 *  @{
 */

/**
 *  @brief Get socket temperature.
 *
 *  @ingroup tagTempQuery
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]     processor_handle Cpu socket which to query
 *
 *  @param[in,out] ptmon - Input buffer to fill temperature value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_temperature(amdsmi_processor_handle processor_handle,
                                                  uint32_t *ptmon);

/** @} End tagTempQuery */

/*****************************************************************************/
/** @defgroup  tagDimmStatistics Dimm statistics
 *  @{
 */

/**
 *  @brief Get DIMM temperature range and refresh rate.
 *
 *  @ingroup tagDimmStatistics
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]     processor_handle Cpu socket which to query
 *
 *  @param[in]     dimm_addr - DIMM address
 *
 *  @param[in,out] rate - Input buffer to fill temperature range and refresh rate value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(amdsmi_processor_handle processor_handle,
                                                                uint8_t dimm_addr,
                                                                amdsmi_temp_range_refresh_rate_t *rate);

/**
 *  @brief Get DIMM power consumption.
 *
 *  @ingroup tagDimmStatistics
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *  @param[in]      dimm_addr - DIMM address
 *  @param[in,out]  dimm_pow - Input buffer to fill power consumption value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_dimm_power_consumption(amdsmi_processor_handle processor_handle,
                                                      uint8_t dimm_addr,
                                                      amdsmi_dimm_power_t *dimm_pow);

/**
 *  @brief Get DIMM thermal sensor value.
 *
 *  @ingroup tagDimmStatistics
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]     dimm_addr - DIMM address
 *
 *  @param[in,out] dimm_temp - Input buffer to fill temperature value
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_dimm_thermal_sensor(amdsmi_processor_handle processor_handle,
                                                   uint8_t dimm_addr,
                                                   amdsmi_dimm_thermal_t *dimm_temp);

/** @} End tagDimmStatistics */

/*****************************************************************************/
/** @defgroup tagXGMIBandwidthCont xGMI bandwidth control
 *  @{
 */

/**
 *  @brief Set xgmi width.
 *
 *  @ingroup tagXGMIBandwidthCont
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]  processor_handle Cpu socket which to query
 *
 *  @param[in]  min - Minimum xgmi width to be set
 *
 *  @param[in]  max - maximum xgmi width to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_xgmi_width(amdsmi_processor_handle processor_handle,
                                          uint8_t min, uint8_t max);

/** @} End tagXGMIBandwidthCont */

/*****************************************************************************/
/** @defgroup tagGMI3WidthCont GMI3 width control
 *  @{
 */

/**
 *  @brief Set gmi3 link width range.
 *
 *  @ingroup tagGMI3WidthCont
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]  processor_handle Cpu socket which to query
 *
 *  @param[in]  min_link_width - minimum link width to be set.
 *
 *  @param[in]  max_link_width - maximum link width to be set.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_gmi3_link_width_range(amdsmi_processor_handle processor_handle,
                                                     uint8_t min_link_width, uint8_t max_link_width);

/** @} End tagGMI3WidthCont */

/*****************************************************************************/
/** @defgroup tagPstateSelect Pstate selection
 *  @{
 */

/**
 *  @brief Enable APB.
 *
 *  @ingroup tagPstateSelect
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_cpu_apb_enable(amdsmi_processor_handle processor_handle);

/**
 *  @brief Disable APB.
 *
 *  @ingroup tagPstateSelect
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]  processor_handle Cpu socket which to query
 *
 *  @param[in]  pstate - pstate value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_cpu_apb_disable(amdsmi_processor_handle processor_handle, uint8_t pstate);

/**
 *  @brief Set NBIO lclk dpm level value.
 *
 *  @ingroup tagPstateSelect
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]  processor_handle Cpu socket which to query
 *
 *  @param[in]  nbio_id - nbio index
 *
 *  @param[in]  min - minimum value to be set
 *
 *  @param[in]  max - maximum value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_socket_lclk_dpm_level(amdsmi_processor_handle processor_handle,
                                                     uint8_t nbio_id, uint8_t min, uint8_t max);

/**
 *  @brief Get NBIO LCLK dpm level.
 *
 *  @ingroup tagPstateSelect
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]      nbio_id - nbio index
 *
 *  @param[in,out]  nbio - Input buffer to fill lclk dpm level
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_lclk_dpm_level(amdsmi_processor_handle processor_handle,
                                                     uint8_t nbio_id, amdsmi_dpm_level_t *nbio);

/**
 *  @brief Set pcie link rate.
 *
 *  @ingroup tagPstateSelect
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]      rate_ctrl - rate control value to be set.
 *
 *  @param[in,out]  prev_mode - Input buffer to fill previous rate control value.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_pcie_link_rate(amdsmi_processor_handle processor_handle,
                                              uint8_t rate_ctrl, uint8_t *prev_mode);

/**
 *  @brief Set df pstate range.
 *
 *  @ingroup tagPstateSelect
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]  processor_handle Cpu socket which to query
 *
 *  @param[in]  max_pstate - maximum pstate value to be set
 *
 *  @param[in]  min_pstate - minimum pstate value to be set
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_set_cpu_df_pstate_range(amdsmi_processor_handle processor_handle,
                                               uint8_t max_pstate, uint8_t min_pstate);

/** @} End tagPstateSelect */

/*****************************************************************************/
/** @defgroup tagBandwidthMon Bandwidth monitor
 *  @{
 */

/**
 *  @brief Get current input output bandwidth.
 *
 *  @ingroup tagBandwidthMon
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]      link - link id and bw type to which io bandwidth to be obtained
 *
 *  @param[in,out]  io_bw - Input buffer to fill bandwidth data
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_current_io_bandwidth(amdsmi_processor_handle processor_handle,
                                                    amdsmi_link_id_bw_type_t link, uint32_t *io_bw);

/**
 *  @brief Get current input output bandwidth.
 *
 *  @ingroup tagBandwidthMon
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in]      link - link id and bw type to which xgmi bandwidth to be obtained
 *
 *  @param[in,out]  xgmi_bw - Input buffer to fill bandwidth data
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_current_xgmi_bw(amdsmi_processor_handle processor_handle,
                                               amdsmi_link_id_bw_type_t link, uint32_t *xgmi_bw);

/** @} End tagBandwidthMon */

/*****************************************************************************/
/** @defgroup tagHSMPMetricsTable HSMP Metrics Table
 *  @{
 */

/**
 *  @brief Get HSMP metrics table version
 *
 *  @ingroup tagHSMPMetricsTable
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]  metrics_version input buffer to return the HSMP metrics table version.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_hsmp_metrics_table_version(amdsmi_processor_handle processor_handle,
                                                      uint32_t *metrics_version);

/**
 *  @brief Get HSMP metrics table
 *
 *  @ingroup tagHSMPMetricsTable
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]  metrics_table input buffer to return the HSMP metrics table.
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_hsmp_metrics_table(amdsmi_processor_handle processor_handle,
                                              amdsmi_hsmp_metrics_table_t *metrics_table);

/** @} End tagHSMPMetricsTable */

/*****************************************************************************/
/** @defgroup tagCPUAuxillary Auxillary functions
 *  @{
 */

/**
 *  @brief Get first online core on socket.
 *
 *  @ingroup tagCPUAuxillary
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[in,out]  pcore_ind - Input buffer to fill first online core on socket data
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_first_online_core_on_cpu_socket(amdsmi_processor_handle processor_handle,
                                                       uint32_t *pcore_ind);

/**
 *  @brief Get CPU family.
 *
 *  @ingroup tagCPUAuxillary
 *
 *  @platform{cpu_bm}
 *
 *  @param[in,out]  cpu_family - Input buffer to return the cpu family
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_family(uint32_t *cpu_family);

/**
 *  @brief Get CPU model.
 *
 *  @ingroup tagCPUAuxillary
 *
 *  @platform{cpu_bm}
 *
 *  @param[in,out]  cpu_model - Input buffer to return the cpu model
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_model(uint32_t *cpu_model);

 /**
 *  @brief Retrieve the CPU processor model name based on the processor index.
 *
 *  @ingroup tagCPUAuxillary
 *
 *  @platform{cpu_bm}
 *
 *  @details
 *  This function obtains the CPU model name associated with the specified processor index
 *  from the list of available processor handles. Before invoking this function, ensure that
 *  the list of processor handles is properly initialized and that the processor type is specified.
 *  This function is to be utilized for RDC and is not part of ESMI library.
 *
 *  @param[in]      processor_handle Cpu socket which to query
 *
 *  @param[out] cpu_info
 *      A pointer to an `amdsmi_cpu_info_t` structure that will be populated with the
 *      CPU processor model information upon successful execution of the function.
 *
 *  @return
 *      ::amdsmi_status_t indicating the result of the operation.
 *      - ::AMDSMI_STATUS_SUCCESS on successful retrieval of the model name.
 *      - A non-zero error code if the operation fails.
 */
amdsmi_status_t amdsmi_get_cpu_model_name(amdsmi_processor_handle processor_handle, amdsmi_cpu_info_t *cpu_info);

/**
 *  @brief Get a description of provided AMDSMI error status for esmi errors.
 *
 *  @ingroup tagCPUAuxillary
 *
 *  @platform{cpu_bm}
 *
 *  @details Set the provided pointer to a const char *, @p status_string, to
 *  a string containing a description of the provided error code @p status.
 *
 *  @param[in]    status - The error status for which a description is desired.
 *
 *  @param[in,out]    status_string - A pointer to a const char * which will be made
 *  to point to a description of the provided error code
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_esmi_err_msg(amdsmi_status_t status, const char **status_string);

/**
 *  @brief Get cpu cores per socket from sys filesystem.
 *
 *  @ingroup tagCPUAuxillary
 *
 *  @platform{cpu_bm}
 *
 *  @param[in]  sock_count - cpu socket count
 *  @param[in,out]  soc_info - Input buffer to return the cpu cores per socket
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_cores_per_socket(uint32_t sock_count, amdsmi_sock_info_t *soc_info);

/**
 *  @brief Get CPU socket count from sys filesystem.
 *
 *  @ingroup tagCPUAuxillary
 *
 *  @platform{cpu_bm}
 *
 *  @param[in,out]  sock_count - Input buffer to return the cpu socket count
 *
 *  @return ::amdsmi_status_t | ::AMDSMI_STATUS_SUCCESS on success, non-zero on fail
 */
amdsmi_status_t amdsmi_get_cpu_socket_count(uint32_t *sock_count);

/** @} End tagCPUAuxillary */

#endif

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __AMDSMI_H__

