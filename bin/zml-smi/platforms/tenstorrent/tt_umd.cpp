#include "tt_umd.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "umd/device/arc/arc_telemetry_reader.hpp"
#include "umd/device/cluster.hpp"
#include "umd/device/cluster_descriptor.hpp"
#include "umd/device/firmware/firmware_info_provider.hpp"
#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/types/arch.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"
#include "umd/device/types/telemetry.hpp"

namespace {

struct ChipEntry {
    int chip_id = 0;
    bool is_remote = false;
    tt::umd::TTDevice *tt_device = nullptr;
    tt::umd::FirmwareInfoProvider *fw = nullptr;
    std::unique_ptr<tt::umd::FirmwareInfoProvider> owned_fw;
};

int64_t opt_u32(const std::optional<uint32_t> &o) {
    return o.has_value() ? static_cast<int64_t>(*o) : TTUMD_ABSENT;
}

int64_t opt_double_milli(const std::optional<double> &o) {
    return o.has_value() ? static_cast<int64_t>(*o * 1000.0) : TTUMD_ABSENT;
}

const char *arch_display_name(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0:
            return "Wormhole";
        case tt::ARCH::BLACKHOLE:
            return "Blackhole";
        case tt::ARCH::QUASAR:
            return "Quasar";
        default:
            return "Tenstorrent";
    }
}

std::string fmt_semver4(const tt::umd::SemVer &v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu.%llu.%llu.%llu", (unsigned long long)v.major, (unsigned long long)v.minor,
                  (unsigned long long)v.patch, (unsigned long long)v.pre_release);
    return buf;
}

std::string fmt_semver3(const tt::umd::SemVer &v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu.%llu.%llu", (unsigned long long)v.major, (unsigned long long)v.minor,
                  (unsigned long long)v.patch);
    return buf;
}

std::string read_m3_fw_version(tt::umd::TTDevice *dev, uint8_t tag) {
    tt::umd::ArcTelemetryReader *tel = dev->get_arc_telemetry_reader();
    if (tel == nullptr || !tel->is_entry_available(tag)) {
        return {};
    }
    
    const uint32_t v = tel->read_entry(tag);
    
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u", (unsigned)((v >> 24) & 0xFF), (unsigned)((v >> 16) & 0xFF),
                  (unsigned)((v >> 8) & 0xFF), (unsigned)(v & 0xFF));
    
    return buf;
}

}

struct ttumd_ctx {
    std::mutex mu; // technically useless but just in case
    std::unique_ptr<tt::umd::Cluster> cluster;
    std::vector<ChipEntry> chips;
};

ttumd_ctx *ttumd_open(void) {
    try {
        auto ctx = std::make_unique<ttumd_ctx>();
        ctx->cluster = std::make_unique<tt::umd::Cluster>(tt::umd::ClusterOptions{});

        tt::umd::ClusterDescriptor *desc = ctx->cluster->get_cluster_description();
        if (desc == nullptr) {
            return nullptr;
        }

        std::vector<tt::ChipId> ordered =
            desc->get_chips_local_first(desc->get_all_chips());

        for (tt::ChipId id : ordered) {
            ChipEntry entry;
            
            entry.chip_id = id;
            entry.is_remote = desc->is_chip_remote(id);
            entry.tt_device = ctx->cluster->get_tt_device(id);
            
            if (entry.tt_device == nullptr) {
                continue;
            }
            
            entry.fw = entry.tt_device->get_firmware_info_provider();
            if (entry.fw == nullptr) {
                entry.owned_fw =
                    tt::umd::FirmwareInfoProvider::create_firmware_info_provider(entry.tt_device);
                entry.fw = entry.owned_fw.get();
            }
            
            if (entry.fw == nullptr) {
                continue;
            }
            
            ctx->chips.push_back(std::move(entry));
        }

        if (ctx->chips.empty()) {
            return nullptr;
        }
	
        return ctx.release();
    } catch (...) {
        return nullptr;
    }
}

void ttumd_close(ttumd_ctx *ctx) {
    delete ctx;  // ~Cluster releases the device
}

uint32_t ttumd_chip_count(ttumd_ctx *ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(ctx->mu);
    return static_cast<uint32_t>(ctx->chips.size());
}

namespace {

// Reads one scalar field under the lock. Bad index or a thrown reader yields `absent`
template <typename T, typename F>
T read_scalar(ttumd_ctx *ctx, uint32_t index, T absent, F &&fn) {
    if (ctx == nullptr) {
        return absent;
    }
    
    std::lock_guard<std::mutex> lock(ctx->mu);
    if (index >= ctx->chips.size()) {
        return absent;
    }
    
    try {
        return static_cast<T>(fn(ctx, ctx->chips[index]));
    } catch (...) {
        return absent;
    }
}

// Copies a value-returning callback's string into `buf` under the lock (mirrors
// read_scalar). Returns 0 on a non-empty string, -1 on empty / throw / bad index.
template <typename F>
int read_str(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n, F &&fn) {
    if (ctx == nullptr || buf == nullptr || n == 0) {
        return -1;
    }
    
    std::lock_guard<std::mutex> lock(ctx->mu);
    if (index >= ctx->chips.size()) {
        return -1;
    }
    
    try {
        const std::string s = fn(ctx, ctx->chips[index]);
        if (s.empty()) {
            return -1;
        }
        
        std::snprintf(buf, n, "%s", s.c_str());
        
        return 0;
    } catch (...) {
        return -1;
    }
}

}

int ttumd_is_remote(ttumd_ctx *ctx, uint32_t index) {
    return read_scalar<int>(ctx, index, -1, [](ttumd_ctx *, ChipEntry &e) { return e.is_remote ? 1 : 0; });
}
int ttumd_device_name(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n) {
    return read_str(ctx, index, buf, n, [](ttumd_ctx *c, ChipEntry &e) -> std::string {
        std::string board = tt::board_type_to_string(e.tt_device->get_board_type());
        
        if (board.empty()) {
            board = "device";
        }
        
        std::string s = std::string(arch_display_name(e.tt_device->get_arch())) + " " + board;
        if (c->chips.size() > 1) {
            s += " - ASIC " + std::to_string(static_cast<int>(e.tt_device->get_asic_location()));
        }
        
        return s;
    });
}
int64_t ttumd_asic_location(ttumd_ctx *ctx, uint32_t index) {
    return read_scalar<int64_t>(ctx, index, TTUMD_ABSENT, [](ttumd_ctx *, ChipEntry &e) {
        return static_cast<int64_t>(e.tt_device->get_asic_location());
    });
}
uint64_t ttumd_board_id(ttumd_ctx *ctx, uint32_t index) {
    return read_scalar<uint64_t>(
        ctx, index, 0ull, [](ttumd_ctx *, ChipEntry &e) { return static_cast<uint64_t>(e.tt_device->get_board_id()); });
}
uint64_t ttumd_asic_id(ttumd_ctx *ctx, uint32_t index) {
    return read_scalar<uint64_t>(ctx, index, 0ull, [](ttumd_ctx *c, ChipEntry &e) -> uint64_t {
        const auto &uids = c->cluster->get_cluster_description()->get_chip_unique_ids();
        auto it = uids.find(e.chip_id);
        
        return it != uids.end() ? static_cast<uint64_t>(it->second) : 0ull;
    });
}
// tt-umd has no capacity API (yet?), so update manually as needed!
int64_t ttumd_mem_total_bytes(ttumd_ctx *ctx, uint32_t index) {
    return read_scalar<int64_t>(ctx, index, TTUMD_ABSENT, [](ttumd_ctx *, ChipEntry &e) -> int64_t {
        constexpr int64_t GiB = 1024LL * 1024 * 1024;
        switch (e.tt_device->get_board_type()) {
            case tt::BoardType::N150:
                return 12 * GiB;
            case tt::BoardType::N300:
                return 12 * GiB;  // per ASIC
            case tt::BoardType::P100:
                return 28 * GiB;
            case tt::BoardType::P150:
                return 32 * GiB;
            default:
                return TTUMD_ABSENT;
        }
    });
}
int ttumd_fw_bundle(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n) {
    return read_str(ctx, index, buf, n,
                    [](ttumd_ctx *, ChipEntry &e) { return fmt_semver4(e.fw->get_firmware_version()); });
}
int ttumd_eth_fw(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n) {
    return read_str(ctx, index, buf, n, [](ttumd_ctx *, ChipEntry &e) -> std::string {
        std::optional<tt::umd::SemVer> v = e.fw->get_eth_fw_version_semver();
        return v.has_value() ? fmt_semver3(*v) : std::string{};
    });
}
int ttumd_cm_fw(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n) {
    return read_str(ctx, index, buf, n, [](ttumd_ctx *, ChipEntry &e) {
        return read_m3_fw_version(e.tt_device, static_cast<uint8_t>(tt::umd::TelemetryTag::CM_FW_VERSION));
    });
}
int ttumd_dm_app(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n) {
    return read_str(ctx, index, buf, n, [](ttumd_ctx *, ChipEntry &e) {
        return read_m3_fw_version(e.tt_device, static_cast<uint8_t>(tt::umd::TelemetryTag::DM_APP_FW_VERSION));
    });
}

namespace {

template <typename F>
int64_t read_metric(ttumd_ctx *ctx, uint32_t index, F &&fn) {
    if (ctx == nullptr) {
        return TTUMD_ABSENT;
    }
    
    std::lock_guard<std::mutex> lock(ctx->mu);
    if (index >= ctx->chips.size()) {
        return TTUMD_ABSENT;
    }
    
    tt::umd::FirmwareInfoProvider *fw = ctx->chips[index].fw;
    if (fw == nullptr) {
        return TTUMD_ABSENT;
    }
    
    try {
        return static_cast<int64_t>(fn(fw));
    } catch (...) {
        return TTUMD_ABSENT;
    }
}

}

int64_t ttumd_temperature_mc(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return fw->get_asic_temperature() * 1000.0; });
}
int64_t ttumd_temperature_limit_mc(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_double_milli(fw->get_thm_limit_shutdown()); });
}
int64_t ttumd_board_temperature_mc(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_double_milli(fw->get_board_temperature()); });
}
int64_t ttumd_dram_temperature_mc(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_double_milli(fw->get_current_max_dram_temperature()); });
}
int64_t ttumd_power_mw(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) {
        auto w = fw->get_tdp();
        return w.has_value() ? static_cast<int64_t>(*w) * 1000 : TTUMD_ABSENT;
    });
}
int64_t ttumd_power_limit_mw(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) {
        auto w = fw->get_board_power_limit();
        return w.has_value() ? static_cast<int64_t>(*w) * 1000 : TTUMD_ABSENT;
    });
}
int64_t ttumd_voltage_mv(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_u32(fw->get_vcore()); });
}
int64_t ttumd_current_ma(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) {
        auto a = fw->get_tdc();
        return a.has_value() ? static_cast<int64_t>(*a) * 1000 : TTUMD_ABSENT;
    });
}
int64_t ttumd_aiclk_mhz(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_u32(fw->get_aiclk()); });
}
int64_t ttumd_arcclk_mhz(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_u32(fw->get_arcclk()); });
}
int64_t ttumd_axiclk_mhz(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_u32(fw->get_axiclk()); });
}
int64_t ttumd_dram_mhz(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) {
        auto s = fw->get_dram_speed();
        return s.has_value() ? static_cast<int64_t>(*s) : TTUMD_ABSENT;
    });
}
int64_t ttumd_heartbeat(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return static_cast<int64_t>(fw->get_heartbeat()); });
}
int64_t ttumd_therm_trip_count(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_u32(fw->get_therm_trip_count()); });
}
int64_t ttumd_fan_rpm(ttumd_ctx *ctx, uint32_t index) {
    return read_metric(ctx, index, [](auto *fw) { return opt_u32(fw->get_fan_rpm()); });
}
