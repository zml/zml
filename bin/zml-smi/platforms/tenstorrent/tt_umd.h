#ifndef ZML_SMI_TT_UMD_H
#define ZML_SMI_TT_UMD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TTUMD_ABSENT INT64_MIN

typedef struct ttumd_ctx ttumd_ctx;

ttumd_ctx *ttumd_open(void);
void ttumd_close(ttumd_ctx *ctx);
uint32_t ttumd_chip_count(ttumd_ctx *ctx);

#define TTUMD_STR_BUF_LEN 64

int ttumd_is_remote(ttumd_ctx *ctx, uint32_t index);
int ttumd_device_name(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n);
int64_t ttumd_asic_location(ttumd_ctx *ctx, uint32_t index);
uint64_t ttumd_board_id(ttumd_ctx *ctx, uint32_t index);
uint64_t ttumd_asic_id(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_mem_total_bytes(ttumd_ctx *ctx, uint32_t index);

int ttumd_fw_bundle(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n);
int ttumd_eth_fw(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n);
int ttumd_cm_fw(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n);
int ttumd_dm_app(ttumd_ctx *ctx, uint32_t index, char *buf, size_t n);

int64_t ttumd_temperature_mc(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_temperature_limit_mc(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_board_temperature_mc(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_dram_temperature_mc(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_power_mw(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_power_limit_mw(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_voltage_mv(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_current_ma(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_aiclk_mhz(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_arcclk_mhz(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_axiclk_mhz(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_dram_mhz(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_heartbeat(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_therm_trip_count(ttumd_ctx *ctx, uint32_t index);
int64_t ttumd_fan_rpm(ttumd_ctx *ctx, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif  // ZML_SMI_TT_UMD_H
