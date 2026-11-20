const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const Tensor = zml.Tensor;

pub const CausalStatsOptions = struct {
    minimum_scale: f32,
    stabilize_with_global: bool,
    scale_factor_exponent: f32 = 10.0,
};

pub const LocationScales = struct {
    locations: Tensor,
    scales: Tensor,
};

/// Causal running mean and std along the last axis (Welford form, in float64).
///
/// Returns means and scales Tensors, both shaped like `data`; position `t` uses
/// the observed values at positions `0..t` only.
/// Unobserved positions get zero weight.
/// The variance uses Bessel's correction and the standard deviation is clamped
/// to `minimum_scale` after the square root.
pub fn computeCausalStatistics(
    data: Tensor,
    mask: Tensor,
    opts: CausalStatsOptions,
    axis: anytype,
) LocationScales {
    const dtype: zml.DataType = .f64;
    const one: Tensor = .scalar(1.0, dtype);

    const w = mask.convert(dtype);
    const x = data.convert(dtype);
    const weighted: Tensor = .mul(w, x);
    const cum_w = w.cumulativeSum(axis);
    const cum_x = weighted.cumulativeSum(axis);
    const denominator = cum_w.clampMin(one);
    const means: Tensor = .div(cum_x, denominator);
    const shifted = means.pad(0, .{ .time = Tensor.Pad{ .low = 1, .high = -1 } });
    const increment: Tensor = .mul(.mul(.sub(x, shifted), .sub(x, means)), w);
    const m2 = increment.cumulativeSum(axis);
    const variance: Tensor = .div(m2, Tensor.sub(denominator, one).clampMin(one));
    const scales = variance.sqrt().clampMin(.scalar(opts.minimum_scale, dtype));

    if (opts.stabilize_with_global) {
        stdx.debug.assert(false, "stabilize_with_global is not implemented", .{});
    }

    return .{
        .locations = means.convert(data.dtype()),
        .scales = scales.convert(data.dtype()),
    };
}

/// Conceptually split a Tensor in equal patches and repeat the last value in in every patch.
pub fn patchwiseBroadcastLast(x: Tensor, axis: anytype, patch_size: u31) Tensor {
    stdx.debug.assert(
        @rem(x.dim(axis), patch_size) == 0,
        "length {} is not a multiple of patch_size {}",
        .{ x.dim(axis), patch_size },
    );
    const last = x.slice(axis, .{ .start = patch_size - 1, .step = patch_size });
    return last.stutter1d(last.axis(axis), patch_size);
}

/// Element-wise inverse hyperbolic sine (StableHLO has no asinh op).
///
/// Uses asinh(x) = sign(x) * log1p(|x| + x² / (1 + sqrt(1 + x²))), which avoids the
/// cancellation of log(x + sqrt(x² + 1)) for negative x and stays accurate near 0.
/// For |x| > 1/sqrt(eps), x² would overflow, so use asinh(x) ≈ sign(x) * (log|x| + ln 2).
pub fn asinh(x: Tensor) Tensor {
    const dtype = x.dtype();
    const a = x.abs();
    const a2: Tensor = .mul(a, a);
    const small: Tensor = Tensor.add(a, .div(a2, a2.addConstant(1).sqrt().addConstant(1))).log1p();
    const large = a.log().addConstant(std.math.ln2);
    const threshold: f64 = 1.0 / @sqrt(@as(f64, std.math.floatEps(f32)));
    const is_large = a.cmp(.GT, .scalar(threshold, dtype));
    return .mul(x.sign(), is_large.select(large, small));
}

/// Scaling function we precomputed scales and locations.
///
/// Computes `asinh((data - loc) / scale)` on observed positions, unobserved positions become 0.
pub fn scaleByAffineAsinh(
    data: Tensor,
    mask: Tensor,
    stats: LocationScales,
    use_asinh: bool,
) Tensor {
    const dtype = data.dtype();
    stdx.debug.assert(dtype == .f32, "Only implemented for f32", .{});
    const eps: zml.DataType.Value = .init(.f32, std.math.floatEps(f32));

    var scaled: Tensor = .div(.sub(data, stats.locations), stats.scales.clampMin(.constant(eps)));
    scaled = mask.select(scaled, .scalar(0, dtype));
    if (use_asinh) {
        // Unobserved positions are already 0 and asinh(0) = 0, so no extra masking is needed.
        scaled = asinh(scaled);
    }
    return scaled;
}
