const zml = @import("zml");

const debug = @import("./debug.zig");

const assertTags = debug.assertTags;
const assertedTags = debug.assertedTags;
const Tensor = zml.Tensor;

pub const Context = struct {
    pub const Masks = struct {
        observed: Tensor, // PY: obs_per_patch
        attended: Tensor, // PY: key_valid
    };

    pub fn forward(observed_patches: Tensor, attend_missing_patches: bool) Masks {
        assertTags(observed_patches, .{ .row, .token });

        // Leading all-unobserved patches are left-padded and are never valid keys.
        const key_valid: Tensor = if (attend_missing_patches)
            // A patch is a valid key if the row's series has started at or before it.
            .logical(observed_patches, .OR, observed_patches.cumulativeMax(.token))
        else
            // A patch is a valid key if it is observed
            observed_patches;
        return .{
            .observed = assertedTags(observed_patches, .{ .row, .token }),
            .attended = assertedTags(key_valid, .{ .row, .token }),
        };
    }
};

pub const Window = struct {
    /// Time mask `(R, L+h, L+h)` for one block without covariates.
    ///
    /// The encoder is causal; every decoder token reads the whole valid context and its
    /// own causal decoder chain; the context never reads the horizon.
    pub fn forward(attended: Tensor, horizon_patches: u32) Tensor {
        assertTags(attended, .{ .row, .token });
        const rows = attended.dim(.row);
        const n_past = attended.dim(.token);
        const n_fut = horizon_patches;

        const key_valid_mask = keyValidityAsMask(attended);
        const enc_enc = assertedTags(encoderCausalMask(key_valid_mask), .{ .row, .token_q, .token_k });
        const enc_dec: Tensor = .zeroes(.init(.{ .row = rows, .token_q = n_past, .token_k = n_fut }, .bool));
        const dec_enc = attended.broad(attended.shape().insert(.token, .{ .token_q = n_fut }))
            .renameTag(.token, .token_k);
        var dec_dec = zml.nn.causalAttnMask(.{ .token_q = n_fut, .token_k = n_fut }, .bool, null);
        dec_dec = dec_dec.broad(dec_dec.shape().insert(.token_q, .{ .row = rows }));

        const top: Tensor = .concatenate(&.{ enc_enc, enc_dec }, .token_k);
        const bottom: Tensor = .concatenate(&.{ dec_enc, dec_dec }, .token_k);
        const out: Tensor = .concatenate(&.{ top, bottom }, .token_q);
        return assertedTags(out, .{ .row, .token_q, .token_k });
    }

    /// Causal encoder-encoder mask `(R, L, L)` over valid keys, with the diagonal always open.
    fn encoderCausalMask(key_valid: Tensor) Tensor {
        assertTags(key_valid, .{ .row, .token_q, .token_k });
        const shape = key_valid.shape();

        const l = key_valid.dim(.token_k);
        const causal = zml.nn.causalAttnMask(.{ .token_q = l, .token_k = l }, .bool, null)
            .broad(shape);
        const eye: Tensor = .cmp(.iota(shape, .token_q), .EQ, .iota(shape, .token_k));
        const out = causal.logical(.AND, key_valid).logical(.OR, eye);

        assertTags(out, .{ .row, .token_q, .token_k });
        return out;
    }

    fn keyValidityAsMask(key_valid: Tensor) Tensor {
        assertTags(key_valid, .{ .row, .token });
        const shape = key_valid.shape();
        const n = shape.dim(.token);
        const out = key_valid.broad(shape.insert(.token, .{ .token_q = n }))
            .renameTag(.token, .token_k);
        return assertedTags(out, .{ .row, .token_q, .token_k });
    }
};

pub const Variate = struct {
    pub fn forward(attended: Tensor, group: Tensor, horizon: u32) Tensor {
        assertTags(attended, .{ .row, .token });
        const out = buildVariateMask(group, attended, horizon);
        return assertedTags(out, .{ .token, .row_q, .row_k });
    }

    /// `(R, R)` forbidden query-to-key row pairs.
    ///
    /// - A covariate row may not read a target row (covariates are exogenous).
    /// - A known-future (bidirectional) row may not read a causal row. Without this, a
    ///   covariate's view of the future could relay a target's own future back into it.
    fn variateFirewall(target_rows: Tensor, bidirectional_rows: Tensor) Tensor {
        assertTags(target_rows, .{.row});
        assertTags(bidirectional_rows, .{.row});
        const r = target_rows.dim(.row);
        const shape: zml.Shape = .init(.{ .row_q = r, .row_k = r }, .bool);

        const target_q = target_rows.renameTag(.row, .row_q).broad(shape);
        const target_k = target_rows.renameTag(.row, .row_k).broad(shape);
        const bidir_q = bidirectional_rows.renameTag(.row, .row_q).broad(shape);
        const bidir_k = bidirectional_rows.renameTag(.row, .row_k).broad(shape);

        const covariate_reads_target = target_q.not().logical(.AND, target_k);
        const bidir_reads_causal = bidir_q.logical(.AND, bidir_k.not());
        const out = covariate_reads_target.logical(.OR, bidir_reads_causal);
        return assertedTags(out, .{ .row_q, .row_k });
    }

    /// `(N_enc + N_dec, R, R)` variate mask, before firewalls and diagonal.
    ///
    /// A row reads another row of the same series; on encoder slots the key row must also
    /// be a valid key. Decoder slots are always valid keys.
    fn variateGroupMask(group_ids: Tensor, enc_key_valid: Tensor, n_dec: u32) Tensor {
        assertTags(group_ids, .{.row});
        assertTags(enc_key_valid, .{ .row, .token });
        const r = enc_key_valid.dim(.row);
        const n_enc = enc_key_valid.dim(.token);
        const shape: zml.Shape = .init(.{ .token = n_enc + n_dec, .row_q = r, .row_k = r }, .bool);

        const group_shape = shape.withDtype(group_ids.dtype());
        const row_eq: Tensor = .cmp(
            group_ids.renameTag(.row, .row_q).broad(group_shape),
            .EQ,
            group_ids.renameTag(.row, .row_k).broad(group_shape),
        );

        const dec_valid = Tensor.scalar(true, .bool).broad(.init(.{ .row = r, .token = n_dec }, .bool));
        const slot_valid: Tensor = .concatenate(&.{ enc_key_valid, dec_valid }, .token);
        const key_valid = slot_valid.renameTag(.row, .row_k).broad(shape);

        const out = row_eq.logical(.AND, key_valid);
        return assertedTags(out, .{ .token, .row_q, .row_k });
    }

    /// `(N_enc + N_dec, R, R)` variate mask without firewalls; the diagonal is always open.
    fn buildVariateMask(group_ids: Tensor, enc_key_valid: Tensor, horizon: u32) Tensor {
        const mask = variateGroupMask(group_ids, enc_key_valid, horizon);
        const idx_shape = mask.shape().withDtype(.i32);
        const eye: Tensor = .cmp(.iota(idx_shape, .row_q), .EQ, .iota(idx_shape, .row_k));
        const out = mask.logical(.OR, eye);
        return assertedTags(out, .{ .token, .row_q, .row_k });
    }

    pub const TwoStreamVariateMaskInputs = struct {
        /// `{.row}` series index of each row.
        group_ids: Tensor,
        /// `{.row, .token}` encoder key validity.
        enc_key_valid: Tensor,
        /// Number of decoder slots.
        n_dec: u32,
        /// `{.row}` bool, rows that are targets.
        target_rows: Tensor,
        /// `{.row}` bool, rows that are known-future (bidirectional).
        bidirectional_rows: Tensor,
    };

    /// `(N_enc + N_dec, R, R)` variate mask.
    ///
    /// A row reads another row of the same series; on encoder slots the key row must also
    /// be a valid key. The firewalls apply on every slot; the diagonal is always open.
    fn buildTwoStreamVariateMask(inputs: TwoStreamVariateMaskInputs) Tensor {
        var mask = variateGroupMask(inputs.group_ids, inputs.enc_key_valid, inputs.n_dec);
        const forbid = variateFirewall(inputs.target_rows, inputs.bidirectional_rows);
        mask = mask.logical(.AND, forbid.not().broad(mask.shape()));

        const idx_shape = mask.shape().withDtype(.i32);
        const eye: Tensor = .cmp(.iota(idx_shape, .row_q), .EQ, .iota(idx_shape, .row_k));
        const out = mask.logical(.OR, eye);
        return assertedTags(out, .{ .token, .row_q, .row_k });
    }
};

pub const Position = struct {
    pub fn forward(observed: Tensor, group: Tensor, horizon_patches: u32) Tensor {
        assertTags(observed, .{ .row, .token });
        const n_past = observed.dim(.token);
        const n_fut = horizon_patches;

        const first_obs = rowFirstObservedPatch(observed, group);
        const position = buildTimePosition(@intCast(n_past + n_fut), first_obs);
        return assertedTags(position, .{ .row, .token });
    }

    fn buildTimePosition(window_size: u63, offsets: Tensor) Tensor {
        assertTags(offsets, .{.row});
        const dtype = offsets.dtype();
        const shape = offsets.shape().append(.{ .token = window_size });

        const pos = Tensor.arange(.{ .end = window_size }, dtype).withTags(.{.token});
        const out: Tensor = .sub(pos.broad(shape), offsets.broad(shape));
        return assertedTags(out, .{ .row, .token });
    }

    /// `(R,)` first patch observed by any row of the same series (0 if none).
    fn rowFirstObservedPatch(obs_per_patch: Tensor, group_ids: Tensor) Tensor {
        // TODO: this could be computed based on the user offset of times series in the tensor.
        assertTags(obs_per_patch, .{ .row, .token });
        assertTags(group_ids, .{.row});
        const r = obs_per_patch.dim(.row);
        const n = obs_per_patch.dim(.token);

        // `{.row, .peer}` whether `.peer` belongs to the same series as `.row`.
        const group_shape: zml.Shape = .init(.{ .row = r, .peer = r }, group_ids.dtype());
        const same_group: Tensor = .cmp(
            group_ids.broad(group_shape),
            .EQ,
            group_ids.renameTag(.row, .peer).broad(group_shape),
        );

        // `{.row, .token}` patches observed by any peer of the row's series.
        const shape: zml.Shape = .init(.{ .row = r, .peer = r, .token = n }, .bool);
        const peer_obs = obs_per_patch.renameTag(.row, .peer).broad(shape);
        const group_obs = same_group.broad(shape).logical(.AND, peer_obs).any(.peer).squeeze(.peer);

        // argMax returns the first maximum, hence the first observed patch (0 if none).
        const first = group_obs.argMax(.token).indices.squeeze(.token);
        return assertedTags(first, .{.row});
    }
};
