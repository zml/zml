/// Tools to load models from https://huggingface.co/karpathy/tinyllamas/
/// Originally made to be run with https://github.com/karpathy/llama2.c
const std = @import("std");

const asynk = @import("async");

const zml = @import("../zml.zig");

const TinyLlamaConfig = extern struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    // llama2.c uses the sign bit of the vocab size to store if we need
    // a second embedding layer.
    vocab: packed struct(u32) {
        size: u31,
        has_lm_head: bool,
    },
    seq_len: u32,
};

/// Open a tinyllama file.
/// For convenience we use the same layer names
/// than the one used by the Llama-3.1 models.
pub fn open(allocator: std.mem.Allocator, model_path: []const u8) !zml.aio.BufferStore {
    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    errdefer res.arena.deinit();
    const arena = res.arena.allocator();

    const file = try asynk.File.open(model_path, .{});
    res.files = try arena.alloc(zml.aio.MemoryMappedFile, 1);
    res.files[0] = try zml.aio.MemoryMappedFile.init(file);

    const c = try file.reader().readStructEndian(TinyLlamaConfig, .little);
    zml.log.info("Read TinyLlamaConfig: {}", .{c});

    const dim = c.dim;
    const hidden_dim = c.hidden_dim;
    const n_layers = c.n_layers;
    const n_heads = c.n_heads;
    const n_kv_heads = c.n_kv_heads;
    const seq_len = c.seq_len;
    const head_size = c.dim / c.n_heads;

    var off: usize = @sizeOf(TinyLlamaConfig);
    try res.buffers.ensureUnusedCapacity(arena, 8 + 9 * n_layers);

    // Map tinyllama naming to Meta llama3 naming.
    // token_embedding_table
    off = newBuff(&res, "model.embed_tokens.weight", .{ .voc = c.vocab.size, .d = dim }, off);
    // rms_att_weight
    off = try splitBuff(&res, "model.layers.{d}.input_layernorm.weight", .{dim}, n_layers, off);
    // wq
    off = try splitBuff(&res, "model.layers.{d}.self_attn.q_proj.weight", .{ dim, n_heads * head_size }, n_layers, off);
    // wk
    off = try splitBuff(&res, "model.layers.{d}.self_attn.k_proj.weight", .{ dim, n_kv_heads * head_size }, n_layers, off);
    // wv
    off = try splitBuff(&res, "model.layers.{d}.self_attn.v_proj.weight", .{ dim, n_kv_heads * head_size }, n_layers, off);
    // wo
    off = try splitBuff(&res, "model.layers.{d}.self_attn.o_proj.weight", .{ n_heads * head_size, dim }, n_layers, off);
    // rms_ffn_weight
    off = try splitBuff(&res, "model.layers.{d}.post_attention_layernorm.weight", .{dim}, n_layers, off);
    // w1
    off = try splitBuff(&res, "model.layers.{d}.mlp.gate_proj.weight", .{ hidden_dim, dim }, n_layers, off);
    // w2
    off = try splitBuff(&res, "model.layers.{d}.mlp.down_proj.weight", .{ dim, hidden_dim }, n_layers, off);
    // w3
    off = try splitBuff(&res, "model.layers.{d}.mlp.up_proj.weight", .{ hidden_dim, dim }, n_layers, off);
    // rms_final_weight
    off = newBuff(&res, "model.norm.weight", .{dim}, off);
    // freq_cis_real (not used)
    off = newBuff(&res, "freq_cis_real", .{ seq_len, head_size / 2 }, off);
    off = newBuff(&res, "freq_cis_imag", .{ seq_len, head_size / 2 }, off);

    // wcls
    if (c.vocab.has_lm_head) {
        off = newBuff(&res, "lm_head.weight", .{ c.vocab.size, c.dim }, off);
    } else {
        res.buffers.putAssumeCapacityNoClobber("lm_head.weight", res.buffers.get("model.embed_tokens.weight").?);
    }

    const weights_size = off;
    std.log.info("Loaded a tinyllama file of {} bytes.\nThis is the parsed configuration of this llama model: {}", .{ weights_size, c });
    if (file.stat() catch null) |stat| {
        zml.meta.assert(weights_size == stat.size, "Expected to have a tinyllama file of {} bytes but file only got {} !\nThis is the parsed configuration of this llama model: {}", .{ weights_size, stat.size, c });
    }

    {
        try res._metadata.ensureUnusedCapacity(arena, 11);
        res._metadata.putAssumeCapacityNoClobber("dim", .{ .int64 = c.dim });
        res._metadata.putAssumeCapacityNoClobber("hidden_dim", .{ .int64 = c.hidden_dim });
        res._metadata.putAssumeCapacityNoClobber("n_layers", .{ .int64 = c.n_layers });
        res._metadata.putAssumeCapacityNoClobber("num_heads", .{ .int64 = c.n_heads });
        res._metadata.putAssumeCapacityNoClobber("num_kv_heads", .{ .int64 = c.n_kv_heads });
        res._metadata.putAssumeCapacityNoClobber("vocab_size", .{ .int64 = c.vocab.size });
        res._metadata.putAssumeCapacityNoClobber("has_lm_head", .{ .boolval = c.vocab.has_lm_head });
        res._metadata.putAssumeCapacityNoClobber("max_seq_len", .{ .int64 = c.seq_len });
        res._metadata.putAssumeCapacityNoClobber("rope_impl", .{ .string = "interleaved" });
        res._metadata.putAssumeCapacityNoClobber("rope_freq_base", .{ .float64 = 10_000 });
        res._metadata.putAssumeCapacityNoClobber("rms_norm_eps", .{ .float64 = 1e-6 });
    }

    return res;
}

fn newBuff(store: *zml.aio.BufferStore, name: []const u8, sh: anytype, offset: usize) usize {
    var shape = zml.Shape.init(sh, .f32);
    const n = shape.byteSize();
    const buff = zml.HostBuffer.fromBytes(shape, store.files[0].data[offset..][0..n]);
    store.buffers.putAssumeCapacityNoClobber(name, buff);
    zml.log.info("Found {s}: {}", .{ name, shape });
    return offset + n;
}

fn splitBuff(store: *zml.aio.BufferStore, comptime fmt: []const u8, sh: anytype, layers: usize, offset: usize) !usize {
    var shape = zml.Shape.init(sh, .f32);
    const n = shape.byteSize();
    var off = offset;
    for (0..layers) |i| {
        const name = try std.fmt.allocPrint(store.arena.allocator(), fmt, .{i});
        const buff = zml.HostBuffer.fromBytes(shape, store.files[0].data[off..][0..n]);
        store.buffers.putAssumeCapacityNoClobber(name, buff);
        off += n;
        if (i == 0) zml.log.info("Found {s}: {}", .{ name, shape });
    }
    return off;
}

pub fn loadTokenizer(allocator: std.mem.Allocator, tokenizer_path: []const u8, vocab_size: u32) !zml.tokenizer.Tokenizer {
    const tokenizer_file = try std.fs.cwd().openFile(tokenizer_path, .{});
    defer tokenizer_file.close();
    var tok_reader = std.io.bufferedReader(tokenizer_file.reader());
    const r = tok_reader.reader();

    const max_token_len = try r.readInt(u32, .little);
    const special_tokens: zml.tokenizer.Tokenizer.SpecialTokens = .{
        .unk = 0,
        .bos = 1,
        .eos = 2,
    };
    var tokenizer = try zml.tokenizer.Tokenizer.init(allocator, vocab_size, max_token_len, null, special_tokens, true);
    var i: u32 = 0;
    while (readToken(&tokenizer, &r)) : (i += 1) {
        // Pass
    } else |_| {
        if (i < vocab_size) {
            zml.log.info("Read {d} words out of {?d}", .{ i, vocab_size });
        }
        tokenizer.vocab_size = i;
    }
    return tokenizer;
}

fn readToken(tokenizer: *zml.tokenizer.Tokenizer, tok_reader: anytype) !void {
    const score: f32 = @bitCast(try tok_reader.readInt(u32, .little));
    const len: usize = @intCast(try tok_reader.readInt(u32, .little));
    try tokenizer.readTokenInto(score, len, tok_reader);
}
