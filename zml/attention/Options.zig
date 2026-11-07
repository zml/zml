/// Query j sees the keys up to token_index + j, as in a decoder. Otherwise every query sees every
/// key, as in an encoder or a diffusion transformer, and token_index is ignored.
causal: bool = true,
