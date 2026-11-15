# Vanilla split reference diagnostic

`vanilla-8db7-one-row-diagnostic.patch` applies only to clean vanilla revision
`8db7ddc46cd2575514408c4ab7dd878f69df2f5f`. With
`STRATA_DIAG_SERIAL_ONE_ROW` set, it forces the server's serial short-prompt
verify loop to commit one input per window. It changes no path when unset.
It does not itself enforce safe diagnostic settings or produce STFR packets.

Use only with bounded prompts, a fresh server state for each forced prefix,
`STRATA_DUMP_FIRST_LOGITS` pointing to a private fresh prefix, and a verified
split/full-resident configuration. Disable MTP, suffix/lookup, caching, prompt
pipelining, merged first windows, vision and control vectors; force all prefix
rows through the serial short-read path. Validate every `REUSED`, `DONE`,
finite full-logit row, argmax and clean process shutdown before packaging any
capture. The existing hook emits raw floats for the final input of each
request, not a continuously captured sequence. Its filenames and logs are
private; the caller is responsible for source/build provenance and safe
failure handling. This diagnostic is not a production mode or parity bypass.
