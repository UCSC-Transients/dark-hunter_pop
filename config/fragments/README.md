# Config fragments
#
# Each stage / domain subagent drafts a YAML fragment here during development (tracked in git).
# The review/integration subagent merges fragments into the single canonical config/config.yaml
# at each checkpoint. Delivered main always has exactly one merged config file.
#
# Naming convention: config/fragments/<domain_or_stage>.yaml
# Example: config/fragments/data_acquisition.yaml
#
# Phase 2 checkpoint (issue #40 / PR #46): domains below are materialized in config/config.yaml.
# Phase 3–4 domains (rv_consistency, triples, companion_nature, population_model) materialized
# under continuous Review/Integration (#64) after children #48/#49/#56/#57 landed.
# Phase 5: inference fragment materialized under continuous Review (#72) after #63 / PR #67.
# Phase 6: benchmarks + diagnostics.sbc / hooks.sbc_recovery materialized under #72 after
# children #69/#70/#71 landed. Loader still merges fragments first (canonical wins).
#
# Do not put secrets in fragments. See ARCHITECTURE.md §7.
