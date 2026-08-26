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
# Phase 5: merge inference fragment when #63 opens/lands; loader still merges fragments first.
#
# Do not put secrets in fragments. See ARCHITECTURE.md §7.
