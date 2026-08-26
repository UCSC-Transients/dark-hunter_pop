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
# Phase 3 Review/Integration (issue #50): merge new fragments (rv_consistency / triples, …)
# into config/config.yaml at each checkpoint. Fragments remain the draft source; loader still
# merges them first.
#
# Do not put secrets in fragments. See ARCHITECTURE.md §7.
