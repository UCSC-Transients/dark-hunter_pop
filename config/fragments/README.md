# Config fragments
#
# Each stage / domain subagent drafts a YAML fragment here during development (tracked in git).
# The review/integration subagent merges fragments into the single canonical config/config.yaml
# at each checkpoint. Delivered main always has exactly one merged config file.
#
# Naming convention: config/fragments/<domain_or_stage>.yaml
# Example: config/fragments/data_acquisition.yaml
#
# Phase 2 checkpoint (issue #40): domains below are materialized in config/config.yaml.
# Fragments remain the draft source for later stage edits; loader still merges them first.
#
# Do not put secrets in fragments. See ARCHITECTURE.md §7.
