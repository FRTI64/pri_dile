# Pilot test_v5 (superseded)

This directory contains the first test_v5 run (2026-10-27, catalog
`096498df...`). The test-only families in that catalog were too cooperative:
every candidate scored 595 on all of them, so the 25% test-only component did
not discriminate.

The catalog was revised to use adversarial test-only families
(`early_harvest`, `frequent_kamikaze`, `probe_harvest`, `hyper_extort`,
`early_probe`, `adaptive_probe`). The manifest hash changed to
`f6705fbb...`; the usage ledger was reset and test_v5 was rerun.

**Do not use these pilot numbers as final results.**
