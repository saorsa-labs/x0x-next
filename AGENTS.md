# x0x-next

Research toward rebuilding x0x around a small verified kernel and written-weights policy plane. The authoritative design is `docs/design-study-2026-11-13.pdf` with condensed versions in `docs/`; the ten open decisions live in `docs/decisions.md` and only David and Ben rule on them. `crates/policy-poc` is the rung-1 POC (spec: `docs/poc-spec.md`). `research/prototypes/` holds the study's Python and C simulation experiments — reference data and evidence, never production code, never executed by agents. Nothing in this repo changes the x0x repo.

Rust quality bar: `cargo fmt --all`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, no `unwrap`/`expect`/`panic!` outside tests. Recipes in `justfile` (`just --list`).

License: MIT OR Apache-2.0.
