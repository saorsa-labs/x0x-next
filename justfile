default:
	just --list

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace --all-features

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

demo:
	cargo run -p policy-poc --bin poc-demo

# Experiment 16 analysis, e.g. `just analyze --holdout-fraction 0.25 .pilot/mac/data`
analyze *ARGS:
	cargo run -q -p exp16-analysis --bin exp16 -- {{ARGS}}

ci: fmt-check clippy test doc
