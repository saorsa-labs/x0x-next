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

ci: fmt-check clippy test doc
