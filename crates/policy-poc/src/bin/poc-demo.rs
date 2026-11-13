//! `poc-demo`: prints the six-step evidence log of the policy plane POC
//! (design study §5; poc-spec §10).

use std::process::ExitCode;

fn main() -> ExitCode {
    match policy_poc::demo::run() {
        Ok(report) => {
            for line in &report.lines {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("poc-demo failed: {e}");
            ExitCode::FAILURE
        }
    }
}
