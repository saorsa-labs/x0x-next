//! `measure-agent`: the Experiment 16 instrument (see the library docs).
//!
//! ```text
//! measure-agent --reflect [--listen <ip:port>]
//! measure-agent --run --config <file.toml> [--decisions <n>] [--no-reflect]
//! measure-agent --self-test [--decisions <n>] [--data-dir <dir>]
//! measure-agent --verify <trace.jsonl | data dir>
//! ```

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use measure_agent::agent::RunOptions;
use measure_agent::reflector::{Faults, Reflector};
use measure_agent::selftest::{self_test, DEFAULT_DECISIONS};
use measure_agent::trace::trace_files;
use measure_agent::{run_agent, verify_trace, Config, MeasureError, DEFAULT_PORT};

const USAGE: &str = "usage:
  measure-agent --reflect [--listen <ip:port>]
  measure-agent --run --config <file.toml> [--decisions <n>] [--no-reflect]
  measure-agent --self-test [--decisions <n>] [--data-dir <dir>]
  measure-agent --verify <trace.jsonl | data dir>";

enum Mode {
    Reflect(SocketAddr),
    Run {
        config: PathBuf,
        decisions: Option<u64>,
        reflect: bool,
    },
    SelfTest {
        decisions: u64,
        data_dir: Option<PathBuf>,
    },
    Verify(PathBuf),
    Help,
    Version,
}

fn parse(args: &[String]) -> Result<Mode, String> {
    let mut mode: Option<&str> = None;
    let mut listen: Option<SocketAddr> = None;
    let mut config: Option<PathBuf> = None;
    let mut decisions: Option<u64> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut verify: Option<PathBuf> = None;
    let mut reflect = true;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "--reflect" | "--run" | "--self-test" => {
                if mode.replace(a.as_str()).is_some() {
                    return Err("choose one mode".to_string());
                }
            }
            "--verify" => {
                if mode.replace("--verify").is_some() {
                    return Err("choose one mode".to_string());
                }
                verify = Some(PathBuf::from(value("--verify")?));
            }
            "--listen" => {
                listen = Some(
                    value("--listen")?
                        .parse()
                        .map_err(|_| "--listen needs ip:port".to_string())?,
                );
            }
            "--config" => config = Some(PathBuf::from(value("--config")?)),
            "--decisions" => {
                decisions = Some(
                    value("--decisions")?
                        .parse()
                        .map_err(|_| "--decisions needs a number".to_string())?,
                );
            }
            "--data-dir" => data_dir = Some(PathBuf::from(value("--data-dir")?)),
            "--no-reflect" => reflect = false,
            "-h" | "--help" => return Ok(Mode::Help),
            "-V" | "--version" => return Ok(Mode::Version),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    match mode {
        Some("--reflect") => {
            Ok(Mode::Reflect(listen.unwrap_or_else(|| {
                SocketAddr::from(([0, 0, 0, 0], DEFAULT_PORT))
            })))
        }
        Some("--run") => Ok(Mode::Run {
            config: config.ok_or("--run needs --config <file>")?,
            decisions,
            reflect,
        }),
        Some("--self-test") => Ok(Mode::SelfTest {
            decisions: decisions.unwrap_or(DEFAULT_DECISIONS),
            data_dir,
        }),
        Some(_) => verify
            .map(Mode::Verify)
            .ok_or_else(|| "--verify needs a path".to_string()),
        None => Ok(Mode::Help),
    }
}

fn verify(path: &Path) -> Result<(), MeasureError> {
    let files = if path.is_dir() {
        trace_files(path)?
    } else {
        vec![path.to_path_buf()]
    };
    for f in files {
        let v = verify_trace(&f)?;
        println!(
            "{}: OK, {} segment(s), {} decisions replayed ({} explored, {} contested, {} copies, {} timeouts), manifest {:?}",
            v.file,
            v.replay.segments,
            v.replay.decisions,
            v.replay.explored,
            v.replay.contested,
            v.replay.copies,
            v.replay.timeouts,
            v.manifest
        );
    }
    Ok(())
}

fn run(mode: Mode) -> Result<(), MeasureError> {
    match mode {
        Mode::Help => println!("{USAGE}"),
        Mode::Version => println!("measure-agent {}", env!("CARGO_PKG_VERSION")),
        Mode::Reflect(addr) => {
            let r = Reflector::spawn(addr, Faults::NONE)?;
            println!("reflecting on udp {}", r.local_addr());
            r.join();
        }
        Mode::Run {
            config,
            decisions,
            reflect,
        } => {
            let cfg = Config::load(&config)?;
            let s = run_agent(
                &cfg,
                RunOptions {
                    max_decisions: decisions,
                    reflect,
                },
            )?;
            println!(
                "run done: {} decisions from #{}, {} explored, {} contested, {} copies, {} pick timeouts",
                s.decisions, s.first_decision, s.explored, s.contested, s.copies, s.pick_timeouts
            );
        }
        Mode::SelfTest {
            decisions,
            data_dir,
        } => print!("{}", self_test(decisions, data_dir)?),
        Mode::Verify(path) => verify(&path)?,
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = match parse(&args) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("measure-agent: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(mode) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("measure-agent: {e}");
            ExitCode::FAILURE
        }
    }
}
