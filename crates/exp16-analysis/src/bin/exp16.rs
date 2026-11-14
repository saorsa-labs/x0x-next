//! `exp16`: the Experiment 16 analysis (see the library docs).
//!
//! ```text
//! exp16 [--split-days F:H | --holdout-fraction X] [--json <path>]
//!       [--bootstrap N] [--seed S] <data dir>...
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use exp16_analysis::analysis::DEFAULT_BOOTSTRAP;
use exp16_analysis::{analyze, load_nodes, report, AnalysisError, Options, Split};

const USAGE: &str = "usage:
  exp16 [--split-days F:H | --holdout-fraction X] [--json <path>]
        [--bootstrap N] [--seed S] <data dir>...

  --split-days F:H       fit on the first F UTC days, score on the next H (default 21:7)
  --holdout-fraction X   score on the last X of each node's decisions (pilot smoke runs)
  --json <path>          also write every number as JSON
  --bootstrap N          bootstrap resamples per node (default 1000)
  --seed S               bootstrap seed (default 16)";

struct Cli {
    opts: Options,
    json: Option<PathBuf>,
    dirs: Vec<PathBuf>,
}

fn usage(msg: impl Into<String>) -> AnalysisError {
    AnalysisError::Usage(msg.into())
}

fn parse(args: &[String]) -> Result<Option<Cli>, AnalysisError> {
    let mut opts = Options::default();
    let mut split: Option<Split> = None;
    let mut json = None;
    let mut dirs = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| usage(format!("{name} needs a value")))
        };
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--split-days" => {
                let v = value(a)?;
                let (f, h) = v
                    .split_once(':')
                    .ok_or_else(|| usage("--split-days takes F:H, e.g. 21:7"))?;
                let parse_days = |s: &str| {
                    s.trim()
                        .parse::<u32>()
                        .map_err(|_| usage(format!("bad day count {s:?}")))
                };
                let new = Split::Days {
                    fit_days: parse_days(f)?,
                    holdout_days: parse_days(h)?,
                };
                if split.replace(new).is_some() {
                    return Err(usage("give one of --split-days and --holdout-fraction"));
                }
            }
            "--holdout-fraction" => {
                let v = value(a)?;
                let fraction = v
                    .parse::<f64>()
                    .map_err(|_| usage(format!("bad fraction {v:?}")))?;
                if split.replace(Split::HoldoutFraction { fraction }).is_some() {
                    return Err(usage("give one of --split-days and --holdout-fraction"));
                }
            }
            "--json" => json = Some(PathBuf::from(value(a)?)),
            "--bootstrap" => {
                let v = value(a)?;
                opts.bootstrap = v
                    .parse()
                    .map_err(|_| usage(format!("bad resample count {v:?}")))?;
            }
            "--seed" => {
                let v = value(a)?;
                opts.seed = v.parse().map_err(|_| usage(format!("bad seed {v:?}")))?;
            }
            s if s.starts_with('-') => return Err(usage(format!("unknown option {s}"))),
            dir => dirs.push(PathBuf::from(dir)),
        }
    }
    if dirs.is_empty() {
        return Err(usage("give at least one data directory"));
    }
    if let Some(s) = split {
        opts.split = s;
    }
    if opts.bootstrap == 0 {
        opts.bootstrap = DEFAULT_BOOTSTRAP;
    }
    opts.split.validate()?;
    Ok(Some(Cli { opts, json, dirs }))
}

fn run(cli: &Cli) -> Result<(), AnalysisError> {
    let nodes = load_nodes(&cli.dirs)?;
    let report = analyze(&nodes, &cli.opts)?;
    print!("{}", report::render(&report));
    if let Some(path) = &cli.json {
        let text = serde_json::to_string_pretty(&report)?;
        std::fs::write(path, text + "\n")
            .map_err(|e| AnalysisError::io(format!("write {}", path.display()), e))?;
        println!("\nJSON report written to {}", path.display());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args) {
        Ok(None) => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Some(cli)) => match run(&cli) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("exp16: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("exp16: {e}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}
