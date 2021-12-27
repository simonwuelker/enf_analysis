mod cli;
mod download;
mod fft;

use crate::cli::Cli;
use crate::download::download;
use anyhow::Result;
use structopt::StructOpt;

fn main() -> Result<()> {
    simple_logger::init_with_level(log::Level::Info)?;

    let args = Cli::from_args();

    if args.download {
        download(&args)?;
    }

    let mut reader = hound::WavReader::open(args.path)?;

    Ok(())
}
