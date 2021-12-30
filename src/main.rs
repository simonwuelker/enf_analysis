mod cli;
mod download;
mod fft;

use crate::cli::Cli;
use crate::download::download;
use crate::fft::fft;
use anyhow::Result;
use structopt::StructOpt;
// use bwavfile::WaveReader; 

fn main() -> Result<()> {
    simple_logger::init_with_level(log::Level::Info)?;

    let args = Cli::from_args();

    if args.download {
        download(&args)?;
    }

    let x = vec![2., 3., 4., 1.];
    let y = fft(x.clone(), false);

    // let mut reader = WaveReader::open(args.path)?;

    Ok(())
}
