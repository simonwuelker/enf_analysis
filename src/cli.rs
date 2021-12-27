use structopt::StructOpt;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(StructOpt)]
pub struct Cli {
    /// path to the input file
    #[structopt(parse(from_os_str))]
    pub path: std::path::PathBuf,

    /// path to a folder containing the dataset
    #[structopt(long, default_value = "data/", parse(from_os_str))]
    pub data_path: std::path::PathBuf,

    /// Whether or not to download the database
    #[structopt(short, long)]
    pub download: bool,
}
