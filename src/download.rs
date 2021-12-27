use crate::cli::Cli;
use anyhow::Result;
use log::*;
use reqwest;
use serde::Deserialize;
use std::io::Read;

#[derive(Deserialize)]
struct Response {
    resources: Vec<Resource>,
}

#[derive(Deserialize)]
struct Resource {
    title: String,
    path: String,
}

/// Download the dataset
pub fn download(args: &Cli) -> Result<()> {
    let client = reqwest::blocking::Client::new();

    // Initial request, get the months where data is available
    let response: Response = client
        .get("https://data.nationalgrideso.com/system/system-frequency-data/datapackage.json")
        .send()?
        .json()?;

    // Save every resource
    log::info!("Starting download");
    for (ix, resource) in response.resources.iter().enumerate() {
        log::info!(
            "[{:02}/{}] Saving '{:<40}' to {:?}",
            ix,
            response.resources.len(),
            resource.title,
            args.data_path
        );
        let bytes = client.get(&resource.path).send()?.bytes()?;

        // wrap the byte array in a Cursor, providing it with a Seek implementation
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
        zip.extract(&args.data_path)?;
    }

    log::info!("Finished downloading the dataset");

    Ok(())
}
