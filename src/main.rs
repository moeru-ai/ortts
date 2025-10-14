mod cli;
use cli::{Cli, Commands};

use clap::Parser;
use human_panic::{setup_panic, metadata};

#[tokio::main]
async fn main() {
  setup_panic!(metadata!().homepage("https://github.com/moeru-ai/ortts/issues"));

  let cli = Cli::parse();

  if let Some(command) = cli.command {
    match command {
      Commands::Serve(_) => todo!(),
      Commands::Run => todo!(),
    }
  } else {
    todo!()
  }
}
