mod commands;

mod cli;
use cli::{Cli, Commands};

use clap::{CommandFactory, Parser};
use human_panic::{metadata, setup_panic};
use ortts_shared::AppError;

#[tokio::main]
async fn main() -> Result<(), AppError> {
  setup_panic!(metadata!().homepage("https://github.com/moeru-ai/ortts/issues"));

  let cli = Cli::parse();

  if let Some(command) = cli.command {
    match command {
      Commands::Serve(args) => commands::serve(args).await,
      Commands::Run => todo!(),
    }
  } else {
    Cli::command().print_help().map_err(std::convert::Into::into)
  }
}
