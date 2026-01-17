use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "ortts", about, author)]
pub struct Cli {
  #[command(subcommand)]
  pub command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
  /// Run a model (TODO)
  Run,
  /// Start ortts
  Serve(ServeArgs),
  /// List models
  #[command(visible_alias = "ls")]
  List,
  /// Remove a model
  #[command(visible_alias = "rm")]
  Remove {
    #[arg(required = true)]
    model: Vec<String>,
  },
}

#[derive(Args, Debug)]
pub struct ServeArgs {
  /// listen on host:port
  #[arg(long, default_value = "127.0.0.1:12775")]
  pub listen: String,
}
