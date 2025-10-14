use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "ortts", about, author)]
pub struct Cli {
  #[command(subcommand)]
  pub command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
  Run,
  Serve(ServeArgs),
}

#[derive(Args, Debug)]
pub struct ServeArgs {
  /// listen on host:port (default: 127.0.0.1:12775)
  #[arg(long, default_value = "127.0.0.1:8080")]
  listen: String,
}
