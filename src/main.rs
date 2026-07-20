#[cfg(feature = "gui")]
mod gui_entry;

mod cli;

fn main() {
    use clap::Parser;

    let args = cli::CliArgs::parse();

    #[cfg(feature = "gui")]
    if !args.cli {
        gui_entry::run();
        return;
    }

    cli::run(args);
}
