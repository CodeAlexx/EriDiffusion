#![cfg(feature = "legacy-bins")]

// NOTE: Legacy binary; gated behind the `legacy-bins` feature so it stays out of default builds. Enable with `--features legacy-bins` if you still rely on it.

fn main() {
    eprintln!(
        "train_sd35_lokr: placeholder stub. SD3.5 LoKr training not implemented in this build."
    );
}
