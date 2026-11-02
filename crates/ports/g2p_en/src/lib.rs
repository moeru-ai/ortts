//! Subset port of the Python g2p_en 2.1.0 package used by GPT-SoVITS.
//!
//! Ports `g2p_en/__init__.py` (re-export of [`G2p`]) and the parts of
//! `g2p_en/g2p.py` that GPT-SoVITS `text/english.py` (`en_G2p`) uses: the
//! `G2p.__init__` vocab tables, the numpy GRU weights from `checkpoint20.npz`,
//! `homograph2features` and `G2p.predict`. See the crate README for the scope.

pub mod g2p;
mod npz;

/// Port of `from .g2p import G2p` in `g2p_en/__init__.py`.
pub use g2p::G2p;
