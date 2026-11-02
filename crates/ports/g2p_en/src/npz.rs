//! Minimal reader for the uncompressed `.npz` archive that
//! `numpy.load("checkpoint20.npz")` reads in `G2p.load_variables`.
//!
//! `checkpoint20.npz` is a zip file written by `np.savez`: every member is
//! stored without compression and holds one `.npy` (format 1.0) array of
//! little-endian `float32` in C order. This reader supports exactly that.

use std::collections::HashMap;

/// A C-order `float32` array read from a `.npy` member, the numpy `ndarray`
/// that `np.load(...)[name]` returns.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
  pub shape: Vec<usize>,
  pub data: Vec<f32>,
}

fn u16_at(b: &[u8], at: usize) -> usize {
  u16::from_le_bytes([b[at], b[at + 1]]) as usize
}

fn u32_at(b: &[u8], at: usize) -> usize {
  u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

/// Port of `numpy.load` for an uncompressed `.npz` of `<f4` arrays. Keys are
/// the member names without the `.npy` suffix, as in `NpzFile.__getitem__`.
pub fn load(bytes: &[u8]) -> HashMap<String, NpyArray> {
  const LOCAL_HEADER: usize = 0x0403_4b50;
  let mut out = HashMap::new();
  let mut pos = 0;
  while pos + 30 <= bytes.len() && u32_at(bytes, pos) == LOCAL_HEADER {
    let flags = u16_at(bytes, pos + 6);
    let method = u16_at(bytes, pos + 8);
    let size = u32_at(bytes, pos + 22);
    let name_len = u16_at(bytes, pos + 26);
    let extra_len = u16_at(bytes, pos + 28);
    assert_eq!(method, 0, "npz member is compressed");
    assert_eq!(flags & 0x08, 0, "npz member uses a data descriptor");
    let name_start = pos + 30;
    let name = std::str::from_utf8(&bytes[name_start..name_start + name_len])
      .expect("npz member name is not UTF-8");
    let data_start = name_start + name_len + extra_len;
    let member = &bytes[data_start..data_start + size];
    let key = name.strip_suffix(".npy").unwrap_or(name).to_owned();
    out.insert(key, read_npy(member));
    pos = data_start + size;
  }
  out
}

/// Port of `numpy.lib.format.read_array` for a C-order `<f4` array.
fn read_npy(b: &[u8]) -> NpyArray {
  assert_eq!(&b[..6], b"\x93NUMPY", "bad npy magic");
  let major = b[6];
  let (header_len, header_start) = match major {
    1 => (u16_at(b, 8), 10),
    _ => (u32_at(b, 8), 12),
  };
  let header = std::str::from_utf8(&b[header_start..header_start + header_len])
    .expect("npy header is not UTF-8");
  assert!(
    header.contains("'descr': '<f4'"),
    "npy dtype is not <f4: {header}"
  );
  assert!(
    header.contains("'fortran_order': False"),
    "npy array is Fortran order: {header}"
  );
  let shape_start =
    header.find("'shape': (").expect("npy header has no shape") + "'shape': (".len();
  let shape_end = shape_start
    + header[shape_start..]
      .find(')')
      .expect("npy shape is not closed");
  let shape: Vec<usize> = header[shape_start..shape_end]
    .split(',')
    .map(str::trim)
    .filter(|s| !s.is_empty())
    .map(|s| s.parse().expect("npy shape is not an integer"))
    .collect();
  let count: usize = shape.iter().product();
  let body = &b[header_start + header_len..];
  assert_eq!(body.len(), count * 4, "npy data size does not match shape");
  let data = body
    .chunks_exact(4)
    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    .collect();
  NpyArray { shape, data }
}
