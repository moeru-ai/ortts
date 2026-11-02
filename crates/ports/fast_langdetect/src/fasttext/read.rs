//! Little-endian binary readers mirroring the `in.read((char*)&x, sizeof(x))`
//! calls of fastText (model files are written on little-endian machines).

use std::io::Read;

pub(crate) fn read_i8<R: Read>(r: &mut R) -> std::io::Result<i8> {
  let mut b = [0u8; 1];
  r.read_exact(&mut b)?;
  Ok(b[0] as i8)
}

pub(crate) fn read_bool<R: Read>(r: &mut R) -> std::io::Result<bool> {
  Ok(read_i8(r)? != 0)
}

pub(crate) fn read_i32<R: Read>(r: &mut R) -> std::io::Result<i32> {
  let mut b = [0u8; 4];
  r.read_exact(&mut b)?;
  Ok(i32::from_le_bytes(b))
}

pub(crate) fn read_i64<R: Read>(r: &mut R) -> std::io::Result<i64> {
  let mut b = [0u8; 8];
  r.read_exact(&mut b)?;
  Ok(i64::from_le_bytes(b))
}

pub(crate) fn read_f64<R: Read>(r: &mut R) -> std::io::Result<f64> {
  let mut b = [0u8; 8];
  r.read_exact(&mut b)?;
  Ok(f64::from_le_bytes(b))
}
