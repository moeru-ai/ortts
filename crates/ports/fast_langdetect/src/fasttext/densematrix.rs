//! Port of fastText `src/densematrix.{h,cc}` and the `Vector` operations that
//! prediction uses (`src/vector.{h,cc}`).

use std::io::Read;

use super::read::read_i64;

/// Port of fastText `class DenseMatrix` (row-major `f32`).
#[derive(Debug, Clone)]
pub struct DenseMatrix {
  m: i64,
  n: i64,
  data: Vec<f32>,
}

impl DenseMatrix {
  /// Port of `DenseMatrix::load(std::istream&)`.
  pub fn load<R: Read>(r: &mut R) -> std::io::Result<Self> {
    let m = read_i64(r)?;
    let n = read_i64(r)?;
    let len = usize::try_from(m * n)
      .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid matrix shape"))?;
    let mut data = vec![0f32; len];
    let mut buf = vec![0u8; 1 << 20];
    let mut filled = 0usize;
    while filled < len {
      let take = (len - filled).min(buf.len() / 4);
      let bytes = &mut buf[..take * 4];
      r.read_exact(bytes)?;
      for (dst, src) in data[filled..filled + take]
        .iter_mut()
        .zip(bytes.chunks_exact(4))
      {
        *dst = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
      }
      filled += take;
    }
    Ok(Self { m, n, data })
  }

  /// Port of `Matrix::size(0)` / `Matrix::size(1)`.
  pub fn size(&self, dim: i64) -> i64 {
    if dim == 0 { self.m } else { self.n }
  }

  fn row(&self, i: i64) -> &[f32] {
    let n = self.n as usize;
    let start = i as usize * n;
    &self.data[start..start + n]
  }

  /// Port of `DenseMatrix::addRowToVector(Vector& x, int32_t i)`:
  /// `x[j] += at(i, j)`.
  pub fn add_row_to_vector(&self, x: &mut [f32], i: i32) {
    for (xj, aij) in x.iter_mut().zip(self.row(i64::from(i))) {
      *xj += *aij;
    }
  }

  /// Port of `DenseMatrix::dotRow(const Vector&, int64_t i)`:
  /// `d += at(i, j) * vec[j]` accumulated in `f32`, with a separate multiply
  /// and add (the reference arm64 wheel does not fuse them into an FMA; the
  /// fixtures match bit for bit only without fusion). fastText throws
  /// `EncounterNanError` on a NaN result, which finite model weights never
  /// produce, so that check is left out.
  pub fn dot_row(&self, vec: &[f32], i: i64) -> f32 {
    let mut d = 0f32;
    for (aij, vj) in self.row(i).iter().zip(vec) {
      d += *aij * *vj;
    }
    d
  }
}

/// Port of `Vector::zero()`.
pub fn vector_zero(x: &mut [f32]) {
  x.fill(0.0);
}

/// Port of `Vector::mul(real a)`.
pub fn vector_mul(x: &mut [f32], a: f32) {
  for v in x.iter_mut() {
    *v *= a;
  }
}
