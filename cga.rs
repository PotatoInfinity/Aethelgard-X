use std::ops::{Add, Sub, Mul};

/// Aethelgard-X Multivector (32-float components for 5D CGA Cl(4,1))
/// The basis is ordered as:
/// 0: 1 (Scalar)
/// 1-5: e1, e2, e3, e+, e- (Vectors)
/// 6-15: Bivectors (e12, e13, e1+, e1-, e23, e2+, e2-, e3+, e3-, e+-)
/// 16-25: Trivectors (e123, e12+, e12-, e13+, e13-, e1+-, e23+, e23-, e2+-, e3+-)
/// 26-30: Quadvectors (e123+, e123-, e12+-, e13+-, e23+-)
/// 31: e123+- (Pseudoscalar)
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Multivector5D {
    pub lanes: [f32; 32],
}

impl Multivector5D {
    pub fn zero() -> Self {
        Self { lanes: [0.0; 32] }
    }

    pub fn new_scalar(s: f32) -> Self {
        let mut m = Self::zero();
        m.lanes[0] = s;
        m
    }

    /// Basis vectors
    pub fn e(i: usize) -> Self {
        let mut m = Self::zero();
        if i >= 1 && i <= 5 {
            m.lanes[i] = 1.0;
        }
        m
    }

    /// Conformal Null Basis: n_infinity = e- + e+
    pub fn n_inf() -> Self {
        let mut m = Self::zero();
        m.lanes[5] = 1.0; // e-
        m.lanes[4] = 1.0; // e+
        m
    }

    /// Conformal Null Basis: n_o = 0.5 * (e- - e+)
    pub fn n_o() -> Self {
        let mut m = Self::zero();
        m.lanes[5] = 0.5;  // e-
        m.lanes[4] = -0.5; // e+
        m
    }

    /// Maps a board coordinate (x, y) to a Conformal Point
    /// P = n_o + x*e1 + y*e2 + 0.5*(x^2 + y^2)*n_inf
    pub fn point(x: f32, y: f32) -> Self {
        let e1 = Self::e(1);
        let e2 = Self::e(2);
        let no = Self::n_o();
        let ninf = Self::n_inf();
        
        no + (e1 * x) + (e2 * y) + (ninf * (0.5 * (x * x + y * y)))
    }

    #[inline(always)]
    pub fn inner_product(&self, other: &Self) -> f32 {
        // Metric for Cl(4,1): e1..e3=+1, e+=+1, e-=-1
        // Using the M4's ability to fuse multiply-adds
        let mut dot = self.lanes[0] * other.lanes[0]; // Scalar
        dot += self.lanes[1] * other.lanes[1]; // e1
        dot += self.lanes[2] * other.lanes[2]; // e2
        dot += self.lanes[3] * other.lanes[3]; // e3
        dot += self.lanes[4] * other.lanes[4]; // e+
        dot -= self.lanes[5] * other.lanes[5]; // e- (Minkowski)
        dot
    }

    /// Reverse operator
    pub fn reverse(&self) -> Self {
        let mut res = *self;
        for i in 0..32 {
            // Number of basis elements k in the blade
            let k = (i as u32).count_ones();
            if k * (k - 1) / 2 % 2 == 1 {
                res.lanes[i] *= -1.0;
            }
        }
        res
    }

    /// Translator Versor: T = 1 - 0.5 * (dx*e1 + dy*e2) * n_inf
    pub fn translator(dx: f32, dy: f32) -> Self {
        let one = Self::new_scalar(1.0);
        let d = Self::e(1) * dx + Self::e(2) * dy;
        let ninf = Self::n_inf();
        one - (d * ninf) * 0.5
    }

    /// Applies a versor transformation: M' = V * M * V_rev
    pub fn transform(&self, versor: &Self) -> Self {
        let rev = versor.reverse();
        (*versor * *self) * rev
    }
    /// Optimized Geometric Product using the Linear Result-Centric Table (Phase 1 Refinement)
    /// This replaces the O(N^2) scatter-write loop with a linear read stream, 
    /// significantly improving cache locality and throughput.
    #[inline(always)]
    pub fn geometric_product(&self, other: &Self) -> Self {
        let mut lanes = [0.0; 32];
        let table = &crate::geometry_tables::GP_MAP;
        
        // The table is ordered by output coefficient 'k'.
        // For each k, there are exactly 32 contributing pairs (sum of products).
        // Total 1024 ops, but linear memory access.
        let mut idx = 0;
        for k in 0..32 {
            let mut acc = 0.0;
            // Unrolling 32 times for auto-vectorization
            for _ in 0..32 {
                let (sign, a, b) = table[idx];
                acc += sign * self.lanes[a] * other.lanes[b];
                idx += 1;
            }
            lanes[k] = acc;
        }
        Self { lanes }
    }

    /// Outer Product (Wedge): A ^ B
    pub fn wedge(&self, other: &Self) -> Self {
        let mut res = Self::zero();
        for (i, &a_val) in self.lanes.iter().enumerate() {
            if a_val == 0.0 { continue; }
            let grade_a = (i as u32).count_ones();
            let table_row = &CAYLEY_TABLE[i];
            for (j, &b_val) in other.lanes.iter().enumerate() {
                if b_val == 0.0 { continue; }
                let grade_b = (j as u32).count_ones();
                let (sign, k) = table_row[j];
                let grade_res = (k as u32).count_ones();
                
                if grade_res == (grade_a + grade_b) {
                    res.lanes[k] += sign * a_val * b_val;
                }
            }
        }
        res
    }

    pub fn dual(&self) -> Self {
        let mut i_inv = Self::zero();
        i_inv.lanes[31] = -1.0; // Pseudoscalar inverse for Cl(4,1)
        *self * i_inv
    }

    /// The Rook Line: P ^ e1 ^ n_inf (Horizontal)
    pub fn rook_blade(p: &Self) -> Self {
        let e1 = Self::e(1);
        let ninf = Self::n_inf();
        p.wedge(&e1).wedge(&ninf)
    }

    /// The Bishop Plane: P ^ (e1+e2) ^ n_inf
    pub fn bishop_blade(p: &Self) -> Self {
        let diag = Self::e(1) + Self::e(2);
        let ninf = Self::n_inf();
        p.wedge(&diag).wedge(&ninf)
    }
}

/// Precomputed Cayley Table for Cl(4,1)
pub static CAYLEY_TABLE: once_cell::sync::Lazy<[[(f32, usize); 32]; 32]> = once_cell::sync::Lazy::new(|| {
    let mut table = [[(0.0, 0); 32]; 32];
    for i in 0..32 {
        for j in 0..32 {
            table[i][j] = basis_product_logic(i, j);
        }
    }
    table
});

fn basis_product_logic(a: usize, b: usize) -> (f32, usize) {
    let mut sign = 1.0;
    let mut a_bits = a;
    for i in 0..5 {
        if (b >> i) & 1 == 1 {
            for j in (i + 1)..5 {
                if (a_bits >> j) & 1 == 1 {
                    sign *= -1.0;
                }
            }
            if (a_bits >> i) & 1 == 1 {
                if i == 4 { // e5 (e-) metric is -1
                    sign *= -1.0;
                }
                a_bits &= !(1 << i);
            } else {
                a_bits |= 1 << i;
            }
        }
    }
    (sign, a_bits)
}

impl Mul for Multivector5D {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.geometric_product(&other)
    }
}

impl Add for Multivector5D {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut res = Self::zero();
        for i in 0..32 {
            res.lanes[i] = self.lanes[i] + other.lanes[i];
        }
        res
    }
}

impl Sub for Multivector5D {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let mut res = Self::zero();
        for i in 0..32 {
            res.lanes[i] = self.lanes[i] - other.lanes[i];
        }
        res
    }
}

impl Mul<f32> for Multivector5D {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        let mut res = Self::zero();
        for i in 0..32 {
            res.lanes[i] = self.lanes[i] * s;
        }
        res
    }
}

/// Lookup table for board points in CGA space
pub static BOARD_SPACE: once_cell::sync::Lazy<[Multivector5D; 64]> = once_cell::sync::Lazy::new(|| {
    let mut table = [Multivector5D { lanes: [0.0; 32] }; 64];
    for r in 0..8 {
        for c in 0..8 {
            table[r * 8 + c] = Multivector5D::point(c as f32, r as f32);
        }
    }
    table
});
