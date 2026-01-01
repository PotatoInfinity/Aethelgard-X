use crate::cga::{Multivector5D, CAYLEY_TABLE};

/// Precomputed tables for Result-Centric Geometric Product.
/// For each output component k (0..32), we list the 32 pairs of (a_idx, b_idx, sign) 
/// that contribute to it.
/// Structure: [Output_Lane][Input_Pair_Index] -> (Sign*Bucket, A_Index, B_Index)
/// We flatten this for cache locality.
pub static GP_MAP: once_cell::sync::Lazy<Vec<(f32, usize, usize)>> = once_cell::sync::Lazy::new(|| {
    // There are 32 output components.
    // For each output 'k', there are 32 pairs of (a, b) such that a * b = +/- k.
    // Total entries = 32 * 32 = 1024.
    // We store them ordered by 'k' to allow linear writing of the result.
    
    let mut map = Vec::with_capacity(1024);
    
    for k in 0..32 {
        for a in 0..32 {
            // We need to find 'b' such that basis(a) * basis(b) = +/- basis(k).
            // In a group, b = a^-1 * k. 
            // We brute force the CAYLEY_TABLE to find the matching pair.
            for b in 0..32 {
                let (sign, res_k) = CAYLEY_TABLE[a][b];
                if res_k == k {
                   map.push((sign, a, b));
                }
            }
        }
    }
    map
});
