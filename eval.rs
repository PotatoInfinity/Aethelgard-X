use cozy_chess::*;
use crate::cga::BOARD_SPACE;

pub struct SquareTensor {
    // Physical dimension d=13 (Empty, P, N, B, R, Q, K * White/Black)
    // Bond dimension chi=10
    // Storage: [Physical; 13][BondL; 10][BondR; 10]
    pub data: [[[f32; 10]; 10]; 13],
}

pub struct GeotensorEvaluator {
    pub tension_weight: f32,
    pub tensors: [SquareTensor; 64],
}

impl GeotensorEvaluator {
    pub fn new(_: Option<&str>) -> Self {
        // Initialize with default (random or basic heuristic) strategic weights
        let mut tensors: Vec<SquareTensor> = Vec::with_capacity(64);
        for _ in 0..64 {
            let mut data = [[[0.0; 10]; 10]; 13];
            // Identity-like bond for empty squares to allow flow
            for c in 0..10 {
                data[0][c][c] = 1.0; 
            }
            // Add slight strategic biases for pieces (normally trained via DMRG)
            for p in 1..13 {
                for c in 0..10 {
                    data[p][c][c] = 0.5;
                }
            }
            tensors.push(SquareTensor { data });
        }

        Self { 
            tension_weight: 1.0,
            tensors: tensors.try_into().unwrap_or_else(|_| panic!("Failed to init tensors")),
        }
    }

    pub fn evaluate(&mut self, board: &Board) -> i32 {
        let us = board.side_to_move();
        
        // 1. Classical Baseline
        let mut score = (self.material_score(board, us) - self.material_score(board, !us)) as f32;

        // 2. Geometric Vision (CGA Blades)
        score += self.calculate_cga_vision(board);

        // 3. Tensor Network Contraction & Entropy
        let (mps_val, entropy) = self.evaluate_mps_with_entropy(board);
        score += mps_val;
        
        // High entropy (tactical tension) favors the side with better mobility
        score += entropy * self.tension_weight;

        score as i32
    }

    fn evaluate_mps_with_entropy(&self, board: &Board) -> (f32, f32) {
        let mut state = [0.0; 10];
        state[0] = 1.0;
        let mut total_entropy = 0.0;

        for &sq_idx in SNAKE_PATH.iter() {
            let p_idx = get_piece_index(board, Square::index(sq_idx));
            let square_tensor = &self.tensors[sq_idx].data[p_idx];
            
            let mut next_state = [0.0; 10];
            let mut norm = 0.0;
            for (curr_bond, &val) in state.iter().enumerate() {
                if val == 0.0 { continue; }
                for next_bond in 0..10 {
                    next_state[next_bond] += val * square_tensor[curr_bond][next_bond];
                }
            }
            
            // Normalize and calculate Local Von Neumann Entropy
            for v in next_state.iter() { norm += v * v; }
            norm = norm.sqrt().max(1e-9);
            for v in next_state.iter_mut() {
                *v /= norm;
                if *v > 0.0 {
                    let p = *v * *v;
                    total_entropy -= p * p.ln();
                }
            }
            state = next_state;
        }

        (state[0] * 100.0, total_entropy * 10.0)
    }

    fn calculate_cga_vision(&self, board: &Board) -> f32 {
        let us = board.side_to_move();
        let mut vision_score = 0.0;
        let occupied = board.occupied();

        for sq in occupied {
            let sq_idx = sq as usize;
            let p_vec = BOARD_SPACE[sq_idx];
            let piece = board.piece_on(sq).unwrap();
            let color = board.color_on(sq).unwrap();
            
            // Define the blade (Line or Plane at infinity)
            let blade = match piece {
                Piece::Rook => Some(crate::cga::Multivector5D::rook_blade(&p_vec)),
                Piece::Bishop => Some(crate::cga::Multivector5D::bishop_blade(&p_vec)),
                Piece::Queen => Some(crate::cga::Multivector5D::rook_blade(&p_vec) + crate::cga::Multivector5D::bishop_blade(&p_vec)),
                _ => None,
            };
            
            if let Some(b) = blade {
                // Collect all targets on this blade line
                let mut targets = Vec::with_capacity(8);
                
                for other_sq in occupied {
                    if other_sq == sq { continue; }
                    let other_p = BOARD_SPACE[other_sq as usize];
                    // Geometric Product Check: If P lies on L, then P ^ L = 0 (or inner product in dual space)
                    // Here we verify intersection using the inner product with the blade
                    let intersection = b.inner_product(&other_p).abs();
                    
                    if intersection < 0.01 {
                        // It's on the line. Calculate Euclidean distance for sorting.
                        let dist = (sq.rank() as i32 - other_sq.rank() as i32).pow(2) + 
                                   (sq.file() as i32 - other_sq.file() as i32).pow(2);
                        targets.push((dist, other_sq));
                    }
                }

                // Sort by distance to simulate ray-casting
                targets.sort_by_key(|k| k.0);

                // Attenuation Loop (The "Wedge" logic)
                let mut opacity = 1.0; 
                for (_, target_sq) in targets {
                    let target_piece = board.piece_on(target_sq).unwrap();
                    let target_color = board.color_on(target_sq).unwrap();
                    
                    // Base value of hitting this square
                    let value = match target_piece {
                         Piece::Pawn => 1.0,
                         Piece::Knight | Piece::Bishop => 3.0,
                         Piece::Rook => 5.0,
                         Piece::Queen => 9.0,
                         Piece::King => 0.0, // Check logic handled elsewhere
                    };

                    if target_color == !us {
                        // Impact: We hit an enemy. Add score weighted by remaining opacity.
                        vision_score += if color == us { 5.0 * value * opacity } else { -5.0 * value * opacity };
                        
                        // Enemy pieces are solid walls
                        opacity = 0.0; 
                    } else {
                        // We hit a friend (X-Ray defense). 
                        vision_score += if color == us { 0.5 * value * opacity } else { -0.5 * value * opacity };
                        
                        // Friendly pieces are semi-transparent (Transparency = 0.2)
                        opacity *= 0.2;
                    }

                    if opacity < 0.05 { break; }
                }
            }
        }
        vision_score
    }

    fn material_score(&self, board: &Board, color: Color) -> i32 {
        let mut s = 0;
        let c = board.colors(color);
        s += (board.pieces(Piece::Pawn) & c).len() as i32 * 100;
        s += (board.pieces(Piece::Knight) & c).len() as i32 * 320;
        s += (board.pieces(Piece::Bishop) & c).len() as i32 * 330;
        s += (board.pieces(Piece::Rook) & c).len() as i32 * 500;
        s += (board.pieces(Piece::Queen) & c).len() as i32 * 900;
        s
    }
}

fn get_piece_index(board: &Board, sq: Square) -> usize {
    match board.piece_on(sq) {
        None => 0,
        Some(p) => {
            let color_offset = if board.color_on(sq) == Some(Color::White) { 0 } else { 6 };
            let piece_val = match p {
                Piece::Pawn => 1,
                Piece::Knight => 2,
                Piece::Bishop => 3,
                Piece::Rook => 4,
                Piece::Queen => 5,
                Piece::King => 6,
            };
            color_offset + piece_val
        }
    }
}

/// The Snake Path: A 1D traversal of the 8x8 board for MPS contraction
pub static SNAKE_PATH: [usize; 64] = [
     0,  1,  2,  3,  4,  5,  6,  7,
    15, 14, 13, 12, 11, 10,  9,  8,
    16, 17, 18, 19, 20, 21, 22, 23,
    31, 30, 29, 28, 27, 26, 25, 24,
    32, 33, 34, 35, 36, 37, 38, 39,
    47, 46, 45, 44, 43, 42, 41, 40,
    48, 49, 50, 51, 52, 53, 54, 55,
    63, 62, 61, 60, 59, 58, 57, 56,
];
