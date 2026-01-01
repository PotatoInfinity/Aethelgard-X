use std::collections::BinaryHeap;
use std::cmp::Ordering;
use cozy_chess::*;

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f32,
    position: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse because BinaryHeap is a max-heap
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct GeodesicField {
    pub costs: [f32; 64],
    pub potentials: [f32; 64],
    pub retro_potentials: [f32; 64],
    pub barriers: std::collections::HashMap<usize, f32>,
}

impl GeodesicField {
    pub fn new() -> Self {
        Self {
            costs: [1.0; 64],
            potentials: [f32::MAX; 64],
            retro_potentials: [f32::MAX; 64],
            barriers: std::collections::HashMap::new(),
        }
    }

    /// Primal Wave: Propagation from origin squares (forward in time)
    pub fn propagate(&mut self, start_sqs: &[usize], piece_type: Option<Piece>, board: &Board) {
        self.potentials.fill(f32::MAX);
        let mut pq = BinaryHeap::new();

        for &sq in start_sqs {
            self.potentials[sq] = 0.0;
            pq.push(State { cost: 0.0, position: sq });
        }

        Self::dijkstra_core(&mut pq, &mut self.potentials, piece_type, board, &self.costs, &self.barriers);
    }

    /// Retrocausal Wave: Propagation backward from the goal (e.g., enemy king)
    pub fn propagate_retro(&mut self, target_sq: usize, board: &Board) {
        self.retro_potentials.fill(f32::MAX);
        let mut pq = BinaryHeap::new();

        self.retro_potentials[target_sq] = 0.0;
        pq.push(State { cost: 0.0, position: target_sq });

        // Goal propagation uses generic piece mobility or "King" as it's the target point
        Self::dijkstra_core(&mut pq, &mut self.retro_potentials, None, board, &self.costs, &self.barriers);
    }

    fn dijkstra_core(
        pq: &mut BinaryHeap<State>, 
        dists: &mut [f32; 64], 
        piece_type: Option<Piece>, 
        board: &Board,
        costs: &[f32; 64],
        barriers: &std::collections::HashMap<usize, f32>,
    ) {
        while let Some(State { cost, position }) = pq.pop() {
            if cost > dists[position] {
                continue;
            }

            for neighbor in get_dynamic_neighbors_static(position, piece_type, board) {
                let base_cost = costs[neighbor];
                let barrier_cost = barriers.get(&neighbor).cloned().unwrap_or(0.0);
                let next_cost = cost + base_cost + barrier_cost;
                
                if next_cost < dists[neighbor] {
                    dists[neighbor] = next_cost;
                    pq.push(State { cost: next_cost, position: neighbor });
                }
            }
        }
    }

    /// Finds the best move target where Primal and Retro waves meet constructively
    pub fn solve_flow(&self, start_sqs: &[usize]) -> Option<usize> {
        let mut best_sq = None;
        let mut min_action = f32::MAX;

        for &sq in start_sqs {
            for neighbor in get_generic_neighbors(sq) {
                // Constructive Interference: S = Primal + Retro
                let action = self.potentials[neighbor] + self.retro_potentials[neighbor];
                if action < min_action {
                    min_action = action;
                    best_sq = Some(neighbor);
                }
            }
        }
        best_sq
    }

    pub fn get_dynamic_neighbors(&self, sq: usize, piece_type: Option<Piece>, board: &Board) -> Vec<usize> {
        get_dynamic_neighbors_static(sq, piece_type, board)
    }

    /// Updates costs based on piece positions and board logic
    pub fn update_costs(&mut self, board: &Board) {
        let us = board.side_to_move();
        let pawns = board.pieces(Piece::Pawn);
        let their_pawns = pawns & board.colors(!us);

        for sq in 0..64 {
            let square = Square::index(sq);
            let mut base_cost = 1.0;

            if let Some(color) = board.color_on(square) {
                if color == us {
                    base_cost = 20.0; // Avoid blocking our own pieces
                } else {
                    base_cost = 0.3; // High attraction to enemy pieces
                }
            }

            // Logarithmic Barrier for Pawn Chains
            // Crossing an opponent's pawn chain is topologically expensive
            let dist_to_chain = self.min_dist_to_bitboard(square, their_pawns);
            if dist_to_chain < 2.0 {
                // S = Phi + ln(1/dist)
                base_cost += (2.0 - dist_to_chain).max(0.1).ln().max(0.0) * 5.0;
            }

            self.costs[sq] = base_cost;
        }
    }

    fn min_dist_to_bitboard(&self, sq: Square, bb: BitBoard) -> f32 {
        let mut min_d = 10.0;
        let r1 = sq.rank() as i32;
        let f1 = sq.file() as i32;
        for other in bb {
            let r2 = other.rank() as i32;
            let f2 = other.file() as i32;
            let d = (((r1 - r2).pow(2) + (f1 - f2).pow(2)) as f32).sqrt();
            if d < min_d { min_d = d; }
        }
        min_d
    }
}

pub fn get_dynamic_neighbors_static(sq: usize, piece_type: Option<Piece>, board: &Board) -> Vec<usize> {
    let mut neighbors = get_generic_neighbors(sq);
    
    match piece_type {
        Some(Piece::Knight) => {
            // Topological Sewing: Knights fold the manifold
            // The Knight "wormholes" to its destination in 1 step
            neighbors.clear(); 
            neighbors.extend_from_slice(&KNIGHT_ADJACENCY[sq]);
        }
        Some(Piece::Rook) | Some(Piece::Bishop) | Some(Piece::Queen) => {
            // Sliding piece logic: add all squares reachable on rays
            let square = Square::index(sq);
            let move_set = match piece_type {
                Some(Piece::Rook) => get_rook_moves(square, board.occupied()),
                Some(Piece::Bishop) => get_bishop_moves(square, board.occupied()),
                _ => get_rook_moves(square, board.occupied()) | get_bishop_moves(square, board.occupied()),
            };
            for target in move_set {
                neighbors.push(target as usize);
            }
        }
        _ => {}
    }
    neighbors
}
pub fn get_generic_neighbors(sq: usize) -> Vec<usize> {
    let mut neighbors = Vec::new();
    let row = sq / 8;
    let col = sq % 8;

    for dr in -1..=1 {
        for dc in -1..=1 {
            if dr == 0 && dc == 0 { continue; }
            let nr = row as i32 + dr;
            let nc = col as i32 + dc;
            if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
                neighbors.push((nr * 8 + nc) as usize);
            }
        }
    }
    neighbors
}

pub static KNIGHT_ADJACENCY: once_cell::sync::Lazy<[Vec<usize>; 64]> = once_cell::sync::Lazy::new(|| {
    let mut table = Vec::with_capacity(64);
    for sq in 0..64 {
        let mut moves = Vec::new();
        let r = (sq / 8) as i32;
        let c = (sq % 8) as i32;
        let diffs = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];
        for (dr, dc) in diffs {
            let nr = r + dr;
            let nc = c + dc;
            if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
                moves.push((nr * 8 + nc) as usize);
            }
        }
        table.push(moves);
    }
    table.try_into().unwrap()
});
