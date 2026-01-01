use cozy_chess::*;
use crate::shadow::ShadowGuard;
use crate::field::GeodesicField;
use crate::eval::GeotensorEvaluator;

pub struct AethelgardX {
    pub board: Board,
    pub shadow: ShadowGuard,
    pub evaluator: GeotensorEvaluator,
    pub nodes: u64,
}

impl AethelgardX {
    pub fn new() -> Self {
        Self {
            board: Board::default(),
            shadow: ShadowGuard::new(),
            evaluator: GeotensorEvaluator::new(None),
            nodes: 0,
        }
    }

    pub fn get_best_move(&mut self) -> Move {
        let side = self.board.side_to_move();
        let enemy_king_sq = self.board.king(!side);
        
        let mut field = GeodesicField::new();
        field.update_costs(&self.board);

        let active_sqs = self.get_our_piece_squares();
        if active_sqs.is_empty() {
            return self.standard_search_fallback();
        }

        let mut attempts = 0;
        loop {
            // Pick a representative piece type for pathfinding
            let piece_type = self.board.piece_on(Square::index(active_sqs[0]));
            
            // Primal Wave (from us)
            field.propagate(&active_sqs, piece_type, &self.board);
            // Retrocausal Wave (from Enemy King)
            field.propagate_retro(enemy_king_sq as usize, &self.board);
            
            if let Some(target_sq) = field.solve_flow(&active_sqs) {
                if let Some(mv) = self.find_move_to_target(target_sq) {
                    let feedback = self.shadow.probe_tactics(&self.board, mv);
                    if feedback.is_safe || attempts > 10 {
                        return mv;
                    } else {
                        // TACTICAL BLUNDER DETECTED: Project into manifold
                        for (sq, mass) in feedback.danger_squares {
                            let current_barrier = field.barriers.entry(sq).or_insert(0.0);
                            *current_barrier += mass;
                        }
                    }
                } else {
                    // Could not find a move to the ideal flow square (obstacle?)
                    field.costs[target_sq] += 10.0;
                }
            } else {
                return self.standard_search_fallback();
            }

            attempts += 1;
            if attempts > 30 {
                return self.standard_search_fallback();
            }
        }
    }

    fn get_our_piece_squares(&self) -> Vec<usize> {
        self.board.colors(self.board.side_to_move())
            .into_iter()
            .map(|sq| sq as usize)
            .collect()
    }

    fn find_move_to_target(&self, target_sq: usize) -> Option<Move> {
        let mut best_move = None;
        self.board.generate_moves(|moves| {
            for mv in moves {
                if mv.to as usize == target_sq {
                    best_move = Some(mv);
                    return true;
                }
            }
            false
        });
        best_move
    }

    fn standard_search_fallback(&mut self) -> Move {
        let mut best_move = None;
        let mut best_score = -i32::MAX;
        
        let mut moves = Vec::new();
        self.board.generate_moves(|mvs| {
            for mv in mvs {
                moves.push(mv);
            }
            false
        });

        for mv in moves {
            let mut next_board = self.board.clone();
            next_board.play(mv);
            // Use the advanced evaluator for the fallback search too
            let score = -self.advanced_search(&next_board, 3, -i32::MAX, i32::MAX);
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
        }
        best_move.expect("No legal moves!")
    }

    fn advanced_search(&mut self, board: &Board, depth: i32, mut alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if depth == 0 {
            return self.evaluator.evaluate(board);
        }

        let mut moves = Vec::new();
        board.generate_moves(|mvs| {
            for mv in mvs {
                moves.push(mv);
            }
            false
        });

        if moves.is_empty() {
            return if board.status() == GameStatus::Drawn { 0 } else { -20000 };
        }

        let mut best_score = -i32::MAX;
        for mv in moves {
            let mut next_board = board.clone();
            next_board.play(mv);
            let score = -self.advanced_search(&next_board, depth - 1, -beta, -alpha);
            if score >= beta { return beta; }
            if score > best_score {
                best_score = score;
                if score > alpha { alpha = score; }
            }
        }
        best_score
    }
}
