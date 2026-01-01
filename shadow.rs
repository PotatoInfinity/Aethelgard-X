use cozy_chess::*;

pub struct ShadowGuard {
    pub nodes: u64,
}

pub struct TacticalFeedback {
    pub is_safe: bool,
    pub danger_squares: Vec<(usize, f32)>,
}
impl ShadowGuard {
    pub fn new() -> Self {
        Self { nodes: 0 }
    }

    /// The Veto Protocol: Checks if a manifold move is tactically "insane"
    pub fn verify_move(&mut self, board: &Board, mv: Move) -> bool {
        self.probe_tactics(board, mv).is_safe
    }

    pub fn probe_tactics(&mut self, board: &Board, mv: Move) -> TacticalFeedback {
        let mut next_board = board.clone();
        next_board.play(mv);

        if next_board.status() == GameStatus::Won {
            return TacticalFeedback { is_safe: true, danger_squares: Vec::new() };
        }
        
        // Find the opponent's best response
        let (score, best_response) = self.search_with_move(&next_board, 4, -30000, 30000);
        
        let is_safe = -score > -50;
        let mut danger_squares = Vec::new();

        if !is_safe {
            // The opponent has a response that hurts us.
            // The destination of that move is a "Danger Square".
            if let Some(res_mv) = best_response {
                danger_squares.push((res_mv.to as usize, 1000.0));
                // Also mark the 'from' square as dangerous if it was a capture
                if board.color_on(res_mv.to).is_some() {
                    danger_squares.push((res_mv.from as usize, 500.0));
                }
            }
        }

        TacticalFeedback { is_safe, danger_squares }
    }

    pub fn search_with_move(&mut self, board: &Board, depth: i32, mut alpha: i32, beta: i32) -> (i32, Option<Move>) {
        self.nodes += 1;
        if depth == 0 {
            return (self.quiescence(board, alpha, beta), None);
        }

        let mut best_move = None;
        let mut best_score = -30000;
        let mut moves = Vec::new();
        board.generate_moves(|mvs| {
            for mv in mvs {
                moves.push(mv);
            }
            false
        });

        if moves.is_empty() {
            return (if board.status() == GameStatus::Drawn { 0 } else { -20000 }, None);
        }

        for mv in moves {
            let mut next_board = board.clone();
            next_board.play(mv);
            let (score, _) = self.search_with_move(&next_board, depth - 1, -beta, -alpha);
            let score = -score;
            if score >= beta { return (beta, Some(mv)); }
            if score > alpha {
                alpha = score;
                best_score = score;
                best_move = Some(mv);
            }
        }
        (best_score, best_move)
    }

    pub fn search(&mut self, board: &Board, depth: i32, alpha: i32, beta: i32) -> i32 {
        self.search_with_move(board, depth, alpha, beta).0
    }

    fn quiescence(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        let stand_pat = self.eval(board);
        if stand_pat >= beta { return beta; }
        if stand_pat > alpha { alpha = stand_pat; }

        let mut moves = Vec::new();
        let occupied = board.occupied();
        board.generate_moves(|mut mvs| {
            mvs.to &= occupied;
            for mv in mvs {
                moves.push(mv);
            }
            false
        });

        for mv in moves {
            let mut next_board = board.clone();
            next_board.play(mv);
            let score = -self.quiescence(&next_board, -beta, -alpha);
            if score >= beta { return beta; }
            if score > alpha { alpha = score; }
        }
        alpha
    }

    fn eval(&self, board: &Board) -> i32 {
        let mut score = 0;
        let us = board.side_to_move();
        let them = !us;

        let pieces = [
            (Piece::Pawn, 100),
            (Piece::Knight, 320),
            (Piece::Bishop, 330),
            (Piece::Rook, 500),
            (Piece::Queen, 900),
        ];

        let our_color = board.colors(us);
        let their_color = board.colors(them);

        for (p, val) in pieces {
            score += (board.pieces(p) & our_color).len() as i32 * val;
            score -= (board.pieces(p) & their_color).len() as i32 * val;
        }
        score
    }
}
