mod geometry_tables;
mod cga;
mod shadow;
mod field;
mod eval;
mod engine;

use std::io::{self, BufRead};
use cozy_chess::*;
use engine::AethelgardX;

fn main() {
    let mut engine = AethelgardX::new();
    let mut board = Board::default();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let cmd = line.map_err(|_| ()).unwrap_or_default();
        let parts: Vec<&str> = cmd.split_whitespace().collect();

        match parts.get(0) {
            Some(&"uci") => {
                println!("id name GeoChess Sentinel V5 (Aethelgard-X)");
                println!("id author Antigravity & User");
                println!("uciok");
            }
            Some(&"isready") => println!("readyok"),
            Some(&"ucinewgame") => {
                board = Board::default();
            }
            Some(&"position") => {
                if let Some(&"startpos") = parts.get(1) {
                    board = Board::default();
                    if let Some(&"moves") = parts.get(2) {
                        for &mv_str in &parts[3..] {
                            if let Ok(mv) = mv_str.parse::<Move>() {
                                board.play(mv);
                            }
                        }
                    }
                }
            }
            Some(&"go") => {
                engine.nodes = 0;
                engine.board = board.clone();
                let best_move = engine.get_best_move();
                println!("bestmove {}", best_move);
            }
            Some(&"quit") => break,
            _ => {}
        }
    }
}
