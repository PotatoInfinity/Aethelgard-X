# **Aethelgard-X: The Technical Realization Whitepaper**
**Engine Class:** Post-Search / Geometric Solver
<img width="2602" height="1412" alt="image" src="https://github.com/user-attachments/assets/5eb07df9-8812-4065-96dd-6a4f60d3638d" />

---

## **1. Mathematical Foundation: The 5D CGA Kernel**

Traditional engines use integers (bitboards). Aethelgard-X uses **Conformal Geometric Algebra (CGA)** over $\mathbb{R}^{4,1}$. This is the physics engine of the board.

### **1.1 The Algebra ($Cl_{4,1}$)**
We operate in a 5D vector space. A "state" is a Multivector.
*   **Basis Vectors:** $e_1, e_2, e_3$ (Euclidean), $e_+$ (Origin), $e_-$ (Infinity).
*   **The Null Basis:** 
    *   $n_0 = \frac{1}{2}(e_- - e_+)$ (Representing the Origin)
    *   $n_\infty = e_- + e_+$ (Representing Infinity)
*   **Total Dimension:** A generic multivector in 5D space has $2^5 = 32$ orthogonal components (1 scalar, 5 vectors, 10 bivectors, 10 trivectors, 5 quadvectors, 1 pseudoscalar).

### **1.2 The Board Mapping (The Encoder)**
Each square $(x,y)$ on the board is mapped to a **Null Vector** $P$ in conformal space.
$$P(x,y) = n_0 + (x e_1 + y e_2) + \frac{1}{2}(x^2 + y^2) n_\infty$$
*   **Rust Optimization:** Pre-compute these 64 vectors (each 32 floats) into a constant lookup table `BOARD_Space[64]`.

### **1.3 The Bivector Field (Piece Representation)**
A piece is not a point; it is a **Blade**.
*   **The Rook Blade:** A line at infinity.
    $$L_{rook} = P \wedge e_1 \wedge n_\infty$$ (Horizontal influence)
*   **The Intersection (Vision):** To check if a square $T$ is attacked by the Rook $L$:
    $$\text{Check} = (L_{rook} \cdot T)$$
    *   In CGA, the inner product of a line and a point is **0** if the point lies on the line.
    *   **Logic:** If `abs(L . T) < Epsilon`, the square is visible. This is a branchless float comparison.

---

## **2. Memory Architecture: The Tensor Network**

We replace the "Evaluation Function" with a **Matrix Product State (MPS)**.

### **2.1 The Tensor Struct**
Each square holds a Tensor describing its local state and its entanglement with neighbors.
```rust
// Physical Dimension (d): 13 (Empty, P, N, B, R, Q, K - White/Black)
// Bond Dimension (chi): 10 (The "depth" of strategic compression)
struct SquareTensor {
    data: Array3<f32>, // Shape: [13, 10, 10] (Physical, Left-Bond, Right-Bond)
}
```

### **2.2 Recursive Contraction (The Evaluation)**
We define the board value $\Psi$ as the trace of the product of all tensors along the "Snake Path" (a path winding through all 64 squares).
*   **The Math:** $\Psi = \text{Tr}(A_1 \cdot A_2 \cdot \dots \cdot A_{64})$
*   **The Alpha Approximation:** Contracting 64 tensors is too slow for real-time. 
    *   **Optimization:** We use **Local Environment Contraction**. We only contract the $3 \times 3$ grid around the move destination to get a "Local Stability Score" ($\Delta S$).
    *   **Global Update:** We update the full network only on "Quiet" positions (no captures), similar to Lazy SMP.

---

## **3. Algorithmic Core: The Geodesic Flow**

This is the replacement for Minimax. We solve a pathfinding problem on a weighted graph.

### **3.1 The Metric Map ($G$)**
Before looking for moves, we generate a 64x64 Cost Matrix.
$$Cost(x, y) = \frac{1}{1 + \text{TensorStability}(y) + \text{MaterialValue}(y)}$$
*   High stability/material = Low Cost (High Gravity).
*   Occupied by Friendly = Infinite Cost (Wall).

### **3.2 The Fast Marching Method (FMM)**
Instead of looking at depth, we propagate a "Wavefront" from the opponent's King.

**Algorithm:**
1.  **Target:** Set $T(\text{EnemyKing}) = 0$. All other $T = \infty$.
2.  **Heap:** Push EnemyKing into a Priority Queue.
3.  **March:**
    ```rust
    while let Some(u) = queue.pop() {
        for v in neighbors(u) {
            let alt = T[u] + Cost(u, v);
            if alt < T[v] {
                T[v] = alt;
                queue.push(v);
            }
        }
    }
    ```
4.  **Gradient:** The "Best Move" is the one that moves from `CurrentSquare` to the neighbor with the lowest $T$ value.

---

## **4. The Safety Logic: The Adjoint Shadow**

This is the "Safety Net" that makes the engine robust.

### **4.1 The Shadow Engine**
A minimal Bitboard engine using `Magic Bitboards` for move generation.
*   **Constraint:** It must run at >10 MN/s (Million Nodes per Second).
*   **Search Type:** Principal Variation Search (PVS) with Quiescence Search.
*   **Depth Cap:** Hard-limited to depth 4.

### **4.2 The Veto Protocol (Asynchronous Rust)**
We use Rust's `crossbeam` channels to link the Manifold (Complex) and Shadow (Fast).

```rust
enum Message {
    Candidate(Move, f32), // Move + Manifold Confidence
    Veto(Move),           // Shadow rejection
    Approval(Move),       // Shadow acceptance
}

// The Supervisor Loop
fn supervisor(manifold: Manifold, shadow: Shadow) {
    let best_geometric_move = manifold.get_geodesic_move();
    
    // The Shadow Check
    let safety_score = shadow.probe(best_geometric_move, depth=4);
    
    if safety_score < -150 (centipawns) {
        // TACTICAL BLUNDER DETECTED
        manifold.apply_infinite_cost(best_geometric_move); // Create a "Wall"
        supervisor(manifold, shadow); // Recursively find the next best path
    } else {
        play_move(best_geometric_move);
    }
}
```

---

## **Code Structure (main.rs)**

```rust
fn main() {
    // 1. Initialize the 5D Manifold (Static Lookup Tables)
    let manifold = Manifold::init();
    
    // 2. Initialize the Shadow (Bitboards)
    let mut shadow = Shadow::init();
    
    // 3. UCI Loop
    loop {
        let input = read_uci();
        let board_state = parse_fen(input);
        
        // A. Update Manifold (CGA Rotors) - Instant
        let field = manifold.update_field(&board_state);
        
        // B. Calculate Gradient (The "Flow")
        let raw_move = field.geodesic_flow();
        
        // C. Shadow Veto (The Safety Net)
        let final_move = if shadow.verify(raw_move) {
            raw_move
        } else {
            // Apply Penalty and Recalculate
            field.add_barrier(raw_move);
            field.geodesic_flow()
        };
        
        println!("bestmove {}", final_move);
    }
}
```


By merging the **Retrocausal Bridge** (knowing the end) with **Simplicial Rigidity** (knowing the structure), Aethelgard-X does not play moves—it **curves the future** until the opponent’s defeat becomes a mathematical necessity of the board’s geometry. Aethelgard-X Alpha is the first engine to move from Searching for a Win to Observing the Inevitability of a Win. 
