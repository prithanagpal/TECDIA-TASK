**Video Frame Unjumbling System**

This project reconstructs the correct chronological order of frames from a jumbled or shuffled video.
The algorithm detects temporal inconsistencies, reversed blocks, and local glitches using a combination of:
Frame similarity analysis
Greedy initialization
Global and local optimization techniques
Reversal detection
Final-tail refinement

The output is a stable, forward-playing video at 30 FPS, closely matching the original sequence.

**Requirements**
Python 3.8+
pip package manager

**Install the following dependencies**
pip install opencv-python numpy

**How to run the code**
Place your input jumbled video in the same folder as the script.
Run the Python script.
Then your reconstructed video will be saved in the same folder as well.

**Algorithm Explanation**
1. Frame Preprocessing (Downsampling + Normalization)
Makes similarity computation fast 
Preserves general appearance
Suitable for MSE-based similarity

2. Similarity Metric (MSE → Smoothness Score)

Similarity between two frames:
sim = 1 / (1 + MSE(frame_i, frame_j))
Low MSE → frames more likely to be similar and consecutive
Extremely fast
Works well on downsampled images
Robust to motion blur and duplicates
No ML model required

3. Similarity Matrix Construction

A full NxN (300×300) matrix is created
This matrix becomes the foundation for:
Ordering
Reversal detection
Optimization

4. Initial Path Construction (Nearest Neighbor Heuristic)

Start from the “most central” frame
and repeatedly pick the most similar next frame.
This creates a rough first ordering, fast.

5. 2-opt Optimization (Local Improvement)

2-opt is a classic TSP optimization:
Randomly reverse a segment of the sequence
Keep it if the overall video smoothness improves
This hels in fixing:
Small reversed sequences
Local ordering errors
Zig-zag patterns

6. Large Reversed Block Detection (Global Optimization)

Some videos might have big segments reversed
Local 2-opt can’t fix big reversals, so the algorithm:
Tries reversing every large block
Accepts the reversal if smoothness improves
Repeats until no further improvement

7. Small Reversed Segment Fixer

After large blocks are corrected, small segment misplacements remain.
This pass checks subsequences 5–40 frames long
and flips them if forward flow improves.

8. End-Glitch Removal

The last 20–40 frames often remain ambiguous.
So we run a strong local 2-opt on only the tail.
This removes:
End-of-video flickering
Local jitter


**Why This Method?**
Generalizes to ANY video without pretraining.
it has high accuracy and is fast
300 frames processed in 6–12 seconds on a laptop.

it handles:
Global reverse
Local reverse
Multi-block reverse
Noise
Low-motion sections


**Design Considerations**
1. Accuracy
Downsampled MSE is stable
Global + local reversal detection
2-opt ensures local smoothness improvement
Tail refinement removes final jitter

2. Time Complexity
Similarity matrix: O(N²)
NN ordering: O(N²)
Large block checking: O(N²)
2-opt: O(iterations × block-size)
With N = 300, all steps remain fast.

3. Parallelism
Similarity matrix can be parallelized if needed
Reversal checks can be parallelized

4. Robustness
Deterministic via random seed
Handles corrupted, blurred, or duplicated frames
Fully generic for any input video


**Execution Time Log**
[INFO] Loading video...
[INFO] Total frames: 300
[TIME] Video loading: 0.998 sec
[TIME] Preprocessing: 0.139 sec
[INFO] Computing similarity matrix...
[TIME] Similarity matrix: 0.457 sec
[STEP] Initial NN path...
[TIME] NN ordering: 0.001 sec
[STEP] Base 2-opt...
[TIME] Base 2-opt: 0.058 sec
[INFO] Detecting large reversed blocks...
[INFO] Flip: start=0, len=160, score 298.8851 → 298.8950
[INFO] Large block detection completed in 2 scans.
[TIME] Large-block fix: 1.935 sec
[STEP] Small reversed blocks...
[TIME] Small-block fix: 0.076 sec
[STEP] Final 2-opt...
[TIME] Final 2-opt: 0.014 sec
[STEP] Tail refinement...
[TIME] Tail refinement: 0.089 sec
[INFO] Saved: reconstructed_final_video.mp4
[TIME] Video writing: 3.680 sec
[DONE] TOTAL TIME: 7.447 sec
