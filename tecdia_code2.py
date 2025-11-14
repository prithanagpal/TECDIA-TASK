import cv2
import numpy as np
import time

# parameters needed
INPUT_VIDEO = "tecdia video.mp4"   #update the filename accordingly
OUTPUT_VIDEO = "reconstructed_final_video.mp4"
DOWNSAMPLE = (64, 64)
MIN_LARGE_BLOCK = 40               
TAIL_SIZE = 30                     
TWO_OPT_ITERS_MAIN = 1200
TWO_OPT_ITERS_SMALL = 300
TWO_OPT_TAIL_ITERS = 2000
SEED = 0

np.random.seed(SEED)

# this is shrinking the frame and normailising the pixel values
# and now preparing it for MSE-based similarity on tiny frames
def preprocess(frame):
    f = cv2.resize(frame, DOWNSAMPLE)
    return f.astype(np.float32) / 255.0

# now we check the MSE similarity 
# lower the MSE, more is the similarity
# also convert MSE to similarity score: sim = 1 / (1 + MSE)
def mse_sim(f1, f2):
    mse = np.mean((f1 - f2) ** 2)
    return 1.0 / (1.0 + mse)

# next we will build symmetric similarity matrix 
def build_similarity(frames_small):
    N = len(frames_small)
    S = np.zeros((N, N), dtype=np.float32)

    print("[INFO] Computing similarity matrix...")
    for i in range(N):
        for j in range(i + 1, N):
            s = mse_sim(frames_small[i], frames_small[j])
            S[i, j] = S[j, i] = s

    np.fill_diagonal(S, -1)
    return S

# now we will use small reversed-segment fixer to see if the video is reversed 
def fix_reversed_segments(S, order, min_len=5, max_len=40):
    N = len(order)
    improved = True

    while improved:
        improved = False

        for length in range(min_len, max_len + 1):
            for start in range(0, N - length):
                end = start + length
                forward_score = 0.0
                reversed_score = 0.0

                for i in range(start, end):
                    forward_score += S[order[i], order[i + 1]]

                temp = order.copy()
                temp[start:end + 1] = list(reversed(temp[start:end + 1]))

                for i in range(start, end):
                    reversed_score += S[temp[i], temp[i + 1]]

                if reversed_score > forward_score + 1e-8:  # tiny tolerance
                    order[start:end + 1] = list(reversed(order[start:end + 1]))
                    improved = True
                    break
            if improved:
                break

    return order

# no we use multi large-block reversed detector for multiple large blocks
# we will attempt to reverse any block length >= MIN_LARGE_BLOCK that improves score.
def fix_large_reversed_blocks(S, order, min_block=MIN_LARGE_BLOCK):
    N = len(order)
    best_order = order.copy()
    best_score = path_score(S, best_order)

    improved_any = True
    iteration = 0
    print("[INFO] Detecting large reversed blocks (may detect multiple)...")
    while improved_any:
        improved_any = False
        iteration += 1
        for L in range(N - 1, min_block - 1, -1):
            # slide window
            for start in range(0, N - L + 1):
                end = start + L
                cand = best_order.copy()
                cand[start:end] = list(reversed(cand[start:end]))
                sc = path_score(S, cand)
                if sc > best_score + 1e-9:
                    print(f"[INFO] Large-block flip: start={start} len={L} improves {best_score:.5f}->{sc:.5f}")
                    best_order = cand
                    best_score = sc
                    improved_any = True
                    break
            if improved_any:
                break
    print(f"[INFO] Large-block detection finished after {iteration} scans.")
    return best_order

# now we initialise the nearest neighbours 
def nn_path(S):
    N = len(S)
    used = set()
    order = []

    start = int(np.argmax(np.sum(S, axis=1)))
    order.append(start)
    used.add(start)
    curr = start

    for _ in range(N - 1):
        nxt = int(np.argmax(S[curr]))
        loop_guard = 0
        while nxt in used:
            S[curr][nxt] = -1
            nxt = int(np.argmax(S[curr]))
            loop_guard += 1
            if loop_guard > N:
                # fallback: pick any unused
                unused = [i for i in range(N) if i not in used]
                if not unused:
                    break
                nxt = unused[0]
                break
        order.append(nxt)
        used.add(nxt)
        curr = nxt

    return order


# now we performing 2-opt optimization, which is used in TSP and sequence reconstruction to improve the smoothness/quality of
#the order
def path_score(S, order):
    return float(sum(S[order[i], order[i + 1]] for i in range(len(order) - 1)))


def two_opt(S, order, iterations=1000):
    N = len(order)
    best = order.copy()
    best_score = path_score(S, best)

    for _ in range(iterations):
        a = np.random.randint(0, N - 2)
        b = np.random.randint(a + 1, N - 1)
        if a >= b:
            continue
        cand = best.copy()
        cand[a:b + 1] = list(reversed(cand[a:b + 1]))
        sc = path_score(S, cand)
        if sc > best_score:
            best = cand
            best_score = sc

    return best

# if any glitches in the last end of the video thenthis function fixes that — usually the final 20–40 frames.
def fix_tail(S, order, tail_size=TAIL_SIZE, iters=TWO_OPT_TAIL_ITERS):
    N = len(order)
    if N <= 6:
        return order
    tail_size = min(tail_size, N // 2)
    start = N - tail_size
    tail = order[start:].copy()

    best = tail.copy()
    best_score = path_score(S, best) 

    for _ in range(iters):
        i = np.random.randint(0, len(tail) - 1)
        j = np.random.randint(i + 1, len(tail))
        cand = best.copy()
        cand[i:j + 1] = list(reversed(cand[i:j + 1]))
        full_cand = order[:start] + cand
        sc = path_score(S, full_cand)
        if sc > best_score + 1e-9:
            best = cand
            best_score = sc

    order[start:] = best
    return order

# now we write final video at 30 FPS as required
def write_video(frames, order, out=OUTPUT_VIDEO):
    if len(frames) == 0:
        print("[WARN] No frames to write.")
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for idx in order:
        writer.write(frames[idx])
    writer.release()
    print("[INFO] Saved:", out)

# now finally we move to main function 
def main():
    t0 = time.time()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {INPUT_VIDEO}")
        return

    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    N = len(frames)
    if N == 0:
        print("[ERROR] No frames found in video.")
        return

    print(f"[INFO] Total frames: {N}")

    frames_small = [preprocess(f) for f in frames]

    S = build_similarity(frames_small)

    print("[STEP] Initial NN path...")
    order = nn_path(S.copy())

    print("[STEP] Base 2-opt refinement...")
    order = two_opt(S, order, iterations=TWO_OPT_ITERS_MAIN)

    order = fix_large_reversed_blocks(S, order)

    print("[STEP] Fixing small reversed chunks...")
    order = fix_reversed_segments(S, order)

    print("[STEP] Final small 2-opt cleanup...")
    order = two_opt(S, order, iterations=TWO_OPT_ITERS_SMALL)

    print("[STEP] Tail refinement (glitch removal)...")
    order = fix_tail(S, order, tail_size=TAIL_SIZE)

    print("[STEP] Writing output video...")
    write_video(frames, order)

    print("[DONE] Total pipeline time: %.2f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
