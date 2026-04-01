import numpy as np

def fuse_center(boxes, scores, topk=4, max_dist=160):

    if boxes is None or len(boxes) == 0:
        return None

    idx = np.argsort(scores)[::-1][:topk]
    boxes = boxes[idx]
    scores = scores[idx]

    cx = (boxes[:,0] + boxes[:,2]) * 0.5
    cy = (boxes[:,1] + boxes[:,3]) * 0.5
    centers = np.stack([cx, cy], axis=1)

    w = scores / (np.sum(scores) + 1e-6)
    ref = np.sum(centers * w[:,None], axis=0)

    dist = np.linalg.norm(centers - ref, axis=1)
    keep = dist < max_dist

    if not np.any(keep):
        # fallback：取最高分
        best = np.argmax(scores)
        return int(cx[best]), int(cy[best]), float(scores[best])

    centers = centers[keep]
    scores = scores[keep]

    final = np.mean(centers, axis=0)
    final_score = float(np.max(scores))

    return int(final[0]), int(final[1]), final_score
