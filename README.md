import time
import numpy as np
import sys

sys.setrecursionlimit(10000000)

def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

# --- 1. MONOTONE CHAIN ---
def monotone_chain(points):
    pts = sorted(points.tolist())
    if len(pts) <= 2: return pts
    upper = []
    for p in pts:
        while len(upper) >= 2 and (upper[-1][0]-upper[-2][0])*(p[1]-upper[-2][1]) - (upper[-1][1]-upper[-2][1])*(p[0]-upper[-2][0]) <= 0:
            upper.pop()
        upper.append(p)
    lower = []
    for p in reversed(pts):
        while len(lower) >= 2 and (lower[-1][0]-lower[-2][0])*(p[1]-lower[-2][1]) - (lower[-1][1]-lower[-2][1])*(p[0]-lower[-2][0]) <= 0:
            lower.pop()
        lower.append(p)
    return upper[:-1] + lower[:-1]

# --- 2. QUICKHULL---
def quickhull(points):
    pts = points.tolist()
    if len(pts) < 3: return pts
    def find_hull(p_list, p1, p2):
        if not p_list: return []
        # En uzak noktayı bulma (Bu kısım 10M'de yavaş kalabilir)
        farthest_pt = max(p_list, key=lambda p: abs((p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])))
        s1 = [p for p in p_list if (farthest_pt[0]-p1[0])*(p[1]-p1[1]) - (farthest_pt[1]-p1[1])*(p[0]-p1[0]) > 0]
        s2 = [p for p in p_list if (p2[0]-farthest_pt[0])*(p[1]-farthest_pt[1]) - (p2[1]-farthest_pt[1])*(p[0]-farthest_pt[0]) > 0]
        return find_hull(s1, p1, farthest_pt) + [farthest_pt] + find_hull(s2, farthest_pt, p2)
    
    min_x = min(pts, key=lambda p: p[0])
    max_x = max(pts, key=lambda p: p[0])
    s1 = [p for p in pts if (max_x[0]-min_x[0])*(p[1]-min_x[1]) - (max_x[1]-min_x[1])*(p[0]-min_x[0]) > 0]
    s2 = [p for p in pts if (min_x[0]-max_x[0])*(p[1]-max_x[1]) - (min_x[1]-max_x[1])*(p[0]-max_x[0]) > 0]
    return [min_x] + find_hull(s1, min_x, max_x) + [max_x] + find_hull(s2, max_x, min_x)

def ugur_4_2_numpy(points):
    idx_min_x, idx_max_x = np.argmin(points[:, 0]), np.argmax(points[:, 0])
    idx_min_y, idx_max_y = np.argmin(points[:, 1]), np.argmax(points[:, 1])
    p1, p2, p3, p4 = points[idx_min_x], points[idx_max_y], points[idx_max_x], points[idx_min_y]

    mask1 = (p2[0]-p1[0])*(points[:, 1]-p1[1]) - (p2[1]-p1[1])*(points[:, 0]-p1[0]) > 0
    mask2 = (p3[0]-p2[0])*(points[:, 1]-p2[1]) - (p3[1]-p2[1])*(points[:, 0]-p2[0]) > 0
    mask3 = (p4[0]-p3[0])*(points[:, 1]-p3[1]) - (p4[1]-p3[1])*(points[:, 0]-p3[0]) > 0
    mask4 = (p1[0]-p4[0])*(points[:, 1]-p4[1]) - (p1[1]-p4[1])*(points[:, 0]-p4[0]) > 0
    
    candidates = points[mask1 | mask2 | mask3 | mask4]

    final_candidates = np.unique(np.vstack([candidates, [p1, p2, p3, p4]]), axis=0)
    return monotone_chain(final_candidates)


if __name__ == "__main__":
    N = 5000000 
    print(f" {N} Noktalık TEST ---")
    pts = np.random.uniform(0, 10000, (N, 2))

    start = time.perf_counter()
    res_u = ugur_4_2_numpy(pts)
    t_u = time.perf_counter() - start
    print(f"Uğur 4.2 (NumPy): {t_u:.4f} sn | Köşe: {len(res_u)}")

    # 2. Monotone Chain
    start = time.perf_counter()
    res_m = monotone_chain(pts)
    t_m = time.perf_counter() - start
    print(f"Monotone Chain : {t_m:.4f} sn | Köşe: {len(res_m)}")

    # 3. Quickhull 
    try:
        start = time.perf_counter()
        res_q = quickhull(pts)
        t_q = time.perf_counter() - start
        print(f" Quickhull      : {t_q:.4f} sn | Köşe: {len(res_q)}")
    except Exception as e:
        print(f" Quickhull: {e}")
