from shinka.edit import apply_diff_patch


patch_str_1 = """
<<<<<<< SEARCH
import numpy as np


def construct_packing():
=======
import numpy as np

# Optional LP solver for radii (used if SciPy is available)
try:
    from scipy.optimize import linprog
except Exception:
    linprog = None


def construct_packing():
>>>>>>> REPLACE
</DIFF>


<NAME>dist_matrix_precompute_for_radii</NAME>
<DESCRIPTION>Speed up the radii computation by precomputing the pairwise distance matrix once and reusing it in both the LP (when available) and the fallback loop. This reduces repeated distance calculations (norms) for the same center pairs and improves runtime reliability without changing the outcome for a fixed set of centers.</DESCRIPTION>
<DIFF>
<<<<<<< SEARCH
    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
    return centers, radii
=======
    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
    return centers, radii
>>>>>>> REPLACE
"""


patch_str_2 = '''
<<<<<<< SEARCH
def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
=======
def compute_max_radii(centers, tol=1e-9, max_iter=1000):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2)

    Returns:
        np.array of shape (n,) with radius of each circle
    """
    n = centers.shape[0]
    # Precompute pairwise distances
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # Boundary distance constraints
    border_dist = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])

    # Initial guess for radii (some slack inside borders)
    x0 = np.clip(border_dist * 0.9, 0.0, border_dist)

    radii = None
    # Try to solve a global max-sum-radii problem using SciPy (SLSQP)
    try:
        from scipy.optimize import minimize
        bounds = [(0.0, bd) for bd in border_dist]

        def objective(r):
            return -np.sum(r)

        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                constraints.append({'type': 'ineq',
                                    'fun': lambda r, i=i, j=j, d=d: d - (r[i] + r[j])})

        res = minimize(objective, x0, bounds=bounds, constraints=constraints,
                       method='SLSQP', options={'ftol': 1e-9, 'maxiter': max_iter})
        if res.success:
            radii = np.clip(res.x, 0.0, border_dist)
    except Exception:
        radii = None

    if radii is not None:
        return radii

    # Fallback simple relaxation if SciPy not available or failed
    radii = np.ones(n)
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
>>>>>>> REPLACE
'''


def test_edit():
    result = apply_diff_patch(
        original_path="tests/circle.py",
        patch_str=patch_str_1,
        patch_dir=None,
    )
    updated_str, num_applied, output_path, error, patch_txt, diff_path = result
    print(error)
    assert num_applied == 2
    assert output_path is None
    assert error is None


def test_edit_2():
    result = apply_diff_patch(
        original_path="tests/circle.py",
        patch_str=patch_str_2,
        patch_dir=None,
    )
    updated_str, num_applied, output_path, error, patch_txt, diff_path = result
    print(error)
    assert num_applied == 1
    assert output_path is None
    assert error is None
