"""
Minimal direct-summation, softened N-body code (NumPy), N-body units (G = M = 1).
Plummer sphere, fixed-step kick-drift-kick leapfrog.
Usage examples:
  python nbody.py --name ref  --seed 1
  python nbody.py --name p12  --seed 1 --perturb 1e-12
  python nbody.py --name perm --seed 1 --variant permute
  python nbody.py --name mat  --seed 1 --variant matmul
"""
import argparse, os, time
import numpy as np

EPS = 0.05          # softening length
DT = 2.0**-7        # fixed time step
SAVE_EVERY = 16     # snapshot every 16 steps = 0.125 time units

def plummer(n, seed):
    rng = np.random.default_rng(seed)
    a = 3.0 * np.pi / 16.0                      # Plummer radius for virial radius = 1
    X = rng.uniform(1e-10, 1 - 1e-10, n)
    r = a / np.sqrt(X ** (-2.0 / 3.0) - 1.0)
    # isotropic positions
    z = rng.uniform(-1, 1, n); phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - z * z)
    pos = np.c_[r * s * np.cos(phi), r * s * np.sin(phi), r * z]
    # velocities by rejection sampling, q = v / v_esc
    q = np.empty(n)
    for i in range(n):
        while True:
            x1, x2 = rng.uniform(0, 1), rng.uniform(0, 0.1)
            if x2 < x1 * x1 * (1 - x1 * x1) ** 3.5:
                q[i] = x1; break
    vesc = np.sqrt(2.0) * (r * r + a * a) ** -0.25
    v = q * vesc
    z = rng.uniform(-1, 1, n); phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - z * z)
    vel = np.c_[v * s * np.cos(phi), v * s * np.sin(phi), v * z]
    m = np.full(n, 1.0 / n)
    pos -= np.average(pos, axis=0, weights=m); vel -= np.average(vel, axis=0, weights=m)
    return m, pos, vel

def acc(pos, m, variant):
    if variant == "matmul":
        # same physics, different arithmetic: pairwise matrix formulation
        sq = np.einsum("ij,ij->i", pos, pos)
        r2 = sq[:, None] + sq[None, :] - 2.0 * (pos @ pos.T)
        np.maximum(r2, 0.0, out=r2)
        w = (r2 + EPS ** 2) ** -1.5
        np.fill_diagonal(w, 0.0)
        w *= m[None, :]
        return w @ pos - w.sum(axis=1)[:, None] * pos
    dx = pos[None, :, :] - pos[:, None, :]
    r2 = np.einsum("ijk,ijk->ij", dx, dx) + EPS ** 2
    w = r2 ** -1.5
    np.fill_diagonal(w, 0.0)
    return np.einsum("ij,ijk->ik", w * m[None, :], dx)

def energy(pos, vel, m):
    ke = 0.5 * np.sum(m * np.einsum("ij,ij->i", vel, vel))
    dx = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", dx, dx) + EPS ** 2)
    iu = np.triu_indices(len(m), 1)
    pe = -np.sum((m[:, None] * m[None, :] / r)[iu])
    return ke, pe

def run(n, seed, name, tmax, perturb, variant, outdir):
    m, pos, vel = plummer(n, seed)
    if perturb:
        pos[0, 0] += perturb           # shift one star (index 0) by `perturb`
    ids = np.arange(n)
    if variant == "permute":
        perm = np.random.default_rng(12345).permutation(n)
        m, pos, vel, ids = m[perm], pos[perm], vel[perm], ids[perm]
    nsteps = int(round(tmax / DT))
    snaps_t, snaps_x, snaps_v, en = [], [], [], []
    a = acc(pos, m, variant)
    t0 = time.time()
    for k in range(nsteps + 1):
        if k % SAVE_EVERY == 0:
            order = np.argsort(ids)                 # always store sorted by star ID
            snaps_t.append(k * DT); snaps_x.append(pos[order].copy()); snaps_v.append(vel[order].copy())
            en.append(energy(pos, vel, m))
        vel += 0.5 * DT * a
        pos += DT * vel
        a = acc(pos, m, variant)
        vel += 0.5 * DT * a
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f"{name}.npz"), t=np.array(snaps_t),
                        x=np.array(snaps_x), v=np.array(snaps_v), energy=np.array(en),
                        n=n, seed=seed, perturb=perturb, variant=variant, eps=EPS, dt=DT)
    print(f"{name}: done in {time.time() - t0:.1f} s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=500); p.add_argument("--tmax", type=float, default=80.0)
    p.add_argument("--perturb", type=float, default=0.0)
    p.add_argument("--variant", default="default", choices=["default", "permute", "matmul"])
    p.add_argument("--out", default="runs")
    a = p.parse_args()
    run(a.n, a.seed, a.name, a.tmax, a.perturb, a.variant, a.out)
