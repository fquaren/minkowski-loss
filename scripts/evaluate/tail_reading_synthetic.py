# Count reading vs rank reading of a tail: noise AND signal, for a model that under-predicts.
# Target and model share the GPD tail shape xi; the model's intensities are scaled by k<1.
# Members = R independent patches from the model law; target = one patch from the true law.
import numpy as np
rng = np.random.default_rng(0)
H, xi, sig, R, P = 128, 0.155, 5.0, 16, 400          # P conditions, R members each
def gpd(U): return sig/xi*(U**-xi - 1)
def uniforms(n, corr):
    z = rng.standard_normal((n, H, H))
    if corr:                                           # Gaussian-smoothed field, 4 px scale
        k = np.fft.fftfreq(H)[:, None]**2 + np.fft.fftfreq(H)[None, :]**2
        z = np.real(np.fft.ifft2(np.fft.fft2(z) * np.exp(-2*(np.pi*4)**2*k)))
        z /= z.std(axis=(1, 2), keepdims=True)
    return 0.5*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3)))
for corr in (False, True):
    uq = {n: np.quantile(gpd(uniforms(200, True)).ravel(), 1 - n/H**2) for n in (5, 20, 100, 500)}
    for k in (1.0, 0.7):
        rows = {n: [] for n in (5, 20, 100, 500)}
        for p in range(P):
            y = gpd(uniforms(1, corr)).ravel()
            f = k*gpd(uniforms(R, corr)).reshape(R, -1)
            ys = np.sort(y)[::-1]; fs = -np.sort(-f, 1)
            for n in rows:
                u = gpd(n / H**2) if not corr else uq[n]       # fixed threshold, as in the reward
                nt = int((y >= u).sum())
                if nt == 0:
                    continue
                c = np.log1p((f >= u).sum(1)); v = np.log1p(fs[:, nt-1])
                rows[n].append((c.mean()-np.log1p(nt), c.std(ddof=1), v.mean()-np.log1p(ys[nt-1]), v.std(ddof=1)))
        print(f"FIXED u  spatially correlated={corr}  k={k}")
        for n, r in rows.items():
            r = np.array(r); sc = np.sqrt((r[:,1]**2).mean()); sv = np.sqrt((r[:,3]**2).mean())
            print(f"  n={n:4d}  count: bias {r[:,0].mean():+.3f} sd {sc:.3f} bias/sd {r[:,0].mean()/sc:+.2f} E|d|/sd {np.abs(r[:,0]).mean()/sc:.2f}"
                  f" | rank: bias {r[:,2].mean():+.3f} sd {sv:.3f} bias/sd {r[:,2].mean()/sv:+.2f} E|d|/sd {np.abs(r[:,2]).mean()/sv:.2f}")
