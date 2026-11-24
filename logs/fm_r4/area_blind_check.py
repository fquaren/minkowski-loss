# Is the quantile_mass area channel blind to predicted values above the target maximum?
import sys; sys.path.insert(0, '.')
import numpy as np, torch
sys.argv = ['x']
from tests.test_reward_form import U, ANNEAL, _cell, _target
from src.losses.minkowski import AnalyticalMinkowskiLoss
from src.trainers.reward_finetune import minkowski_reward
geom = AnalyticalMinkowskiLoss(physical_thresholds=np.array(U, np.float32), pixel_size_km=2.0,
                               topology_mode="euler", area_mode="ste")
y = _cell(30.0)                       # target max ~30 mm: thresholds 31.7, 50.1, 79 are above it
tg = _target(geom, y)
def members(spike):
    p = torch.cat([_cell(30.0, seed=s) for s in range(4)], 0)
    p[:, 0, 24, 24] += spike          # raise the peak pixel of every member
    return p.unsqueeze(1)             # [M,1,1,H,W]
for name, cw in [("area-only [1,0,0]", [1., 0., 0.]), ("full [1,1,1]", [1., 1., 1.])]:
    geom.channel_weights = torch.tensor(cw)
    out = []
    for s in [0.0, 20.0, 100.0, 300.0]:
        p = members(s).requires_grad_(True)
        r = minkowski_reward(geom, p, tg, "energy", ANNEAL, 0.0, "quantile_mass", y)
        r.backward()
        out.append(f"spike +{s:.0f}: reward {r.item():.5f}, dR/dpeak {p.grad[:,0,0,24,24].mean().item():+.2e}")
    print(name); print("  " + "\n  ".join(out))
