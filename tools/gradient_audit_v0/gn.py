import json, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/work/fquareng/ch2/minkowski-loss")
from src.utils import load_config, load_scaler_val, set_seed
from src.data.datasets import DeterministicSRDataset
from src.losses.competing import build_structural_loss
from src.models.unet import LogSpaceResidualUNet
from torch.utils.data import DataLoader, WeightedRandomSampler
set_seed(0)
cfg = load_config("/work/fquareng/ch2/minkowski-loss/config.yaml")
sv = load_scaler_val(cfg); ds = json.load(open(cfg["DEM_STATS"]))
dset = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], cfg["TRAIN_METADATA_FILE"], (ds["dem_mean"], ds["dem_std"]), sv, split="train", data_percentage=2.0, topology_mode="euler")
dl = DataLoader(dset, batch_size=128, sampler=WeightedRandomSampler(dset.sample_weights, len(dset), replacement=True), num_workers=4)  # was 8: node34 allows 8 cores in total
dev = "cuda"; mv = torch.tensor(sv, device=dev)
losses = {n: build_structural_loss(n, cfg, dev) for n in ["minkowski","spectral","ssim","wetarea","opticalflow"]}
W = {"minkowski":1e-4,"spectral":5e-4,"ssim":1.6e-4,"wetarea":2e-2,"opticalflow":4e-5}
ckpts = {"vanilla":"UNet_Ana_20260623_144958","mink1e-4":"UNet_Ana_20260731_100128"}
NB = int(sys.argv[1]) if len(sys.argv)>1 else 6
batches = []
for i,b in enumerate(dl):
    batches.append(b)
    if len(batches)>=NB: break
def gnorm(model, loss):
    model.zero_grad(); loss.backward(retain_graph=True)
    return torch.sqrt(sum((p.grad.float()**2).sum() for p in model.parameters() if p.grad is not None)).item()
for tag, run in ckpts.items():
    m = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
    m.load_state_dict(torch.load(f"/work/fquareng/ch2/minkowski-loss/runs/sr_analytical/{run}/unet_best.pth", map_location=dev, weights_only=False)["model_state_dict"])
    m.train()
    res = {k: [] for k in ["mse"]+list(losses)}; vals = {k: [] for k in res}
    for X,Y,G in batches:
        X,Y,G = X.to(dev),Y.to(dev),G.to(dev)
        P = m(X)
        pp = F.relu(torch.expm1(torch.clamp(P[:,0:1]*mv,max=7.0))); tp = F.relu(torch.expm1(torch.clamp(Y[:,0:1]*mv,max=7.0)))
        rp = F.relu(torch.expm1(torch.clamp(X[:,0:1]*mv,max=7.0)))
        lm = F.mse_loss(P,Y); res["mse"].append(gnorm(m,lm)); vals["mse"].append(lm.item())
        for n,fn in losses.items():
            la = fn(P[:,0:1],Y[:,0:1],pp,tp,gamma_target=G,reference_phys=rp,anneal_factor=0.1)
            res[n].append(gnorm(m,la)); vals[n].append(la.item())
    g0 = np.median(res["mse"]); v0 = np.median(vals["mse"])
    print(f"== {tag}: MSE value {v0:.3e}  ||grad MSE|| {g0:.3e}")
    for n in losses:
        g = np.median(res[n]); v = np.median(vals[n])
        print(f"{n:12s} val={v:.3e} ||g||={g:.3e}  lambda_gn=||gMSE||/||gaux||={g0/g:.3e}  used w={W[n]:.1e}  w*||g||/||gMSE||={W[n]*g/g0:.3e}  w*L/MSE={W[n]*v/v0:.3e}")
