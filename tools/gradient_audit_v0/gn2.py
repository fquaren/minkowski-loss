exec(open("tools/gradient_audit_v0/gn.py").read().split("ckpts = ")[0])
NB=6
batches=[]
for b in dl:
    batches.append(b)
    if len(batches)>=NB: break
runs = {"vanilla":"UNet_Ana_20260623_144958","mink":"UNet_Ana_20260731_100128","spectral":"UNet_Ana_20260829_033325","ssim":"UNet_Ana_20260824_155753","wetarea":"UNet_Ana_20260826_041128","opticalflow":"UNet_Ana_20260827_151319","vanilla34ep":"UNet_Ana_20260722_081033"}
def flat(m): return torch.cat([p.grad.flatten().float() for p in m.parameters()])
out={}
for tag,run in runs.items():
    m = LogSpaceResidualUNet(in_channels=2, out_channels=1).to(dev)
    import os
    pth=f"/work/fquareng/ch2/minkowski-loss/runs/sr_analytical/{run}/unet_best.pth"
    if not os.path.exists(pth): print("no ckpt",tag); continue
    m.load_state_dict(torch.load(pth, map_location=dev, weights_only=False)["model_state_dict"]); m.train()
    vals={k:[] for k in ["mse","mae_phys"]+list(losses)}; cos={k:[] for k in losses}
    for X,Y,G in batches:
        X,Y,G=X.to(dev),Y.to(dev),G.to(dev); P=m(X)
        pp=F.relu(torch.expm1(torch.clamp(P[:,0:1]*mv,max=7.0))); tp=F.relu(torch.expm1(torch.clamp(Y[:,0:1]*mv,max=7.0))); rp=F.relu(torch.expm1(torch.clamp(X[:,0:1]*mv,max=7.0)))
        lm=F.mse_loss(P,Y); vals["mse"].append(lm.item()); vals["mae_phys"].append((pp-tp).abs().mean().item())
        m.zero_grad(); lm.backward(retain_graph=True); gm=flat(m).clone()
        for n,fn in losses.items():
            la=fn(P[:,0:1],Y[:,0:1],pp,tp,gamma_target=G,reference_phys=rp,anneal_factor=0.1); vals[n].append(la.item())
            if tag=="vanilla":
                m.zero_grad(); la.backward(retain_graph=True); ga=flat(m); cos[n].append(F.cosine_similarity(gm,ga,dim=0).item())
    print(tag, " ".join(f"{k}={np.mean(v):.4e}" for k,v in vals.items()))
    if tag=="vanilla": print("  cos(gMSE,gaux):", " ".join(f"{k}={np.mean(v):+.3f}" for k,v in cos.items()))
