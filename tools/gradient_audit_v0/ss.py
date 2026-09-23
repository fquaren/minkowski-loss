exec(open("tools/gradient_audit_v0/gn.py").read().split("ckpts = ")[0])
from src.losses.competing import ImageSSIMLoss, FFT2DKernelCRPSLoss
b=next(iter(dl)); Y=b[1].to(dev)[:,0:1]
tp=F.relu(torch.expm1(torch.clamp(Y*mv,max=7.0)))
print("SSIM loss(target,target) eps=1e-6:", ImageSSIMLoss()(Y,Y).item(), " eps=1e-12:", ImageSSIMLoss(eps=1e-12)(Y,Y).item())
print("SSIM loss(zeros,target):", ImageSSIMLoss()(torch.zeros_like(Y),Y).item())
print("spectral(target,target)", FFT2DKernelCRPSLoss()(Y,Y).item(), " spectral(0,target)", FFT2DKernelCRPSLoss()(torch.zeros_like(Y),Y).item())
print("frac pixels > drizzle", (tp>0.1).float().mean().item(), "sv", sv)
