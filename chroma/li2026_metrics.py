"""Float64 CUDA evaluation of the same CIEDE2000 and valid-window SSIM."""
import torch
import torch.nn.functional as F


def lab(rgb):
    rgb = rgb.clamp(0, 1)
    linear = torch.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055).pow(2.4))
    matrix = rgb.new_tensor([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]])
    xyz = (linear @ matrix.T) / rgb.new_tensor([.95047, 1., 1.08883])
    delta = 6/29
    value = torch.where(xyz > delta**3, xyz.pow(1/3), xyz/(3*delta**2) + 4/29)
    return torch.stack((116*value[..., 1]-16, 500*(value[..., 0]-value[..., 1]),
                        200*(value[..., 1]-value[..., 2])), dim=-1)


def delta_e(a, b):
    l1, a1, b1 = a.unbind(-1)
    l2, a2, b2 = b.unbind(-1)
    c1, c2 = torch.hypot(a1, b1), torch.hypot(a2, b2)
    c7 = ((c1+c2)/2).pow(7)
    g = .5*(1-torch.sqrt(c7/(c7+25**7)))
    ap1, ap2 = (1+g)*a1, (1+g)*a2
    cp1, cp2 = torch.hypot(ap1, b1), torch.hypot(ap2, b2)
    hp1 = torch.remainder(torch.rad2deg(torch.atan2(b1, ap1)), 360)
    hp2 = torch.remainder(torch.rad2deg(torch.atan2(b2, ap2)), 360)
    dl, dc = l2-l1, cp2-cp1
    zero = cp1*cp2 <= 1e-15
    dh = torch.where(zero, 0., hp2-hp1)
    dh = torch.where(dh > 180, dh-360, dh)
    dh = torch.where(dh < -180, dh+360, dh)
    dh = 2*torch.sqrt(cp1*cp2)*torch.sin(torch.deg2rad(dh/2))
    lm, cm = (l1+l2)/2, (cp1+cp2)/2
    hs = hp1+hp2
    hm = torch.where(zero, hs, hs/2)
    hm = torch.where((~zero) & ((hp1-hp2).abs()>180) & (hs<360), (hs+360)/2, hm)
    hm = torch.where((~zero) & ((hp1-hp2).abs()>180) & (hs>=360), (hs-360)/2, hm)
    t = (1-.17*torch.cos(torch.deg2rad(hm-30))+.24*torch.cos(torch.deg2rad(2*hm))
         +.32*torch.cos(torch.deg2rad(3*hm+6))-.20*torch.cos(torch.deg2rad(4*hm-63)))
    theta = 30*torch.exp(-((hm-275)/25).square())
    rc = 2*torch.sqrt(cm.pow(7)/(cm.pow(7)+25**7))
    sl = 1+.015*(lm-50).square()/torch.sqrt(20+(lm-50).square())
    sc, sh = 1+.045*cm, 1+.015*cm*t
    rt = -torch.sin(torch.deg2rad(2*theta))*rc
    lt, ct, ht = dl/sl, dc/sc, dh/sh
    return torch.sqrt((lt.square()+ct.square()+ht.square()+rt*ct*ht).clamp_min(0))


@torch.inference_mode()
def perceptual(reference, candidate, device='cuda'):
    height, width = reference.shape[:2]
    de_sum = torch.zeros((), dtype=torch.float64, device=device)
    ss_sum = torch.zeros_like(de_sum)
    ss_count = 0
    coordinates = torch.arange(-5, 6, dtype=torch.float64, device=device)
    kernel = torch.exp(-coordinates.square()/(2*1.5**2))
    kernel = kernel/kernel.sum()
    kx = kernel.reshape(1, 1, 1, 11).expand(5, 1, 1, 11).contiguous()
    ky = kernel.reshape(1, 1, 11, 1).expand(5, 1, 11, 1).contiguous()
    coefficients = kernel.new_tensor([.299, .587, .114])
    for top in range(0, height, 256):
        bottom = min(height, top+256)
        y0, y1 = max(0, top-5), min(height, bottom+5)
        a = torch.as_tensor(reference[y0:y1].copy(), device=device, dtype=torch.float64)
        b = torch.as_tensor(candidate[y0:y1].copy(), device=device, dtype=torch.float64)
        de_sum += delta_e(lab(a[top-y0:bottom-y0]/255), lab(b[top-y0:bottom-y0]/255)).sum()
        x, y = a @ coefficients, b @ coefficients
        moments = torch.stack((x, y, x*x, y*y, x*y)).unsqueeze(0)
        blurred = F.conv2d(F.conv2d(moments, kx, groups=5), ky, groups=5)[0]
        mx, my, xx, yy, xy = blurred.unbind(0)
        vx, vy, covariance = xx-mx*mx, yy-my*my, xy-mx*my
        score = ((2*mx*my+2.55**2)*(2*covariance+7.65**2) /
                 ((mx*mx+my*my+2.55**2)*(vx+vy+7.65**2)))
        lo, hi = max(top, 5)-(y0+5), min(bottom, height-5)-(y0+5)
        if hi > lo:
            valid = score[lo:hi]
            ss_sum += valid.sum()
            ss_count += valid.numel()
    return {'ciede2000': float(de_sum.item()/(height*width)),
            'ssim_luma': float(ss_sum.item()/ss_count)}
