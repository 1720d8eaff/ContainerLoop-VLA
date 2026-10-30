"""Calibrated RGB-D frame loading, Brown-Conrady unprojection and clock diagnostics."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Camera:
    fx:float
    fy:float
    cx:float
    cy:float
    distortion:tuple=(0.,0.,0.,0.,0.)
    depth_scale_m:float=.001

    def __post_init__(self):
        if len(self.distortion)!=5 or not np.isfinite([self.fx,self.fy,self.cx,self.cy,self.depth_scale_m,*self.distortion]).all() or min(self.fx,self.fy,self.depth_scale_m)<=0:raise ValueError('invalid calibration')

    def unproject(self,u,v,depth):
        xd=(np.asarray(u)-self.cx)/self.fx;yd=(np.asarray(v)-self.cy)/self.fy
        x,y=xd.copy(),yd.copy();k1,k2,p1,p2,k3=self.distortion
        for _ in range(12):
            r2=x*x+y*y;radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
            x,y=(xd-2*p1*x*y-p2*(r2+2*x*x))/radial,(yd-p1*(r2+2*y*y)-2*p2*x*y)/radial
        z=np.asarray(depth)*self.depth_scale_m
        if np.any(z<=0) or not np.isfinite(np.r_[np.ravel(x),np.ravel(y),np.ravel(z)]).all():raise ValueError('invalid depth or distortion')
        return np.stack([x*z,y*z,z],axis=-1)


def read_rgbd(rgb_path,depth_path):
    from PIL import Image
    rgb=np.asarray(Image.open(rgb_path).convert('RGB'))
    depth=np.load(depth_path,allow_pickle=False) if Path(depth_path).suffix=='.npy' else np.asarray(Image.open(depth_path))
    if depth.ndim!=2 or rgb.shape[:2]!=depth.shape or not np.isfinite(depth).all():raise ValueError('RGB-D size or values invalid')
    return rgb,depth


def corrected_times(timestamps,scale=1.,offset_s=0.,expected_period_s=1/30):
    t=np.asarray(timestamps,dtype=float)
    if t.ndim!=1 or not np.isfinite(t).all() or not np.isfinite([scale,offset_s,expected_period_s]).all() or scale<=0 or expected_period_s<=0:raise ValueError('invalid clock parameters')
    t=t*scale+offset_s
    if np.any(np.diff(t)<=0):raise ValueError('timestamps must increase')
    gaps=np.flatnonzero(np.diff(t)>1.5*expected_period_s).tolist()
    return t,{'gap_after_frame_indices':gaps,'clock_scale':scale,'clock_offset_s':offset_s,'expected_period_s':expected_period_s}
