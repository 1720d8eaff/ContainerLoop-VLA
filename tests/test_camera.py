import numpy as np
from PIL import Image
from containerloop.camera import Camera,read_rgbd,corrected_times


def test_rgb_depth_and_clock_gaps(tmp_path):
    Image.fromarray(np.full((4,5,3),17,dtype=np.uint8)).save(tmp_path/'rgb.png')
    np.save(tmp_path/'depth.npy',np.full((4,5),1000,dtype=np.uint16))
    rgb,depth=read_rgbd(tmp_path/'rgb.png',tmp_path/'depth.npy')
    assert rgb[0,0,0]==17
    np.testing.assert_allclose(Camera(100,100,2,2).unproject(2,2,depth[2,2]),[0,0,1])
    times,info=corrected_times([0,.033,.1],offset_s=.02)
    assert info['gap_after_frame_indices']==[1] and times[0]==.02
