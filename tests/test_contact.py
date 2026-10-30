import numpy as np
from containerloop.contact_benchmark import episode


def test_contact_simulation_is_measured_and_condition_sensitive(tmp_path):
    flags={'feedback':True,'geometry':True,'smoothing':True}
    c={'id':'unit','mass_kg':.08,'friction':1.0,'offset_xy_m':[0,0]}
    result=episode(c,flags,tmp_path/'trace.csv')
    assert result['source']=='mujoco_simulation' and result['steps']==2400
    assert result['contact_frames']>0 and result['wall_time_s']>0
    assert (tmp_path/'trace.csv').stat().st_size>1000
    far=episode({**c,'offset_xy_m':[.28,.28]}, {**flags,'feedback':False})
    assert not far['success']
    assert not np.allclose(result['final_position_m'],far['final_position_m'])
