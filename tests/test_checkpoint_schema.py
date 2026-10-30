import numpy as np
import pytest
from containerloop.checkpoint import assemble,validate_actions


def test_actual_observation_arrays_and_action_validation():
    obs=assemble({'wrist':np.full((1,2,16,16,3),128,dtype=np.uint8)}, {'joints':np.ones((1,2,7))},'pick')
    assert obs['video']['wrist'][0,0,0,0,0]==128
    assert obs['state']['joints'].dtype==np.float32
    assert validate_actions({'arm':np.ones((1,4,7))})['arm'].shape==(1,4,7)
    with pytest.raises(ValueError): validate_actions({'arm':np.full((1,4,7),np.nan)})
