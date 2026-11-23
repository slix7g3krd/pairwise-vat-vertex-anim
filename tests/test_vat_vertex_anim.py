import pytest
import graphics.vat_vertex_anim as mod

CASES = [
    ("frame_rate_map", (1,)),
    ("loop_seam", (1,)),
    ("normal_precision", (1,)),
    ("instance_share", (1,)),
    ("determinism", (1,)),
    ("texture_lifetime", (1,)),
    ("sample", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("blend", (1,)),
    ("metrics", (1,)),
    ("batch_upload", (1,)),
    ("lod_switch", (1,)),
    ("bounds", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
