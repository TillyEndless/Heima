import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/progressive_suite"))
import train_protocol as t
class T:
    eos_token="<eos>"; pad_token_id=0; eos_token_id=0
    def __call__(self, s, add_special_tokens=False): return {"input_ids": list(range(1, max(2, len(str(s).split())+1)))}
    def convert_tokens_to_ids(self,x): return {k:i+10 for i,k in enumerate(t.TOKENS)}[x]
    def decode(self,ids,skip_special_tokens=False): return " ".join(map(str,ids))
def row(): return {"sample_id":"0","question":"q","gold_cot":"a b c d e f g h","answer":"#### 1","cot_chunks":["a b","c d","e f","g h"]}
def test_progressive_stage_construction():
    ids,labs,segs=t.make_sequence(T(),row(),"P1_PROGRESSIVE_4_TYPED",2)
    assert segs.count("think")==2 and any(s=="text3" for s in segs)
def test_oneshot_typed_has_four_think_and_boundary():
    ids,labs,segs=t.make_sequence(T(),row(),"P2_ONESHOT_4_TYPED",4)
    assert segs.count("think")==4 and segs.count("boundary")==2
def test_no_latent_ntp_masks_think():
    ids,labs,segs=t.make_sequence(T(),row(),"P5_PROGRESSIVE_4_TYPED_NO_LATENT_NTP",4)
    assert all(l==-100 for l,s in zip(labs,segs) if s=="think")
def test_fixed_k():
    ids,labs,segs=t.make_sequence(T(),row(),"FIXED_K64",4)
    assert segs.count("think")==64
