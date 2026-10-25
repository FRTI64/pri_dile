#!/usr/bin/env python3
"""Additional opponent types: memory-one, FSMs, reactive, ZD and tuned smart variants."""
from __future__ import annotations
import random
import ipd_extended as ext
from ipd_sim import C, D, Strategy
from ipd_extended import MemoryOne, Reactive, RandomFSM, ExtortZD, GenerousZD
from ipd_field import GenerousTFT, NoisyTFT, Joss, RandomP
from search_common import ParamPolicy, ParamHarvest
from train_multiobj import ParamResp
from p2_skeleton_search import GapStartPolicy
from run_smart_opponents import SmartAdapter, SMART

class _Wrap(Strategy):
    def __init__(self, name, inner):
        self.name=name; self.inner=inner
    def reset(self): self.inner.reset()
    def act(self,t,mine,theirs):
        if hasattr(self.inner,'act'): return self.inner.act(t,mine,theirs)
        return self.inner.move(t+1,mine,theirs)

def _memone(seed):
    r=random.Random(seed)
    # moderate TFT-like random memory-one
    p=(r.uniform(.7,1), r.uniform(0,.5), r.uniform(.4,1), r.uniform(0,.3))
    return lambda rr: MemoryOne(p, C, f"MemOne.F{seed}", rr)

def _reactive(seed):
    r=random.Random(seed)
    return lambda rr: Reactive(C, r.uniform(.4,1), r.uniform(0,.6),
                               f"Reactive.F{seed}", rr)

PLUS=[]
def add(name, weight, fac, family="plus"):
    PLUS.append(ext.TypeSpec(name, family, weight, fac))

for i in range(8):
    add(f"MemOne.F{i}", .8, _memone(100+i))
for i in range(8):
    add(f"Reactive.F{i}", .8, _reactive(200+i))
for i in range(6):
    seed=300+i
    add(f"FSM.F{i}", .7, lambda r, seed=seed: RandomFSM(random.Random(seed), 3))
for chi in (1.2, 1.5, 2.0, 2.5, 3.0):
    add(f"Extort.F{chi}", .5, lambda r, chi=chi: ExtortZD(chi, r))
for k in (1.2, 2.0, 3.0):
    add(f"GenZD.F{k}", .4, lambda r, k=k: GenerousZD(k, r))
for g in (.02,.07,.13,.22,.30,.45,.60):
    add(f"GTFT.F{g}", .5, lambda r,g=g: GenerousTFT(g, True, r))
for p in (.015,.04,.07,.11,.16,.22,.30):
    add(f"Joss.F{p}", .4, lambda r,p=p: Joss(p,r))
    add(f"Noisy.F{p}", .4, lambda r,p=p: NoisyTFT(p,r))
for p in (.08,.18,.28,.42,.55,.75,.85):
    add(f"Random.F{p}", .3, lambda r,p=p: RandomP(p,r))

# Tuned smart variants as opponents.
FINAL_TUNED={
 "P_UNPROVOKED":.2395699546614404,"P_MUTUAL":.6399155368157192,
 "P_RESPONSIVE":.9544467968114831,"MIN_C":5,"MIN_D":2,
 "RESP_GAP":.3871088255996472,"DEFECT_RATE":.1170465108152342,
 "PEACE_TESTS":1,"PEACE_COOLDOWN":9}
RESP_TUNED={"g_unprov":.31711381663242244,"g_mutual":.26173130084075913,
            "g_resp":.9804446329081643,"min_c":5,"min_d":2,
            "resp":.46788262082576104,"rate":.197064176521839,
            "max_tests":3,"cooldown":14}
add("FinalTuned", 1.2,
    lambda r: SmartAdapter("FinalTuned", ParamPolicy(seed=r.randrange(1<<30), **FINAL_TUNED)),
    family="smart")
add("RespTuned", 1.0,
    lambda r: SmartAdapter("RespTuned", ParamResp(seed=r.randrange(1<<30), **RESP_TUNED)),
    family="smart")
for start in (21, 35, 50, 75):
    add(f"Harvest{start}", .7,
        lambda r,start=start: SmartAdapter(f"Harvest{start}",
             ParamHarvest(seed=r.randrange(1<<30), HARVEST_START=start)),
        family="smart")
for th in (.45,.55,.64):
    add(f"Gap{th}", .6,
        lambda r,th=th: SmartAdapter(f"Gap{th}", GapStartPolicy(1,th)),
        family="smart")

ALL_SPECS = list(ext.TYPE_BANK) + list(SMART) + list(PLUS)
# De-duplicate by name.
_by_name={s.name:s for s in ALL_SPECS}
ALL_SPECS=list(_by_name.values())
