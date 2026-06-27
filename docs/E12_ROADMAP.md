# E12 — Retention policy + dynamic keys (stateful LLM path)

**Status:** in progress  
**North star:** Frozen model θ, system prompt once, **memory in external state** — not LoRA, not context replay.

---

## Two parallel tracks (same north star)

| Track | Mechanism | What it stores | Relation to KV inject |
|-------|-----------|----------------|------------------------|
| **Greenfield (E12)** | Kernel PUT/GET + LOG | Typed facts `fact.*`, dynamic `user.*` | Discrete, auditable — **what must be exactly true** |
| **Lane C** | Block merge → `session.carrier.h` | Compressed turn history in carrier | Evolution of PRI KV — **continuous session state** without replaying tokens |

We test both. Convergence target: **carrier for fluency + kernel for truth**.

---

## Memory tiers (no runtime training)

```
Every turn     → OBS + SEAL        → LOG (audit — “remember everything observed”)
Policy PLANT   → PUT + SEAL        → STORAGE (consolidated — retrievable)
GET            → WORKING.last_read → ephemeral hot read
Chitchat       → OBS only          → no new STORAGE slot
```

**Retention policy** = model/templates propose PLANT; kernel accepts or REVERT. **θ never changes.**

---

## E12a (now): dynamic `user.*` keys

| Feature | Module |
|---------|--------|
| Location plants | `memory/dynamic_plant.py` — “I live in Amsterdam” → `user.location` |
| Location queries | “where do I live?” → GET `user.location` |
| Generic remember | “remember favorite_color is blue” → `user.favorite_color` |
| Schema | `user.*` string slots, overwrite allowed |
| Policy | `deploy/policy.dynamic.json` (+ overflow/promote) |
| Live chat | `chat_v1` uses dynamic policy + encoder respects explicit slot/value |

---

## E12b (next): learned retention head

- Classifier: `SEAL | OBS_ONLY` on open text (train on synthetic “worth remembering” corpus)
- Proposes `(key, value)` without fixed slot list
- Still kernel-gated — no LoRA

---

## E12c (Lane C bridge)

See [`E12_LANE_C_BRIDGE.md`](E12_LANE_C_BRIDGE.md).

- `HybridSession`: greenfield `MachineState` + optional Lane C `SessionCarrier`
- Turn flow: NL → kernel ops → merge turn block into carrier
- System prompt once; neither skills nor history re-injected as text

---

## Commands

```bash
python -m greenfield.eval_e12_dynamic
python -m greenfield.validate_release   # + e12_dynamic gate
python -m greenfield.eval_chat_v1
```

---

## Exit criteria

- “I live in Amsterdam” → PUT `user.location`; “where do I live?” → correct answer from STORAGE
- Italian chitchat / unsupported Q → OBS-only, no spurious PUT
- 16+ gates green; Space live chat uses `policy.dynamic.json`
