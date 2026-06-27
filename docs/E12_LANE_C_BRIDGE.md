# E12 — Greenfield ↔ Lane C bridge (design)

**Goal:** One session, two memories — **exact facts in kernel**, **compressed narrative in carrier**.

---

## Why both

| | Greenfield kernel | Lane C carrier |
|--|-------------------|----------------|
| **Analog** | Hippocampal index + ledger | Cortical summary |
| **Recall** | GET by key — exact | Conditioned decode — fluent |
| **Cost** | O(1) per fact read | O(1) merge per turn vs O(n) text replay |
| **PRI/KV** | Not KV paste | Native merge replaces inject-resume |

LoRA is **out of scope** — both paths mutate **session state**, not **θ**.

---

## Target turn loop (production shape)

```
1. user_text
2. greenfield: OBS → parse → PUT/GET/CHITCHAT → STORAGE', LOG'
3. lane_c: encode turn block → merge(session.carrier, block) → carrier'
4. render: GET values from STORAGE + decode with carrier' (future joint renderer)
5. return reply, HybridSession{machine_state, carrier, blocks}
```

System prompt: set once at `genesis` / `new_session`. Skills: `PUT` tool handles + `RUN` — never re-read markdown.

---

## Implementation phases

| Phase | Deliverable |
|-------|-------------|
| **E12c.0** | Design + `session_bridge.py` stub (types only) |
| **E12c.1** | Export greenfield log tail → turn block text for Lane C training data |
| **E12c.2** | Dual eval: G-recall (kernel) + G-geo (carrier parity) on same sessions |
| **E12c.3** | Unified REPL / Space tab “Hybrid session” |

---

## What we are not doing

- Session LoRA / online fine-tune
- PRI raw KV paste as memory policy (decode optimization only, deferred in ARCHITECTURE.md)
- Replacing kernel STORAGE with carrier alone (loses audit + exact recall)

---

## Reference code

- Lane C runtime: `lane_c/runtime/session.py` — `send()`, carrier fast-path
- Greenfield chat: `greenfield/chat_v1.py` — `ChatSessionState`
- Bridge stub: `greenfield/session_bridge.py`
