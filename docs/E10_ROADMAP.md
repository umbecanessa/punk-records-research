# E10 — Kernel-native chatbot v1

**Status:** green (13/13 release gates)  
**Stack:** E6 opcodes · E9a NL transformer · E9b renderer transformer · deterministic kernel

---

## Architecture

```
User messages (free-form, multi-turn)
        │
        ▼
  E9a LearnedEventParser  ──► PLANT | QUERY | CHITCHAT
        │
        ├── CHITCHAT ──► OBS only (no storage write)
        ├── PLANT    ──► OBS → PUT → SEAL  (memory in STORAGE)
        └── QUERY    ──► OBS → GET → RENDER (E9b reads storage)
        │
        ▼
  TokenLedger — savings vs chat-LLM context replay
```

Language never writes state. The renderer only reads `RENDER` keys.

---

## Modules

| Piece | Path |
|-------|------|
| Session runner | `greenfield/chat_v1.py` |
| Eval + bench JSON | `greenfield/eval_chat_v1.py` |
| Space tab | `space/app.py` → Chat v1 (E10) |
| Gate | `chat_v1` in `validate_release` |

---

## Exit criteria

- Multi-turn script: plants + chitchat + queries
- `query_hits == queries`, `reverts == 0`
- Token savings ratio ≥ 35% (grows with session length)
- Memory survives across turns in STORAGE (not context)

---

## Commands

```bash
python -m greenfield.eval_chat_v1
python -m greenfield.validate_release
```

---

## Scale path (post-v1)

| Rung | NL front | Renderer | Window |
|------|----------|----------|--------|
| v1 (now) | 621K transformer | 830K transformer | 12-char |
| v1.1 | 50M | 350M | byte/char stream |
| v2 | 350M–1B | 1–2B | open phrasing corpus |

Hub publish when ready: **`validate_release` green → publish**.
