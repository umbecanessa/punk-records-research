# Punk Records Research — Agent Kernel ABI (v0.1)

Public opcode and storage summary for the release weights bundle.

## Machine model

```
LOG (immutable)       hash-linked event chain
STORAGE (mutable)     typed slots — writes only via kernel
WORKING (ephemeral)   percept + hot reads — cleared on SEAL / REVERT
KERNEL                (state, op, args) → state' | REVERT
RENDERER (untrusted)  reads storage → text; never writes state
```

Training ground truth: simulator world `W`, not chat logs.  
Eval ground truth: `read(state, key) == W[key]`.

## Core opcodes

| Opcode | Role |
|--------|------|
| **OBS** | Record observation into working percept |
| **GET** | Read slot into working set |
| **PUT** | Write fact slot (requires evidence ref to prior OBS) |
| **FOCUS** | Set hot key list |
| **RUN** | Execute bound tool handle |
| **STEP** | Advance plan pointer |
| **SEAL** | Checkpoint working → log anchor |
| **REVERT** | Roll back to last seal |
| **RENDER** | Syscall out — untrusted text generation |

## Storage (v0)

- `fact.*` string slots — write-once unless policy allows overwrite
- `plan` — step stack for tool / multi-hop scripts
- Overflow (stage F): hot cap evicts oldest facts to **cold store**; GET resolves hot then cold

## Curriculum stages (synthetic eval)

| Stage | Stress |
|-------|--------|
| A | plant → query |
| B | plant + filler → query |
| C | tool RUN → query |
| D | distractor PUT (kernel reverts) |
| E | plan + tool + filler |
| F | many facts → overflow → cold GET |

## Policy files

Shipped under `greenfield/deploy/` and in the Hub weights bundle under `policies/`.

See `hub/stack.json` for the default release combination.
