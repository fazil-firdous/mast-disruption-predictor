# Hand check of the loss-event word matches (scripts/candidate_notes.py)

The script, with its word lists fixed before this check, printed 30 seeded random comments with a loss-event word from the candidates and 30 from the flux-exhausted endings (server run, 2026-11-06). Each was read once and judged: **Y** the comment says the plasma was lost (VDE, locked mode or event that ends the shot, an early or premature termination), **N** it does not (an end time only, a mode the plasma survived, a negation the rule missed), **?** unclear. A non-expert reading of plain-language comments, not a physics judgement of the shot.

| Group | Y | ? | N | Precision (Y / 30) | With ? as Y |
|---|---|---|---|---|---|
| candidate | 26 | 2 | 2 | 87% (Wilson 70-95%) | 93% |
| flux_exhausted | 10 | 6 | 14 | 33% (Wilson 19-51%) | 53% |

Applied to the shares with a loss-event word (candidates 407 of 3,962 = 10.3%; flux-exhausted 107 of 3,561 = 3.0%): about 9% of candidates (~350 shots) against about 1.0-1.6% of flux-exhausted endings are described by the operators as losing the plasma. A lower bound for unrecorded disruptions among the candidates, since most comments do not describe the ending at all.

## Candidates
| Shot | Judgement | Why |
|---|---|---|
| 11957 | Y | "event at 317ms is desastorous (locked mode?)" |
| 12243 | Y | VDE at 363 ms |
| 13853 | Y | terminated early |
| 14298 | Y | FAs lost, then the discharge terminates (hardware-triggered loss) |
| 14504 | N | end time with the NBI, "good shot" |
| 15288 | Y | VDE (an intended VDE study, ":-)") |
| 15293 | Y | VDE (intended, ":-)") |
| 15320 | Y | "looks like just a VDE" |
| 15796 | Y | "but terminated at ~300ms" |
| 17134 | Y | terminated by locked mode |
| 17192 | Y | terminated earlier, locked mode |
| 17412 | Y | IRE tripped FAs causing slow VDE |
| 17930 | ? | "terminated completely when previous shot lost H mode" (truncated) |
| 18506 | Y | VDE at 235 ms |
| 19525 | Y | VDE at ~0.15 s |
| 20099 | Y | terminated earlier than reference |
| 21005 | Y | terminated 0.190 (reference 0.320) |
| 21504 | Y | "but terminates shortly after 300 ms" |
| 22402 | ? | "shot terminates at 300 ms" (end time only?) |
| 22667 | Y | plasma ended early again |
| 25142 | Y | terminated earlier than reference |
| 26086 | Y | large instability, plasma terminates |
| 26276 | Y | VDE at 0.292 |
| 26387 | Y | discharge terminates early |
| 28072 | N | end time only |
| 28207 | Y | VDE triggered by IRE |
| 29037 | Y | IRE kills the shot |
| 29629 | Y | terminates early |
| 30267 | Y | terminated early |
| 30380 | Y | IRE that kills the shot |

## Flux-exhausted endings
| Shot | Judgement | Why |
|---|---|---|
| 12390 | N | "better behaved termination" |
| 12438 | N | possible locked mode mid-shot |
| 12455 | N | "no sign of spontaneous locked mode" (negation missed: 3 words between) |
| 13010 | Y | terminated by locked mode |
| 13512 | N | "locked mode that didn't lock" |
| 13527 | N | quasi locked mode, density loss mid-shot |
| 15889 | Y | lost due to long locked mode |
| 16704 | Y | finished by locked mode |
| 16957 | N | signs of a locked mode |
| 17422 | ? | locked mode at 340 ms |
| 17830 | Y | killed by locked mode |
| 18774 | ? | "reason for termination not obvious" |
| 19540 | Y | terminates earlier |
| 19541 | N | terminates earlier "(running out of flux faster)": flux exhaustion in the operator's words |
| 19895 | Y | terminates more abruptly |
| 20618 | Y | ends with locked mode |
| 20729 | Y | early termination |
| 21461 | N | possible locked mode |
| 21703 | ? | "standard locked mode" |
| 21704 | N | "no more obvious sign of a locked mode" (negation missed) |
| 21768 | ? | short shot, no real-time density |
| 22775 | N | the H-mode phase, not the plasma, terminated |
| 23428 | N | coils generate lock mode (error-field experiment) |
| 23496 | Y | IRE followed by loss of vertical control |
| 23802 | N | kicked out of H-mode |
| 26156 | ? | "lead to lock mode" |
| 28046 | N | H-mode terminates |
| 28655 | Y | VDE at end |
| 29012 | ? | locked mode onset |
| 30360 | N | end time; IRE spoils the shot |
