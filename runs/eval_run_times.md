# Evaluation run times

Generated 2026-10-24 21:48 from `runs/*_eval_seed*/state.json` (Train All Personas, 400 generations per run, runs written since 2026-10-24 18:53).

**Time** is the sum of every generation's wall-clock duration, so it includes the slowdown from running ~12 runs in parallel; trainer start-up (a few seconds) is excluded. Runs marked *partial* were stopped before reaching the generation budget.

## Total per persona and seed (all 4 games)

| Persona | 101 | 202 | 303 | 404 | 505 | Total |
|---|---|---|---|---|---|---|
| experienced | 1:23:59 | 55:56 | 51:48 | 1:01:37 | 1:01:32 | 5:14:52 |
| novice | 1:08:41 | 1:01:24 | 1:07:07 | 49:26 | 1:08:07 | 5:14:45 |
| speedrunner | 1:20:44 | 54:10 | 49:45 | 58:25 | 1:00:45 | 5:03:48 |
| clumsy | 35:47 | 38:22 | 42:11 | 39:37 | 34:39 | 3:10:35 |
| killer | 1:04:05 | 1:25:46 | 1:28:26 | 55:28 | 1:02:28 | 5:56:14 |
| collector | 1:35:30 | 1:06:37 | 1:18:37 | 1:00:33 | 1:08:34 | 6:09:52 |

\* includes a partial or missing run.

## Average per game and persona (completed runs only)

| Game | experienced | novice | speedrunner | clumsy | killer | collector |
|---|---|---|---|---|---|---|
| Mario 1-1 | 32:56 (n=5) | 32:51 (n=5) | 30:31 (n=5) | 17:47 (n=5) | 33:53 (n=5) | 38:30 (n=5) |
| Meat Boy 1-3 | 1:23 (n=5) | 1:58 (n=5) | 1:22 (n=5) | 1:45 (n=5) | 1:53 (n=5) | 1:55 (n=5) |
| Bomberman 09 | 9:32 (n=5) | 8:25 (n=5) | 9:27 (n=5) | 4:26 (n=5) | 13:59 (n=5) | 13:58 (n=5) |
| Mega Man Cut Man | 19:08 (n=5) | 19:44 (n=5) | 19:26 (n=5) | 14:09 (n=5) | 21:29 (n=5) | 19:35 (n=5) |

## Every run

| Seed | Game | Persona | Generations | Time | Gens with a win | First win |
|---|---|---|---|---|---|---|
| 101 | Mario 1-1 | experienced | 400 | 26:22 | 393 | 8 |
| 101 | Mario 1-1 | novice | 400 | 30:46 | 395 | 6 |
| 101 | Mario 1-1 | speedrunner | 400 | 22:37 | 393 | 8 |
| 101 | Mario 1-1 | clumsy | 400 | 16:57 | 10 | 21 |
| 101 | Mario 1-1 | killer | 400 | 34:31 | 385 | 16 |
| 101 | Mario 1-1 | collector | 400 | 39:13 | 395 | 6 |
| 101 | Meat Boy 1-3 | experienced | 400 | 1:24 | 392 | 9 |
| 101 | Meat Boy 1-3 | novice | 400 | 1:38 | 225 | 176 |
| 101 | Meat Boy 1-3 | speedrunner | 400 | 1:24 | 392 | 9 |
| 101 | Meat Boy 1-3 | clumsy | 400 | 1:33 | 15 | 16 |
| 101 | Meat Boy 1-3 | killer | 400 | 1:37 | 0 | — |
| 101 | Meat Boy 1-3 | collector | 400 | 1:36 | 0 | — |
| 101 | Bomberman 09 | experienced | 400 | 12:40 | 345 | 56 |
| 101 | Bomberman 09 | novice | 400 | 3:52 | 0 | — |
| 101 | Bomberman 09 | speedrunner | 400 | 12:41 | 345 | 56 |
| 101 | Bomberman 09 | clumsy | 400 | 3:14 | 0 | — |
| 101 | Bomberman 09 | killer | 400 | 9:51 | 352 | 49 |
| 101 | Bomberman 09 | collector | 400 | 9:47 | 352 | 49 |
| 101 | Mega Man Cut Man | experienced | 400 | 43:34 | 0 | — |
| 101 | Mega Man Cut Man | novice | 400 | 32:25 | 0 | — |
| 101 | Mega Man Cut Man | speedrunner | 400 | 44:02 | 0 | — |
| 101 | Mega Man Cut Man | clumsy | 400 | 14:02 | 0 | — |
| 101 | Mega Man Cut Man | killer | 400 | 18:06 | 0 | — |
| 101 | Mega Man Cut Man | collector | 400 | 44:54 | 0 | — |
| 202 | Mario 1-1 | experienced | 400 | 38:09 | 400 | 1 |
| 202 | Mario 1-1 | novice | 400 | 36:28 | 398 | 3 |
| 202 | Mario 1-1 | speedrunner | 400 | 35:25 | 400 | 1 |
| 202 | Mario 1-1 | clumsy | 400 | 17:21 | 124 | 2 |
| 202 | Mario 1-1 | killer | 400 | 35:47 | 385 | 16 |
| 202 | Mario 1-1 | collector | 400 | 39:52 | 394 | 7 |
| 202 | Meat Boy 1-3 | experienced | 400 | 1:22 | 399 | 2 |
| 202 | Meat Boy 1-3 | novice | 400 | 1:21 | 0 | — |
| 202 | Meat Boy 1-3 | speedrunner | 400 | 1:23 | 399 | 2 |
| 202 | Meat Boy 1-3 | clumsy | 400 | 1:24 | 23 | 23 |
| 202 | Meat Boy 1-3 | killer | 400 | 1:25 | 320 | 81 |
| 202 | Meat Boy 1-3 | collector | 400 | 1:25 | 320 | 81 |
| 202 | Bomberman 09 | experienced | 400 | 3:58 | 317 | 84 |
| 202 | Bomberman 09 | novice | 400 | 8:57 | 0 | — |
| 202 | Bomberman 09 | speedrunner | 400 | 4:14 | 317 | 84 |
| 202 | Bomberman 09 | clumsy | 400 | 4:46 | 1 | 327 |
| 202 | Bomberman 09 | killer | 400 | 12:51 | 0 | — |
| 202 | Bomberman 09 | collector | 400 | 12:43 | 0 | — |
| 202 | Mega Man Cut Man | experienced | 400 | 12:26 | 0 | — |
| 202 | Mega Man Cut Man | novice | 400 | 14:37 | 0 | — |
| 202 | Mega Man Cut Man | speedrunner | 400 | 13:08 | 0 | — |
| 202 | Mega Man Cut Man | clumsy | 400 | 14:50 | 0 | — |
| 202 | Mega Man Cut Man | killer | 400 | 35:43 | 0 | — |
| 202 | Mega Man Cut Man | collector | 400 | 12:37 | 0 | — |
| 303 | Mario 1-1 | experienced | 400 | 32:59 | 397 | 4 |
| 303 | Mario 1-1 | novice | 400 | 37:34 | 398 | 3 |
| 303 | Mario 1-1 | speedrunner | 400 | 32:20 | 397 | 4 |
| 303 | Mario 1-1 | clumsy | 400 | 20:38 | 92 | 7 |
| 303 | Mario 1-1 | killer | 400 | 41:28 | 396 | 5 |
| 303 | Mario 1-1 | collector | 400 | 45:37 | 395 | 6 |
| 303 | Meat Boy 1-3 | experienced | 400 | 1:28 | 396 | 5 |
| 303 | Meat Boy 1-3 | novice | 400 | 2:12 | 87 | 314 |
| 303 | Meat Boy 1-3 | speedrunner | 400 | 1:25 | 396 | 5 |
| 303 | Meat Boy 1-3 | clumsy | 400 | 2:30 | 34 | 32 |
| 303 | Meat Boy 1-3 | killer | 400 | 1:58 | 0 | — |
| 303 | Meat Boy 1-3 | collector | 400 | 2:02 | 0 | — |
| 303 | Bomberman 09 | experienced | 400 | 5:22 | 0 | — |
| 303 | Bomberman 09 | novice | 400 | 9:22 | 310 | 91 |
| 303 | Bomberman 09 | speedrunner | 400 | 5:13 | 0 | — |
| 303 | Bomberman 09 | clumsy | 400 | 3:46 | 1 | 22 |
| 303 | Bomberman 09 | killer | 400 | 19:06 | 197 | 204 |
| 303 | Bomberman 09 | collector | 400 | 19:10 | 197 | 204 |
| 303 | Mega Man Cut Man | experienced | 400 | 11:58 | 0 | — |
| 303 | Mega Man Cut Man | novice | 400 | 18:00 | 0 | — |
| 303 | Mega Man Cut Man | speedrunner | 400 | 10:46 | 0 | — |
| 303 | Mega Man Cut Man | clumsy | 400 | 15:17 | 0 | — |
| 303 | Mega Man Cut Man | killer | 400 | 25:54 | 0 | — |
| 303 | Mega Man Cut Man | collector | 400 | 11:49 | 0 | — |
| 404 | Mario 1-1 | experienced | 400 | 37:56 | 400 | 1 |
| 404 | Mario 1-1 | novice | 400 | 31:13 | 364 | 37 |
| 404 | Mario 1-1 | speedrunner | 400 | 35:06 | 400 | 1 |
| 404 | Mario 1-1 | clumsy | 400 | 17:16 | 158 | 2 |
| 404 | Mario 1-1 | killer | 400 | 30:22 | 385 | 16 |
| 404 | Mario 1-1 | collector | 400 | 34:15 | 391 | 10 |
| 404 | Meat Boy 1-3 | experienced | 400 | 1:20 | 398 | 3 |
| 404 | Meat Boy 1-3 | novice | 400 | 2:20 | 109 | 292 |
| 404 | Meat Boy 1-3 | speedrunner | 400 | 1:20 | 398 | 3 |
| 404 | Meat Boy 1-3 | clumsy | 400 | 1:44 | 5 | 21 |
| 404 | Meat Boy 1-3 | killer | 400 | 1:26 | 0 | — |
| 404 | Meat Boy 1-3 | collector | 400 | 1:27 | 0 | — |
| 404 | Bomberman 09 | experienced | 400 | 9:58 | 257 | 144 |
| 404 | Bomberman 09 | novice | 400 | 7:38 | 372 | 29 |
| 404 | Bomberman 09 | speedrunner | 400 | 9:30 | 257 | 144 |
| 404 | Bomberman 09 | clumsy | 400 | 6:38 | 10 | 130 |
| 404 | Bomberman 09 | killer | 400 | 12:04 | 375 | 26 |
| 404 | Bomberman 09 | collector | 400 | 12:10 | 375 | 26 |
| 404 | Mega Man Cut Man | experienced | 400 | 12:23 | 0 | — |
| 404 | Mega Man Cut Man | novice | 400 | 8:16 | 0 | — |
| 404 | Mega Man Cut Man | speedrunner | 400 | 12:30 | 0 | — |
| 404 | Mega Man Cut Man | clumsy | 400 | 14:00 | 0 | — |
| 404 | Mega Man Cut Man | killer | 400 | 11:36 | 0 | — |
| 404 | Mega Man Cut Man | collector | 400 | 12:42 | 0 | — |
| 505 | Mario 1-1 | experienced | 400 | 29:13 | 397 | 4 |
| 505 | Mario 1-1 | novice | 400 | 28:12 | 388 | 13 |
| 505 | Mario 1-1 | speedrunner | 400 | 27:09 | 397 | 4 |
| 505 | Mario 1-1 | clumsy | 400 | 16:42 | 136 | 1 |
| 505 | Mario 1-1 | killer | 400 | 27:19 | 325 | 76 |
| 505 | Mario 1-1 | collector | 400 | 33:33 | 392 | 9 |
| 505 | Meat Boy 1-3 | experienced | 400 | 1:19 | 395 | 6 |
| 505 | Meat Boy 1-3 | novice | 400 | 2:18 | 357 | 44 |
| 505 | Meat Boy 1-3 | speedrunner | 400 | 1:19 | 395 | 6 |
| 505 | Meat Boy 1-3 | clumsy | 400 | 1:36 | 30 | 23 |
| 505 | Meat Boy 1-3 | killer | 400 | 3:00 | 0 | — |
| 505 | Meat Boy 1-3 | collector | 400 | 3:04 | 0 | — |
| 505 | Bomberman 09 | experienced | 400 | 15:41 | 0 | — |
| 505 | Bomberman 09 | novice | 400 | 12:16 | 142 | 259 |
| 505 | Bomberman 09 | speedrunner | 400 | 15:35 | 0 | — |
| 505 | Bomberman 09 | clumsy | 400 | 3:47 | 2 | 293 |
| 505 | Bomberman 09 | killer | 400 | 16:04 | 330 | 71 |
| 505 | Bomberman 09 | collector | 400 | 16:04 | 330 | 71 |
| 505 | Mega Man Cut Man | experienced | 400 | 15:20 | 0 | — |
| 505 | Mega Man Cut Man | novice | 400 | 25:21 | 0 | — |
| 505 | Mega Man Cut Man | speedrunner | 400 | 16:42 | 0 | — |
| 505 | Mega Man Cut Man | clumsy | 400 | 12:35 | 0 | — |
| 505 | Mega Man Cut Man | killer | 400 | 16:05 | 0 | — |
| 505 | Mega Man Cut Man | collector | 400 | 15:53 | 0 | — |
