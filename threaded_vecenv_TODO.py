"""Threaded VecEnv for memory-bound training (TO BUILD after punch run).

Problem: SubprocVecEnv costs one full MuJoCo scene + ONNX session PER WORKER
(~600MB each on G1). 4 workers + trainer ~= 3GB = box ceiling; 8 OOMs.

Design: N envs in ONE process, stepped on a ThreadPoolExecutor. MuJoCo's
mj_step and onnxruntime both release the GIL, so CPU parallelism survives.
Scene + ONNX session are built once per env (models are per-env state, but
the ONNX session can be SHARED read-only across envs since LocoBase29
inference is stateless given obs — one session, N LocoBase29 instances).

Interface: same as DummyVecEnv (SB3 VecEnv subclass), drop-in for train scripts.

Expected savings: 4 envs ~= 1.2GB instead of 2.4GB; 8 envs fits in <2GB.
Also halves boxing-training memory (2 sims per env) later.

TODO:
- [ ] class ThreadedVecEnv(VecEnv): reset/step_async/step_wait/close
- [ ] share ort.InferenceSession across env instances (thread-safe for run())
- [ ] benchmark vs SubprocVecEnv (steps/s + RSS)
- [ ] swap into train_g1_punch.py / future boxing trainer
"""
