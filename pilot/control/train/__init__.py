"""T4 — PPO training harness for the drone-racing policy.

Trains against `pilot.control.surrogate.env.VecSurrogate`; the deployment wrapper
(`pilot/control/policies/`) imports `network`, `framestack` and `normalize` from here
so that inference reproduces the training-time pipeline exactly.

The pipeline order is load-bearing and identical on both sides:

    raw 73-D observation  ->  ObsNormalizer (frozen stats)  ->  FrameStack(k)  ->  net

See `README.md` in this directory.
"""
