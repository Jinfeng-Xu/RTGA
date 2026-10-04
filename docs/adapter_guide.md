# Adapter Guide

RTGA is model-agnostic. To connect a SchNet, PaiNN, NequIP-like, or other MLIP
backbone, implement the `AdaptivePotential` protocol:

```python
from rtga.interfaces import AdaptationInfo, ForcePrediction, MDState

class MyPotential:
    def risk_features(self, state: MDState):
        # Return calibrated diagnostic features, for example SSL losses,
        # representation distances, force proxy statistics, or their concat.
        ...

    def evaluate(self, state: MDState) -> ForcePrediction:
        # Return energy and forces for the current atomic configuration.
        ...

    def adapt(self, state: MDState, weight: float) -> AdaptationInfo:
        # Apply one local repair step. The weight is in [0, 1] and can scale
        # learning rate, loss, update frequency, or a trust-region size.
        ...
```

Recommended production logging fields:

- `step`, `time`, `temperature`, `energy`, `force_norm`
- `risk_score`, raw risk feature vector, and threshold
- pre-force update weight
- guard reason and guard update weight
- accepted force norm after recomputation
- failure reason, if any
- checkpoint hash, data split, seed, timestep, thermostat, and command line

The included controller check writes a subset of these fields to JSON.
