from TrainingRegimes.standard.HoverRegime import HoverRegime
import numpy as np


def test_hover_regime():
    regime = HoverRegime('models/UAV/scene.xml')
    state = regime.reset()
    state.append_historical_data()
    state.append_observation()
    while not state.done:
        state = regime.step(state, np.array([0.0, 0.0, 0.0, 0.0]))
        state.append_historical_data()
        state.append_observation()
    print(state.reward)
    print(state.done)
    print(state.metrics)

