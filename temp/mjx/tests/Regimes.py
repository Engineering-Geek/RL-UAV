import jax.numpy as jnp
import jax
from temp.mjx.SimpleRegimes import HoverRegime
from brax.mjx.base import State as MjxState
from brax.base import State as BraxState


def test_hover_regime():
    # Initialize a HoverRegime object
    hover_regime = HoverRegime('../../models/skydio_x2/scene.xml', jnp.array([0, 0, 0]), 1.0, 1.0)
    brax_state = hover_regime.reset(jax.random.PRNGKey(0))
    brax_state = hover_regime.step(brax_state=brax_state, action=jnp.array([0, 0, 0, 0]))
    assert isinstance(brax_state, BraxState)
    assert isinstance(brax_state.pipeline_state, MjxState)
