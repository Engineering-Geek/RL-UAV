from mujoco import MjModel, MjData, Renderer, MjvOption, mj_step


class BaseRegime:
    def __init__(self, scene_file_path: str, **kwargs):
        self.model: MjModel = MjModel.from_xml_path(scene_file_path)
        self.data: MjData = MjData(self.model)
        self.renderer: Renderer = Renderer(self.model, MjvOption())

    def reset(self, rng: jnp.ndarray) -> MjData:
        self.data.qpos = jax.random.uniform(rng, (self.model.nq,), minval=-0.1, maxval=0.1)
        self.data.qvel = jax.random.uniform(rng, (self.model.nv,), minval=-0.1, maxval=0.1)
        return self.data

    def step(self, mjx_data: MjData, action: jnp.ndarray) -> MjData:
        mj_step(self.model, mjx_data, action)
        return mjx_data




