import time

import jax.numpy as np
import cv2
import jax
from jax import device_get
from tqdm import tqdm
from typing import List

from TrainingRegimes import HoverRegime

# initialize the state
rng = jax.random.PRNGKey(0)

# grab a trajectory
n_steps = 10000
render_every = 2

env = HoverRegime('models/skydio_x2/scene.xml', np.array([0, 0, 5]), 1.0, 1.0)
state = env.reset(rng)
reset = jax.jit(env.reset)
step = jax.jit(env.step)
# reset = env.reset
# step = env.step
frames: List[np.ndarray] = []

for _ in tqdm(range(n_steps)):
    state = step(state, np.array([0.0] * 4))

    # Write the frame to the video file (adjust the frame size as needed)
    frame = device_get(state.obs[3])
    frames.append(frame)


cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 600, 600)
for i, frame in enumerate(frames):
    cv2.imshow('frame', frame)
    time.sleep(0.01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()



