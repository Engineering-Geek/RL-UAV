from typing import List, Tuple

from Environments.multi_agent.core.BaseDrone import BaseDrone


class SimpleDrone(BaseDrone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ---------------------------------------------------------
        
        # region Reward Flags
        
        # Flags for rewards and penalties
        self._hit_floor = False
        self._got_hit = False
        self._hit_target = False
        self._out_of_bounds = False
        self._bullet_out_of_bounds = False
        # endregion
        
        # ---------------------------------------------------------
        
        # region Reward Variables and Functions
        
        # Reward and Penalty parameters
        self.alive_reward = kwargs.get("alive_reward", 1.0)                     # Continuous reward for staying alive
        self.hit_reward = kwargs.get("hit_reward", 10.0)                        # Sparse Reward for hitting the target
        self.hit_penalty = kwargs.get("hit_penalty", 10.0)                      # Sparse Penalty for getting hit
        self.floor_penalty = kwargs.get("floor_penalty", 10.0)                  # Sparse Penalty for hitting the floor
        self.out_of_bounds_penalty = kwargs.get("out_of_bounds_penalty", 10.0)  # Sparse Penalty for going out of bounds
        self.bullet_out_of_bounds_penalty = (
            kwargs.get("bullet_out_of_bounds_penalty", 10.0))                   # Sparse Penalty for bullet going out of bounds
        # endregion
        
    def got_hit(self):
        self._got_hit = True
    
    def hit_floor(self):
        self._hit_floor = True
        self.reset()        # Respawning the drone after hitting the floor
    
    def on_bullet_out_of_bounds(self):
        self._bullet_out_of_bounds = True
    
    def on_out_of_bounds(self):
        self._out_of_bounds = True
    
    def scored_hit(self):
        self._hit_target = True
    
    def aiming_at(self, drones: List[Tuple[BaseDrone, float]]):
        pass
    
    def aimed_at(self, drones: List[Tuple[BaseDrone, float]]):
        pass
    
    def env_update(self):
        pass
    
    def reset_params(self):
        self._hit_floor = False
        self._got_hit = False
        self._hit_target = False
        self._out_of_bounds = False
        self._bullet_out_of_bounds = False
        
    @property
    def reward(self):
        reward = 0
        if self._hit_floor:
            reward -= self.floor_penalty
        if self._got_hit:
            reward -= self.hit_penalty
        if self._hit_target:
            reward += self.hit_reward
        if self._out_of_bounds:
            reward -= self.out_of_bounds_penalty
        if self._bullet_out_of_bounds:
            reward -= self.bullet_out_of_bounds_penalty
        if self.alive:
            reward += self.alive_reward * self.model.opt.timestep
        return reward

    @property
    def done(self):
        return self._hit_floor or self._got_hit
    
    @property
    def truncated(self) -> bool:
        return self._out_of_bounds
    
    @property
    def info(self):
        return {
            "hit_floor": self._hit_floor,
            "got_hit": self._got_hit,
            "hit_target": self._hit_target,
            "out_of_bounds": self._out_of_bounds,
            "bullet_out_of_bounds": self._bullet_out_of_bounds,
        }

