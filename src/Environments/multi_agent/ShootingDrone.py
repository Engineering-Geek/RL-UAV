from typing import List, Tuple

from numpy import cos

from Environments.multi_agent.core.BaseDrone import BaseDrone


class SimpleShooter(BaseDrone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ---------------------------------------------------------
        
        # region Reward Flags
        
        # Flags for sparse rewards and penalties and actions
        self._hit_floor = False
        self._got_hit = False
        self._hit_target = False
        self._out_of_bounds = False
        self._bullet_out_of_bounds = False
        
        # Rewards and Penalties for Continuous actions
        self._aimed_at_angles: List[float] = []
        self._aiming_at_angles: List[float] = []
        # endregion
        
        # ---------------------------------------------------------
        
        # region Reward Variables and Functions
        
        # Reward and Penalty parameters
        self.staying_alive_reward = kwargs.get("staying_alive_reward", 1.0)     # Reward for staying alive per second
        self.hit_reward = kwargs.get("hit_reward", 10.0)                        # Reward for hitting the target
        self.hit_penalty = kwargs.get("hit_penalty", 10.0)                      # Penalty for getting hit
        self.floor_penalty = kwargs.get("floor_penalty", 10.0)                  # Penalty for hitting the floor
        self.out_of_bounds_penalty = kwargs.get("out_of_bounds_penalty", 10.0)  # Penalty for going out of bounds
        self.bullet_out_of_bounds_penalty = (
            kwargs.get("bullet_out_of_bounds_penalty", 10.0))                   # Penalty for bullet going out of bounds
        self.aim_reward = kwargs.get("aim_reward", 10.0)                          # Reward for aiming at the target
        self.aim_penalty = kwargs.get("aim_penalty", 10.0)                        # Penalty for being aimed at
        self._shooting_penalty = kwargs.get("shooting_penalty", 1.0)            # Penalty for shooting
        # endregion
    
    def got_hit(self):
        self._got_hit = True
        self.reset()        # Respawning the drone after getting hit
    
    def hit_floor(self):
        self._hit_floor = True
        self.reset()        # Respawning the drone after hitting the floor
    
    def on_bullet_out_of_bounds(self):
        self._bullet_out_of_bounds = True
    
    def on_out_of_bounds(self):
        self._out_of_bounds = True
        self.reset()        # Respawning the drone after going out of bounds
    
    def scored_hit(self):
        self._hit_target = True
    
    def aiming_at(self, drones: List[Tuple[BaseDrone, float]]):
        self._aiming_at_angles = [angle for drone, angle in drones]
    
    def aimed_at(self, drones: List[Tuple[BaseDrone, float]]):
        self._aimed_at_angles = [angle for drone, angle in drones]
    
    def env_update(self):
        pass
    
    def reset_params(self):
        self._hit_floor = False
        self._got_hit = False
        self._hit_target = False
        self._out_of_bounds = False
        self._bullet_out_of_bounds = False
        
        # No need to reset the aiming and aimed at angles because they are updated every step
        # self._aimed_at_angles = []
        # self._aiming_at_angles = []
    
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
            reward += self.staying_alive_reward * self.model.opt.timestep
        if self.just_shot:
            reward -= self._shooting_penalty
        
        # Formula for Aiming Reward
        for angle in self._aiming_at_angles:
            reward += self.aim_reward * cos(angle) * self.model.opt.timestep
        
        # Formula for Aimed At Penalty
        for angle in self._aimed_at_angles:
            reward -= self.aim_penalty * cos(angle) * self.model.opt.timestep
            
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

