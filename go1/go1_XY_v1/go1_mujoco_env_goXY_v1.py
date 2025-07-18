from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
from pathlib import Path

from pathlib import Path
import os


DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}


class Go1MujocoEnv(MujocoEnv):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, ctrl_type=None, **kwargs):

        model_path = Path(os.path.expanduser(f"scene_position.xml"))

        

        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # Perform an action every 10 frames (dt(=0.002) * 10 = 0.02 seconds -> 50hz action rate)
            observation_space=None,  
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 60,
        }
        self._last_render_time = -1.0
        self.test_mode=kwargs.get("test_mode", False)
        self._max_episode_time_sec = 15.0
        self._step = 0

        # Pesos de las recompensas
        self.reward_weights = {
            "linear_vel_tracking": 00,  
            "angular_vel_tracking": 1.0,
            "healthy": 0.05,  
            "feet_airtime": 1, 
            "target_xy":4.0,
            "header_xy":1.0,
            "velocity_xy": 2.0,
            "termination": 4.0,
        }

        # Pesos de las penalizaciones
        self.cost_weights = {
            "torque": 0.0005,
            "vertical_vel": 0.4,  
            "xy_angular_vel": 0.001,  
            "action_rate": 0.01,
            "joint_limit": 1.0, 
            "joint_velocity": 0.01, 
            "joint_acceleration":5e-6 ,
            "orientation": 0.05,
        }

        self._curriculum_base = 0.3
        self._gravity_vector = np.array(self.model.opt.gravity)
        self._default_joint_position = np.array(self.model.key_ctrl[0])


        #posicion a ir xy
        self._target_position=[-4,4]
        self.distance_umbral=0.5
        self.velocity_treshold=0.2
        self._unhealthy_counter =0

        self._target_position_min = np.array([-5.0, -5.0])
        self._target_position_max = np.array([5.0, 5.0])


        # vx (m/s), vy (m/s), wz (rad/s)
        self._desired_velocity_min = np.array([-1, 0, -1])
        self._desired_velocity_max = np.array([1, 0, 1])
        self._desired_velocity = self._sample_desired_vel()  
        self._obs_scale = {
            "linear_velocity": 5.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self._tracking_velocity_sigma = 0.8 # controla la tolerancia a cambios 

        # Para detrminar si es un estado terminal
        self._healthy_z_range = (0.22, 0.65)
        self._healthy_pitch_range = (-np.deg2rad(10), np.deg2rad(10))
        self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]


        dof_position_limit_multiplier = 0.9  # No penaliza el 90% del rango
        ctrl_range_offset = (0.5 * (1 - dof_position_limit_multiplier)* (self.model.actuator_ctrlrange[:, 1]- self.model.actuator_ctrlrange[:, 0]))
        # El primer valor se ignora
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1

        # Action: 12 torque values
        self._last_action = np.zeros(12)

        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        
        self.feet_site = [
            "FR",
            "FL",
            "RR",
            "RL",
        ]
        self._feet_site_name_to_id = {f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)for f in self.feet_site}
        
       
        self._main_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk")

    def step(self, action):
        self._step += 1
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        
        terminated = not self.is_healthy 
        
        #terminated si se llega a las pos target a una distancia umbral(0.5)
        if np.linalg.norm(self.data.qpos[:2] - self._target_position) < self.distance_umbral:
            terminated=True
            reward += 10*self.reward_weights["termination"]
            
        truncated = self._step >= (self._max_episode_time_sec / self.dt)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        

        if not self.is_healthy:
            self._unhealthy_counter += 1

        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
            1.0 / self.metadata["render_fps"]
        ):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action

        return observation, reward, terminated, truncated, info

    
    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z

        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= state[4] <= max_roll

        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= state[5] <= max_pitch

        return is_healthy

    @property
    def projected_gravity(self):
        w, x, y, z = self.data.qpos[3:7]
        euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z))
        projected_gravity_not_normalized = (
            np.dot(self._gravity_vector, euler_orientation) * euler_orientation
        )
        if np.linalg.norm(projected_gravity_not_normalized) == 0:
            return projected_gravity_not_normalized
        else:
            return projected_gravity_not_normalized / np.linalg.norm(
                projected_gravity_not_normalized
            )

    @property
    def feet_contact_forces(self):
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)

    ######### Positive Reward functions #########
    @property
    def linear_velocity_tracking_reward(self):
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - self.data.qvel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        vel_sqr_error = np.square(self._desired_velocity[2] - self.data.qvel[5])
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)


    @property
    def feet_air_time_reward(self):
        #Premia pasos largos (> 0.1s), penaliza pasos cortos, fomenta simetría y coordinación
        feet_contact_force_mag = self.feet_contact_forces  # tamaño (4,)
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        min_air_time = 0.1 
        stride_reward = (self._feet_air_time - min_air_time) * first_contact
        
        stride_reward = np.maximum(stride_reward, 0.0)

        # Penalización por pasos cortos
        short_steps = (self._feet_air_time < min_air_time) * first_contact
        short_step_penalty = np.sum(short_steps) * 0.05  # penaliza -0.1 por cada pata que no cumplió

        # Agrupar patas por lateralidad
        right_air = self._feet_air_time[[0, 2]]  # FR, RR
        left_air  = self._feet_air_time[[1, 3]]  # FL, RL

        # Penalización por asimetría entre lados
        asym_penalty = np.abs(np.mean(left_air) - np.mean(right_air))
        asym_penalty = np.clip(asym_penalty, 0.0, 1.0)

        # Bonificación por coordinación
        num_contacts = np.sum(curr_contact)
        coordination_bonus = 1.0 - np.clip((num_contacts - 2) / 2.0, 0.0, 1.0)

        # Recompensa base por pasos largos
        air_time_reward = np.sum(stride_reward)


        # Aplicar penalizaciones y bonificaciones
        air_time_reward *= (1.0 - 0.5 * asym_penalty)
        air_time_reward *= coordination_bonus

        # Bonus si las patas traseras están participando bien
        rear_stride_bonus = np.sum(stride_reward[[2, 3]])  # RR, RL
        rear_bonus = np.clip(rear_stride_bonus / (min_air_time * 2), 0.0, 1.0)
        air_time_reward *= 0.5 + 0.5 * rear_bonus

        # Aplicar penalización por pasos cortos
        air_time_reward -= short_step_penalty

        # Acotar entre -1 y 1
        air_time_reward = np.clip(air_time_reward, -1.0, 1.0)

        # Resetear tiempos en el aire para patas que hicieron contacto
        self._feet_air_time *= ~contact_filter

        return air_time_reward




    @property
    def healthy_reward(self):
        return self.is_healthy
    
    # Recompensa para fomentar ir a xy
    @property
    def target_position_reward(self):
        current_pos = self.data.qpos[:2]  # posición actual [x, y]
        distance = np.linalg.norm(current_pos - self._target_position)
        reward = np.exp(-distance / self._tracking_velocity_sigma)
        
        return reward
    

    # Recompensa para orinetarse ahcia el target
    @property
    def heading_alignment_reward(self):
        

        # Dirección normalizada hacia el objetivo (plano XY)
        current_pos = self.data.qpos[:2]
        to_target = self._target_position - current_pos
        dist = np.linalg.norm(to_target)

        
        dir_to_target = to_target / dist

        w, x, y, z = self.data.qpos[3:7]
        _, _, yaw = self.euler_from_quaternion(w, x, y, z)

        forward = np.array([np.cos(yaw), np.sin(yaw)])

        cos_sim = np.dot(forward, dir_to_target)
        
        return (cos_sim + 1.0) / 2.0


    # Recompensa por avanzar hacia objetivo
    @property
    def reward_velocity_towards_target(self):
        direction = self._target_position - self.data.qpos[:2]
        direction_norm = np.linalg.norm(direction)

        direction = direction / direction_norm  # Vector unitario hacia el objetivo
        velocity = self.data.qvel[:2]
        forward_velocity = np.dot(velocity, direction)  

        # Se normaliza con la vel máxima
        max_velocity = self._desired_velocity_max[0]
        normalized = np.clip(forward_velocity / max_velocity, -1.0, 1.0)
        
        return normalized 

    ######### Negative Reward functions #########

    
    @property
    def non_flat_base_cost(self):
        # Penaliza por no estar paraleleo al suelo
        return np.sum(np.square(self.projected_gravity[:2]))

    @property
    def joint_limit_cost(self):
        # Penaliza por salirse de los limites establecidos de las articulaciones
        out_of_range = (self._soft_joint_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        # Penaliza el uso de torque
        return np.sum(np.square(self.data.qfrc_actuator[-12:]))

    @property
    def vertical_velocity_cost(self):
        # Penaliza vel en z
        return np.square(self.data.qvel[2])

    @property
    def xy_angular_velocity_cost(self):
        # Penaliza vel angular en x,y
        return np.sum(np.square(self.data.qvel[3:5]))

    
    def action_rate_cost(self, action):
        # Penaliza moviminetos bruscos
        return np.sum(np.square(self._last_action - action))

    @property
    def acceleration_cost(self):
        # Penaliza aceleraciones
        return np.sum(np.square(self.data.qacc[6:]))


    @property
    def curriculum_factor(self):
        # Permite que el entorno se complique con el tiempo
        return self._curriculum_base**0.997

    

    def _calc_reward(self, action):

        # Positive Rewards
        linear_vel_tracking_reward = (self.linear_velocity_tracking_reward* self.reward_weights["linear_vel_tracking"])
        angular_vel_tracking_reward = (self.angular_velocity_tracking_reward* self.reward_weights["angular_vel_tracking"])
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = (self.feet_air_time_reward * self.reward_weights["feet_airtime"])
        target_xy_reward=self.target_position_reward * self.reward_weights["target_xy"]

        reward_velocity_towards_target=self.reward_velocity_towards_target * self.reward_weights["velocity_xy"]

        reward_heading_alignment=self.heading_alignment_reward * self.reward_weights["header_xy"]

        
        rewards = (
                linear_vel_tracking_reward
                + target_xy_reward
                + reward_heading_alignment
                + reward_velocity_towards_target
                + angular_vel_tracking_reward
                + healthy_reward
                + feet_air_time_reward
            )


        # Negative Costs
        ctrl_cost = self.torque_cost * self.cost_weights["torque"]
        action_rate_cost = (self.action_rate_cost(action) * self.cost_weights["action_rate"])
        vertical_vel_cost = (self.vertical_velocity_cost * self.cost_weights["vertical_vel"])
        xy_angular_vel_cost = (self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"])
        joint_limit_cost = self.joint_limit_cost * self.cost_weights["joint_limit"]
        joint_acceleration_cost = (self.acceleration_cost * self.cost_weights["joint_acceleration"])
        orientation_cost = self.non_flat_base_cost * self.cost_weights["orientation"]


        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_acceleration_cost
            + orientation_cost
            
        )
        
        reward = rewards - self.curriculum_factor * costs
        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        # The first three indices are the global x,y,z position of the trunk of the robot
        # The second four are the quaternion representing the orientation of the robot
        # The above seven values are ignored since they are privileged information
        # The remaining 12 values are the joint positions
        # The joint positions are relative to the starting position
        dofs_position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]

        # The first three values are the global linear velocity of the robot
        # The second three are the angular velocity of the robot
        # The remaining 12 values are the joint velocities
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]

        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.projected_gravity

        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],
                base_angular_velocity * self._obs_scale["angular_velocity"],
                projected_gravity,
                desired_vel * self._obs_scale["linear_velocity"],
                dofs_position * self._obs_scale["dofs_position"],
                dofs_velocity * self._obs_scale["dofs_velocity"],
                last_action,
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        return curr_obs
    
    def reset_model(self):
        # Reseta la posicion incial con ruido
        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[
            0
        ] + self._reset_noise_scale * self.np_random.standard_normal(
            *self.data.ctrl.shape
        )

        # Reseta las variables
        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0
        self._unhealthy_counter =0

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return desired_vel

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians