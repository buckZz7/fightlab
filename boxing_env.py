"""FightLab boxing environment for self-play RL.

Two humanoid robots in an arena. Each gets:
- Observation: joint positions, velocities, opponent state, contact forces
- Action: joint torques for shoulders, elbows, hips, knees
- Reward: damage dealt, staying upright, proximity to opponent
- Termination: knockdown (torso below threshold), timeout

Domain randomization: mass, friction, motor gear jitter per episode.
"""
import mujoco
import numpy as np
import json, os
from collections import deque

# Boxing arena XML (same as demo but with motors for RL control)
ARENA_XML = """
<mujoco model="fightlab_arena">
  <compiler angle="radian" coordinate="local"/>
  <option integrator="RK4" timestep="0.002"/>
  <option gravity="0 0 -9.81"/>
  
  <default>
    <joint damping="0.5" armature="0.1" frictionloss="0.1"/>
    <geom conaffinity="1" condim="3" friction="1.0 0.5 0.5"/>
  </default>
  
  <!-- Actuators -->
  <actuator>
    <motor name="l_sh1" joint="l_sh1" ctrlrange="-50 50"/>
    <motor name="l_el1" joint="l_el1" ctrlrange="-50 50"/>
    <motor name="r_sh1" joint="r_sh1" ctrlrange="-50 50"/>
    <motor name="r_el1" joint="r_el1" ctrlrange="-50 50"/>
    <motor name="l_hip1" joint="l_hip1" ctrlrange="-50 50"/>
    <motor name="l_kn1" joint="l_kn1" ctrlrange="-50 50"/>
    <motor name="r_hip1" joint="r_hip1" ctrlrange="-50 50"/>
    <motor name="r_kn1" joint="r_kn1" ctrlrange="-50 50"/>
    <motor name="neck1" joint="neck1" ctrlrange="-50 50"/>
    <motor name="l_sh2" joint="l_sh2" ctrlrange="-50 50"/>
    <motor name="l_el2" joint="l_el2" ctrlrange="-50 50"/>
    <motor name="r_sh2" joint="r_sh2" ctrlrange="-50 50"/>
    <motor name="r_el2" joint="r_el2" ctrlrange="-50 50"/>
    <motor name="l_hip2" joint="l_hip2" ctrlrange="-50 50"/>
    <motor name="l_kn2" joint="l_kn2" ctrlrange="-50 50"/>
    <motor name="r_hip2" joint="r_hip2" ctrlrange="-50 50"/>
    <motor name="r_kn2" joint="r_kn2" ctrlrange="-50 50"/>
    <motor name="neck2" joint="neck2" ctrlrange="-50 50"/>
  </actuator>
  
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    
    <!-- Arena boundary (invisible) -->
    <geom name="wall_n" type="box" pos="0 2.5 1" size="2.5 0.05 1" rgba="0.5 0.5 0.5 0.3" contype="0" conaffinity="0"/>
    <geom name="wall_s" type="box" pos="0 -2.5 1" size="2.5 0.05 1" rgba="0.5 0.5 0.5 0.3" contype="0" conaffinity="0"/>
    <geom name="wall_e" type="box" pos="2.5 0 1" size="0.05 2.5 1" rgba="0.5 0.5 0.5 0.3" contype="0" conaffinity="0"/>
    <geom name="wall_w" type="box" pos="-2.5 0 1" size="0.05 2.5 1" rgba="0.5 0.5 0.5 0.3" contype="0" conaffinity="0"/>
    
    <!-- Robot 1 (Blue) -->
    <body name="robot1" pos="-0.8 0 1.2">
      <freejoint name="root1"/>
      <geom name="torso1" type="capsule" size="0.12 0.3" rgba="0.2 0.4 0.8 1"/>
      <body name="head1" pos="0 0 0.45">
        <joint name="neck1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.1" rgba="0.2 0.4 0.8 1"/>
      </body>
      <!-- Left arm -->
      <body name="l_shoulder1" pos="0.15 0 0.25">
        <joint name="l_sh1" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="0.04 0.15" rgba="0.2 0.4 0.8 1"/>
        <body name="l_elbow1" pos="0 0 -0.15">
          <joint name="l_el1" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size="0.035 0.12" rgba="0.2 0.4 0.8 1"/>
          <body name="l_hand1" pos="0 0 -0.12">
            <geom type="sphere" size="0.045" rgba="0.2 0.4 0.8 1"/>
          </body>
        </body>
      </body>
      <!-- Right arm -->
      <body name="r_shoulder1" pos="-0.15 0 0.25">
        <joint name="r_sh1" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="0.04 0.15" rgba="0.2 0.4 0.8 1"/>
        <body name="r_elbow1" pos="0 0 -0.15">
          <joint name="r_el1" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size="0.035 0.12" rgba="0.2 0.4 0.8 1"/>
          <body name="r_hand1" pos="0 0 -0.12">
            <geom type="sphere" size="0.045" rgba="0.2 0.4 0.8 1"/>
          </body>
        </body>
      </body>
      <!-- Left leg -->
      <body name="l_hip1" pos="0.08 0 -0.35">
        <joint name="l_hip1" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="0.05 0.25" rgba="0.2 0.4 0.8 1"/>
        <body name="l_knee1" pos="0 0 -0.25">
          <joint name="l_kn1" type="hinge" axis="0 1 0" range="0 2"/>
          <geom type="capsule" size="0.04 0.2" rgba="0.2 0.4 0.8 1"/>
          <body name="l_foot1" pos="0 0 -0.2">
            <geom type="box" size="0.08 0.04 0.02" rgba="0.2 0.4 0.8 1"/>
          </body>
        </body>
      </body>
      <!-- Right leg -->
      <body name="r_hip1" pos="-0.08 0 -0.35">
        <joint name="r_hip1" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="0.05 0.25" rgba="0.2 0.4 0.8 1"/>
        <body name="r_knee1" pos="0 0 -0.25">
          <joint name="r_kn1" type="hinge" axis="0 1 0" range="0 2"/>
          <geom type="capsule" size="0.04 0.2" rgba="0.2 0.4 0.8 1"/>
          <body name="r_foot1" pos="0 0 -0.2">
            <geom type="box" size="0.08 0.04 0.02" rgba="0.2 0.4 0.8 1"/>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Robot 2 (Red) -->
    <body name="robot2" pos="0.8 0 1.2">
      <freejoint name="root2"/>
      <geom name="torso2" type="capsule" size="0.12 0.3" rgba="0.8 0.2 0.2 1"/>
      <body name="head2" pos="0 0 0.45">
        <joint name="neck2" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="sphere" size="0.1" rgba="0.8 0.2 0.2 1"/>
      </body>
      <body name="l_shoulder2" pos="0.15 0 0.25">
        <joint name="l_sh2" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="0.04 0.15" rgba="0.8 0.2 0.2 1"/>
        <body name="l_elbow2" pos="0 0 -0.15">
          <joint name="l_el2" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size="0.035 0.12" rgba="0.8 0.2 0.2 1"/>
          <body name="l_hand2" pos="0 0 -0.12">
            <geom type="sphere" size="0.045" rgba="0.8 0.2 0.2 1"/>
          </body>
        </body>
      </body>
      <body name="r_shoulder2" pos="-0.15 0 0.25">
        <joint name="r_sh2" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="0.04 0.15" rgba="0.8 0.2 0.2 1"/>
        <body name="r_elbow2" pos="0 0 -0.15">
          <joint name="r_el2" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size="0.035 0.12" rgba="0.8 0.2 0.2 1"/>
          <body name="r_hand2" pos="0 0 -0.12">
            <geom type="sphere" size="0.045" rgba="0.8 0.2 0.2 1"/>
          </body>
        </body>
      </body>
      <body name="l_hip2" pos="0.08 0 -0.35">
        <joint name="l_hip2" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="0.05 0.25" rgba="0.8 0.2 0.2 1"/>
        <body name="l_knee2" pos="0 0 -0.25">
          <joint name="l_kn2" type="hinge" axis="0 1 0" range="0 2"/>
          <geom type="capsule" size="0.04 0.2" rgba="0.8 0.2 0.2 1"/>
          <body name="l_foot2" pos="0 0 -0.2">
            <geom type="box" size="0.08 0.04 0.02" rgba="0.8 0.2 0.2 1"/>
          </body>
        </body>
      </body>
      <body name="r_hip2" pos="-0.08 0 -0.35">
        <joint name="r_hip2" type="hinge" axis="0 1 0" range="-2 2"/>
        <geom type="capsule" size="0.05 0.25" rgba="0.8 0.2 0.2 1"/>
        <body name="r_knee2" pos="0 0 -0.25">
          <joint name="r_kn2" type="hinge" axis="0 1 0" range="0 2"/>
          <geom type="capsule" size="0.04 0.2" rgba="0.8 0.2 0.2 1"/>
          <body name="r_foot2" pos="0 0 -0.2">
            <geom type="box" size="0.08 0.04 0.02" rgba="0.8 0.2 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

class BoxingEnv:
    """Self-play boxing environment for two humanoid robots."""
    
    def __init__(self, randomize=True, max_steps=2000):
        self.model = mujoco.MjModel.from_xml_string(ARENA_XML)
        self.data = mujoco.MjData(self.model)
        self.randomize = randomize
        self.max_steps = max_steps
        self.step_count = 0
        
        # Joint indices for each robot
        self.joints_1 = ['l_sh1', 'l_el1', 'r_sh1', 'r_el1', 'l_hip1', 'l_kn1', 'r_hip1', 'r_kn1', 'neck1']
        self.joints_2 = ['l_sh2', 'l_el2', 'r_sh2', 'r_el2', 'l_hip2', 'l_kn2', 'r_hip2', 'r_kn2', 'neck2']
        
        # Body indices
        self.body_1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot1')
        self.body_2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'robot2')
        self.hand_ids_1 = {
            'l_hand1': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'l_hand1'),
            'r_hand1': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'r_hand1'),
        }
        self.hand_ids_2 = {
            'l_hand2': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'l_hand2'),
            'r_hand2': mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'r_hand2'),
        }
        
        # Contact tracking
        self.punches_landed = 0
        self.punches_thrown = 0
        self.contact_history = deque(maxlen=50)
        
        # Damage tracking
        self.hp_1 = 100.0
        self.hp_2 = 100.0
        self.damage_cooldown = 0
        
        # Base masses for domain randomization
        self.base_mass_1 = self.model.body_mass[self.body_1]
        self.base_mass_2 = self.model.body_mass[self.body_2]
        self.base_friction = self.model.geom_friction[0].copy()
        
    def reset(self):
        """Reset environment with domain randomization."""
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.punches_landed = 0
        self.punches_thrown = 0
        self.hp_1 = 100.0
        self.hp_2 = 100.0
        self.damage_cooldown = 0
        self.contact_history.clear()
        
        # Domain randomization
        if self.randomize:
            # Mass jitter (±20%)
            mass_jitter = np.random.uniform(0.8, 1.2)
            self.model.body_mass[self.body_1] = self.base_mass_1 * mass_jitter
            self.model.body_mass[self.body_2] = self.base_mass_2 * mass_jitter
            
            # Friction jitter (±30%)
            friction_jitter = np.random.uniform(0.7, 1.3)
            for i in range(self.model.ngeom):
                self.model.geom_friction[i] = self.base_friction * friction_jitter
            
            # Motor gear jitter (±10%)
            for i in range(self.model.nu):
                self.model.actuator_gear[i, 0] *= np.random.uniform(0.9, 1.1)
        
        # Random starting positions (freejoint has 7 DOF: 3 pos + 4 quat)
        jnt1 = self.model.joint('root1')
        jnt2 = self.model.joint('root2')
        idx1 = int(jnt1.qposadr[0])
        idx2 = int(jnt2.qposadr[0])
        self.data.qpos[idx1:idx1+3] = [np.random.uniform(-1.2, -0.4), 0.0, 1.2]
        self.data.qpos[idx2:idx2+3] = [np.random.uniform(0.4, 1.2), 0.0, 1.2]
        # Quaternion: identity (upright)
        self.data.qpos[idx1+3:idx1+7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[idx2+3:idx2+7] = [1.0, 0.0, 0.0, 0.0]
        
        # Random initial joint angles (small perturbations)
        for j in self.joints_1 + self.joints_2:
            jnt = self.model.joint(j)
            self.data.qpos[jnt.qposadr[0]] = np.random.uniform(-0.2, 0.2)
        
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()
    
    def _get_obs(self):
        """Get observation for both agents."""
        obs = {
            'agent_1': self._get_agent_obs(1),
            'agent_2': self._get_agent_obs(2),
        }
        return obs
    
    def _get_agent_obs(self, agent_id):
        """Get observation for one agent."""
        if agent_id == 1:
            my_body = self.body_1
            opp_body = self.body_2
            my_joints = self.joints_1
            my_hp = self.hp_1
            opp_hp = self.hp_2
            my_hands = self.hand_ids_1
            opp_hands = self.hand_ids_2
        else:
            my_body = self.body_2
            opp_body = self.body_1
            my_joints = self.joints_2
            my_hp = self.hp_2
            opp_hp = self.hp_1
            my_hands = self.hand_ids_2
            opp_hands = self.hand_ids_1
        
        # Own body state
        my_pos = self.data.xpos[my_body][:3]
        my_quat = self.data.xquat[my_body][:4]
        root_jnt = self.model.joint(f'root{agent_id}')
        qvel_idx = int(root_jnt.dofadr[0])
        my_vel = self.data.qvel[qvel_idx:qvel_idx+3]
        my_angvel = self.data.qvel[qvel_idx+3:qvel_idx+6]
        
        # Own joint states
        my_joint_pos = []
        my_joint_vel = []
        for j in my_joints:
            jnt = self.model.joint(j)
            my_joint_pos.append(self.data.qpos[int(jnt.qposadr[0])])
            my_joint_vel.append(self.data.qvel[int(jnt.dofadr[0])])
        opp_pos = self.data.xpos[opp_body][:3]
        opp_root_jnt = self.model.joint(f'root{3-agent_id}')
        opp_qvel_idx = int(opp_root_jnt.dofadr[0])
        opp_vel = self.data.qvel[opp_qvel_idx:opp_qvel_idx+3]
        
        # Relative position
        rel_pos = opp_pos - my_pos
        dist = np.linalg.norm(rel_pos[:2])
        
        # Hand positions (for punching)
        my_hand_pos = [self.data.xpos[h][:3] for h in my_hands.values()]
        opp_hand_pos = [self.data.xpos[h][:3] for h in opp_hands.values()]
        
        # Contact forces on hands (punching force)
        my_hand_forces = []
        for hand_name, hand_id in my_hands.items():
            # Sum contact forces at hand
            force = 0.0
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 == hand_id or contact.geom2 == hand_id:
                    force += np.linalg.norm(contact.frame[3:6])  # Normal force
            my_hand_forces.append(force)
        
        # Compose observation vector
        obs = np.concatenate([
            my_pos,                    # 3
            my_quat,                   # 4
            my_vel,                    # 3
            my_angvel,                 # 3
            my_joint_pos,              # 9
            my_joint_vel,              # 9
            opp_pos,                   # 3
            opp_vel,                   # 3
            rel_pos,                   # 3
            [dist],                    # 1
            [my_hp / 100.0],           # 1
            [opp_hp / 100.0],          # 1
            np.array(my_hand_forces) / 100.0,  # 2
        ])
        
        return obs
    
    def step(self, actions):
        """Step environment with actions from both agents."""
        # Apply actions (joint torques)
        if 'agent_1' in actions:
            self._apply_action(actions['agent_1'], 1)
        if 'agent_2' in actions:
            self._apply_action(actions['agent_2'], 2)
        
        # Physics step
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        
        # Check for contacts (punches)
        contact_events = self._check_contacts()
        
        # Update damage
        self._update_damage(contact_events)
        
        # Compute rewards
        rewards = self._compute_rewards(contact_events)
        
        # Check termination
        done = self._check_done()
        
        # Get observations
        obs = self._get_obs()
        
        info = {
            'hp_1': self.hp_1,
            'hp_2': self.hp_2,
            'punches_landed': self.punches_landed,
            'contacts': contact_events,
        }
        
        return obs, rewards, done, info
    
    def _apply_action(self, action, agent_id):
        """Apply action (joint torques) to one agent."""
        if agent_id == 1:
            joints = self.joints_1
        else:
            joints = self.joints_2
        
        for i, joint_name in enumerate(joints):
            if i < len(action):
                jnt = self.model.joint(joint_name)
                # Find actuator for this joint
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                if act_id >= 0:
                    self.data.ctrl[act_id] = np.clip(action[i], -50, 50)
    
    def _check_contacts(self):
        """Check for hand-to-body contacts (punches)."""
        events = []
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Check if contact involves a hand and an opponent body part
            hands = {**self.hand_ids_1, **self.hand_ids_2}
            bodies = {self.body_1: 1, self.body_2: 2}
            
            hand_geom = None
            body_geom = None
            attacker = None
            defender = None
            
            for hand_name, hand_id in hands.items():
                if contact.geom1 == hand_id or contact.geom2 == hand_id:
                    hand_geom = hand_id
                    # Determine attacker from hand name
                    if hand_name.endswith('1'):
                        attacker = 1
                    else:
                        attacker = 2
            
            for body_id, owner in bodies.items():
                if (contact.geom1 == body_id or contact.geom2 == body_id) and body_id != hand_geom:
                    body_geom = body_id
                    defender = owner
            
            if hand_geom is not None and body_geom is not None and attacker != defender:
                force = np.linalg.norm(contact.frame[3:6])
                events.append({
                    'attacker': attacker,
                    'defender': defender,
                    'force': force,
                    'hand': hand_geom,
                    'body': body_geom,
                })
        
        self.contact_history.extend(events)
        return events
    
    def _update_damage(self, contact_events):
        """Update HP based on contacts."""
        if self.damage_cooldown > 0:
            self.damage_cooldown -= 1
            return
        
        for event in contact_events:
            force = event['force']
            
            # Damage scales with force, capped
            damage = min(15.0, force * 0.5)
            
            # Reduce cooldown for weak hits
            if force < 5.0:
                damage *= 0.5
            
            if event['attacker'] == 1:
                self.hp_2 -= damage
                self.punches_landed += 1
            else:
                self.hp_1 -= damage
                self.punches_landed += 1
            
            # Cooldown to prevent multi-hit spam
            self.damage_cooldown = 10
    
    def _compute_rewards(self, contact_events):
        """Compute rewards for both agents."""
        reward_1 = 0.0
        reward_2 = 0.0
        
        # Punch reward
        for event in contact_events:
            if event['attacker'] == 1:
                reward_1 += 10.0  # Landed a punch
                reward_2 -= 5.0   # Got hit
            else:
                reward_2 += 10.0
                reward_1 -= 5.0
        
        # Staying upright reward
        torso_1_z = self.data.xpos[self.body_1][2]
        torso_2_z = self.data.xpos[self.body_2][2]
        
        if torso_1_z > 0.8:
            reward_1 += 0.1
        else:
            reward_1 -= 1.0
        
        if torso_2_z > 0.8:
            reward_2 += 0.1
        else:
            reward_2 -= 1.0
        
        # Proximity reward (encourage engagement)
        dist = np.linalg.norm(self.data.xpos[self.body_1][:2] - self.data.xpos[self.body_2][:2])
        if dist < 1.5:
            reward_1 += 0.05
            reward_2 += 0.05
        elif dist > 3.0:
            reward_1 -= 0.1
            reward_2 -= 0.1
        
        # HP differential reward
        reward_1 += (self.hp_1 - self.hp_2) * 0.01
        reward_2 += (self.hp_2 - self.hp_1) * 0.01
        
        return {'agent_1': reward_1, 'agent_2': reward_2}
    
    def _check_done(self):
        """Check if episode is done."""
        # Knockdown check
        torso_1_z = self.data.xpos[self.body_1][2]
        torso_2_z = self.data.xpos[self.body_2][2]
        
        if torso_1_z < 0.5:  # Torso below 0.5m = knockdown
            return True
        if torso_2_z < 0.5:
            return True
        
        # HP check
        if self.hp_1 <= 0 or self.hp_2 <= 0:
            return True
        
        # Timeout
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def get_state(self):
        """Get raw state for analysis."""
        return {
            'robot1_pos': self.data.xpos[self.body_1].copy(),
            'robot2_pos': self.data.xpos[self.body_2].copy(),
            'hp_1': self.hp_1,
            'hp_2': self.hp_2,
            'step': self.step_count,
            'contacts': len(self.contact_history),
        }


def demo():
    """Demo the environment with random actions."""
    env = BoxingEnv(randomize=True, max_steps=1000)
    obs = env.reset()
    
    print(f"Observation shape: {obs['agent_1'].shape}")
    print(f"Initial state: {env.get_state()}")
    
    # Random actions for 100 steps
    for i in range(100):
        actions = {
            'agent_1': np.random.uniform(-10, 10, 9),
            'agent_2': np.random.uniform(-10, 10, 9),
        }
        obs, rewards, done, info = env.step(actions)
        
        if i % 20 == 0:
            print(f"Step {i}: HP1={info['hp_1']:.1f}, HP2={info['hp_2']:.1f}, R1={rewards['agent_1']:.2f}, R2={rewards['agent_2']:.2f}")
        
        if done:
            print(f"Episode ended at step {i}")
            print(f"Final state: {env.get_state()}")
            break
    
    print("\nEnvironment works!")

if __name__ == '__main__':
    demo()
