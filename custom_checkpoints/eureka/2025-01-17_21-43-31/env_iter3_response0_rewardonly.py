@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between current orientation and goal orientation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Convert quaternion difference to angle (in radians)
    angular_distance = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))
    
    # Temperature parameters for transforming rewards
    orientation_temperature = 5.0
    angvel_temperature = 2.0

    # Reward for minimizing the angular distance with emphasis
    orientation_reward = 1.0 - torch.tanh(orientation_temperature * angular_distance) 

    # Penalty for the angular velocity magnitude
    angular_velocity_penalty = torch.exp(-angvel_temperature * torch.norm(object_angvel, dim=1))

    # New component: Scaled reward for achieving reduction in angular distance over time, normalized over possible scale
    progress_reward = 1.0 - orientation_reward

    # Combine the rewards
    total_reward = orientation_reward + 0.03 * angular_velocity_penalty + 0.1 * progress_reward

    # Return the total reward and individual components
    reward_components = {
        "orientation_reward": orientation_reward,
        "angular_velocity_penalty": angular_velocity_penalty,
        "progress_reward": progress_reward
    }

    return total_reward, reward_components
