@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between current orientation and goal orientation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Convert quaternion difference to angle (in radians)
    angular_distance = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))

    # Re-defined transform temperatures for improved scaling
    orientation_temperature = 1.5
    angvel_temperature = 1.0

    # Enhance orientation reward transformation
    orientation_reward = 1.0 - torch.exp(-orientation_temperature * (1.0 - angular_distance))
    
    # Redefine angular velocity penalty transformation
    angular_velocity_penalty = torch.exp(-angvel_temperature * torch.norm(object_angvel, dim=1))

    # Normalization temperature and factor for improved scaling
    total_reward_temperature = 0.3
    
    # Combine the rewards with adjusted weights for more significant feedback
    total_reward = orientation_reward - 0.5 * angular_velocity_penalty

    # Apply an overall exponential scaling to the reward to standardize impact
    total_reward = torch.exp(-total_reward_temperature * total_reward)

    # Return the total reward and components
    reward_components = {
        "orientation_reward": orientation_reward,
        "angular_velocity_penalty": angular_velocity_penalty
    }

    return total_reward, reward_components
