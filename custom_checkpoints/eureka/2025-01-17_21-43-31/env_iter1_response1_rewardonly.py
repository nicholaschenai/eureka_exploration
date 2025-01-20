@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between current orientation and goal orientation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Convert quaternion difference to angle (in radians)
    angular_distance = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))

    # Temperature parameters for transforming rewards
    orientation_temperature = 2.0
    angvel_temperature = 1.0

    # Reward for minimizing the angular distance
    orientation_reward = torch.exp(-orientation_temperature * angular_distance)

    # Penalty for the magnitude of angular velocity
    angular_velocity_penalty = torch.exp(-angvel_temperature * torch.norm(object_angvel, dim=1))

    # Combine the rewards
    total_reward = orientation_reward + 0.05 * angular_velocity_penalty

    # Return the total reward and individual components
    reward_components = {
        "orientation_reward": orientation_reward,
        "angular_velocity_penalty": angular_velocity_penalty
    }

    return total_reward, reward_components
