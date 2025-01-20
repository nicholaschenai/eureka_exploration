@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, episode_step: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between current orientation and goal orientation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Calculate the angle from quaternion difference
    angular_distance = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))
    
    # Temperature parameters for transforming components
    orientation_temperature = 1.5
    angular_velocity_temperature = 0.5
    time_bonus_temperature = 0.01

    # Reward for minimizing the angular distance
    orientation_reward = torch.exp(-orientation_temperature * angular_distance)

    # (Optional) Refine penalty for angular velocity's magnitude, or discard it
    angular_velocity_penalty = -torch.norm(object_angvel, dim=1)

    # Encourage achieving the target orientation sooner
    time_bonus = torch.exp(-time_bonus_temperature * episode_step.float())

    # Combine the rewards with consideration to optimization scale
    total_reward = orientation_reward + 0.1 * angular_velocity_penalty + 0.2 * time_bonus

    # Return the total reward and individual components
    reward_components = {
        "orientation_reward": orientation_reward,
        "angular_velocity_penalty": angular_velocity_penalty,
        "time_bonus": time_bonus
    }

    return total_reward, reward_components
