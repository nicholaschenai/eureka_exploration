@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, episode_progress: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between current orientation and goal orientation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Convert quaternion difference to angle (in radians)
    angular_distance = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))

    # Temperature parameter for orientation transformation
    orientation_temperature = 3.0

    # Reward for minimizing the angular distance
    orientation_reward = torch.exp(-orientation_temperature * angular_distance)

    # New penalty to discourage longer episodes without progress
    episode_length_penalty = 0.01 * episode_progress

    # Total reward calculation
    total_reward = orientation_reward - episode_length_penalty

    # Include an explicit incentive for being near the goal
    near_goal_bonus = torch.where(angular_distance < 0.1, torch.tensor(0.5, device=object_rot.device), torch.tensor(0.0, device=object_rot.device))
    total_reward += near_goal_bonus

    # Return the total reward and individual components
    reward_components = {
        "orientation_reward": orientation_reward,
        "episode_length_penalty": episode_length_penalty,
        "near_goal_bonus": near_goal_bonus
    }

    return total_reward, reward_components
