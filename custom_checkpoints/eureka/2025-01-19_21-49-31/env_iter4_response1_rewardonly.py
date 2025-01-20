@torch.jit.script
def compute_reward(
    object_rot: torch.Tensor, 
    goal_rot: torch.Tensor, 
    object_angvel: torch.Tensor, 
    object_pos: torch.Tensor, 
    goal_pos: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    
    # Adjusted quaternion-based orientation error
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2.0 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))
    
    # Rewritten Orientation Reward: normalizing over pi and scaled
    orientation_reward_temperature: float = 10.0
    orientation_reward = 1.0 - (angle_diff / torch.pi)
    orientation_reward = torch.exp(-orientation_reward_temperature * orientation_reward)

    # Re-evaluate Angular Velocity Penalty for better scaling
    angular_velocity_penalty_weight: float = 0.5
    angular_velocity_penalty = torch.exp(-angular_velocity_penalty_weight * torch.norm(object_angvel, dim=1))
    
    # Scaled Distance to Goal Reward
    position_diff = torch.norm(object_pos - goal_pos, dim=1)
    position_reward_temperature: float = 0.5
    distance_to_goal_reward = torch.exp(-position_reward_temperature * position_diff)

    # Enhanced Task Completion Bonus: lowering threshold and adjusting the reward
    goal_threshold = 0.5  # Looser threshold
    task_completion_bonus = (angle_diff < goal_threshold).float() * 5.0  # Increased reward

    # Combine components for total reward, balancing different parts
    total_reward = (
        2.0 * orientation_reward +
        0.5 * angular_velocity_penalty +
        1.0 * distance_to_goal_reward +
        5.0 * task_completion_bonus
    )

    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'distance_to_goal_reward': distance_to_goal_reward,
        'task_completion_bonus': task_completion_bonus
    }

    return total_reward, reward_dict
