@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, object_pos: torch.Tensor, goal_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Quaternion-based orientation error
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2.0 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))
    
    # Orientation alignment: increase temperature to make it more sensitive
    orientation_reward_temperature: float = 5.0
    orientation_reward = torch.exp(-orientation_reward_temperature * angle_diff)

    # Angular velocity penalty to discourage excessive motion
    angular_velocity_penalty_weight: float = 0.2
    angular_velocity_penalty = torch.exp(-torch.norm(object_angvel * angular_velocity_penalty_weight, dim=1))

    # Distance to goal: re-scale to encourage reaching the goal
    position_diff = torch.norm(object_pos - goal_pos, dim=1)
    position_reward_temperature: float = 0.5
    distance_to_goal_reward = torch.exp(-position_reward_temperature * position_diff)

    # Task completion bonus: adjust the condition for a practical threshold
    goal_threshold = 0.2  # Radian threshold for bonus
    task_completion_bonus = (angle_diff < goal_threshold).float()

    # Combine components for total reward
    total_reward = (
        1.0 * orientation_reward +
        0.2 * angular_velocity_penalty +
        0.5 * distance_to_goal_reward +
        2.0 * task_completion_bonus
    )

    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'distance_to_goal_reward': distance_to_goal_reward,
        'task_completion_bonus': task_completion_bonus
    }

    return total_reward, reward_dict
