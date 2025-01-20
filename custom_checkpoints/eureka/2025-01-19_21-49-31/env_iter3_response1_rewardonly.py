@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_pos: torch.Tensor, goal_pos: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate orientation alignment
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2.0 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))
  
    # Refined orientation reward
    orientation_reward_temperature: float = 5.0
    orientation_reward = torch.exp(-orientation_reward_temperature * angle_diff)

    # Angular velocity constraint to prevent too rapid spinning
    angular_velocity_penalty: float = 0.2
    angular_velocity_constraint = torch.exp(-torch.norm(object_angvel * angular_velocity_penalty, dim=1))

    # Consider angular proximity rather than position
    angle_to_goal_penalty: float = 0.5
    angular_proximity_reward = torch.exp(-angle_to_goal_penalty * angle_diff)
    
    # New task completion mechanism: bonus increases as we closely align orientation
    completion_threshold: float = 0.05  
    task_completion_bonus = (angle_diff < completion_threshold).float()

    # Aggregate reward
    total_reward = (
        1.5 * orientation_reward +
        0.5 * angular_velocity_constraint +
        2.0 * angular_proximity_reward +
        3.0 * task_completion_bonus
    )

    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_constraint': angular_velocity_constraint,
        'angular_proximity_reward': angular_proximity_reward,
        'task_completion_bonus': task_completion_bonus
    }

    return total_reward, reward_dict
