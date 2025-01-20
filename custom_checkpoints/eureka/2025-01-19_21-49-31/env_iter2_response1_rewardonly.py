@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_pos: torch.Tensor, goal_pos: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the rotation error using quaternion multiplication
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2.0 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))

    # Orientation alignment reward with adjusted sensitivity
    orientation_reward_temperature: float = 1.0
    orientation_reward = torch.exp(-orientation_reward_temperature * angle_diff)

    # Reformulated angular velocity constraint to maintain relevance
    angular_velocity_constraint_weight: float = 0.5
    angular_velocity_constraint = torch.exp(-torch.norm(object_angvel * angular_velocity_constraint_weight, dim=1))

    # Reintroduce task distance component
    position_diff = torch.norm(object_pos - goal_pos, dim=1)
    position_reward_temperature: float = 0.8
    distance_to_goal_reward = torch.exp(-position_reward_temperature * position_diff)

    # Add a new task completion bonus for achieving a near-goal orientation
    goal_thres = 0.1  # radian threshold for goal orientation
    task_completion_bonus = (angle_diff < goal_thres).float()

    # Total reward as a combination of components
    total_reward = (
        1.0 * orientation_reward +
        0.3 * angular_velocity_constraint +
        0.5 * distance_to_goal_reward +
        0.2 * task_completion_bonus
    )

    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_constraint': angular_velocity_constraint,
        'distance_to_goal_reward': distance_to_goal_reward,
        'task_completion_bonus': task_completion_bonus
    }

    return total_reward, reward_dict
