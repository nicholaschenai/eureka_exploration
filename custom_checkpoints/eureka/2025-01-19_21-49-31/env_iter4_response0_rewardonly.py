@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, object_pos: torch.Tensor, goal_pos: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute angular difference as continuous penalty
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2.0 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))
    
    # Updated orientation reward based on angular distance
    orientation_penalty_temperature: float = 3.0
    orientation_reward = torch.exp(-orientation_penalty_temperature * angle_diff)

    # Penalize high angular velocities
    angular_velocity_penalty_temperature: float = 1.0
    angular_velocity_penalty = torch.exp(-angular_velocity_penalty_temperature * torch.norm(object_angvel, dim=1))

    # Reward for being close to the goal position
    position_diff = torch.norm(object_pos - goal_pos, dim=1)
    position_reward_temperature: float = 2.0
    distance_to_goal_reward = torch.exp(-position_reward_temperature * position_diff)

    # Task completion bonus to encourage reaching desirable orientations
    goal_threshold = 0.1  # Tighter threshold for precision
    task_completion_bonus = (angle_diff < goal_threshold).float() * 5.0  # Larger bonus for success

    # Adjusted total reward combining different aspects
    total_reward = (
        2.0 * orientation_reward +
        1.0 * angular_velocity_penalty +
        1.5 * distance_to_goal_reward +
        task_completion_bonus
    )

    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'distance_to_goal_reward': distance_to_goal_reward,
        'task_completion_bonus': task_completion_bonus
    }

    return total_reward, reward_dict
