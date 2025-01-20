@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor, success_indicator: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the rotation error using quaternion multiplication
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))

    # Revised orientation alignment reward
    orientation_reward_temperature: float = 0.2  # Increased the emphasis
    orientation_reward = torch.exp(-orientation_reward_temperature * angle_diff)

    # Modified angular velocity penalty to be slightly less prominent
    angular_velocity_penalty_temperature: float = 0.1
    angular_velocity_penalty = torch.exp(-angular_velocity_penalty_temperature * torch.norm(object_angvel, dim=1))

    # Task completion score based on success indicator (simulating environment signal)
    task_completion_bonus = 0.5 * success_indicator

    # Total reward is a weighted sum of individual components
    total_reward = orientation_reward + 0.1 * angular_velocity_penalty + task_completion_bonus

    # Create a dictionary for each individual reward component
    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'task_completion_bonus': task_completion_bonus
    }

    return total_reward, reward_dict
