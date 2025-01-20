@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_pos: torch.Tensor, goal_pos: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the rotation error using quaternion multiplication
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))  # Extract angle from quaternion

    # Orientation alignment reward
    orientation_reward_temperature: float = 0.5
    orientation_reward = torch.exp(-orientation_reward_temperature * angle_diff)

    # Reformulate angular velocity penalty
    angular_velocity_penalty_weight: float = 0.1
    angular_velocity_penalty = torch.exp(-torch.norm(object_angvel * angular_velocity_penalty_weight, dim=1))

    # New task completion reward based on position difference
    position_diff = torch.norm(object_pos - goal_pos, dim=1)
    position_reward_temperature: float = 0.5
    position_reward = torch.exp(-position_reward_temperature * position_diff)

    # Total reward is a weighted sum of individual components
    total_reward = 1.0 * orientation_reward + 0.1 * angular_velocity_penalty + 0.5 * position_reward

    # Create a dictionary for each individual reward component
    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'position_reward': position_reward
    }

    return total_reward, reward_dict
