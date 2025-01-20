@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the rotation error using quaternion multiplication
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))  # Extract angle from quaternion

    # Orientation alignment reward
    orientation_reward_temperature: float = 1.0
    orientation_reward = torch.exp(-orientation_reward_temperature * angle_diff)

    # Angular velocity penalty to encourage smooth rotations
    angular_velocity_penalty_temperature: float = 0.05
    angular_velocity_penalty = torch.exp(-angular_velocity_penalty_temperature * torch.norm(object_angvel, dim=1))

    # Success reward for achieving alignment within threshold
    success_threshold: float = 0.1  # Arbitrary threshold for success
    success_reward = torch.where(angle_diff < success_threshold, torch.tensor(1.0, device=object_rot.device), torch.tensor(0.0, device=object_rot.device))

    # Total reward calculation
    total_reward = 1.0 * orientation_reward + 0.1 * angular_velocity_penalty + 5.0 * success_reward

    # Reward component tracking
    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_penalty': angular_velocity_penalty,
        'success_reward': success_reward
    }

    return total_reward, reward_dict
