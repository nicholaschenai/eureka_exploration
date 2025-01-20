@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Calculate the rotation error using quaternion multiplication
    rotation_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    angle_diff = 2 * torch.acos(torch.clamp(rotation_diff[:, 0], -1.0, 1.0))  # Extract angle from quaternion

    # Orientation alignment reward (negative of angle difference)
    orientation_reward_temperature: float = 0.1
    orientation_reward = torch.exp(-orientation_reward_temperature * angle_diff)

    # Optionally, penalize angular velocity to encourage smooth rotation towards the goal
    angular_velocity_penalty_temperature: float = 0.05
    angular_velocity_penalty = torch.exp(-angular_velocity_penalty_temperature * torch.norm(object_angvel, dim=1))

    # Total reward is a weighted sum of individual components
    total_reward = orientation_reward + 0.2 * angular_velocity_penalty

    # Create a dictionary for each individual reward component
    reward_dict = {
        'orientation_reward': orientation_reward,
        'angular_velocity_penalty': angular_velocity_penalty
    }

    return total_reward, reward_dict
