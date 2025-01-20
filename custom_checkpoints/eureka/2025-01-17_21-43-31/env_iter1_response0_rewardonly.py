@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotational difference and normalize
    rot_diff_tensor = quat_mul(object_rot, quat_conjugate(goal_rot))[:, :4]
    rot_diff = torch.norm(rot_diff_tensor[:, :3], dim=1)

    # Temperature parameters for transforming rewards
    rot_diff_temperature = 5.0  # Reduced to make a more noticeable change
    angvel_temperature = 0.5  # Reduced to smooth since previous was high
    zone_stable_temperature = 2.0

    # Reward for minimizing the rotational difference
    # Using a reward function that emphasizes minimization
    orientation_reward = 1.0 - torch.tanh(rot_diff_temperature * rot_diff)

    # Small penalty for any angular velocity to encourage stabilizing behavior
    angular_velocity_penalty = 1.0 - torch.tanh(angvel_temperature * torch.norm(object_angvel, dim=1))

    # Encourage maintaining stability when alignment is within a "zone"
    goal_alignment_threshold = 0.1
    within_stable_zone = torch.where(rot_diff < goal_alignment_threshold, torch.tensor(1.0, device=object_rot.device), torch.tensor(0.0, device=object_rot.device))
    zone_stability_reward = within_stable_zone * torch.exp(-zone_stable_temperature * torch.norm(object_angvel, dim=1))

    # Combine the rewards for a balanced score
    total_reward = orientation_reward * 0.5 + angular_velocity_penalty * 0.3 + zone_stability_reward * 0.2

    # Return the total reward and individual components for analysis
    reward_components = {
        "orientation_reward": orientation_reward,
        "angular_velocity_penalty": angular_velocity_penalty,
        "zone_stability_reward": zone_stability_reward
    }

    return total_reward, reward_components
