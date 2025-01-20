@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the rotational difference between the current object orientation and goal orientation
    rot_diff_tensor = quat_mul(object_rot, quat_conjugate(goal_rot))[:, :4]  # Only take the first 4 elements as it's a quaternion
    rot_diff = torch.norm(rot_diff_tensor[:, :3], dim=1)  # Measure angular difference

    # Temperature parameters for transforming rewards
    rot_diff_temperature = 10.0
    angvel_temperature = 1.0

    # Reward for minimizing the rotational difference
    orientation_reward = torch.exp(-rot_diff_temperature * rot_diff)

    # Penalty for the magnitude of angular velocity
    angular_velocity_penalty = torch.exp(-angvel_temperature * torch.norm(object_angvel, dim=1))

    # Combine the rewards
    total_reward = orientation_reward + 0.1 * angular_velocity_penalty

    # Return the total reward and individual components
    reward_components = {
        "orientation_reward": orientation_reward,
        "angular_velocity_penalty": angular_velocity_penalty
    }

    return total_reward, reward_components
