@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion distance between current orientation and goal orientation
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    
    # Convert quaternion difference to angle (in radians)
    angular_distance = 2.0 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))

    # Setting new temperature parameters
    orientation_temperature = 5.0
    angvel_temperature = 1.0

    # Enhance orientation reward sensitivity
    orientation_reward = torch.exp(-orientation_temperature * angular_distance)

    # Stronger penalization for high angular velocity, encouraging slower, controlled spins
    angular_velocity_penalty = 1.0 - torch.exp(-angvel_temperature * torch.norm(object_angvel, dim=1))

    # Combine the rewards with adjusted weights
    total_reward = 3.0 * orientation_reward + 0.5 * angular_velocity_penalty

    # Return the total reward and individual components
    reward_components = {
        "orientation_reward": orientation_reward,
        "angular_velocity_penalty": angular_velocity_penalty
    }

    return total_reward, reward_components
