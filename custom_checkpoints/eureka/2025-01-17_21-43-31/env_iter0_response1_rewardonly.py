@torch.jit.script
def compute_reward(object_rot: torch.Tensor, goal_rot: torch.Tensor, object_angvel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Compute the quaternion angle error between object and goal orientation
    orientation_error = quat_angle_error(object_rot, goal_rot)

    # Encourage minimizing the orientation error
    orientation_error_reward = 1.0 - torch.clamp(orientation_error / torch.pi, 0.0, 1.0)

    # Encourage positive angular velocity magnitude to ensure spinning
    angvel_norm = torch.norm(object_angvel, dim=-1, p=2)
    angvel_reward = angvel_norm / (1.0 + angvel_norm)

    # Total Reward
    total_reward = orientation_error_reward + angvel_reward

    # Transformations with temperature factors
    temperature_orientation = 0.5
    orientation_exp_reward = torch.exp(-temperature_orientation * orientation_error)

    temperature_velocity = 0.1
    angvel_exp_reward = torch.exp(temperature_velocity * angvel_reward)

    # Combined Reward with exponential transformations
    transformed_total_reward = orientation_exp_reward + angvel_exp_reward

    # Reward components dictionary
    reward_components = {
        "orientation_error_reward": orientation_error_reward,
        "angvel_reward": angvel_reward,
        "orientation_exp_reward": orientation_exp_reward,
        "angvel_exp_reward": angvel_exp_reward,
    }

    return transformed_total_reward, reward_components
