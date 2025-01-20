@torch.jit.script
def compute_reward(root_states: torch.Tensor, dof_vel: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant state information
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute direction to target and normalize
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    normalized_to_target = torch.nn.functional.normalize(to_target, dim=-1)
    
    # Speed component: Project velocity onto the target direction
    speed_along_target = (velocity * normalized_to_target).sum(dim=-1)
    
    # Maintain upright position (modified to scale appropriately with additional temperature)
    target_z = 1.0
    z_deviation = torch.abs(torso_position[:, 2] - target_z)
    upright_temp = 1.0
    upright_bonus = torch.exp(-upright_temp * z_deviation)

    # Stability penalty and joint smoothness reward (to encourage smooth motion)
    stability_penalty_weight = 1.0
    stability_penalty = stability_penalty_weight * z_deviation
    
    dof_velocity_penalty_weight = 0.1
    joint_smoothness_reward = torch.exp(-dof_velocity_penalty_weight * torch.norm(dof_vel, p=1, dim=-1))
    
    # Calculate overall reward
    speed_reward_weight = 1.0
    upright_bonus_weight = 0.5
    joint_smoothness_weight = 0.4
    
    speed_temp = 0.2
    
    speed_reward = speed_reward_weight * torch.exp(speed_temp * speed_along_target)
    upright_reward = upright_bonus_weight * upright_bonus
    joint_smoothness_reward = joint_smoothness_weight * joint_smoothness_reward
    
    # Total reward
    total_reward = speed_reward + upright_reward + joint_smoothness_reward - stability_penalty
    
    rewards = {
        "speed_reward": speed_reward,
        "upright_reward": upright_reward,
        "joint_smoothness_reward": joint_smoothness_reward,
        "stability_penalty": stability_penalty
    }

    return total_reward, rewards
