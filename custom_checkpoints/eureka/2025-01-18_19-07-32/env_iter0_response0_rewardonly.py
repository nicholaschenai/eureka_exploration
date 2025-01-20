@torch.jit.script
def compute_reward(root_states: torch.Tensor, potentials: torch.Tensor, prev_potentials: torch.Tensor, vel_loc: torch.Tensor, roll: torch.Tensor, yaw: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Reward for forward velocity (maximize forward speed)
    forward_velocity_reward = vel_loc[:, 0]  # Assuming forward direction is along the x-axis
    
    # Penalty for unnecessary rotational motions (both roll and yaw)
    roll_penalty = torch.abs(roll)
    yaw_penalty = torch.abs(yaw)

    # Reward or penalty based on change in potential, encouraging moving towards a target if defined
    potential_difference = potentials - prev_potentials
    potential_reward = potential_difference

    # Temperature parameters for exponential scaling to adjust sensitivity
    velocity_temp = 0.1
    roll_temp = 0.05
    yaw_temp = 0.05
    potential_temp = 0.1

    # Applying exponential scaling to the individual components
    scaled_velocity_reward = torch.exp(forward_velocity_reward * velocity_temp) - 1.0
    scaled_roll_penalty = -torch.exp(roll_penalty * roll_temp) + 1.0
    scaled_yaw_penalty = -torch.exp(yaw_penalty * yaw_temp) + 1.0
    scaled_potential_reward = torch.exp(potential_reward * potential_temp) - 1.0

    # Total reward combining all components
    total_reward = scaled_velocity_reward + scaled_roll_penalty + scaled_yaw_penalty + scaled_potential_reward

    # Dictionary to store individual reward components
    reward_dict = {
        "scaled_velocity_reward": scaled_velocity_reward,
        "scaled_roll_penalty": scaled_roll_penalty,
        "scaled_yaw_penalty": scaled_yaw_penalty,
        "scaled_potential_reward": scaled_potential_reward,
    }

    return total_reward, reward_dict
