import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from env import TicTacToe
from agent import QLearningAgent
from minimax import MinimaxAgent

def get_opponent_action(game, opponent_type, opponent_agent, minimax_agent, difficulty=1.0):
    """Enhanced opponent action selection with difficulty scaling"""
    actions = game.available_actions()
    if not actions:
        return None
    
    if opponent_type == 'random':
        return random.choice(actions)
    
    elif opponent_type == 'self':
        return opponent_agent.choose_action(game.normalize_state(), actions, explore=True)
    
    elif opponent_type == 'human-like':
        # Scale human-like mistakes based on difficulty
        if random.random() > difficulty:
            return random.choice(actions)
        return human_like_move(game)
    
    else:  # minimax
        # For minimax, difficulty affects depth (though Tic-Tac-Toe is simple enough)
        return minimax_agent.choose_action(game)

def human_like_move(game):
    """Enhanced human-like move simulation with more realistic patterns"""
    board = game.board
    actions = game.available_actions()
    player = game.current_player
    
    # 1. Immediate win
    for a in actions:
        board[a] = player
        if game.check_winner() == player:
            board[a] = ' '
            return a
        board[a] = ' '
    
    # 2. Block opponent
    opponent = 'O' if player == 'X' else 'X'
    for a in actions:
        board[a] = opponent
        if game.check_winner() == opponent:
            board[a] = ' '
            return a
        board[a] = ' '
    # 3. Random choice with strategic bias
    if len(actions) == 1:
        return actions[0]  # Only one move available
    # len(actions) == 2:
        
    
    # 4. Strategic positions with some randomness
    preferred_order = [4, 0, 2, 6, 8, 1, 3, 5, 7]  # center > corners > edges
    preferred_available = [a for a in preferred_order if a in actions]
    if preferred_available:
        # 80% chance to pick best available, 20% to pick randomly
        return preferred_available[0] if random.random() < 0.8 else random.choice(preferred_available)
    
    return random.choice(actions)

def calculate_reward(game, action, player):
    """Enhanced reward calculation with more nuanced shaping"""
    board = game.board.copy()
    board[action] = player
    opponent = 'O' if player == 'X' else 'X'
    
    # Immediate game outcome rewards
    if game.check_winner() == player:
        return 1.0  # Win
    elif game.check_winner() == opponent:
        return -1.0  # Loss
    elif game.check_winner() == 'Draw':
        return 0.3  # Draw
    
    # Intermediate strategic rewards
    reward = 0
    
    # Reward for creating potential winning lines
    winning_lines = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), 
                    (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    
    for line in winning_lines:
        line_state = [board[i] for i in line]
        if line_state.count(player) == 2 and line_state.count(' ') == 1:
            reward += 0.2  # Potential win next move
        
        if line_state.count(opponent) == 2 and line_state.count(' ') == 1:
            reward -= 0.3  # Potential loss next move
    
    # Reward for center control
    if action == 4:  # Center position
        reward += 0.1
    
    # Reward for corner positions
    elif action in [0, 2, 6, 8]:
        reward += 0.05
    
    return reward

def train(episodes=200000, alpha=0.3, gamma=0.95, epsilon=1.0,
          epsilon_decay=0.9995, epsilon_min=0.01, opponent_type="progressive"):
    """
    Enhanced training function with improved progressive difficulty and monitoring.
    """
    # Initialize agent with enhanced settings
    agent = QLearningAgent(
        alpha=alpha,
        alpha_min=0.01,
        alpha_decay=0.999995,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        use_double_q=True,
        use_prioritized_replay=True,
        replay_capacity=100000,
        batch_size=64,
        initial_q=0.1
    )
    
    minimax_agent = MinimaxAgent()
    opponent_agent = QLearningAgent() if opponent_type in ['self', 'progressive'] else None
    
    # Enhanced metrics tracking
    metrics = {
        'episodes': [],
        'win_rates': [],
        'exploration_rate': [],
        'learning_rate': [],
        'avg_reward': [],
        'q_values': []
    }
    
    checkpoint = max(1, episodes // 100)
    results = {'win': 0, 'draw': 0, 'loss': 0}
    checkpoint_results = {'win': 0, 'draw': 0, 'loss': 0}
    total_rewards = []
    
    # Early stopping variables
    best_win_rate = 0
    patience = 20
    patience_counter = 0
    
    progress_bar = trange(episodes, desc="Training")
    
    for episode in progress_bar:
        # Determine opponent configuration
        current_opponent, difficulty = get_training_configuration(episode, opponent_type)
        
        game = TicTacToe()
        state = game.normalize_state()
        episode_rewards = []
        
        while True:
            # Agent's move
            actions = game.available_actions()
            if not actions:
                break
                
            action = agent.choose_action(state, actions, explore=True)
            game.make_move(action)
            next_state = game.normalize_state()
            
            # Enhanced reward calculation
            reward = calculate_reward(game, action, 'X')
            episode_rewards.append(reward)
            
            winner = game.check_winner()
            next_actions = [] if winner else game.available_actions()
            
            # Store experience
            agent.update(state, action, reward, next_state, next_actions)
            
            if winner:
                update_results(results, checkpoint_results, winner, 'X')
                break
                
            state = next_state

            # Switch player
            game.switch_player()
            
            # Opponent's move
            if game.available_actions():
                opp_action = get_opponent_action(
                    game, current_opponent, opponent_agent, minimax_agent, difficulty
                )
                if opp_action is not None:
                    game.make_move(opp_action)
                    
                    winner = game.check_winner()
                    if winner:
                        update_results(results, checkpoint_results, winner, 'X')
                        break
                        
                    state = game.normalize_state()
                    game.switch_player()
        
        # Update metrics
        total_rewards.append(np.mean(episode_rewards))
        
        # Progress reporting
        if episode % 100 == 0:
            progress_bar.set_postfix({
                'win%': f"{results['win']/(episode+1):.1%}",
                'ε': f"{agent.epsilon:.3f}",
                'α': f"{agent.alpha:.4f}",
                'avg_r': f"{np.mean(total_rewards[-100:]):.2f}"
            })
        
        # Checkpoint saving
        if episode % checkpoint == 0 and episode > 0:
            total_games = sum(checkpoint_results.values())
            metrics['episodes'].append(episode)
            metrics['win_rates'].append(checkpoint_results['win'] / total_games)
            metrics['exploration_rate'].append(agent.epsilon)
            metrics['learning_rate'].append(agent.alpha)
            metrics['avg_reward'].append(np.mean(total_rewards[-checkpoint:]))
            
            # Track Q-value growth
            if agent.Q:
                sample_state = random.choice(list(agent.Q.keys()))
                metrics['q_values'].append(np.mean(list(agent.Q[sample_state].values())))
            
            checkpoint_results = {'win': 0, 'draw': 0, 'loss': 0}
            
            # Early stopping check
            current_win_rate = metrics['win_rates'][-1]
            if current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                patience_counter = 0
                agent.save('best_agent.pkl')  # Save best performing agent
            else:
                patience_counter += 1
                if patience_counter >= patience and episode > episodes//2:
                    print(f"\nEarly stopping at episode {episode} with win rate {best_win_rate:.1%}")
                #     break
    
    # Final processing
    agent.save('trained_agent.pkl')
    plot_enhanced_training_results(metrics, opponent_type)
    
    return results

def get_training_configuration(episode, opponent_type):
    """Determine opponent type and difficulty based on training progress"""
    if opponent_type != 'progressive':
        return opponent_type, 1.0  # Full difficulty
    
    # Progressive difficulty schedule
    progress = min(episode / 200000, 1.0)  # Normalized progress
    
    if progress < 0.25:  # First 25% - random opponents
        return 'random', 0.0
    elif progress < 0.5:  # 25-50% - mix random and human-like
        if random.random() < (progress - 0.25)*4:
            return 'human-like', 0.5
        return 'random', 0.0
    elif progress < 0.75:  # 50-75% - human-like and minimax
        if random.random() < (progress - 0.5)*4:
            return 'minimax', 0.75
        return 'human-like', min(0.5 + (progress - 0.5)*2, 1.0)
    else:  # Final 25% - mostly minimax
        if random.random() < 0.9:
            return 'minimax', 1.0
        return 'human-like', 1.0

def update_results(results, checkpoint_results, winner, agent_mark):
    """Update result trackers based on game outcome"""
    if winner == agent_mark:
        results['win'] += 1
        checkpoint_results['win'] += 1
    elif winner == 'Draw':
        results['draw'] += 1
        checkpoint_results['draw'] += 1
    else:
        results['loss'] += 1
        checkpoint_results['loss'] += 1

def plot_enhanced_training_results(metrics, opponent_type):
    """Enhanced plotting with more detailed visualizations"""
    plt.figure(figsize=(15, 10))
    
    # Main win rate plot
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episodes'], metrics['win_rates'], 'g-', linewidth=2)
    plt.title(f'Win Rate vs {opponent_type.capitalize()} Opponent', fontsize=12)
    plt.xlabel('Training Episodes', fontsize=10)
    plt.ylabel('Win Rate', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1)
    
    # Learning dynamics subplot
    plt.subplot(2, 2, 2)
    plt.plot(metrics['episodes'], metrics['exploration_rate'], 'b-', label='Exploration (ε)')
    plt.plot(metrics['episodes'], metrics['learning_rate'], 'r-', label='Learning Rate (α)')
    plt.title('Learning Parameters', fontsize=12)
    plt.xlabel('Training Episodes', fontsize=10)
    plt.ylabel('Parameter Value', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Reward tracking
    plt.subplot(2, 2, 3)
    plt.plot(metrics['episodes'], metrics['avg_reward'], 'm-')
    plt.title('Average Reward per Episode', fontsize=12)
    plt.xlabel('Training Episodes', fontsize=10)
    plt.ylabel('Average Reward', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Q-value growth
    if metrics['q_values']:
        plt.subplot(2, 2, 4)
        plt.plot(metrics['episodes'], metrics['q_values'], 'c-')
        plt.title('Average Q-Value Growth', fontsize=12)
        plt.xlabel('Training Episodes', fontsize=10)
        plt.ylabel('Average Q-Value', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an enhanced Q-learning Tic-Tac-Toe agent')
    parser.add_argument('--episodes', type=int, default=10000, help='Total training episodes')
    parser.add_argument('--alpha', type=float, default=0.3, help='Initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995, help='Epsilon decay rate')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--opponent', type=str, default='progressive',
                      choices=['random', 'self', 'minimax', 'human-like', 'progressive'],
                      help='Opponent type for training')
    
    args = parser.parse_args()
    
    results = train(
        episodes=50000,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        opponent_type=args.opponent
    )
    
    total = sum(results.values())
    print("\nFinal Training Results:")
    print(f"Wins: {results['win']} ({results['win']/total:.1%})")
    print(f"Draws: {results['draw']} ({results['draw']/total:.1%})")
    print(f"Losses: {results['loss']} ({results['loss']/total:.1%})")