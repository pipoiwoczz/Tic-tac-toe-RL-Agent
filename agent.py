import random
import pickle
import numpy as np
from collections import deque
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class QLearningAgent:
    """Enhanced Q-learning agent for Tic-Tac-Toe with advanced features."""
    
    def __init__(self, 
                 alpha=0.3, 
                 alpha_min=0.01, 
                 alpha_decay=0.999995,
                 gamma=0.95, 
                 epsilon=1.0, 
                 epsilon_decay=0.9995, 
                 epsilon_min=0.01,
                 initial_q=0.1,
                 use_double_q=False,
                 use_prioritized_replay=False,
                 replay_capacity=10000,
                 batch_size=32):
        
        # Learning parameters
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_q = initial_q
        
        # Q-tables
        self.Q = {}
        self.use_double_q = use_double_q
        if use_double_q:
            self.Q2 = {}
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.priorities = deque(maxlen=replay_capacity)
            self.max_priority = 1.0
        
        # Tracking
        self.training_steps = 0

    def get_q(self, state, action, use_secondary=False):
        """Get Q-value for a state-action pair with optimistic initialization."""
        q_table = self.Q2 if (use_secondary and self.use_double_q) else self.Q
        
        if state not in q_table:
            q_table[state] = {a: self.initial_q for a in range(9)}
        if action not in q_table[state]:
            q_table[state][action] = self.initial_q
        return q_table[state][action]
        

    def choose_action(self, state, actions, explore=True):
        """
        Enhanced action selection with Boltzmann exploration and strategic preferences.
        
        Args:
            state: Current board state
            actions: List of available actions
            explore: Whether to include exploration
            
        Returns:
            Selected action
        """
        if not actions:
            return None

        # Check if agent is in a winning position
        for action in actions:
            temp = list(state)
            temp[action] = 'X'  # Assume agent is 'X'
            if self.check_win(temp, 'X'):
                return action  # Take winning move immediately

        # Check if opponent is about to win
        for action in actions:
            temp = list(state)
            temp[action] = 'O'  # Assume opponent is 'O'
            if self.check_win(temp, 'O'):
                return action  # Block opponent's winning move
            
        # Exploration with strategic preferences
        if explore and random.random() < self.epsilon:
            # Weight actions by strategic value (center > corners > edges)
            weights = []
            for a in actions:
                if a == 4: weight = 1.5    # Center
                elif a in [0, 2, 6, 8]: weight = 1.2  # Corners
                else: weight = 1.0  # Edges
                weights.append(weight)
            return int(random.choices(actions, weights=weights, k=1)[0])
        
        # Boltzmann exploration during exploitation
        qs = np.array([self.get_q(state, a) for a in actions])
        temperature = max(0.1, self.epsilon)  # Temperature proportional to exploration
        exp_qs = np.exp(qs / temperature)
        probs = exp_qs / exp_qs.sum()
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, next_actions):
        """Update Q-values using experience."""
        # Store experience in replay buffer
        experience = (state, action, reward, next_state, next_actions)
        self.replay_buffer.push(experience)
        
        if self.use_prioritized_replay:
            # Initialize with maximum priority
            self.priorities.append(self.max_priority)
        
        # Learn from replay buffer
        self.update_from_replay()

    def update_from_replay(self):
        """Update Q-values using experiences from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch from replay buffer
        if self.use_prioritized_replay:
            # Prioritized experience replay sampling
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.replay_buffer), 
                                     size=self.batch_size,
                                     p=probs)
            batch = [self.replay_buffer.buffer[i] for i in indices]
        else:
            # Uniform sampling
            batch = self.replay_buffer.sample(self.batch_size)
        
        # Update Q-values for each experience in the batch
        max_next_q = 0.0
        for state, action, reward, next_state, next_actions in batch:
            if not next_actions:
                # Terminal state: no future action to learn from
                target = reward
            else:
                if self.use_double_q:
                    if random.random() < 0.5:
                        max_next_action = max(next_actions, key=lambda a: self.get_q(next_state, a))
                        target = reward + self.gamma * self.get_q(next_state, max_next_action, use_secondary=True)
                        old_q = self.get_q(state, action)
                        self.Q[state][action] = old_q + self.alpha * (target - old_q)
                    else:
                        max_next_action = max(next_actions, key=lambda a: self.get_q(next_state, a, use_secondary=True))
                        target = reward + self.gamma * self.get_q(next_state, max_next_action)
                        old_q = self.get_q(state, action, use_secondary=True)
                        if state not in self.Q2:
                            self.Q2[state] = {a: self.initial_q for a in range(9)}
                        self.Q2[state][action] = old_q + self.alpha * (target - old_q)
                else:
                    max_next_q = max([self.get_q(next_state, a) for a in next_actions])
                    target = reward + self.gamma * max_next_q
                    old_q = self.get_q(state, action)
                    self.Q[state][action] = old_q + self.alpha * (target - old_q)

             # Compute td_error after Q updated
            if self.use_prioritized_replay:
                # Ensure priorities match replay buffer size
                while len(self.priorities) > len(self.replay_buffer):
                    self.priorities.popleft()
                while len(self.priorities) < len(self.replay_buffer):
                    self.priorities.append(self.max_priority)

                probs = np.array(self.priorities)
                probs = probs / probs.sum()
                indices = np.random.choice(len(self.replay_buffer), 
                                        size=self.batch_size,
                                        p=probs)
                batch = [self.replay_buffer.buffer[i] for i in indices]
            else:
                batch = self.replay_buffer.sample(self.batch_size)
            
        self.training_steps += 1
        self.decay_parameters()

    def decay_parameters(self):
        """Decay learning rate and exploration rate over time."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)

    def estimate_win_probability(self, state, action, depth=1):
        """
        Estimate win probability by looking ahead.
        Returns value between -1 (certain loss) and 1 (certain win).
        """
        board = list(state)
        board[action] = 'X'  # Assume we're player X
        
        # Check immediate win
        if self.check_win(board, 'X'):
            return 1.0

            
        # Check immediate loss if opponent plays optimally
        opponent = 'O'
        opponent_actions = [i for i, cell in enumerate(board) if cell == ' ']
        for a in opponent_actions:
            board[a] = opponent
            if self.check_win(board, opponent):
                return -1.0
            board[a] = ' '
        
        # Recursive lookahead
        if depth > 0:
            best_opponent_value = 1.0  # Worst case for us
            for a in opponent_actions:
                board[a] = opponent
                next_state = ''.join(board)
                next_actions = [i for i, cell in enumerate(board) if cell == ' ']
                if next_actions:
                    qs = [self.get_q(next_state, na) for na in next_actions]
                    our_value = max(qs)
                    opponent_value = -self.estimate_win_probability(next_state, 
                                                                  np.argmax(qs), 
                                                                  depth-1)
                    best_opponent_value = min(best_opponent_value, opponent_value)
                board[a] = ' '
            return best_opponent_value
        
        return 0.0  # Neutral if no immediate outcome

    @staticmethod
    def check_win(board, player):
        """Check if the specified player has won."""
        wins = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),    # columns
            (0, 4, 8), (2, 4, 6)                # diagonals
        ]
        return any(all(board[i] == player for i in combo) for combo in wins)

    def save(self, filename='q_table.pkl'):
        """Save Q-table(s) to file."""
        try:
            data = {
                'Q': self.Q,
                'training_steps': self.training_steps,
                'epsilon': self.epsilon,
                'alpha': self.alpha
            }
            if self.use_double_q:
                data['Q2'] = self.Q2
                
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    def load(self, filename='q_table.pkl'):
        """Load Q-table(s) from file."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.Q = data.get('Q', {})
                if self.use_double_q:
                    self.Q2 = data.get('Q2', {})
                self.training_steps = data.get('training_steps', 0)
                self.epsilon = data.get('epsilon', self.epsilon)
                self.alpha = data.get('alpha', self.alpha)
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Starting with empty Q-table.")
            self.Q = {}
            if self.use_double_q:
                self.Q2 = {}
        except Exception as e:
            print(f"Error loading Q-table: {e}")
            self.Q = {}
            if self.use_double_q:
                self.Q2 = {}

    def get_stats(self):
        """Return training statistics."""
        return {
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'states_learned': len(self.Q),
            'replay_memory': len(self.replay_buffer)
        }