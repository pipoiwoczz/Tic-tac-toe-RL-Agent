# env.py

class TicTacToe:
    """A 3x3 Tic-Tac-Toe game environment."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Initialize an empty board with X as the first player."""
        self.board = [' '] * 9
        self.current_player = 'X'
        return self.get_state()

    def get_state(self):
        """Return the current board state as a string."""
        return ''.join(self.board)

    def normalize_state(self):
        """
        Normalize board state to a canonical form using rotations and reflections.
        
        Returns:
            str: Canonical state representation.
        """
        board_str = self.get_state()
        # Rotations
        rotations = [board_str]
        temp_board = self.board[:]
        for _ in range(3):
            temp_board = [temp_board[6], temp_board[3], temp_board[0],
                         temp_board[7], temp_board[4], temp_board[1],
                         temp_board[8], temp_board[5], temp_board[2]]
            rotations.append(''.join(temp_board))
        # Reflections
        reflections = [
            ''.join([self.board[2], self.board[1], self.board[0],
                    self.board[5], self.board[4], self.board[3],
                    self.board[8], self.board[7], self.board[6]]),  # Horizontal
            ''.join([self.board[6], self.board[7], self.board[8],
                    self.board[3], self.board[4], self.board[5],
                    self.board[0], self.board[1], self.board[2]])   # Vertical
        ]
        return min([board_str] + rotations + reflections)

    def available_actions(self):
        """Return a list of empty cell indices (0-8)."""
        return [i for i, cell in enumerate(self.board) if cell == ' ']

    def make_move(self, action):
        """
        Apply a move to the board.
        
        Args:
            action (int): Position (0-8) to place the current player's symbol.
            
        Returns:
            bool: True if move is valid, False otherwise.
        """
        if action < 0 or action > 8:
            raise ValueError("Action must be an integer between 0 and 8")
        if self.board[action] != ' ':
            return False
        self.board[action] = self.current_player
        return True

    def switch_player(self):
        """Switch the current player (X to O or vice versa)."""
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        """Check if there is a winner or draw."""
        wins = [(0,1,2), (3,4,5), (6,7,8),
                (0,3,6), (1,4,7), (2,5,8),
                (0,4,8), (2,4,6)]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                return self.board[a]
        if ' ' not in self.board:
            return 'Draw'
        return None