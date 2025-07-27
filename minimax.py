# minimax.py
class MinimaxAgent:
    def choose_action(self, game):
        def minimax(board, is_maximizing, player, opponent):
            winner = game.check_winner()
            if winner == player: return 1
            if winner == opponent: return -10
            if winner == 'Draw': return 0
            if is_maximizing:
                best = -float('inf')
                for action in game.available_actions():
                    game.make_move(action)
                    score = minimax(board, False, player, opponent)
                    game.board[action] = ' '
                    best = max(best, score)
                return best
            else:
                best = float('inf')
                for action in game.available_actions():
                    game.make_move(action)
                    score = minimax(board, True, player, opponent)
                    game.board[action] = ' '
                    best = min(best, score)
                return best
            
        best_action = None
        best_score = -float('inf')
        for action in game.available_actions():
            game.make_move(action)
            score = minimax(game.board, False, game.current_player, 'O' if game.current_player == 'X' else 'X')
            game.board[action] = ' '
            if score > best_score:
                best_score = score
                best_action = action

        return best_action