import tkinter as tk
from env import TicTacToe
from agent import QLearningAgent


class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe: Human (O) vs Agent (X)")

        self.status_label = tk.Label(self.root, text="Your turn (O)", font=("Helvetica", 14))
        self.status_label.pack()

        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.reset_button = tk.Button(self.root, text="Reset Game", command=self.reset_game, font=("Helvetica", 12))
        self.reset_button.pack(pady=10)

        self.buttons = []
        self.agent = QLearningAgent()

        try:
            self.agent.load("best_agent.pkl")
            self.agent.epsilon = 0  # Disable exploration for testing
        except FileNotFoundError:
            self.status_label.config(text="No trained model found.")
            self.disable_board()
            return

        self.game = TicTacToe()
        self.create_board()

        # If agent goes first
        if self.game.current_player == 'X':
            self.root.after(300, self.agent_move)

    def create_board(self):
        for i in range(9):
            btn = tk.Button(self.frame, text=' ', font=("Helvetica", 24),
                            width=5, height=2,
                            command=lambda idx=i: self.human_move(idx))
            btn.grid(row=i // 3, column=i % 3)
            self.buttons.append(btn)

    def human_move(self, idx):
        if self.game.board[idx] != ' ' or self.game.current_player != 'O':
            return
        self.game.make_move(idx)
        self.update_button(idx, 'O')
        self.after_move()

    def agent_move(self):
        if self.game.current_player != 'X':
            return
        state = self.game.get_state()
        available = self.game.available_actions()

        if not available:
            self.end_game("Draw")
            return

        action = self.agent.choose_action(state, available)
        if action not in range(9):
            print(f"Warning: Invalid agent action: {action}")
            self.end_game("O")  # Let human win if agent fails
            return

        self.game.make_move(action)
        self.update_button(action, 'X')
        self.after_move()

    def after_move(self):
        winner = self.game.check_winner()
        if winner:
            self.end_game(winner)
            return
        self.game.switch_player()
        if self.game.current_player == 'X':
            self.status_label.config(text="Agent's turn...")
            self.root.after(300, self.agent_move)
        else:
            self.status_label.config(text="Your turn (O)")

    def update_button(self, idx, symbol):
        self.buttons[idx].config(text=symbol, state='disabled')

    def end_game(self, winner):
        for btn in self.buttons:
            btn.config(state='disabled')
        if winner == 'Draw':
            self.status_label.config(text="It's a draw!")
        elif winner == 'X':
            self.status_label.config(text="Agent wins!")
        else:
            self.status_label.config(text="You win!")

    def reset_game(self):
        self.game.reset()
        for i in range(9):
            self.buttons[i].config(text=' ', state='normal')
        self.status_label.config(text="Your turn (O)")
        if self.game.current_player == 'X':
            self.root.after(300, self.agent_move)

    def disable_board(self):
        for i in range(9):
            btn = tk.Button(self.frame, text=' ', font=("Helvetica", 24),
                            width=5, height=2, state='disabled')
            btn.grid(row=i // 3, column=i % 3)
            self.buttons.append(btn)


def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
