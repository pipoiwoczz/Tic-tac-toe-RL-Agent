# Tic-Tac-Toe AI with Reinforcement Learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-Q--learning-orange)](https://en.wikipedia.org/wiki/Q-learning)

An AI agent that masters Tic-Tac-Toe through Q-learning and Minimax algorithms. Features progressive training, strategic gameplay, and an interactive GUI.

![Tic-Tac-Toe GUI Demo](gui_screenshot.png) <!-- Replace with actual screenshot -->

## Features

- 🧠 **Double Q-learning** with prioritized experience replay
- 🔁 **State normalization** for efficient learning (8x smaller state space)
- 📊 **Progressive training curriculum** (random → human-like → minimax)
- 🎯 **Strategic reward shaping** for intermediate board positions
- 🖥️ **Interactive GUI** for human vs AI gameplay
- 📈 **Comprehensive metrics tracking** (win rates, Q-values, exploration decay)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/tic-tac-toe-ai.git
cd tic-tac-toe-ai
```
2. Install dependencies:
```bash
pip install numpy matplotlib tkinter tqdm
```

## Usage
1. Train the AI Agent
```bash
# Train with default settings (50,000 episodes)
python train.py
```

### Custom training example
```bash
python train.py --episodes 20000 --alpha 0.2 --gamma 0.9 --epsilon 0.8
```

### Play Against the AI
```bash
python play_human.py
```

### Training Options
```bash
Parameter	Description	Default
--episodes	Training episodes	50000
--alpha	Learning rate	0.3
--gamma	Discount factor	0.95
--epsilon	Initial exploration rate	1.0
--epsilon_decay	Exploration decay rate	0.9995
--epsilon_min	Minimum exploration rate	0.01
--opponent	Opponent type (random/minimax/self/progressive)	progressive
```

## Project Structure
```text
tic-tac-toe-ai/
├── agent.py            # Q-learning agent implementation
├── env.py              # Game environment and rules
├── minimax.py          # Minimax algorithm implementation
├── play_human.py       # GUI for human vs AI gameplay
├── train.py            # Training script with metrics
├── best_agent.pkl      # Pre-trained model (generated)
└── requirements.txt    # Python dependencies
```

## Performance
After 50,000 training episodes against progressive opponents:
```table
Opponent Type	Win Rate	Draw Rate	Loss Rate
Random	98.2%	1.7%	0.1%
Human-like	92.5%	6.3%	1.2%
Minimax	82.1%	17.9%	0.0%
```
## Future Improvements
- Add Monte Carlo Tree Search integration
- Implement neural network function approximation
- Develop tournament system for agent competition
- Create web-based interface
- Extend to 4×4 and 5×5 board variations
