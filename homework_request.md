HW3-3: Enhance DQN for random mode WITH Training Tips [30%]

Convert the DQN model from PyTorch to either:
Keras, or
PyTorch Lightning
Bonus points for integrating training techniques to stabilize/improve learning
(e.g., gradient clipping, learning rate scheduling, etc.)

write tf.keras to progressively solve:

Static Mode

Player Mode

Random Mode

using DQN mechanisms S1 to S5 incrementally.

For each mode:

first analyze the environment difficulty

analyze training instability symptoms

identify which DQN weakness appears

determine which mechanism is necessary

implement ONLY the required schemes

skip unnecessary advanced schemes

continue progressively to the next mode

The implementation must evolve incrementally:

Stage 1:
Basic DQN

S1 Replay Buffer

S2 Target Network

Stage 2:
Extend Stage 1

S3 Double DQN

S4 Dueling DQN

Stage 3:
Extend Stage 2

S5 Prioritized Experience Replay

stabilization tricks

The generated code must:

use TensorFlow and tf.keras

use custom GradientTape training loops

avoid model.fit()

provide complete runnable implementations

include replay buffer logic

include target synchronization

include TD target computation

include epsilon-greedy exploration

include reward/loss visualization

include training analysis and debugging discussion

The generated explanation must clearly explain:

what failure symptom appears

why current mechanisms fail

why the selected scheme solves the problem

why other schemes are skipped

The final output should resemble a real DRL engineering project suitable for:

homework submission

DRL education

debugging practice

reinforcement learning research prototyping.