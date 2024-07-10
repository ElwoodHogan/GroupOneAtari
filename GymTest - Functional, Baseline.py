import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Create the environment
env = gym.make('CartPole-v1')

# Build the neural network
model = Sequential([
    Dense(24, input_shape=(4,), activation='relu'),  # Input shape matches the array's length
    Dense(24, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Start interacting with the environment
output = env.reset()
state = output[0] if isinstance(output, tuple) else output  # Extract state if it's part of a tuple

for _ in range(1000):
    # Prepare state for prediction
    state_array = np.array(state).reshape(1, -1)  # Reshape state to fit model input
    action = np.argmax(model.predict(state_array)[0])
    
    # Perform action
    output = env.step(action)
    next_state = output[0] if isinstance(output, tuple) else output  # Extract next state if it's part of a tuple
    reward, done, info = output[1], output[2], output[3]  # Unpack other elements normally

    if done:
        output = env.reset()
        state = output[0] if isinstance(output, tuple) else output  # Reset and extract new state
    else:
        state = next_state  # Update state

env.close()  # Close the environment
