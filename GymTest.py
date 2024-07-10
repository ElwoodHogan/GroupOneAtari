import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Create the environment
env = gym.make('CartPole-v1')

# Build the neural network
model = Sequential([
    Dense(24, input_shape=(4,), activation='relu'),  # Adjust to match the expected state dimensions
    Dense(24, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

episodes = 10
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        print("State before processing:", state)  # Debugging to check the state format

        # Ensure state is a flat, homogeneous NumPy array
        if isinstance(state, tuple):
            # If state is a tuple, assume the array we need is the first element
            state = state[0]

        # Further debugging to confirm the content and shape
        print("State to be converted to array:", state)

        try:
            state_array = np.array(state, dtype=np.float32).reshape(1, -1)
        except Exception as e:
            print("Error while reshaping:", e)
            break  # Break the loop to avoid further errors

        action = np.argmax(model.predict(state_array)[0])

        # Execute action
        results = env.step(action)
        if isinstance(results, tuple) and len(results) == 5:
            next_state, reward, done, _, info = results  # Adjust unpacking if environment returns 5 items
        else:
            print("Unexpected format from env.step:", results)
            break

        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1} completed with total reward: {total_reward}")

env.close()
