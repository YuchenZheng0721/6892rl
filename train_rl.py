import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import os
import glob
import random
from collections import deque
import time
import argparse
from PIL import Image, ImageDraw
import gym
from gym import spaces
import tensorflow_probability as tfp

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


class ImageNavigationEnv(gym.Env):
    def __init__(self, image_path, agent_radius=5, debug=False):
        super(ImageNavigationEnv, self).__init__()

        self.original_image = cv2.imread(image_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original_image.shape[:2]

        self.visualization_image = np.copy(self.original_image)

        self.agent_radius = agent_radius

        self.debug = debug

        self.blue_lower = np.array([0, 0, 150])
        self.blue_upper = np.array([100, 100, 255])

        self.yellow_lower = np.array([200, 200, 0])
        self.yellow_upper = np.array([255, 255, 100])

        self.green_lower = np.array([0, 150, 0])
        self.green_upper = np.array([100, 255, 100])

        self.red_lower = np.array([150, 0, 0])
        self.red_upper = np.array([255, 100, 100])

        self._create_masks()

        self.start_position = self._find_start_position()

        self.current_position = np.array(self.start_position, dtype=np.float32)

        self.target_position = self._find_target_position()

        self.max_distance = np.sqrt(self.width ** 2 + self.height ** 2)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=self.original_image.shape, dtype=np.uint8),
            'position': spaces.Box(low=np.array([0, 0]), high=np.array([self.width, self.height]), dtype=np.float32),
            'target_mask': spaces.Box(low=0, high=1, shape=(self.height, self.width), dtype=np.uint8),
            'goal_direction': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })

        self.trajectory = [self.current_position.copy()]

        self.max_steps = 1000
        self.current_step = 0

        self.target_visible = False

        self.prev_goal_direction = self._calculate_goal_direction()

        self.first_time_target_visible = False

    def _create_masks(self):
        self.floor_mask = cv2.inRange(self.original_image, self.blue_lower, self.blue_upper)

        self.moving_objects_mask = cv2.inRange(self.original_image, self.yellow_lower, self.yellow_upper)

        self.target_mask = cv2.inRange(self.original_image, self.green_lower, self.green_upper)

        self.obstacles_mask = cv2.inRange(self.original_image, self.red_lower, self.red_upper)

        self.all_obstacles_mask = cv2.bitwise_or(self.moving_objects_mask, self.obstacles_mask)

        self.safe_area_mask = cv2.bitwise_not(self.all_obstacles_mask)

    def _find_start_position(self):
        edge_points = np.where(self.floor_mask > 0)

        if len(edge_points[0]) == 0:
            return (self.agent_radius, self.agent_radius)

        x_coords = edge_points[1]
        y_coords = edge_points[0]

        min_x = np.min(x_coords)
        candidate_indices = np.where(x_coords == min_x)[0]
        best_index = candidate_indices[np.argmax(y_coords[candidate_indices])]

        x = x_coords[best_index]
        y = y_coords[best_index]

        return (x, y)

    def _find_target_position(self):
        target_points = np.where(self.target_mask > 0)

        if len(target_points[0]) == 0:
            return (self.width - self.agent_radius, self.agent_radius)

        y_center = np.mean(target_points[0])
        x_center = np.mean(target_points[1])

        return (int(x_center), int(y_center))

    def _is_safe_position(self, position):
        x, y = int(position[0]), int(position[1])

        circle_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(circle_mask, (x, y), self.agent_radius, 255, -1)

        overlap = cv2.bitwise_and(circle_mask, self.all_obstacles_mask)
        return np.count_nonzero(overlap) == 0

    def _get_min_distance_to_obstacle(self, position):
        obstacle_points = np.column_stack(np.where(self.all_obstacles_mask > 0))
        if obstacle_points.size == 0:
            return None

        distances = np.linalg.norm(obstacle_points - np.array([int(position[1]), int(position[0])]), axis=1)
        return np.min(distances)

    def _is_inside_boundary(self, position):
        x, y = int(position[0]), int(position[1])

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        contours, _ = cv2.findContours(self.floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False

        largest_contour = max(contours, key=cv2.contourArea)
        x_min, y_min, w, h = cv2.boundingRect(largest_contour)

        if (x_min + self.agent_radius <= x <= x_min + w - self.agent_radius and
                y_min + self.agent_radius <= y <= y_min + h - self.agent_radius):
            return True
        else:
            return False

    def _is_target_visible(self, position):
        x, y = int(position[0]), int(position[1])
        target_x, target_y = int(self.target_position[0]), int(self.target_position[1])

        line_points = self._bresenham_line(x, y, target_x, target_y)

        for px, py in line_points:
            if 0 <= px < self.width and 0 <= py < self.height:
                if self.all_obstacles_mask[py, px] > 0:
                    return False

        return True

    def _bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def _is_at_target(self, position):
        x, y = int(position[0]), int(position[1])

        circle_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(circle_mask, (x, y), self.agent_radius, 255, -1)

        overlap = cv2.bitwise_and(circle_mask, self.target_mask)
        return np.count_nonzero(overlap) > 0

    def _collides_with_obstacle(self, position):
        x, y = int(position[0]), int(position[1])

        circle_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(circle_mask, (x, y), self.agent_radius, 255, -1)

        overlap = cv2.bitwise_and(circle_mask, self.all_obstacles_mask)
        return np.count_nonzero(overlap) > 0

    def _calculate_goal_direction(self):
        direction = np.array(self.target_position) - self.current_position
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm
        return np.zeros(2)

    def _get_observation(self):
        self.visualization_image = np.copy(self.original_image)

        cv2.circle(self.visualization_image,
                   (int(self.current_position[0]), int(self.current_position[1])),
                   self.agent_radius, (255, 255, 255), -1)

        for i in range(1, len(self.trajectory)):
            cv2.line(self.visualization_image,
                     (int(self.trajectory[i - 1][0]), int(self.trajectory[i - 1][1])),
                     (int(self.trajectory[i][0]), int(self.trajectory[i][1])),
                     (0, 0, 255), 2)

        goal_direction = self._calculate_goal_direction()

        self.target_visible = self._is_target_visible(self.current_position)

        if self.target_visible:
            cv2.line(self.visualization_image,
                     (int(self.current_position[0]), int(self.current_position[1])),
                     (int(self.target_position[0]), int(self.target_position[1])),
                     (0, 255, 255), 1)

        normalized_target_mask = self.target_mask.astype(np.float32) / 255.0

        return {
            'image': self.original_image,
            'position': self.current_position,
            'target_mask': normalized_target_mask,
            'goal_direction': goal_direction
        }

    def reset(self):
        self.reached_goal = False
        self.current_position = np.array(self.start_position, dtype=np.float32)

        self.trajectory = [self.current_position.copy()]

        self.current_step = 0

        self.target_visible = self._is_target_visible(self.current_position)
        self.first_time_target_visible = False

        self.prev_goal_direction = self._calculate_goal_direction()

        return self._get_observation()

    def step(self, action):
        if hasattr(self, 'reached_goal') and self.reached_goal:
            observation = self._get_observation()
            done = True
            reward = 0.0
            info = {
                'reached_target': True,
                'collision': False,
                'out_of_bounds': False,
                'step': self.current_step,
                'distance_to_target': 0.0,
                'min_dist_to_obstacle': self._get_min_distance_to_obstacle(self.current_position),
                'target_visible': self.target_visible
            }
            return observation, reward, done, info

        self.current_step += 1

        max_step_size = 10.0

        if self.current_step <= 50:
            move_direction = np.array(self.target_position) - self.current_position
            move_direction = move_direction / (np.linalg.norm(move_direction) + 1e-8)

            noise_scale = max(0.2 * (1.0 - self.current_step / 50.0), 0.05)
            noise = np.random.normal(scale=noise_scale, size=2)
            move_direction = move_direction + noise
            move_direction = move_direction / (np.linalg.norm(move_direction) + 1e-8)

            action = 0.7 * move_direction + 0.3 * action
            action = action / (np.linalg.norm(action) + 1e-8)
        else:
            if self.target_visible:
                move_direction = np.array(self.target_position) - self.current_position
                move_direction = move_direction / (np.linalg.norm(move_direction) + 1e-8)

                action = 0.2 * action + 0.8 * move_direction
                action = action / (np.linalg.norm(action) + 1e-8)

        action = action / (np.linalg.norm(action) + 1e-8)
        dx = action[0] * max_step_size
        dy = action[1] * max_step_size

        new_position = self.current_position + np.array([dx, dy])

        if self._is_inside_boundary(new_position) and not self._collides_with_obstacle(new_position):
            self.current_position = new_position

        self.trajectory.append(self.current_position.copy())

        observation = self._get_observation()

        reached_target = self._is_at_target(self.current_position)
        collision = self._collides_with_obstacle(self.current_position)
        out_of_bounds = not self._is_inside_boundary(self.current_position)

        done = collision or out_of_bounds or self.current_step >= self.max_steps

        if len(self.trajectory) >= 2:
            prev_position = self.trajectory[-2]
            prev_distance = np.linalg.norm(prev_position - self.target_position)
        else:
            prev_distance = np.linalg.norm(self.current_position - self.target_position)

        current_distance = np.linalg.norm(self.current_position - self.target_position)
        distance_delta = prev_distance - current_distance

        reward = 20.0 * distance_delta

        current_goal_dir = self._calculate_goal_direction()
        action_norm = np.linalg.norm(action)
        if action_norm > 0:
            action_dir = action / action_norm
            direction_alignment = np.dot(action_dir, current_goal_dir)
            reward += 5.0 * (direction_alignment + 1.0)

        if self.target_visible and not self.first_time_target_visible:
            reward += 50.0
            self.first_time_target_visible = True
        elif self.target_visible:
            reward += 2.0

        reward -= 0.005

        min_dist_to_obstacle = self._get_min_distance_to_obstacle(self.current_position)
        if min_dist_to_obstacle is not None:
            if min_dist_to_obstacle < 100.0:
                penalty = np.exp(-(min_dist_to_obstacle / 20.0))
                reward -= penalty * 5.0

        if reached_target:
            reward += 200.0
            done = True
            self.reached_goal = True
        elif collision:
            reward = -100.0
        elif out_of_bounds:
            reward = -100.0

        info = {
            'reached_target': reached_target,
            'collision': collision,
            'out_of_bounds': out_of_bounds,
            'step': self.current_step,
            'distance_to_target': current_distance,
            'min_dist_to_obstacle': min_dist_to_obstacle,
            'target_visible': self.target_visible
        }

        return observation, reward, done, info

    def render(self, mode='human', save_path=None):
        if mode == 'human':
            plt.figure(figsize=(10, 10))
            plt.imshow(self.visualization_image)
            plt.title(f'Step: {self.current_step}, Target Visible: {self.target_visible}')

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)

            plt.show()

        return self.visualization_image

    def close(self):
        pass


def preprocess_image(image, target_size=(84, 84)):
    processed_image = cv2.resize(image, target_size)
    processed_image = processed_image / 255.0
    return processed_image


def preprocess_mask(mask, target_size=(84, 84)):
    processed_mask = cv2.resize(mask, target_size)
    processed_mask = processed_mask.astype(np.float32) / 255.0
    return processed_mask


class PPOAgent:
    def __init__(self, state_dim, action_dim, action_bound, batch_size=64, actor_lr=0.0003, critic_lr=0.001, gamma=0.99,
                 lam=0.95, clip_ratio=0.2, target_kl=0.01, epochs=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.epochs = epochs

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.actor_lr)

        self.buffer = PPOBuffer(self.state_dim, self.action_dim, batch_size, gamma, lam)

    def _build_actor(self):
        image_inputs = keras.Input(shape=self.state_dim)

        target_mask_inputs = keras.Input(shape=(self.state_dim[0], self.state_dim[1], 1))

        goal_direction_inputs = keras.Input(shape=(2,))

        conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")(image_inputs)
        conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")(conv1)
        conv3 = layers.Conv2D(64, 3, strides=1, activation="relu")(conv2)

        mask_conv1 = layers.Conv2D(32, 8, strides=4, activation="sigmoid")(target_mask_inputs)
        mask_conv2 = layers.Conv2D(64, 4, strides=2, activation="sigmoid")(mask_conv1)
        mask_conv3 = layers.Conv2D(64, 3, strides=1, activation="sigmoid")(mask_conv2)

        enhanced_features = layers.Multiply()([conv3, mask_conv3])

        combined_features = layers.Add()([conv3, enhanced_features])

        flatten = layers.Flatten()(combined_features)

        concat_features = layers.Concatenate()([flatten, goal_direction_inputs])

        dense1 = layers.Dense(512, activation="relu")(concat_features)
        dense2 = layers.Dense(256, activation="relu")(dense1)

        mu = layers.Dense(self.action_dim, activation="tanh")(dense2)
        mu = layers.Lambda(lambda x: x * self.action_bound)(mu)

        log_std = layers.Dense(self.action_dim, activation="linear")(dense2)
        log_std = layers.Lambda(lambda x: tf.clip_by_value(x, -20, 2))(log_std)
        std = layers.Lambda(lambda x: tf.exp(x))(log_std)

        model = keras.Model(inputs=[image_inputs, target_mask_inputs, goal_direction_inputs], outputs=[mu, std])

        return model

    def _build_critic(self):
        image_inputs = keras.Input(shape=self.state_dim)

        target_mask_inputs = keras.Input(shape=(self.state_dim[0], self.state_dim[1], 1))

        goal_direction_inputs = keras.Input(shape=(2,))

        conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")(image_inputs)
        conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")(conv1)
        conv3 = layers.Conv2D(64, 3, strides=1, activation="relu")(conv2)

        mask_conv1 = layers.Conv2D(32, 8, strides=4, activation="sigmoid")(target_mask_inputs)
        mask_conv2 = layers.Conv2D(64, 4, strides=2, activation="sigmoid")(mask_conv1)
        mask_conv3 = layers.Conv2D(64, 3, strides=1, activation="sigmoid")(mask_conv2)

        enhanced_features = layers.Multiply()([conv3, mask_conv3])

        combined_features = layers.Add()([conv3, enhanced_features])

        flatten = layers.Flatten()(combined_features)

        concat_features = layers.Concatenate()([flatten, goal_direction_inputs])

        dense1 = layers.Dense(512, activation="relu")(concat_features)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        value = layers.Dense(1)(dense2)

        model = keras.Model(inputs=[image_inputs, target_mask_inputs, goal_direction_inputs], outputs=value)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.critic_lr), loss="mse")

        return model

    def get_action(self, state, target_mask, goal_direction, deterministic=False):
        state_batch = np.expand_dims(state, axis=0)

        target_mask_batch = np.expand_dims(target_mask, axis=0)
        if len(target_mask_batch.shape) == 3:
            target_mask_batch = np.expand_dims(target_mask_batch, axis=-1)

        goal_direction_batch = np.expand_dims(goal_direction, axis=0)

        mu, std = self.actor.predict([state_batch, target_mask_batch, goal_direction_batch])
        mu, std = mu[0], std[0]

        if deterministic:
            return mu

        action = np.random.normal(mu, std)
        action = np.clip(action, -self.action_bound, self.action_bound)

        return action

    def get_value(self, state, target_mask, goal_direction):
        state_batch = np.expand_dims(state, axis=0)

        target_mask_batch = np.expand_dims(target_mask, axis=0)
        if len(target_mask_batch.shape) == 3:
            target_mask_batch = np.expand_dims(target_mask_batch, axis=-1)

        goal_direction_batch = np.expand_dims(goal_direction, axis=0)

        return self.critic.predict([state_batch, target_mask_batch, goal_direction_batch])[0, 0]

    def store(self, state, target_mask, goal_direction, action, reward, next_state, next_target_mask,
              next_goal_direction, done, value, log_prob=None):
        full_state = {
            'image': state,
            'target_mask': target_mask,
            'goal_direction': goal_direction
        }

        full_next_state = {
            'image': next_state,
            'target_mask': next_target_mask,
            'goal_direction': next_goal_direction
        }

        self.buffer.store(full_state, action, reward, full_next_state, done, value, log_prob)

    def train(self):
        self.buffer.finish_path()

        states, actions, advantages, returns, old_values = self.buffer.get()

        images = np.array([s['image'] for s in states])
        target_masks = np.array([s['target_mask'] for s in states])
        target_masks = np.expand_dims(target_masks, axis=-1)
        goal_directions = np.array([s['goal_direction'] for s in states])

        actor_loss = self._train_actor(images, target_masks, goal_directions, actions, advantages)

        critic_loss = self._train_critic(images, target_masks, goal_directions, returns)

        self.buffer.clear()

        return actor_loss, critic_loss

    def _train_actor(self, images, target_masks, goal_directions, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.actor([images, target_masks, goal_directions], training=True)

            dist = tfp.distributions.Normal(mu, std)

            log_probs = dist.log_prob(actions)
            log_probs = tf.reduce_sum(log_probs, axis=1)

            old_log_probs = self.buffer.log_probs
            ratio = tf.exp(log_probs - old_log_probs)

            clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_adv))

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        return loss.numpy()

    def _train_critic(self, images, target_masks, goal_directions, returns):
        return self.critic.train_on_batch([images, target_masks, goal_directions], returns)


class PPOBuffer:
    def __init__(self, state_dim, action_dim, size, gamma=0.99, lam=0.95):
        self.state_buf = []
        self.next_state_buf = []

        self.action_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.value_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, state, action, reward, next_state, done, value, log_prob=None):
        assert self.ptr < self.max_size

        if self.ptr >= len(self.state_buf):
            self.state_buf.append(state)
            self.next_state_buf.append(next_state)
        else:
            self.state_buf[self.ptr] = state
            self.next_state_buf[self.ptr] = next_state

        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.value_buf[self.ptr] = value
        if log_prob is not None:
            self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def finish_path(self, last_value=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buf[path_slice], last_value)
        values = np.append(self.value_buf[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        self.ret_buf[path_slice] = self._discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x, discount):
        discounted_sum = np.zeros_like(x)
        discounted_sum[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):
            discounted_sum[t] = x[t] + discount * discounted_sum[t + 1]
        return discounted_sum

    def get(self):
        assert self.ptr > 0, "Buffer is empty"

        data_slice = slice(0, self.ptr)

        states = self.state_buf[:self.ptr]

        adv_mean, adv_std = np.mean(self.adv_buf[data_slice]), np.std(self.adv_buf[data_slice])
        self.adv_buf[data_slice] = (self.adv_buf[data_slice] - adv_mean) / (adv_std + 1e-8)

        return (states, self.action_buf[data_slice],
                self.adv_buf[data_slice], self.ret_buf[data_slice],
                self.value_buf[data_slice])

    def clear(self):
        self.ptr, self.path_start_idx = 0, 0
        self.state_buf = []
        self.next_state_buf = []


def train(env, agent, episodes=1, max_steps=1000, render_interval=50):
    best_reward = -float('inf')
    rewards_history = []
    success_history = []

    max_buffer_fill = min(agent.batch_size, agent.buffer.max_size)

    for episode in range(1, episodes + 1):
        obs = env.reset()
        state = preprocess_image(obs['image'])
        target_mask = preprocess_mask(obs['target_mask'])
        goal_direction = obs['goal_direction']

        episode_reward = 0
        success = False

        if agent.buffer.ptr >= max_buffer_fill:
            agent.buffer.clear()

        for step in range(max_steps):
            action = agent.get_action(state, target_mask, goal_direction)

            next_obs, reward, done, info = env.step(action)

            next_state = preprocess_image(next_obs['image'])
            next_target_mask = preprocess_mask(next_obs['target_mask'])
            next_goal_direction = next_obs['goal_direction']

            if agent.buffer.ptr < max_buffer_fill:
                value = agent.get_value(state, target_mask, goal_direction)
                agent.store(
                    state, target_mask, goal_direction,
                    action, reward,
                    next_state, next_target_mask, next_goal_direction,
                    done, value
                )

            state = next_state
            target_mask = next_target_mask
            goal_direction = next_goal_direction
            episode_reward += reward

            if done:
                success = info['reached_target']
                break

        if agent.buffer.ptr >= agent.batch_size:
            actor_loss, critic_loss = agent.train()
            print(f"Episode {episode}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

        rewards_history.append(episode_reward)
        success_history.append(1 if success else 0)

        print(
            f"Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, Success: {success}, Target Visible: {info.get('target_visible', False)}")

        if episode % render_interval == 0 or episode == 1:
            env.render(save_path=f'training_images/episode_{episode}.png')

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.actor.save_weights("best_actor.weights.h5")
            agent.critic.save_weights("best_critic.weights.h5")
            print(f"Best model saved with reward: {best_reward:.2f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    window_size = min(1000, len(success_history))
    success_rate = [np.mean(success_history[max(0, i - window_size):i + 1]) for i in range(len(success_history))]
    plt.plot(success_rate)
    plt.title('Success Rate (Moving Average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    plt.tight_layout()
    plt.savefig('training_results.png')

    return rewards_history, success_rate


def test(env, agent, episodes=10, render=True):
    success_count = 0
    rewards = []

    for episode in range(1, episodes + 1):
        obs = env.reset()
        state = preprocess_image(obs['image'])
        target_mask = preprocess_mask(obs['target_mask'])
        goal_direction = obs['goal_direction']

        episode_reward = 0
        success = False

        while True:
            action = agent.get_action(state, target_mask, goal_direction, deterministic=True)

            next_obs, reward, done, info = env.step(action)

            state = preprocess_image(next_obs['image'])
            target_mask = preprocess_mask(next_obs['target_mask'])
            goal_direction = next_obs['goal_direction']

            episode_reward += reward

            if done:
                success = info['reached_target']
                break

        rewards.append(episode_reward)
        if success:
            success_count += 1

        print(f"Test Episode {episode}/{episodes}, Reward: {episode_reward:.2f}, Success: {success}")

        if render:
            env.render(save_path=f'test_images/episode_{episode}.png')

    success_rate = success_count / episodes
    avg_reward = np.mean(rewards)
    print(f"Test Results: Success Rate: {success_rate:.2f}, Average Reward: {avg_reward:.2f}")

    return success_rate, avg_reward


def create_env_from_image(image_path, agent_radius=5):
    return ImageNavigationEnv(image_path, agent_radius=agent_radius)


def main():
    class Args:
        image_dir = 'images'
        episodes = 1000
        batch_size = 64
        render_interval = 100
        test = False
        test_episodes = 10

    args = Args()

    os.makedirs('training_images', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)

    image_files = glob.glob(os.path.join(args.image_dir, '*.png'))
    if not image_files:
        print(f"No images found in {args.image_dir}")
        return

    initial_env = create_env_from_image(image_files[0])

    state_dim = (84, 84, 3)
    action_dim = 2
    action_bound = 1.0

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        batch_size=args.batch_size
    )

    if args.test:
        agent.actor.load_weights("best_actor.weights.h5")
        agent.critic.load_weights("best_critic.weights.h5")

        for image_file in image_files:
            print(f"Testing on {image_file}")
            env = create_env_from_image(image_file)
            test(env, agent, episodes=args.test_episodes)
    else:
        total_episode_count = 0
        total_rewards = []
        total_success = []

        for episode in range(args.episodes):
            image_file = random.choice(image_files)
            print(f"Training episode {episode + 1} on map: {os.path.basename(image_file)}")

            env = create_env_from_image(image_file)

            rewards, success_rate = train(env, agent,
                                          episodes=1,
                                          max_steps=env.max_steps,
                                          render_interval=args.render_interval)

            total_rewards.extend(rewards)
            total_success.extend(success_rate)
            total_episode_count += 1

            if (episode + 1) % 100 == 0:
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                plt.plot(total_rewards)
                plt.title(f'Overall Rewards (Episodes: {total_episode_count})')
                plt.xlabel('Episode')
                plt.ylabel('Reward')

                plt.subplot(1, 2, 2)
                window_size = min(1000, len(total_success))
                avg_success = [np.mean(total_success[max(0, i - window_size):i + 1]) for i in range(len(total_success))]
                plt.plot(avg_success)
                plt.title('Overall Success Rate (Moving Average)')
                plt.xlabel('Episode')
                plt.ylabel('Success Rate')

                plt.tight_layout()
                plt.savefig(f'training_overall_{total_episode_count}.png')
                plt.close()


if __name__ == "__main__":
    main()