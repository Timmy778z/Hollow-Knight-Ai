# Hollow-Knight-Ai Code

env.py:

import gym
from gym import spaces
import numpy as np
from utils import capture_screen_for_model, get_hp, get_geo, detect_enemies, detect_boss_defeat, exploration_reward
import pyautogui
import time

class HollowKnightEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(6)  # 0-left,1-right,2-jump,3-attack,4-dash,5-do nothing
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)
        self.last_hp = 100
        self.last_geo = 0

    def step(self, action):
        self._take_action(action)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        return obs, reward, done, {}

    def reset(self):
        time.sleep(1)
        self.last_hp = 100
        self.last_geo = 0
        return self._get_obs()

    def _take_action(self, action):
        if action == 0:
            pyautogui.keyDown('left'); time.sleep(0.1); pyautogui.keyUp('left')
        elif action == 1:
            pyautogui.keyDown('right'); time.sleep(0.1); pyautogui.keyUp('right')
        elif action == 2:
            pyautogui.press('space')  # jump
        elif action == 3:
            pyautogui.press('z')  # attack
        elif action == 4:
            pyautogui.press('x')  # dash
        elif action == 5:
            pass

    def _get_obs(self):
        return capture_screen_for_model()

    def _compute_reward(self):
        current_hp = get_hp()
        current_geo = get_geo()
        enemies_hit = detect_enemies()
        reward = 0

        # Survival reward
        reward += (current_hp - self.last_hp) * 2
        self.last_hp = current_hp

        # Geo reward
        reward += (current_geo - self.last_geo) * 0.1
        self.last_geo = current_geo

        # Enemy hits
        reward += enemies_hit * 1.0

        # Boss defeat
        if detect_boss_defeat():
            reward += 100

        # Exploration reward
        reward += exploration_reward()

        return reward

    def _check_done(self):
        return self.last_hp <= 0

utils.py:

import pygetwindow as gw
import pyautogui
import cv2
import numpy as np
from PIL import Image
import pytesseract

discovered_positions = set()

def get_game_window():
    windows = gw.getWindowsWithTitle("Hollow Knight")
    if not windows:
        raise Exception("Hollow Knight window not found!")
    win = windows[0]
    return win.left, win.top, win.width, win.height

def capture_game_window():
    left, top, width, height = get_game_window()
    img = pyautogui.screenshot(region=(left, top, width, height))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def capture_screen_for_model():
    img = capture_game_window()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (84, 84))
    return np.expand_dims(img, axis=-1)

def get_hp():
    left, top, width, height = get_game_window()
    hp_region = (left + int(0.05*width), top + int(0.05*height), int(0.2*width), int(0.03*height))
    img = pyautogui.screenshot(region=hp_region)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    hp_pixels = cv2.countNonZero(img)
    return hp_pixels / (hp_region[2]*hp_region[3])

def get_geo():
    left, top, width, height = get_game_window()
    geo_region = (left + int(0.7*width), top + int(0.05*height), int(0.15*width), int(0.05*height))
    img = pyautogui.screenshot(region=geo_region)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(img, config='--psm 7 digits')
    try:
        return int(''.join(filter(str.isdigit, text)))
    except:
        return 0

def detect_enemies():
    img = capture_game_window()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,100,100])
    upper_red = np.array([10,255,255])
    mask = cv2.inRange(img, lower_red, upper_red)
    count = cv2.countNonZero(mask)
    return count // 50

def detect_boss():
    img = capture_game_window()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Placeholder: check top-center area for boss HP bar
    return np.count_nonzero(img[0:20, img.shape[1]//2 - 100: img.shape[1]//2 + 100]) > 50

def detect_boss_defeat():
    return not detect_boss()

def get_player_position():
    return pyautogui.position()

def exploration_reward():
    pos = get_player_position()
    grid_pos = (pos[0] // 50, pos[1] // 50)
    if grid_pos not in discovered_positions:
        discovered_positions.add(grid_pos)
        return 5
    return 0

Train.py:

from stable_baselines3 import PPO
from env import HollowKnightEnv

env = HollowKnightEnv()
model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=50000)
model.save("hollow_knight_ai_model")

run_ai.py:

from stable_baselines3 import PPO
from env import HollowKnightEnv

env = HollowKnightEnv()
model = PPO.load("hollow_knight_ai_model")

obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

requirements.txt:
opencv-python
pyautogui
numpy
stable-baselines3
torch
gym
pillow
pytesseract
pygetwindow
