import random
import numpy as np
import pygame
from gymnasium import spaces



# 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]



class SnakeCore:
    def __init__(self, render_mode=None, fps=None, max_steps=None,
                width=1000, height=700, cell=10, seed=None, half_size=3,
                penalty_range=1, penalty_max=0.01):
        self.render_mode = render_mode
        self.fps = fps if fps is not None else 25
        self.max_steps = max_steps

        self.width = int(width)
        self.height = int(height)

        # for watcher/evaluator
        self.WIDTH = self.width
        self.HEIGHT = self.height

        self.cell = int(cell)
        self.grid_w = self.width // self.cell
        self.grid_h = self.height // self.cell

        self.rng = random.Random(seed)
        
        # Configurable observation window size
        self.half_size = int(half_size)
        grid_size = (2 * self.half_size + 1) ** 2  # e.g., 5x5 = 25
        obs_size = grid_size + 2  # grid + 2 direction values

        # Penalty configuration
        self.penalty_range = int(penalty_range)  # How far penalties propagate (e.g., 2 = 5x5 area)
        self.penalty_max = float(penalty_max)    # Maximum penalty at distance 1 from danger

        # Gym-style spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_size,), dtype=np.float32)

        # Game state
        self.snake_pos = None
        self.snake_body = None
        self.food_pos = None
        self.direction = 'RIGHT'
        self.change_to = 'RIGHT'
        self.score = 0

        self._steps = 0
        self.ate = False
        self.dead = False

        self._prev_d = None
        self._no_progress_steps = 0
        self._dir_hist = []
        
        # Store the last local grid for move_delta calculation
        grid_dim = 2 * self.half_size + 1
        self.local_grid_array = np.zeros((grid_dim, grid_dim), dtype=np.float32)

        self.reset(seed=seed)


    def _check_danger(self, grid_x, grid_y):
        """Check if position (grid_x, grid_y) is dangerous (wall or body)."""
        # Wall check
        if grid_x < 0 or grid_x >= self.grid_w or grid_y < 0 or grid_y >= self.grid_h:
            return True
        # Body check
        pos = [grid_x * self.cell, grid_y * self.cell]
        return pos in self.snake_body


    def _get_cell_value_delta(self, gx, gy, fx, fy, current_dist):
        """
        Calculate cell value based on distance delta.
        
        Delta = current_dist - grid_cell_dist
        - Positive delta = moving to this cell gets you CLOSER to food
        - Negative delta = moving to this cell gets you FARTHER from food
        - Large negative = dangerous (wall/body)
        
        Args:
            gx, gy: Grid coordinates of cell
            fx, fy: Food grid coordinates
            current_dist: Current distance from head to food
        
        Returns:
            float: Cell value
        """
        # Check if dangerous (wall or body) - very negative value
        if gx < 0 or gx >= self.grid_w or gy < 0 or gy >= self.grid_h:
            return -10.0  # Large negative for walls
        
        pos = [gx * self.cell, gy * self.cell]
        if pos in self.snake_body:
            return -10.0  # Large negative for body
        
        # Calculate distance from this cell to food
        grid_cell_dist = ((gx - fx)**2 + (gy - fy)**2) ** 0.5
        
        # Delta: positive = gets closer, negative = gets farther
        delta = current_dist - grid_cell_dist
        
        return delta


    def get_action_space(self):
        return self.action_space


    def get_observation_space(self):
        return self.observation_space


    def get_fps(self):
        return self.fps


    def get_surface_size(self):
        return (self.width, self.height)


    def get_max_steps(self):
        return self.max_steps


    # ---------- helpers ----------
    def _spawn_food(self):
        # place on an empty grid cell
        while True:
            fx = self.rng.randrange(1, self.grid_w) * self.cell
            fy = self.rng.randrange(1, self.grid_h) * self.cell
            if [fx, fy] not in self.snake_body:
                self.food_pos = [fx, fy]
                return


    def _start_snake(self):
        self.snake_pos = [100, 50]
        self.snake_body = [
            [100, 50],
            [100 - self.cell, 50],
            [100 - (2 * self.cell), 50]
        ]
        self.direction = 'RIGHT'
        self.change_to = self.direction


    def _obs(self):
        hx = self.snake_pos[0] // self.cell
        hy = self.snake_pos[1] // self.cell
        fx = self.food_pos[0] // self.cell
        fy = self.food_pos[1] // self.cell

        # Current head distance to food
        current_dist = ((hx - fx)**2 + (hy - fy)**2) ** 0.5

        # Initialize both the main grid and a separate grid for penalties
        grid_dim = 2 * self.half_size + 1
        local_grid_2d = np.zeros((grid_dim, grid_dim), dtype=np.float32)
        penalty_grid = np.zeros((grid_dim, grid_dim), dtype=np.float32)
        
        # --- Single Pass with Linear Fade Penalty ---
        for dy in range(-self.half_size, self.half_size + 1):
            for dx in range(-self.half_size, self.half_size + 1):
                # 1. Calculate the base value for the current cell
                gx = hx + dx
                gy = hy + dy
                cell_value = self._get_cell_value_delta(gx, gy, fx, fy, current_dist)
                
                arr_y = dy + self.half_size
                arr_x = dx + self.half_size
                local_grid_2d[arr_y, arr_x] = cell_value
                
                # 2. If this cell is a danger zone, add penalties for its neighbors
                if cell_value < 0:
                    # Propagate penalties in a square radius defined by penalty_range
                    for ndy in range(-self.penalty_range, self.penalty_range + 1):
                        for ndx in range(-self.penalty_range, self.penalty_range + 1):
                            # Don't penalize the danger cell itself
                            if ndy == 0 and ndx == 0:
                                continue
                            
                            neighbor_y = arr_y + ndy
                            neighbor_x = arr_x + ndx
                            
                            # Check bounds before applying the penalty
                            if 0 <= neighbor_y < grid_dim and 0 <= neighbor_x < grid_dim:
                                #_dist = np.sqrt(ndx**2 + ndy**2)
                                #_dist = max(abs(ndx), abs(ndy))
                                _dist = 1
                                penalty = self.penalty_max / _dist
                                penalty_grid[neighbor_y, neighbor_x] -= penalty
        
        # After the loop, apply all collected penalties to the main grid
        local_grid_2d += penalty_grid
        
        # Store the final grid for the move_delta calculation in the step() function
        self.local_grid_array = local_grid_2d.copy()
        
        # Flatten the 2D grid into a 1D list for the observation space
        local_grid = local_grid_2d.flatten().tolist()
        
        # Add the snake's current direction to the observation
        if self.direction == 'UP':
            dir_x, dir_y = 0, -1
        elif self.direction == 'DOWN':
            dir_x, dir_y = 0, 1
        elif self.direction == 'LEFT':
            dir_x, dir_y = -1, 0
        else: # 'RIGHT'
            dir_x, dir_y = 1, 0
        
        obs = local_grid + [
            (dir_x + 1) / 2.0,
            (dir_y + 1) / 2.0
        ]
        
        return np.array(obs, dtype=np.float32)


    # ---------- required API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self._steps = 0
        self.score = 0
        self.ate = False
        self.dead = False
        self._no_progress_steps = 0
        self._dir_hist = []
        self._start_snake()
        self._spawn_food()

        # initialize distance baseline
        self._prev_d = self._compute_distance()

        return self._obs()


    def _compute_distance(self):
        hx = (self.snake_pos[0] / self.cell) / self.grid_w
        hy = (self.snake_pos[1] / self.cell) / self.grid_h
        fx = (self.food_pos[0] / self.cell) / self.grid_w
        fy = (self.food_pos[1] / self.cell) / self.grid_h
        return float(((hx - fx)**2 + (hy - fy)**2) ** 0.5)


    def step(self, action):
        # Store OLD head position BEFORE moving
        old_hx = self.snake_pos[0] // self.cell
        old_hy = self.snake_pos[1] // self.cell
        fx = self.food_pos[0] // self.cell
        fy = self.food_pos[1] // self.cell
        
        # map discrete action to direction change
        if action == 0: self.change_to = 'UP'
        elif action == 1: self.change_to = 'RIGHT'
        elif action == 2: self.change_to = 'DOWN'
        elif action == 3: self.change_to = 'LEFT'


        turning = False
        # prevent instant 180
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
            turning = True
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
            turning = True
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
            turning = True
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'
            turning = True
        
        # move
        if self.direction == 'UP':
            self.snake_pos[1] -= self.cell
        if self.direction == 'DOWN':
            self.snake_pos[1] += self.cell
        if self.direction == 'LEFT':
            self.snake_pos[0] -= self.cell
        if self.direction == 'RIGHT':
            self.snake_pos[0] += self.cell


        # NEW head position AFTER moving
        new_hx = self.snake_pos[0] // self.cell
        new_hy = self.snake_pos[1] // self.cell
        
        # Calculate which cell the snake moved INTO (relative to OLD head position)
        # This is the cell value from the previous observation
        move_delta = 0.0
        if self.local_grid_array is not None:
            dx = new_hx - old_hx  # Will be -1, 0, or 1
            dy = new_hy - old_hy  # Will be -1, 0, or 1
            
            # Convert to array indices (center is at half_size, half_size)
            arr_x = self.half_size + dx
            arr_y = self.half_size + dy
            
            # Check bounds and get the cell value
            grid_dim = 2 * self.half_size + 1
            if 0 <= arr_x < grid_dim and 0 <= arr_y < grid_dim:
                move_delta = float(self.local_grid_array[arr_y, arr_x])


        # grow / tail
        self.snake_body.insert(0, list(self.snake_pos))
        self.ate = (self.snake_pos[0] == self.food_pos[0] and
                    self.snake_pos[1] == self.food_pos[1])
        if self.ate:
            self.score += 1
            self._spawn_food()
        else:
            self.snake_body.pop()


        # death checks
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.width - self.cell:
            self.dead = True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.height - self.cell:
            self.dead = True
        if not self.dead:
            for block in self.snake_body[1:]:
                if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                    self.dead = True
                    break


        self._steps += 1
        terminated = bool(self.dead)
        if self.max_steps is not None and int(self.max_steps) > 0:
            if self._steps >= int(self.max_steps):
                terminated = True


        base = 1.0 if self.ate else 0.0


        # --- distance & progress ---
        cur_d = self._compute_distance()
        prev_d = self._prev_d if self._prev_d is not None else cur_d
        dist_delta = prev_d - cur_d  # >0 closer, <0 away
        moved_closer = dist_delta > 1e-6


        self._no_progress_steps = 0 if moved_closer else (self._no_progress_steps + 1)
        self._prev_d = cur_d


        # oscillation check
        self._dir_hist.append(self.direction)
        if len(self._dir_hist) > 4:
            self._dir_hist.pop(0)
        oscillating = (
            len(self._dir_hist) == 4
            and self._dir_hist[0] == self._dir_hist[2]
            and self._dir_hist[1] == self._dir_hist[3]
            and self._dir_hist[0] != self._dir_hist[1]
        )


        # Normalized positions for reward
        hx_norm = (self.snake_pos[0] / self.cell) / self.grid_w
        hy_norm = (self.snake_pos[1] / self.cell) / self.grid_h
        fx_norm = (self.food_pos[0] / self.cell) / self.grid_w
        fy_norm = (self.food_pos[1] / self.cell) / self.grid_h


        info = {
            'score': int(self.score),
            'length': int(len(self.snake_body)),
            'ate': bool(self.ate),
            'moved_closer': bool(moved_closer),
            'no_progress_steps': int(self._no_progress_steps),
            'oscillating': bool(oscillating),
            'dist_delta': float(dist_delta),
            'head_x': float(hx_norm),
            'head_y': float(hy_norm),
            'food_x': float(fx_norm),
            'food_y': float(fy_norm),
            'move_delta': float(move_delta),  # Cell value the snake moved into
            'turning' : bool(turning)
        }


        return self._obs(), base, terminated, info


    def render(self, surface, blit_only=False):
        cell = self.cell
        surface.fill((0, 0, 0))

        # Draw grid lines
        for gx in range(0, self.width, cell):
            pygame.draw.line(surface, (30, 30, 30), (gx, 0), (gx, self.height))
        for gy in range(0, self.height, cell):
            pygame.draw.line(surface, (30, 30, 30), (0, gy), (self.width, gy))
        # Draw snake body
        for i, pos in enumerate(self.snake_body):
            color = (50, 200, 50) if i else (50, 220, 240)
            pygame.draw.rect(surface, color, pygame.Rect(pos[0], pos[1], cell, cell))
        
        # Draw food
        pygame.draw.rect(surface, (255, 255, 255),
                        pygame.Rect(self.food_pos[0], self.food_pos[1], cell, cell))

        if False:
        # Visualize local_grid_array as colored dots
            if self.local_grid_array is not None and self.snake_pos is not None:
                hx = self.snake_pos[0] // self.cell
                hy = self.snake_pos[1] // self.cell
                grid_dim = 2 * self.half_size + 1
                
                for arr_y in range(grid_dim):
                    for arr_x in range(grid_dim):
                        # Get the cell value
                        cell_value = self.local_grid_array[arr_y, arr_x]
                        
                        # Calculate world grid position
                        dx = arr_x - self.half_size
                        dy = arr_y - self.half_size
                        world_gx = hx + dx
                        world_gy = hy + dy
                        
                        # Convert to pixel coordinates (center of cell)
                        pixel_x = world_gx * cell + cell // 2
                        pixel_y = world_gy * cell + cell // 2
                        
                        # Skip if out of bounds
                        if world_gx < 0 or world_gx >= self.grid_w or world_gy < 0 or world_gy >= self.grid_h:
                            continue
                        
                        # Color mapping based on cell value
                        if cell_value < -5.0:
                            # Danger (wall/body): bright red
                            color = (255, 0, 0)
                            radius = 6
                        elif cell_value < 0:
                            # Negative (penalized area): orange to red gradient
                            intensity = min(255, int(abs(cell_value) * 50))
                            color = (255, 255 - intensity, 0)
                            radius = 4
                        elif cell_value > 0:
                            # Positive (closer to food): green gradient
                            intensity = min(255, int(cell_value * 100))
                            color = (0, intensity, 0)
                            radius = 5
                        else:
                            # Zero: dim gray
                            color = (80, 80, 80)
                            radius = 2
                        
                        # Draw the dot
                        pygame.draw.circle(surface, color, (pixel_x, pixel_y), radius)
        # Draw snake body
        for i, pos in enumerate(self.snake_body):
            color = (50, 200, 50) if i else (50, 220, 240)
            pygame.draw.rect(surface, color, pygame.Rect(pos[0], pos[1], cell, cell))
        
        # Draw food
        pygame.draw.rect(surface, (255, 255, 255),
                        pygame.Rect(self.food_pos[0], self.food_pos[1], cell, cell))
        # Draw grid lines
        for gx in range(0, self.width, cell):
            pygame.draw.line(surface, (30, 30, 30), (gx, 0), (gx, self.height))
        for gy in range(0, self.height, cell):
            pygame.draw.line(surface, (30, 30, 30), (0, gy), (self.width, gy))
        


    def close(self):
        pass
