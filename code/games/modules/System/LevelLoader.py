import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
from .Map_parameters import (
    TILE_AIR, TILE_GROUND, TILE_PLATFORM, TILE_GOAL, TILE_SPIKE, TILE_QBLOCK,
    COLOR_SKY, COLOR_GROUND, COLOR_PLATFORM, COLOR_GOAL, COLOR_SPIKE, 
    COLOR_QBLOCK, TILE_SIZE
)
from .Tile import Tile, create_tile
from .GameObject import GameObject
from .Enemy import Enemy
from .Coin import Coin
from .QuestionBlock import QuestionBlock
from .Powerup import Powerup
from .SpatialHash import SpatialHash

@dataclass
class LevelData:
    """
    Data Transfer Object to hold all level assets.
    """
    tiles: List[List[Tile]] = field(default_factory=list)
    grid: List[List[int]] = field(default_factory=list)
    enemies: List[Enemy] = field(default_factory=list)
    coins: List[Coin] = field(default_factory=list)
    qblocks: List[QuestionBlock] = field(default_factory=list)
    powerups: List[Powerup] = field(default_factory=list)
    player_start: Tuple[float, float] = (100.0, 350.0)
    rows: int = 0
    cols: int = 0
    width: float = 0.0
    height: float = 0.0
    static_hash: SpatialHash = field(default_factory=lambda: SpatialHash(64))

class LevelLoader:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
        
        # Variable to hold path to the directory as requested
        self.level_path = os.path.join(self.base_dir, "levels")
            
    def load_level(self, source: Union[Dict[str, Any], str]) -> LevelData:
        """
        Orchestrates loading using either a YAML config dictionary OR a direct filename string.
        
        Args:
            source: Dict containing 'level_file' etc., OR a string "stage_1.txt"
        """
        data = LevelData()
        filename = ""
        config = {}

        # 1. Determine input type
        if isinstance(source, dict):
            config = source
            # If coming from YAML, it might be "levels/stage_1.txt" or "stage_1.txt"
            raw_file = config.get('level_file', '')
            filename = os.path.basename(raw_file) # Ensure we just get the name
        elif isinstance(source, str):
            filename = os.path.basename(source)
            config = {} # No dynamic config if loading by string
        
        # 2. Build full path using the directory variable
        txt_path = os.path.join(self.level_path, filename)
        
        if os.path.exists(txt_path):
            self._parse_ascii_map(txt_path, data)
        else:
            print(f"[LevelLoader] Error: Level file {txt_path} not found.")
            return data

        # 3. Spawn Dynamic Entities if config exists (Additive to ASCII spawns)
        if config:
            self._spawn_entities_from_yaml(config, data)

        return data

    def _parse_ascii_map(self, path: str, data: LevelData):
        with open(path, "r") as file:
            lines = [ln.rstrip("\n") for ln in file.readlines()]

        data.rows = len(lines)
        data.cols = max(len(ln) for ln in lines) if data.rows else 0
        data.width = data.cols * TILE_SIZE
        data.height = data.rows * TILE_SIZE
        
        data.grid = [[TILE_AIR for _ in range(data.cols)] for _ in range(data.rows)]
        data.tiles = [[None for _ in range(data.cols)] for _ in range(data.rows)]
        data.static_hash.clear()

        for row in range(data.rows):
            curr_row = lines[row]
            for col in range(len(curr_row)):
                ascii_char = curr_row[col]
                tile_type = TILE_AIR
                color = COLOR_SKY
                solid = False
                
                if ascii_char == '#': 
                    tile_type = TILE_GROUND; color = COLOR_GROUND; solid = True
                elif ascii_char == '=': 
                    tile_type = TILE_PLATFORM; color = COLOR_PLATFORM; solid = True
                elif ascii_char == 'G': 
                    tile_type = TILE_GOAL; color = COLOR_GOAL; solid = False
                elif ascii_char == '^': 
                    tile_type = TILE_SPIKE; color = COLOR_SPIKE; solid = False
                elif ascii_char == '?': 
                    tile_type = TILE_QBLOCK; color = COLOR_QBLOCK; solid = True
                    qb = QuestionBlock(gObj=GameObject(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE, True), contains="coin")
                    data.qblocks.append(qb)
                    data.static_hash.insert(qb)

                # --- Dynamic Spawns from ASCII ---
                elif ascii_char == 'C':
                    data.coins.append(Coin(gObj=GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 16, 16, True)))
                elif ascii_char == 'E':
                    data.enemies.append(Enemy(GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 25, 20, True), vx=-60.0))
                
                elif ascii_char == 'P':
                    data.player_start = (float(col * TILE_SIZE), float(row * TILE_SIZE))

                data.grid[row][col] = tile_type
                if tile_type != TILE_AIR:
                    new_tile = create_tile(tile_type, col * TILE_SIZE, row * TILE_SIZE, solid, color)
                    data.tiles[row][col] = new_tile
                    
                    if solid or tile_type in (TILE_SPIKE, TILE_GOAL):
                        data.static_hash.insert(new_tile)

