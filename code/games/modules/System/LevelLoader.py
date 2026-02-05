import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union

# Import EntityType to assign IDs correctly
from .EntityType import EntityType

from ..Parameters.Map_parameters import (
    TILE_AIR, TILE_GROUND, TILE_PLATFORM, TILE_GOAL, TILE_SPIKE, TILE_QBLOCK,
    COLOR_SKY, COLOR_GROUND, COLOR_PLATFORM, COLOR_GOAL, COLOR_SPIKE, 
    COLOR_QBLOCK, TILE_SIZE
)
from ..Objects.Tile import Tile, create_tile
from ..Objects.GameObject import GameObject
from ..Objects.Enemy import Enemy
from ..Objects.Coin import Coin
from ..Objects.QuestionBlock import QuestionBlock
from ..Objects.Powerup import Powerup
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
            # Current file: .../games/modules/System/LevelLoader.py
            # Up 1: .../games/modules/System
            # Up 2: .../games/modules
            # Up 3: .../games/ (This is where 'levels' folder is)
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir
        
        self.level_path = os.path.join(self.base_dir, "levels")
            
    def load_level(self, source: Union[Dict[str, Any], str]) -> LevelData:
        """
        Orchestrates loading using either a YAML config dictionary OR a direct filename string.
        """
        data = LevelData()
        filename = ""
        config = {}

        # 1. Determine input type
        if isinstance(source, dict):
            config = source
            # If coming from YAML, it might be "levels/stage_1.txt" or "stage_1.txt"
            raw_file = config.get('file', '') # Changed from 'level_file' to 'file' to match YAML
            filename = os.path.basename(raw_file)
        elif isinstance(source, str):
            filename = os.path.basename(source)
            config = {} 
        
        # 2. Build full path
        txt_path = os.path.join(self.level_path, filename)
        
        if os.path.exists(txt_path):
            self._parse_ascii_map(txt_path, data)
        else:
            print(f"[LevelLoader] Warning: Level file {txt_path} not found. Returning empty/default data.")

        # 3. Spawn Dynamic Entities if config exists (Additive to ASCII spawns)
        # Check for 'dynamics' key as per YAML structure
        if config and 'dynamics' in config:
            self._spawn_entities_from_yaml(config['dynamics'], data)

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
                    qb.gObj.type_id = EntityType.QBLOCK
                    data.qblocks.append(qb)
                    data.static_hash.insert(qb)

                # --- Dynamic Spawns from ASCII ---
                elif ascii_char == 'C':
                    c = Coin(gObj=GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 16, 16, True))
                    c.gObj.type_id = EntityType.COIN
                    data.coins.append(c)
                elif ascii_char == 'E':
                    e = Enemy(GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 25, 20, True), vx=-60.0)
                    e.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(e)
                
                elif ascii_char == 'P':
                    data.player_start = (float(col * TILE_SIZE), float(row * TILE_SIZE))

                data.grid[row][col] = tile_type
                if tile_type != TILE_AIR:
                    new_tile = create_tile(tile_type, col * TILE_SIZE, row * TILE_SIZE, solid, color)
                    # Assign EntityType to Tile for PhysicsManager resolution
                    if tile_type == TILE_SPIKE: new_tile.type_id = EntityType.SPIKE
                    elif tile_type == TILE_GOAL: new_tile.type_id = EntityType.GOAL
                    else: new_tile.type_id = EntityType.TILE
                    
                    data.tiles[row][col] = new_tile
                    
                    if solid or tile_type in (TILE_SPIKE, TILE_GOAL):
                        data.static_hash.insert(new_tile)

    def _spawn_entities_from_yaml(self, dynamics: Dict[str, Any], data: LevelData):
        """
        Parses the 'dynamics' section of the YAML config.
        """
        # 1. Enemies
        if 'enemies' in dynamics:
            for e in dynamics['enemies']:
                x = e.get('x', 0)
                y = e.get('y', 0)
                vx = e.get('vx', -60.0)
                enemy = Enemy(GameObject(x, y, 25, 20, True), vx=vx)
                enemy.gObj.type_id = EntityType.ENEMY
                data.enemies.append(enemy)
        
        # 2. Coins
        if 'coins' in dynamics:
            for c in dynamics['coins']:
                x = c.get('x', 0)
                y = c.get('y', 0)
                coin = Coin(gObj=GameObject(x, y, 16, 16, True))
                coin.gObj.type_id = EntityType.COIN
                data.coins.append(coin)

        # 3. Powerups
        if 'powerups' in dynamics:
            for p in dynamics['powerups']:
                x = p.get('x', 0)
                y = p.get('y', 0)
                kind = p.get('type', 'mushroom')
                pup = Powerup(gObj=GameObject(x, y, 20, 20, True), kind=kind)
                pup.gObj.type_id = EntityType.POWERUP
                data.powerups.append(pup)