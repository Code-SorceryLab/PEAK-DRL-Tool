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
    """
    Responsible for parsing level files (TXT and YAML) and creating the corresponding
    game objects, tiles, and initial state data.
    """
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Current file: .../games/modules/System/LevelLoader.py
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir
        
        self.level_path = os.path.join(self.base_dir, "levels")
        
        # --- DICTIONARY MAPPING FOR STATIC TILES ---
        # Char -> (TileType, Color, Solid, EntityType)
        self.TILE_MAP = {
            '#': (TILE_GROUND, COLOR_GROUND, True, EntityType.TILE),
            '=': (TILE_PLATFORM, COLOR_PLATFORM, True, EntityType.TILE),
            'G': (TILE_GOAL, COLOR_GOAL, False, EntityType.GOAL),
            '^': (TILE_SPIKE, COLOR_SPIKE, False, EntityType.SPIKE),
        }

    def load_level(self, source: Union[Dict[str, Any], str]) -> LevelData:
        """
        Orchestrates loading using either a YAML config dictionary OR a direct filename string.
        
        1. Determines if the source is a Dictionary (YAML config) or String (file path).
        2. Constructs the full file path to the ASCII map file.
        3. Calls _parse_ascii_map to generate the grid and static geometry.
        4. If a YAML config was provided, calls _spawn_entities_from_yaml to add extra dynamic objects.
        5. Returns the fully populated LevelData object.
        """
        data = LevelData()
        filename = ""
        config = {}

        # 1. Determine input type
        if isinstance(source, dict):
            config = source
            # If coming from YAML, it might be "levels/stage_1.txt" or "stage_1.txt"
            raw_file = config.get('file', '') 
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
        if config and 'dynamics' in config:
            self._spawn_entities_from_yaml(config['dynamics'], data)

        return data

    def _parse_ascii_map(self, path: str, data: LevelData):
        """
        Parses a text file character by character to build the level.
        
        1. Reads lines from the file to determine dimensions (rows/cols).
        2. Initializes the grid, tile arrays, and static spatial hash.
        3. Iterates through every character in the file:
           - Uses TILE_MAP to create static tiles (#, =, G, ^).
           - Checks for special characters (?, C, E, P) to spawn Entities (Question blocks, Coins, Enemies, Player).
        4. Inserts static objects into the SpatialHash for collision detection.
        """
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
                
                # 1. Handle Static Tiles via Dictionary
                if ascii_char in self.TILE_MAP:
                    t_type, color, solid, e_type = self.TILE_MAP[ascii_char]
                    
                    data.grid[row][col] = t_type
                    new_tile = create_tile(t_type, col * TILE_SIZE, row * TILE_SIZE, solid, color)
                    new_tile.type_id = e_type
                    data.tiles[row][col] = new_tile
                    
                    if solid or t_type in (TILE_SPIKE, TILE_GOAL):
                        data.static_hash.insert(new_tile)

                # 2. Handle Complex Entities (QBlocks, Enemies, Start Pos)
                elif ascii_char == '?': 
                    # QBlock is both an entity and a solid tile
                    data.grid[row][col] = TILE_QBLOCK
                    qb = QuestionBlock(gObj=GameObject(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE, True), contains="coin")
                    qb.gObj.type_id = EntityType.QBLOCK
                    data.qblocks.append(qb)
                    data.static_hash.insert(qb)
                    # We don't add it to data.tiles[] usually if it's treated as a pure object, 
                    # but if physics checks grid, we set grid val above.

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

    def _spawn_entities_from_yaml(self, dynamics: Dict[str, Any], data: LevelData):
        """
        Parses the 'dynamics' section of the YAML config.
        Allows adding entities (Enemies, Coins, Powerups) at specific coordinates
        defined in the YAML, supplemental to the ASCII map.
        """
        if 'enemies' in dynamics:
            for e in dynamics['enemies']:
                x = e.get('x', 0); y = e.get('y', 0); vx = e.get('vx', -60.0)
                enemy = Enemy(GameObject(x, y, 25, 20, True), vx=vx)
                enemy.gObj.type_id = EntityType.ENEMY
                data.enemies.append(enemy)
        
        if 'coins' in dynamics:
            for c in dynamics['coins']:
                x = c.get('x', 0); y = c.get('y', 0)
                coin = Coin(gObj=GameObject(x, y, 16, 16, True))
                coin.gObj.type_id = EntityType.COIN
                data.coins.append(coin)

        if 'powerups' in dynamics:
            for p in dynamics['powerups']:
                x = p.get('x', 0); y = p.get('y', 0); kind = p.get('type', 'mushroom')
                pup = Powerup(gObj=GameObject(x, y, 20, 20, True), kind=kind)
                pup.gObj.type_id = EntityType.POWERUP
                data.powerups.append(pup)