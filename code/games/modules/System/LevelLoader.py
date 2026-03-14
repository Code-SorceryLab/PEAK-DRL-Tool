import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union

# Import EntityType to assign IDs correctly
from .EntityType import EntityType

from ..Parameters.Map_parameters import (
    TILE_AIR, TILE_GROUND, TILE_PLATFORM, TILE_GOAL, TILE_SPIKE, TILE_QBLOCK, TILE_PIT,
    COLOR_SKY, COLOR_GROUND, COLOR_PLATFORM, COLOR_GOAL, COLOR_SPIKE, 
    COLOR_QBLOCK, TILE_SIZE
)
from ..Objects.Tile import Tile, create_tile
from ..Objects.Spike import Spike
from ..Objects.MovingPlatform import MovingPlatform
from ..Objects.GameObject import GameObject
from ..Objects.Enemy import Enemy
from ..Objects.Koopa import Koopa        # ← add this
from ..Objects.Coin import Coin
from ..Objects.QuestionBlock import QuestionBlock
from ..Objects.Mushroom import Mushroom
from ..Objects.LifeUp import LifeUp
from ..Objects.StarPowerUp import StarPowerUp
from ..Objects.FireFlower import FireFlower
from ..Objects.Goal import Goal
from ..Objects.Ladder import Ladder
from .SpatialHash import SpatialHash

@dataclass
class LevelData:
    """
    Data Transfer Object to hold all level assets.

    Spatial hashes owned here:
      static_hash  — immovable geometry (ground, platforms, spikes, qblocks).
                     Built once at load time, never rebuilt mid-episode.

    Moving-platform and enemy/collectible hashes are owned by PhysicsManager
    and rebuilt every frame in platformer_core.step() because those objects move.
    """
    tiles:            List[List[Tile]]          = field(default_factory=list)
    grid:             List[List[int]]           = field(default_factory=list)
    enemies:          List[Enemy]               = field(default_factory=list)
    coins:            List[Coin]               = field(default_factory=list)
    qblocks:          List[QuestionBlock]      = field(default_factory=list)
    powerups:         List[Any]              = field(default_factory=list)
    goals:            List[Goal]               = field(default_factory=list)
    ladders:          List[Ladder]             = field(default_factory=list)
    pits:             List[Any]               = field(default_factory=list)
    moving_platforms: List[MovingPlatform]     = field(default_factory=list)
    projectiles:      List[Any]              = field(default_factory=list)
    player_start:     Tuple[float, float]      = (100.0, 350.0)
    rows:             int                      = 0
    cols:             int                      = 0
    width:            float                    = 0.0
    height:           float                    = 0.0
    static_hash:      SpatialHash              = field(default_factory=lambda: SpatialHash(64))

class LevelLoader:
    """
    Responsible for parsing level files (TXT and YAML) and creating the corresponding
    game objects, tiles, and initial state data.
    """
    def __init__(self, base_dir=None, tile_size=None):
        if base_dir is None:
            # Current file: .../games/modules/System/LevelLoader.py
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir
        
        self.level_path = os.path.join(self.base_dir, "levels")
        self.tile_size = tile_size or TILE_SIZE
        
        # --- DICTIONARY MAPPING FOR STATIC TILES ---
        # Char -> (TileType, Color, Solid, EntityType)
        # Note: '^' (Spike) is handled separately to instantiate the Spike class.
        self.TILE_MAP = {
            '#': (TILE_GROUND,   COLOR_GROUND,   True,  EntityType.TILE),
            '=': (TILE_PLATFORM, COLOR_PLATFORM, True,  EntityType.TILE),
            'D': (TILE_GOAL,     COLOR_GOAL,     False, EntityType.GOAL), # Boss Door
        }

        # QBlock char → what the block contains
        # '?' = coin, '>' = star, '<' = mushroom, 'F' = flower, 'L' = life
        self.QBLOCK_CONTAINS = {
            '?': 'coin',
            '>': 'star',
            '<': 'mushroom',
            'F': 'flower',
            'L': 'life',
        }

    def load_level(self, source: Union[Dict[str, Any], str]) -> LevelData:
        """
        Orchestrates loading using either a YAML config dictionary OR a direct filename string.
        
        1. Determines if the source is a Dictionary (YAML config) or String (file path).
        2. Constructs the full file path to the ASCII map file.
        3. Calls _parse_ascii_map to generate the grid and static geometry.
        4. Loads a per-level sidecar YAML ([level].yaml) if it exists alongside the .txt.
        5. If a YAML config was provided, calls _spawn_entities_from_yaml for extra dynamics.
        6. Inserts spikes into hazard_hash, moving platforms into dynamic_hash.
        7. Returns the fully populated LevelData object.
        """
        data     = LevelData()
        filename = ""
        config   = {}

        # If config has tile_size, use it
        if isinstance(source, dict) and 'tile_size' in source:
            self.tile_size = source['tile_size']

        # 1. Determine input type
        if isinstance(source, dict):
            config   = source
            raw_file = config.get('file', '')
            filename = os.path.basename(raw_file)
        elif isinstance(source, str):
            config = {}
            # Absolute or existing path → use directly; otherwise join with level_path
            if os.path.isabs(source) or os.path.exists(source):
                txt_path = source
            else:
                filename = os.path.basename(source)
                txt_path = os.path.join(self.level_path, filename)

        # 2. Build full path (dict source only — str source sets txt_path above)
        if isinstance(source, dict):
            if not filename:
                print(f"[LevelLoader] Error: Level config has no 'file' entry — cannot load. Check game_config.yaml.")
                return data
            # If raw_file is an absolute path that exists, use it directly.
            # This allows editor play-test injection without copying files.
            if os.path.isabs(raw_file) and os.path.exists(raw_file):
                txt_path = raw_file
            else:
                txt_path = os.path.join(self.level_path, filename)

        # Guard: never try to open a directory as a file
        if os.path.isfile(txt_path):
            self._parse_ascii_map(txt_path, data)
        elif os.path.isdir(txt_path):
            print(f"[LevelLoader] Error: '{txt_path}' is a directory, not a level file. The level is missing a 'file:' entry in game_config.yaml.")
        else:
            print(f"[LevelLoader] Warning: Level file '{txt_path}' not found.")

        # 3. Load sidecar YAML ([level_name].yaml next to the .txt)
        sidecar_path = txt_path.rsplit('.', 1)[0] + '.yaml'
        sidecar_dynamics: Dict[str, Any] = {}
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f:
                    sidecar_data = yaml.safe_load(f) or {}
                sidecar_dynamics = sidecar_data.get('dynamics', {}) or {}
            except Exception as e:
                print(f"[LevelLoader] Sidecar YAML error ({sidecar_path}): {e}")

        # 4. Merge config dynamics + sidecar dynamics then spawn
        merged_dynamics: Dict[str, Any] = {}
        if sidecar_dynamics:
            self._dict_merge(merged_dynamics, sidecar_dynamics)
        if config and 'dynamics' in config:
            self._dict_merge(merged_dynamics, config['dynamics'])

        if merged_dynamics:
            self._spawn_entities_from_yaml(merged_dynamics, data)

        return data

    def _dict_merge(self, base: dict, override: dict):
        """Recursively merge override into base (list values are extended)."""
        for k, v in override.items():
            if k in base and isinstance(base[k], list) and isinstance(v, list):
                base[k].extend(v)
            elif k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._dict_merge(base[k], v)
            else:
                base[k] = v

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
        data.width = data.cols * self.tile_size
        data.height = data.rows * self.tile_size
        
        data.grid = [[TILE_AIR for _ in range(data.cols)] for _ in range(data.rows)]
        data.tiles = [[None for _ in range(data.cols)] for _ in range(data.rows)]
        data.static_hash.clear()

        for row in range(data.rows):
            curr_row = lines[row]
            for col in range(len(curr_row)):
                ascii_char = curr_row[col]

                if ascii_char in ('.', ' '):
                    data.grid[row][col] = TILE_AIR
                    continue
                
                # 1. Handle Spikes with dedicated Spike class
                if ascii_char == '^':
                    spike = Spike.from_tile(col * self.tile_size, row * self.tile_size)
                    data.grid[row][col] = TILE_SPIKE
                    data.tiles[row][col] = spike
                    # Spikes go into static_hash only — PhysicsManager._resolve_player_world
                    # already detects EntityType.SPIKE via _get_tile_rects_near and routes
                    # to core._handle_death(). No separate hazard_hash needed.
                    data.static_hash.insert(spike)

                # 2. Handle other Static Tiles via Dictionary
                elif ascii_char == 'H':
                    ladder = Ladder.from_tile(col * self.tile_size, row * self.tile_size, self.tile_size)
                    data.grid[row][col] = TILE_AIR
                    data.ladders.append(ladder)

                # 3. Handle other Static Tiles via Dictionary
                elif ascii_char in self.TILE_MAP:
                    t_type, color, solid, e_type = self.TILE_MAP[ascii_char]
                    
                    data.grid[row][col] = t_type
                    new_tile = create_tile(t_type, col * self.tile_size, row * self.tile_size, solid, color)
                    
                    # IMPORTANT: new_tile.type_id is the int constant (TILE_GROUND,
                    # TILE_PLATFORM etc.) set by create_tile — do NOT overwrite it.
                    # _get_tile_rects_near reads item.type_id to distinguish platforms
                    # from ground tiles for one-way collision. Overwriting with
                    # EntityType.TILE loses that information.
                    # Only set the gObj EntityType for dispatch/filter purposes.
                    new_tile.gObj.type_id = e_type
                    
                    data.tiles[row][col] = new_tile
                    
                    if solid:
                        data.static_hash.insert(new_tile)

                # 4. Handle Complex Entities (QBlocks, Enemies, Start Pos)
                elif ascii_char in self.QBLOCK_CONTAINS:
                    contains = self.QBLOCK_CONTAINS[ascii_char]
                    data.grid[row][col] = TILE_QBLOCK
                    qb = QuestionBlock(gObj=GameObject(col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size, True), contains=contains)
                    qb.gObj.type_id = EntityType.QBLOCK
                    data.qblocks.append(qb)
                    data.static_hash.insert(qb)

                elif ascii_char == 'C':
                    c = Coin(gObj=GameObject(col * self.tile_size + self.tile_size//2, row * self.tile_size + self.tile_size//2, self.tile_size//2, self.tile_size//2, True))
                    c.gObj.type_id = EntityType.COIN
                    data.coins.append(c)

                elif ascii_char == 'E':
                    e = Enemy(GameObject(col * self.tile_size + 8, row * self.tile_size + 8, 25, 20, True), vx=-60.0)
                    e.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(e)

                elif ascii_char == 'k':
                    k = Koopa(gObj=GameObject(col * self.tile_size + 8, row * self.tile_size + 8, 22, 30, True))
                    k.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(k)

                elif ascii_char == 'K':
                    k = Koopa(gObj=GameObject(col * self.tile_size + 8, row * self.tile_size + 8, 22, 30, True), flying=True)
                    k.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(k)
                
                elif ascii_char == 'P':
                    data.player_start = (float(col * self.tile_size), float(row * self.tile_size))

                elif ascii_char == 'G' or ascii_char == 'D':
                    g = Goal(gObj=GameObject(col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size, True))
                    data.goals.append(g)

                elif ascii_char == 'M':
                    # Met spawn (generic Enemy for now, MegaManCore will convert)
                    e = Enemy(GameObject(col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size, True), vx=0.0)
                    e.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(e)

                elif ascii_char == 'B':
                    # Bat spawn (generic Enemy for now, MegaManCore will convert)
                    e = Enemy(GameObject(col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size, True), vx=0.0)
                    e.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(e)

                elif ascii_char == 'O':
                    # PIT: non-solid kill zone. Transparent in game, visible in editor.
                    # Not inserted into static_hash (no collision blocking) — inserted
                    # into hazard_hash by platformer_core.load_level() so the observation
                    # hazard channel picks it up, and PhysicsManager triggers death on overlap.
                    pit_obj = GameObject(
                        float(col * self.tile_size), float(row * self.tile_size),
                        self.tile_size, self.tile_size, False  # solid=False
                    )
                    pit_obj.type_id = EntityType.PIT
                    data.grid[row][col] = TILE_PIT
                    data.pits.append(pit_obj)

    def _spawn_entities_from_yaml(self, dynamics: Dict[str, Any], data: LevelData):
        """
        Parses the 'dynamics' section of a YAML config or sidecar file.
        Supports: enemies, coins, powerups, moving_platforms.
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
            _KIND_MAP = {
                'mushroom': Mushroom,
                'star':     StarPowerUp,
                'flower':   FireFlower,
                'life':     LifeUp,
            }
            for p in dynamics['powerups']:
                x    = p.get('x', 0)
                y    = p.get('y', 0)
                kind = p.get('type', 'mushroom')
                cls  = _KIND_MAP.get(kind, Mushroom)
                pup  = cls(gObj=GameObject(x, y, 20, 20, True))
                pup.gObj.type_id = EntityType.POWERUP
                data.powerups.append(pup)

        if 'moving_platforms' in dynamics:
            for mp_data in dynamics['moving_platforms']:
                start  = mp_data.get('start',  [0, 0])
                end    = mp_data.get('end',    [64, 0])
                speed  = float(mp_data.get('speed',  80.0))
                width  = int(mp_data.get('width',  self.tile_size * 3))
                height = int(mp_data.get('height', self.tile_size // 2))
                plat = MovingPlatform.from_points(
                    start=tuple(start),
                    end=tuple(end),
                    speed=speed,
                    width=width,
                    height=height,
                )
                data.moving_platforms.append(plat)
                # NOTE: Moving platforms are NOT inserted into static_hash.
                # static_hash is for immovable geometry only — anything in it is
                # assumed to never change position. Platforms move every frame and
                # are instead managed through PhysicsManager.platform_hash, which
                # is rebuilt each step() in platformer_core.
