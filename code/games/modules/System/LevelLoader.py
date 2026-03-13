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
from ..Objects.Koopa import Koopa
from ..Objects.Coin import Coin
from ..Objects.QuestionBlock import QuestionBlock
from ..Objects.Mushroom import Mushroom
from ..Objects.LifeUp import LifeUp
from ..Objects.StarPowerUp import StarPowerUp
from ..Objects.FireFlower import FireFlower
from ..Objects.Goal import Goal
from .SpatialHash import SpatialHash

# ── MISSING IMPORTS RESTORED HERE ──
from ..Objects.SlopeTile import SlopeTile, SLOPE_CHAR_MAP
from ..Objects.Spring import Spring

@dataclass
class LevelData:
    """
    Data Transfer Object to hold all level assets.
    """
    tiles:            List[List[Tile]]          = field(default_factory=list)
    grid:             List[List[int]]           = field(default_factory=list)
    enemies:          List[Enemy]               = field(default_factory=list)
    coins:            List[Coin]               = field(default_factory=list)
    qblocks:          List[QuestionBlock]      = field(default_factory=list)
    powerups:         List[Any]              = field(default_factory=list)
    goals:            List[Goal]               = field(default_factory=list)
    pits:             List[Any]               = field(default_factory=list)
    moving_platforms: List[MovingPlatform]     = field(default_factory=list)
    projectiles:      List[Any]              = field(default_factory=list)
    
    # ── MISSING ARRAYS RESTORED HERE ──
    slope_tiles:      List[SlopeTile]          = field(default_factory=list)
    springs:          List[Spring]             = field(default_factory=list)
    
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
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        else:
            self.base_dir = base_dir
        
        self.level_path = os.path.join(self.base_dir, "levels")
        
        self.TILE_MAP = {
            '#': (TILE_GROUND,   COLOR_GROUND,   True,  EntityType.TILE),
            '=': (TILE_PLATFORM, COLOR_PLATFORM, True,  EntityType.TILE),
        }

        self.QBLOCK_CONTAINS = {
            '?': 'coin',
            '>': 'star',
            '<': 'mushroom',
            'F': 'flower',
            'L': 'life',
        }

    def load_level(self, source: Union[Dict[str, Any], str]) -> LevelData:
        data     = LevelData()
        filename = ""
        config   = {}

        if isinstance(source, dict):
            config   = source
            raw_file = config.get('file', '')
            filename = os.path.basename(raw_file)
        elif isinstance(source, str):
            config = {}
            if os.path.isabs(source) or os.path.exists(source):
                txt_path = source
            else:
                filename = os.path.basename(source)
                txt_path = os.path.join(self.level_path, filename)

        if isinstance(source, dict):
            if not filename:
                print(f"[LevelLoader] Error: Level config has no 'file' entry — cannot load. Check game_config.yaml.")
                return data
            if os.path.isabs(raw_file) and os.path.exists(raw_file):
                txt_path = raw_file
            else:
                txt_path = os.path.join(self.level_path, filename)

        if os.path.isfile(txt_path):
            self._parse_ascii_map(txt_path, data)
        elif os.path.isdir(txt_path):
            print(f"[LevelLoader] Error: '{txt_path}' is a directory, not a level file.")
        else:
            print(f"[LevelLoader] Warning: Level file '{txt_path}' not found.")

        sidecar_path = txt_path.rsplit('.', 1)[0] + '.yaml'
        sidecar_dynamics: Dict[str, Any] = {}
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f:
                    sidecar_data = yaml.safe_load(f) or {}
                sidecar_dynamics = sidecar_data.get('dynamics', {}) or {}
            except Exception as e:
                print(f"[LevelLoader] Sidecar YAML error ({sidecar_path}): {e}")

        merged_dynamics: Dict[str, Any] = {}
        if sidecar_dynamics:
            self._dict_merge(merged_dynamics, sidecar_dynamics)
        if config and 'dynamics' in config:
            self._dict_merge(merged_dynamics, config['dynamics'])

        if merged_dynamics:
            self._spawn_entities_from_yaml(merged_dynamics, data)

        return data

    def _dict_merge(self, base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], list) and isinstance(v, list):
                base[k].extend(v)
            elif k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._dict_merge(base[k], v)
            else:
                base[k] = v

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
                
                # 1. Handle Spikes
                if ascii_char == '^':
                    spike = Spike.from_tile(col * TILE_SIZE, row * TILE_SIZE)
                    data.grid[row][col] = TILE_SPIKE
                    data.tiles[row][col] = spike
                    data.static_hash.insert(spike)

                # 2. Handle Sonic Slope Tiles 
                elif ascii_char in SLOPE_CHAR_MAP:
                    slope_type = SLOPE_CHAR_MAP[ascii_char]
                    slope_tile = SlopeTile.create(col, row, slope_type, TILE_SIZE)
                    slope_tile.gObj.type_id = EntityType.TILE
                    data.grid[row][col] = slope_tile.type_id
                    data.tiles[row][col] = slope_tile
                    data.slope_tiles.append(slope_tile)
                    data.static_hash.insert(slope_tile)

                # 3. Handle Springs 
                elif ascii_char == 'S':
                    s = Spring(gObj=GameObject(col * TILE_SIZE, row * TILE_SIZE + (TILE_SIZE // 2), TILE_SIZE, TILE_SIZE // 2, True))
                    data.springs.append(s)

                # 4. Handle other Static Tiles via Dictionary
                elif ascii_char in self.TILE_MAP:
                    t_type, color, solid, e_type = self.TILE_MAP[ascii_char]
                    
                    data.grid[row][col] = t_type
                    new_tile = create_tile(t_type, col * TILE_SIZE, row * TILE_SIZE, solid, color)
                    new_tile.gObj.type_id = e_type
                    data.tiles[row][col] = new_tile
                    
                    if solid:
                        data.static_hash.insert(new_tile)

                # 5. Handle Complex Entities (QBlocks, Enemies, Start Pos)
                elif ascii_char in self.QBLOCK_CONTAINS:
                    contains = self.QBLOCK_CONTAINS[ascii_char]
                    data.grid[row][col] = TILE_QBLOCK
                    qb = QuestionBlock(gObj=GameObject(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE, True), contains=contains)
                    qb.gObj.type_id = EntityType.QBLOCK
                    data.qblocks.append(qb)
                    data.static_hash.insert(qb)

                elif ascii_char == 'C':
                    c = Coin(gObj=GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 16, 16, True))
                    c.gObj.type_id = EntityType.COIN
                    data.coins.append(c)

                elif ascii_char == 'E':
                    e = Enemy(GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 25, 20, True), vx=-60.0)
                    e.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(e)

                elif ascii_char == 'k':
                    k = Koopa(gObj=GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 22, 30, True))
                    k.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(k)

                elif ascii_char == 'K':
                    k = Koopa(gObj=GameObject(col * TILE_SIZE + 8, row * TILE_SIZE + 8, 22, 30, True), flying=True)
                    k.gObj.type_id = EntityType.ENEMY
                    data.enemies.append(k)
                
                elif ascii_char == 'P':
                    data.player_start = (float(col * TILE_SIZE), float(row * TILE_SIZE))

                elif ascii_char == 'G':
                    g = Goal(gObj=GameObject(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE, True))
                    data.goals.append(g)

                elif ascii_char == 'O':
                    pit_obj = GameObject(
                        float(col * TILE_SIZE), float(row * TILE_SIZE),
                        TILE_SIZE, TILE_SIZE, False
                    )
                    pit_obj.type_id = EntityType.PIT
                    data.grid[row][col] = TILE_PIT
                    data.pits.append(pit_obj)

    def _spawn_entities_from_yaml(self, dynamics: Dict[str, Any], data: LevelData):
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
                width  = int(mp_data.get('width',  TILE_SIZE * 3))
                height = int(mp_data.get('height', TILE_SIZE // 2))
                plat = MovingPlatform.from_points(
                    start=tuple(start),
                    end=tuple(end),
                    speed=speed,
                    width=width,
                    height=height,
                )
                data.moving_platforms.append(plat)