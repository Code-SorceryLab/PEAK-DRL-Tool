from typing import List, Tuple, Dict, Set
import math

class SpatialHash:
    def __init__(self, cell_size: float):
        """"
        Cell size defines the width and height of each grid cell of one bucket. 
        Ideally, this is slightly larger than the average size of objects being tracked.
        """
        self.cell_size = cell_size
        self.contents: Dict[Tuple[int, int], List[any]] = {}
    
    
    # -- Grid Management --    
    def clear(self):
        self.contents.clear()
    
    def _get_cell_coords(self, position: Tuple[float, float]) -> Tuple[int, int]:
        x, y = position
        return int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size))
    
    def insert(self, obj):
        """
        Inserts an object into all cells it overlaps. 
        Assumes obj has a .x, .y, .width, and .height attributes or an obj with these . 
        """
        
        # This will Handle wrapped GameObjects vs Direct GameObjects
        target = obj.gObj if hasattr(obj, 'gObj') else obj
        
        if not target.active:
            return
        
        start_cx, start_cy = self._get_cell_coords((target.x, target.y))
        end_cx, end_cy = self._get_cell_coords((target.x + target.width, target.y + target.height))
        
        for cx in range (start_cx, end_cx + 1):
            for cy in range (start_cy, end_cy + 1):
                cell = (cx, cy)
                if cell not in self.contents:
                    self.contents[cell] = []
                self.contents[cell].append(obj)
    
    def query(self, obj) -> Set[any]:
        """
        Returns a set of unique objects that resides in the same cells as the given object.
        """
        target = obj.gObj if hasattr(obj, 'gObj') else obj
        return self.query_rect(target.x, target.y, target.width, target.height, ignore_obj=obj)

    def query_rect(self, x: float, y: float, w: float, h: float, ignore_obj=None) -> Set[any]:
        """
        Returns all objects in buckets overlapping the given rectangle.
        Useful for Camera Culling.
        """
        found_objects = []
        seen_ids = set()
        
                
        start_cx, start_cy = self._get_cell_coords((x, y))
        end_cx, end_cy = self._get_cell_coords((x + w, y + h))
        
        for cx in range(start_cx, end_cx + 1):
            for cy in range(start_cy, end_cy + 1):
                cell = (cx, cy)
                if cell in self.contents:
                    for neighbor in self.contents[cell]:
                        # Use id() to track uniqueness, preventing "unhashable type" errors
                        nid = id(neighbor)
                        if nid not in seen_ids and neighbor is not ignore_obj:
                            seen_ids.add(nid)
                            found_objects.append(neighbor)
                            
                            
        return found_objects
