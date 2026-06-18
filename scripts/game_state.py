"""
game_state.py — SHANKPIT game state serializer + action decoder.

Bridges the SHANKPIT wire protocol to GPT-2 natural-language token format.

Two public functions:
  serialize_snapshot(data, self_id, tick, scene_id) -> str
  decode_action(action_str) -> dict

Wire format (PacketSnapshot from apps2/server-go):
  [0]    type = 0x02
  [1]    entity count
  [2+]   entity_count × 18 bytes:
           [0]    client_id  uint8
           [1]    scene_id   uint8
           [2:6]  x          float32 LE
           [6:10] y          float32 LE
           [10:14] z         float32 LE
           [14:18] yaw       float32 LE

Note: Go server snapshot does not include health/weapon/flags. Those fields
will appear here once the server snapshot is extended (S40 series).
"""

import math
import struct

PACKET_SNAPSHOT = 0x02
ENTITY_SIZE = 18

WEAPON_NAMES = {0: "knife", 1: "magnum", 2: "ar", 3: "shotgun", 4: "sniper", 5: "katana"}
WEAPON_IDX   = {v: k for k, v in WEAPON_NAMES.items()}

DIRECTION_SECTORS = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
]


def _yaw_to_direction(yaw_deg):
    """Convert yaw (0=north, clockwise) to compass word."""
    idx = int((yaw_deg % 360) / 45 + 0.5) % 8
    return DIRECTION_SECTORS[idx]


def _dist2d(ax, az, bx, bz):
    return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)


def _parse_entities(data):
    """Return list of dicts from raw PacketSnapshot bytes."""
    if len(data) < 2 or data[0] != PACKET_SNAPSHOT:
        return []
    count = data[1]
    entities = []
    off = 2
    for _ in range(count):
        if off + ENTITY_SIZE > len(data):
            break
        cid, sid = data[off], data[off + 1]
        x, y, z, yaw = struct.unpack_from("<ffff", data, off + 2)
        entities.append({"id": cid, "scene_id": sid, "x": x, "y": y, "z": z, "yaw": yaw})
        off += ENTITY_SIZE
    return entities


def serialize_snapshot(data, self_id=None, tick=0, scene_id=0):
    """
    Convert a raw PacketSnapshot byte buffer to a GPT-2 state token string.

    Args:
        data:     bytes — raw UDP packet starting with PACKET_SNAPSHOT byte
        self_id:  int|None — client ID of the bot (None = no self entity)
        tick:     int — current game tick (for context)
        scene_id: int — current scene

    Returns:
        str — natural language state string, stable for a fixed input
    """
    entities = _parse_entities(data)
    self_ent = None
    enemies = []
    for e in entities:
        if e["id"] == self_id:
            self_ent = e
        else:
            enemies.append(e)

    parts = [f"shankpit state tick:{tick} scene:{scene_id}"]

    if self_ent:
        sx, sz = round(self_ent["x"], 1), round(self_ent["z"], 1)
        parts.append(f"self pos:{sx},{sz} yaw:{_yaw_to_direction(self_ent['yaw'])}")
    else:
        parts.append("self pos:? yaw:?")

    if enemies:
        enemy_tokens = []
        for e in enemies[:4]:  # cap at 4 to keep token count bounded
            ex, ez = round(e["x"], 1), round(e["z"], 1)
            dist = round(_dist2d(
                self_ent["x"] if self_ent else 0,
                self_ent["z"] if self_ent else 0,
                e["x"], e["z"],
            ), 1)
            enemy_tokens.append(f"pos:{ex},{ez} dist:{dist} yaw:{_yaw_to_direction(e['yaw'])}")
        parts.append("enemy: " + " | ".join(enemy_tokens))
    else:
        parts.append("enemy: none")

    return "\n".join(parts)


def decode_action(action_str):
    """
    Parse a GPT-2 action token string into a UserCmd field dict.

    Expected format (all fields optional):
      move:northeast fwd:0.8 str:0.2 yaw_delta:-2.3 shoot:1 jump:0 crouch:0 weapon:ar

    Returns dict with keys:
      fwd (float), str (float), yaw_delta (float),
      shoot (bool), jump (bool), crouch (bool), weapon_idx (int)
    """
    result = {
        "fwd": 0.0,
        "str": 0.0,
        "yaw_delta": 0.0,
        "shoot": False,
        "jump": False,
        "crouch": False,
        "weapon_idx": 1,  # Magnum default
    }

    MOVE_TO_FWD_STR = {
        "north":     ( 1.0,  0.0),
        "northeast": ( 0.7,  0.7),
        "east":      ( 0.0,  1.0),
        "southeast": (-0.7,  0.7),
        "south":     (-1.0,  0.0),
        "southwest": (-0.7, -0.7),
        "west":      ( 0.0, -1.0),
        "northwest": ( 0.7, -0.7),
    }

    for token in action_str.split():
        if ":" not in token:
            continue
        key, val = token.split(":", 1)
        try:
            if key == "move":
                fwd, st = MOVE_TO_FWD_STR.get(val, (0.0, 0.0))
                result["fwd"] = fwd
                result["str"] = st
            elif key == "fwd":
                result["fwd"] = float(val)
            elif key == "str":
                result["str"] = float(val)
            elif key == "yaw_delta":
                result["yaw_delta"] = float(val)
            elif key == "shoot":
                result["shoot"] = bool(int(val))
            elif key == "jump":
                result["jump"] = bool(int(val))
            elif key == "crouch":
                result["crouch"] = bool(int(val))
            elif key == "weapon":
                result["weapon_idx"] = WEAPON_IDX.get(val, 1)
        except (ValueError, KeyError):
            pass  # malformed token — skip

    return result


def encode_action(fwd=0.0, str_=0.0, yaw_delta=0.0,
                  shoot=False, jump=False, crouch=False, weapon_idx=1):
    """
    Encode bot inputs back to an action token string (for replay logging).
    """
    weapon = WEAPON_NAMES.get(weapon_idx, "magnum")
    return (
        f"fwd:{round(fwd,2)} str:{round(str_,2)} yaw_delta:{round(yaw_delta,2)} "
        f"shoot:{int(shoot)} jump:{int(jump)} crouch:{int(crouch)} weapon:{weapon}"
    )
