"""Unit tests for game_state.py — serialize_snapshot + decode_action."""

import struct
import unittest
from game_state import (
    PACKET_SNAPSHOT,
    ENTITY_SIZE,
    serialize_snapshot,
    decode_action,
    encode_action,
)


def _make_snapshot(*entities):
    """Build a raw PacketSnapshot buffer from a list of (id, scene, x, y, z, yaw) tuples."""
    buf = bytearray([PACKET_SNAPSHOT, len(entities)])
    for eid, sid, x, y, z, yaw in entities:
        buf += bytes([eid, sid])
        buf += struct.pack("<ffff", x, y, z, yaw)
    return bytes(buf)


class TestSerializeSnapshot(unittest.TestCase):

    def test_empty_snapshot_no_self(self):
        data = bytes([PACKET_SNAPSHOT, 0])
        out = serialize_snapshot(data)
        self.assertIn("enemy: none", out)
        self.assertIn("self pos:?", out)

    def test_single_enemy_no_self(self):
        data = _make_snapshot((3, 0, 20.0, 5.0, 15.0, 90.0))
        out = serialize_snapshot(data, self_id=None, tick=100, scene_id=2)
        self.assertIn("tick:100", out)
        self.assertIn("scene:2", out)
        self.assertIn("enemy:", out)
        self.assertIn("pos:20.0,15.0", out)

    def test_self_entity_extracted(self):
        data = _make_snapshot(
            (1, 0, 10.0, 5.0, 8.0, 0.0),    # self
            (2, 0, 22.0, 5.0, 18.0, 180.0),  # enemy
        )
        out = serialize_snapshot(data, self_id=1)
        self.assertIn("self pos:10.0,8.0", out)
        self.assertIn("yaw:north", out)
        # enemy should not include self
        self.assertNotIn("pos:10.0,8.0", out.split("enemy:")[1])

    def test_distance_computed(self):
        data = _make_snapshot(
            (1, 0,  0.0, 5.0,  0.0, 0.0),   # self at origin
            (2, 0, 30.0, 5.0, 40.0, 0.0),   # enemy at (30,40) → dist=50
        )
        out = serialize_snapshot(data, self_id=1)
        self.assertIn("dist:50.0", out)

    def test_stable_for_fixed_input(self):
        data = _make_snapshot((5, 0, 14.0, 5.0, 8.0, 45.0))
        out1 = serialize_snapshot(data, self_id=None, tick=500, scene_id=0)
        out2 = serialize_snapshot(data, self_id=None, tick=500, scene_id=0)
        self.assertEqual(out1, out2)

    def test_wrong_packet_type_returns_empty(self):
        bad = bytes([0x01, 0])  # PacketUserCmd, not PacketSnapshot
        out = serialize_snapshot(bad)
        self.assertIn("enemy: none", out)

    def test_caps_enemies_at_four(self):
        entities = [(i, 0, float(i * 5), 5.0, float(i * 5), 0.0) for i in range(1, 8)]
        data = _make_snapshot(*entities)
        out = serialize_snapshot(data, self_id=None)
        # At most 4 enemy entries separated by " | "
        enemy_section = out.split("enemy: ")[1]
        self.assertLessEqual(enemy_section.count("pos:"), 4)


class TestDecodeAction(unittest.TestCase):

    def test_full_action_string(self):
        r = decode_action("move:northeast fwd:0.8 str:0.2 yaw_delta:-2.3 shoot:1 jump:0 crouch:0 weapon:ar")
        self.assertAlmostEqual(r["fwd"], 0.8)
        self.assertAlmostEqual(r["str"], 0.2)
        self.assertAlmostEqual(r["yaw_delta"], -2.3)
        self.assertTrue(r["shoot"])
        self.assertFalse(r["jump"])
        self.assertFalse(r["crouch"])
        self.assertEqual(r["weapon_idx"], 2)  # ar=2

    def test_move_token_sets_fwd_str(self):
        r = decode_action("move:north")
        self.assertAlmostEqual(r["fwd"], 1.0)
        self.assertAlmostEqual(r["str"], 0.0)

    def test_weapon_names(self):
        for name, idx in [("knife", 0), ("magnum", 1), ("ar", 2), ("shotgun", 3), ("sniper", 4), ("katana", 5)]:
            r = decode_action(f"weapon:{name}")
            self.assertEqual(r["weapon_idx"], idx)

    def test_defaults_on_empty_string(self):
        r = decode_action("")
        self.assertEqual(r["fwd"], 0.0)
        self.assertEqual(r["weapon_idx"], 1)
        self.assertFalse(r["shoot"])

    def test_malformed_tokens_skipped(self):
        r = decode_action("fwd:0.5 GARBAGE shoot:1")
        self.assertAlmostEqual(r["fwd"], 0.5)
        self.assertTrue(r["shoot"])

    def test_roundtrip_encode_decode(self):
        original = encode_action(fwd=0.9, str_=-0.3, yaw_delta=5.0, shoot=True, weapon_idx=4)
        decoded = decode_action(original)
        self.assertAlmostEqual(decoded["fwd"], 0.9)
        self.assertAlmostEqual(decoded["str"], -0.3)
        self.assertAlmostEqual(decoded["yaw_delta"], 5.0)
        self.assertTrue(decoded["shoot"])
        self.assertEqual(decoded["weapon_idx"], 4)

    def test_roundtrip_stable(self):
        s1 = encode_action(fwd=0.5, str_=0.0, shoot=False, weapon_idx=2)
        s2 = encode_action(**{k: decode_action(s1)[k] if k != "str_" else decode_action(s1)["str"]
                              for k in ["fwd", "shoot", "weapon_idx"]} | {"str_": decode_action(s1)["str"]})
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    unittest.main()
