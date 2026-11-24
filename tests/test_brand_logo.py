"""Prevent a broken or truncated embedded SearchBridge bird emblem."""
from __future__ import annotations

import base64
from pathlib import Path
import re
import struct
import unittest
import zlib


class BrandLogoTests(unittest.TestCase):
    def test_embedded_png_is_complete_and_valid(self):
        path = Path(__file__).resolve().parents[1] / "assets/searchbridge-robin.svg"
        source = path.read_text(encoding="utf-8")
        match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', source)
        self.assertIsNotNone(match, "Expected original PNG data in emblem SVG")
        data = base64.b64decode(match.group(1), validate=True)
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        position = 8
        chunks = []
        while position + 12 <= len(data):
            length = struct.unpack(">I", data[position:position + 4])[0]
            tag = data[position + 4:position + 8]
            body = data[position + 8:position + 8 + length]
            crc = struct.unpack(">I", data[position + 8 + length:position + 12 + length])[0]
            self.assertEqual(zlib.crc32(tag + body) & 0xffffffff, crc)
            chunks.append(tag)
            position += length + 12
            if tag == b"IEND":
                break
        self.assertIn(b"IHDR", chunks)
        self.assertIn(b"IDAT", chunks)
        self.assertEqual(chunks[-1], b"IEND")
        self.assertEqual(position, len(data))


if __name__ == "__main__":
    unittest.main()
