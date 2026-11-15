import unittest

from serve.heterogeneous_long_context import (MAX_LONG_CONTEXT, LONG_COMMAND_BYTES,
    command_cap, require_mode, valid_cells, validate_capacity)


class LongContextTests(unittest.TestCase):
    def test_staged_contexts(self):
        for n in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
            self.assertTrue(valid_cells(n))
            validate_capacity(n, n)
        for n in (0, 2047, 65535, 262145, True, 262144.0):
            self.assertFalse(valid_cells(n))
        with self.assertRaises(ValueError):
            validate_capacity(262144, 131072)

    def test_explicit_mode(self):
        require_mode(True, True, 'int8', True, 128, False)
        require_mode(False, False, 'f16', False, 0, True)
        for args in ((True, False, 'int8', True, 128, False),
                     (True, True, 'f16', True, 128, False),
                     (True, True, 'int8', False, 128, False),
                     (True, True, 'int8', True, 0, False),
                     (True, True, 'int8', True, 2049, False),
                     (True, True, 'int8', True, 128, True),
                     (1, True, 'int8', True, 128, False)):
            with self.assertRaises(ValueError):
                require_mode(*args)

    def test_protocol_size(self):
        self.assertEqual(command_cap(8192, False), 65536)
        self.assertEqual(command_cap(32768, False), 384 * 1024)
        with self.assertRaises(ValueError):
            command_cap(MAX_LONG_CONTEXT, False)
        self.assertEqual(command_cap(MAX_LONG_CONTEXT, True), LONG_COMMAND_BYTES)
        wire = '1 ' + '2147483647 ' * (MAX_LONG_CONTEXT-1)
        self.assertLessEqual(len(wire), LONG_COMMAND_BYTES)


if __name__ == '__main__':
    unittest.main()
