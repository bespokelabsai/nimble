"""Fitted temperatures apply only to the exact checkpoint they were fitted on."""

import unittest

from nimble.scoring.calibration import fitted_temperature

NIMBLE = "bespokelabs/Bespoke-Nimble-9B"
REVISION = "93ec5d6ff1a9cd31d6cc0e0c58d312465d36de7c"


class FittedTemperatureTests(unittest.TestCase):
    def test_only_the_exact_checkpoint_has_a_temperature(self):
        self.assertEqual(fitted_temperature(NIMBLE, REVISION), 2.179078721266035)
        self.assertIsNone(fitted_temperature(NIMBLE, "main"))
        self.assertIsNone(fitted_temperature("Qwen/Qwen3.5-9B", REVISION))


if __name__ == "__main__":
    unittest.main()
