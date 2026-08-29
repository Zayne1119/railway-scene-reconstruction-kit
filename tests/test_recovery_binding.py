from __future__ import annotations

import unittest

from railway_recon.recovery_binding import bind_recovery_registry


class RecoveryBindingTests(unittest.TestCase):
    def test_binding_targets_both_rail_assets_without_geometry_claim(self) -> None:
        observation_id = "s0100_0150m:TRACK-0004"
        registry = [
            {
                "id": f"TRACK-0004-RAIL-{side}",
                "type": "Rail",
                "parameters": {"source_observation_ids": [observation_id]},
                "sources": [],
            }
            for side in ("LEFT", "RIGHT")
        ]
        recovery = {
            "recoveries": [
                {
                    "observation_id": observation_id,
                    "status": "fixed_center_evidence_recovered",
                    "support": {"joint_support_ratio": 0.96},
                }
            ]
        }
        updated, mapping = bind_recovery_registry(registry, recovery, "abc123")
        self.assertEqual(
            mapping[observation_id],
            ["TRACK-0004-RAIL-LEFT", "TRACK-0004-RAIL-RIGHT"],
        )
        for asset in updated:
            self.assertFalse(
                asset["parameters"]["targeted_recovery_geometry_changed"]
            )
            self.assertEqual(
                asset["targetedRecoveryEvidence"][0]["observationId"],
                observation_id,
            )

    def test_binding_rejects_unresolved_recovery(self) -> None:
        with self.assertRaisesRegex(ValueError, "unresolved"):
            bind_recovery_registry(
                [],
                {
                    "recoveries": [
                        {
                            "observation_id": "obs",
                            "status": "unresolved",
                        }
                    ]
                },
                "abc123",
            )


if __name__ == "__main__":
    unittest.main()
