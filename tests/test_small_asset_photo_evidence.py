import unittest

from railway_recon.small_asset_photo_evidence import (
    build_small_asset_vertical_proxies,
)


class SmallAssetPhotoEvidenceTests(unittest.TestCase):
    def test_long_component_uses_three_review_anchors(self) -> None:
        small_assets = {
            "segment_id": "s0_50m",
            "sides": [
                {
                    "side": "right",
                    "candidates": [
                        {
                            "id": "FENCE-001",
                            "candidate_class": "railing_or_fence_candidate",
                            "minimum_s_c_z_m": [0.0, 8.0, 1.0],
                            "maximum_s_c_z_m": [50.0, 8.2, 2.4],
                            "centroid_s_c_z_m": [25.0, 8.1, 1.7],
                            "span_s_c_z_m": [50.0, 0.2, 1.4],
                        }
                    ],
                }
            ],
        }
        vertical, provenance = build_small_asset_vertical_proxies(
            small_assets, {"frame": {"origin_xy": [1.0, 2.0]}}
        )
        self.assertEqual(len(vertical["candidates"]), 3)
        self.assertEqual(
            [item["longitudinal_position_m"] for item in vertical["candidates"]],
            [5.0, 25.0, 45.0],
        )
        self.assertTrue(all(value["source_candidate_id"] == "FENCE-001" for value in provenance.values()))

    def test_compact_component_uses_one_review_anchor(self) -> None:
        small_assets = {
            "segment_id": "s0_50m",
            "sides": [
                {
                    "side": "right",
                    "candidates": [
                        {
                            "id": "COLUMN-001",
                            "candidate_class": "unresolved_above_platform_component",
                            "minimum_s_c_z_m": [4.0, 3.0, 1.0],
                            "maximum_s_c_z_m": [4.8, 3.7, 4.0],
                            "centroid_s_c_z_m": [4.4, 3.35, 2.5],
                            "span_s_c_z_m": [0.8, 0.7, 3.0],
                        }
                    ],
                }
            ],
        }
        vertical, _ = build_small_asset_vertical_proxies(
            small_assets, {"frame": {"origin_xy": [1.0, 2.0]}}
        )
        self.assertEqual(len(vertical["candidates"]), 1)
        self.assertEqual(vertical["candidates"][0]["id"], "COLUMN-001")
        self.assertAlmostEqual(vertical["candidates"][0]["footprint_m"], 0.8)


if __name__ == "__main__":
    unittest.main()
