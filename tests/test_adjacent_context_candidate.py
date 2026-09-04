from railway_recon.adjacent_context_candidate import (
    _context_asset_type,
    _next_context_sequence,
)


def test_context_asset_types_follow_source_namespace_and_component() -> None:
    assert _context_asset_type("TRACK--TRACK-1-RAIL-LEFT") == "rail"
    assert _context_asset_type("TRACK--TRACK-1-SLEEPERS") == "sleeper_group"
    assert _context_asset_type("TRACK--TRACK-1-BED") == "track_bed"
    assert (
        _context_asset_type("STATION--CANOPY--RIGHT-CANOPY-ROOF-01")
        == "canopy_roof_context"
    )
    assert (
        _context_asset_type("STATION--CANOPY--RIGHT-CANOPY-GRID-01")
        == "canopy_column_context"
    )
    assert (
        _context_asset_type("CONDUCTOR--MESSENGER-WIRE-01")
        == "catenary_conductor_context"
    )
    assert _context_asset_type("TRACK-0001-RAIL-LEFT") == "rail"
    assert (
        _context_asset_type("ADJACENT-NEXT-RIGHT-CANOPY-ROOF-SURFACE-01")
        == "canopy_roof_context"
    )
    assert (
        _context_asset_type("ADJACENT-NEXT-RIGHT-PLATFORM-TRANSITION")
        == "platform_transition_context"
    )


def test_next_context_sequence_skips_existing_context_ids() -> None:
    registry = {
        "assets": [
            {"id": "ORIGINAL-001"},
            {"id": "ADJCTX-004-RAIL"},
            {"id": "ADJCTX-012-PLATFORM"},
        ]
    }
    assert _next_context_sequence(registry) == 13
