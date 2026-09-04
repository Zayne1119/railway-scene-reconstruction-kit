from railway_recon.mesh_object_registry import _classify_node, _safe_asset_id


def test_mesh_object_classifier_covers_corridor_component_names() -> None:
    assert _classify_node("TRACKGRAPH--TRACK-0001-RAIL-LEFT")[0] == "rail"
    assert _classify_node("SEG2050-PLATFORM--S2050-PLATFORM-001-TOP")[0] == "platform_surface"
    assert _classify_node("SEG2050-CANOPY--RIGHT-CANOPY-GRID-004-CAPITAL")[0] == "canopy_capital"
    assert _classify_node("SEG2050-CANOPY--RIGHT-CANOPY-GRID-004-LOCAL-UNDERROOF")[0] == "canopy_underroof_connector"
    assert _classify_node("S2050-OPPOSITE-CANOPY-COLUMN-01")[0] == "canopy_column"
    assert _classify_node("CORE-SIGNS--S2100-RIGHT-PLATFORM-SIGN-001")[0] == "station_information_sign"
    assert _classify_node("SCENE--CATENARY-MAST-0001-CANTILEVER")[0] == "catenary_cantilever"
    assert _classify_node("SCENE--CATENARY-MAST-0001-POSITIONER")[0] == "catenary_positioner"
    assert _classify_node("CATENARY-MAST-AUTO-0001-TOPOLOGY-CONSTRAINED-STEEL")[0] == "catenary_cantilever"
    assert _classify_node("SCENE--CATENARY-MAST-0001-INSULATOR-01")[0] == "catenary_insulator"
    assert _classify_node("SCENE--CONTEXT-LOW-CONFIDENCE-001")[0] == "context_building_mass"
    assert _classify_node("SCENE--STATION-ENTRY-GLASS-01")[0] == "station_glazing"


def test_long_mesh_node_gets_stable_registry_identifier() -> None:
    node = "A" + "-VERY-LONG-MESH-NODE" * 10
    asset_id = _safe_asset_id(node)
    assert len(asset_id) == 80
    assert asset_id == _safe_asset_id(node)
