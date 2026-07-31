"""Tests for the schema derived from the codebook."""

from __future__ import annotations

from vision_zero.data.schema import (
    DERIVED,
    TERNARY_SEVERITY,
    columns,
    dtypes,
    feature_candidates,
    leaks,
    ordered_categories,
)


def test_every_column_has_a_dtype() -> None:
    assert set(dtypes()) == set(columns())


def test_integer_coded_variables_are_nullable_integers() -> None:
    """Nullable, so that an unspecified value stays null instead of forcing float."""
    for name in ("WEATHER", "RD_COND", "MONTH", "NUM_VEH", "LIGHT", "YEAR"):
        assert dtypes()[name] == "Int64", name


def test_variables_with_non_integer_values_are_categories() -> None:
    for name in ("SEVERITY", "ASPECT", "ACCDN_TYPE", "REGION", "HOUR", "PED"):
        assert dtypes()[name] == "category", name


def test_speed_limit_is_categorical_despite_being_ordinal() -> None:
    """One of its values is '<50', so it cannot be an integer column."""
    assert dtypes()["SPD_LIM"] == "category"
    assert "SPD_LIM" in ordered_categories()


def test_identifiers_are_strings() -> None:
    assert dtypes()["ID"] == "string"


def test_ordinal_categories_keep_their_documented_order() -> None:
    order = ordered_categories()["SPD_LIM"]
    assert order[0] == "<50"
    assert order[-1] == "100"

    severity = ordered_categories()["SEVERITY"]
    assert severity[0] == "Material damage below the reporting threshold"
    assert severity[-1] == "Fatal or serious"


def test_ternary_severity_merges_the_two_material_damage_levels() -> None:
    """They differ by a dollar threshold on the estimate, not by outcome."""
    mapping = TERNARY_SEVERITY.mapping
    below_threshold = "Material damage below the reporting threshold"
    assert mapping["Material damage only"] == mapping[below_threshold]
    assert mapping["Minor"] == 1
    assert mapping["Fatal or serious"] == 2


def test_derived_columns_map_every_documented_source_value() -> None:
    from vision_zero.data import load_codebook

    book = load_codebook()
    for derived in DERIVED:
        documented = set(book[derived.source].values)
        mapped = {str(key) for key in derived.mapping}
        assert documented <= mapped, f"{derived.name} does not cover all {derived.source} values"


def test_leakage_is_symmetric_where_it_should_be() -> None:
    assert "NUM_VICTIMS" in leaks("TNRY_SEV")
    assert "TNRY_SEV" in leaks("NUM_VICTIMS")
    assert "NUM_VEH" in leaks("MULT_VEH")
    assert "MULT_VEH" in leaks("NUM_VEH")


def test_unknown_target_has_no_declared_leakage() -> None:
    assert leaks("WEATHER") == frozenset()


def test_feature_candidates_exclude_the_target_and_what_determines_it() -> None:
    features = feature_candidates("TNRY_SEV")
    assert "TNRY_SEV" not in features
    assert "SEVERITY" not in features
    assert "NUM_VICTIMS" not in features
    assert "VICTIMS" not in features
    assert "WEATHER" in features


def test_feature_candidates_exclude_identifiers() -> None:
    assert "ID" not in feature_candidates("TNRY_SEV")


def test_feature_candidates_honour_explicit_exclusions() -> None:
    features = feature_candidates("TNRY_SEV", exclude=frozenset({"YEAR", "REGION"}))
    assert "YEAR" not in features
    assert "REGION" not in features


def test_accident_type_leaks_the_road_user_flags() -> None:
    """A pedestrian collision is exactly one whose PED flag is set."""
    features = feature_candidates("ACCDN_TYPE")
    for flag in ("PED", "BICYC", "MTRCYC", "HVY_VEH", "LT_TRK"):
        assert flag not in features


def test_the_pedestrian_indicator_is_treated_as_an_outcome() -> None:
    """`PED` records a pedestrian *victim*, not a pedestrian *involved*.

    The other road-user indicators say a vehicle of some kind took part. `PED`
    says someone was hurt, which is part of the answer rather than part of the
    question. A model given it is rewarded for restating the target.
    """
    from vision_zero.data.schema import OUTCOME_VARIABLES

    assert "PED" in OUTCOME_VARIABLES
    for target in ("TNRY_SEV", "SEVERITY", "VICTIMS", "NUM_VICTIMS"):
        assert "PED" in leaks(target), target
        assert "PED" not in feature_candidates(target), target


def test_involvement_indicators_remain_available() -> None:
    """These record who took part, which is a circumstance and not an outcome."""
    features = feature_candidates("TNRY_SEV")
    for indicator in ("BICYC", "MTRCYC", "HVY_VEH", "LT_TRK"):
        assert indicator in features, indicator


def test_accident_type_remains_available_for_severity() -> None:
    """A collision *with* a pedestrian describes the event, not its outcome."""
    assert "ACCDN_TYPE" in feature_candidates("TNRY_SEV")


def test_every_outcome_variable_is_excluded_from_every_severity_target() -> None:
    from vision_zero.data.schema import OUTCOME_VARIABLES

    for target in ("TNRY_SEV", "SEVERITY", "VICTIMS", "NUM_VICTIMS"):
        candidates = set(feature_candidates(target))
        assert not (candidates & OUTCOME_VARIABLES), target


def test_every_usable_column_is_assigned_to_a_block() -> None:
    """A column in no block would vanish from the ablation without complaint.

    Checked against the whole schema rather than one experiment's features, so
    that adding a variable to the codebook and forgetting to place it fails
    here rather than silently narrowing an analysis.
    """
    from vision_zero.data.schema import FEATURE_BLOCKS, IDENTIFIERS, OUTCOME_VARIABLES, columns

    assigned = {name for block in FEATURE_BLOCKS.values() for name in block}
    accounted = assigned | set(IDENTIFIERS) | set(OUTCOME_VARIABLES) | {"YEAR"}
    unassigned = set(columns()) - accounted
    assert not unassigned, f"columns in no block: {sorted(unassigned)}"


def test_no_feature_appears_in_two_blocks() -> None:
    from vision_zero.data.schema import FEATURE_BLOCKS

    members = [name for block in FEATURE_BLOCKS.values() for name in block]
    assert len(members) == len(set(members))


def test_blocks_reference_only_real_columns() -> None:
    from vision_zero.data.schema import DERIVED, FEATURE_BLOCKS, columns

    known = set(columns()) | {derived.name for derived in DERIVED}
    for name, block in FEATURE_BLOCKS.items():
        assert block <= known, f"{name} references unknown columns: {sorted(block - known)}"
