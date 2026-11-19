"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink (Einrasten und Begrenzen), jedes Preset erzeugt gültige Settings."""

import pytest

import rau_constants as C
import rau_evaluation as ev
import rau_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_every_preset_states_every_setting_so_that_switching_never_leaves_a_stale_value():
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name


def test_preset_values_are_valid_for_their_controls():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            elif not isinstance(value, bool):
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_settings_field():
    import dataclasses

    assert set(P.SETTING_SPECS) == {f.name for f in dataclasses.fields(ev.Settings)}


def test_the_stepped_values_lie_on_the_slider_grid():
    for key, spec in P.SETTING_SPECS.items():
        if spec.step is None:
            continue
        for value in {spec.default, *(p[key] for p in C.PRESETS.values() if key in p)}:
            k = (value - spec.lo) / spec.step
            assert abs(k - round(k)) < 1e-9, (key, value)


def test_the_stepped_values_of_the_standard_case_are_the_documented_ones():
    s = ev.Settings()
    assert (s.plan_n, s.plan_gap, s.form_m, s.form_rho, s.plan_double) == (8, 300, 50, 0.5, False)
    assert (s.lam_home, s.lam_away, s.duel_elo, s.away_rule, s.extra_time) == (1.5, 1.1, 100, False, True)
    assert (s.perf_n, s.judges, s.drift, s.anchor) == (10, 5, 0.05, 0.0)
    assert (s.pen_p, s.pen_gap, s.pen_d, s.pen_order) == (0.75, 0.05, 0.05, "ABAB")
    assert (s.alpha, s.power, s.design) == (5, 80, "zwei Lagen")


@pytest.mark.parametrize("key, raw, want", [
    ("form_m", 70, 75), ("form_m", 1000, 150), ("form_m", -5, 0),
    ("lam_home", 1.52, 1.5), ("lam_home", 9.0, 2.5),
    ("form_rho", 0.97, 0.95), ("plan_n", 7, 6), ("plan_n", 1, 4), ("runs", 6000, 5000), ("noise_games", 30, 20),
    ("design", "gegen den Fairwert", "gegen den Fairwert"), ("design", "unsinn", "zwei Lagen"),
    ("pen_order", "ABBA", "ABBA"), ("pen_order", "x", "ABAB"),
])
def test_permalink_values_are_clamped_and_snapped(key, raw, want):
    assert P.snap(raw, P.SETTING_SPECS[key]) == want


def test_to_bool_reads_both_spellings_and_rejects_garbage():
    assert P.to_bool("True") and P.to_bool("1") and not P.to_bool("False") and not P.to_bool("0")
    with pytest.raises(ValueError):
        P.to_bool("vielleicht")


def test_url_parameters_are_unique():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))
