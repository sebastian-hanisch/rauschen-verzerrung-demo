"""Orakel-Tests (ORACLE-PLAYBOOK §2 und §5): jeder Kern der Demo gegen einen anderen Rechenweg.

1. Benötigte Zahl und Teststärke: `scipy.stats.norm` und `scipy.stats.binom` (Summe über beide Binomialverteilungen Zelle für Zelle) und eine unabhängige Simulation
   mit `numpy.random.default_rng`, gegen Formel und `rau_power`.
2. Hin- und Rückspiel: `scipy.stats.skellam` (geschlossene Form ohne Regel) und eine Vierfachschleife über alle vier Torzahlen mit `scipy.stats.poisson` (mit Regel und Verlängerung),
   die Verlängerung in der Schleife über beide Torzahlen.
3. Elfmeterschießen: Aufzählung aller 2^(2R) Schussfolgen der Regelzeit mit eigener Zählung und Sudden Death als Summe über 600 Paare (statt geschlossener Form).
4. Auftrittsreihenfolge: Schleifenrechnung in reinem Python für die Wertungen, Auftrittsreihenfolge als Listenaufbau, exakter Wert für zwei Auftretende mit `scipy.integrate.quad`.
5. Spielplan und Form: vollständige Aufzählung aller Ergebnisbäume einer Liga mit vier Mannschaften (reines Python, ein eigener Fixpunkt für den Startzustand) gegen die Simulation.
6. Rauschgrenze: Aufzählung aller Ergebnisse für drei und vier Mannschaften.
"""

import itertools
import math

import numpy as np
import pytest

import rau_duel as D
import rau_judges as J
import rau_noise as NO
import rau_plan as PL
import rau_power as PW
import rau_shootout as SO
from rau_scenario import circle_rounds, elo_ladder

scipy_stats = pytest.importorskip("scipy.stats")
scipy_integrate = pytest.importorskip("scipy.integrate")


# --- 1. Benötigte Zahl und Teststärke -------------------------------------------------------------------------------------------------------------------

def test_quantiles_agree_with_scipy():
    for q in (0.5, 0.8, 0.9, 0.975, 0.995):
        assert PW.z_quantile(q) == pytest.approx(scipy_stats.norm.ppf(q), abs=1e-9)


def _scipy_power(p1, p2, n, alpha):
    """Exakte Teststärke des Zweistichproben-Tests: Zelle für Zelle über scipy.stats.binom (Handrechnung der Teststatistik, nicht der Code der Demo)."""
    zc = scipy_stats.norm.ppf(1 - alpha / 2)
    x = np.arange(n + 1)
    f1 = scipy_stats.binom.pmf(x, n, p1)
    f2 = scipy_stats.binom.pmf(x, n, p2)
    total = 0.0
    for a in range(n + 1):
        for b in range(n + 1):
            pooled = (a + b) / (2 * n)
            if pooled in (0.0, 1.0):
                continue
            z = abs(a / n - b / n) / math.sqrt(2 * pooled * (1 - pooled) / n)
            if z >= zc - 1e-12:
                total += f1[a] * f2[b]
    return total


@pytest.mark.parametrize("p1, p2, n, alpha", [(0.6, 0.4, 60, 0.05), (0.7, 0.55, 40, 0.1), (0.5, 0.5, 80, 0.05), (0.9, 0.7, 25, 0.01), (0.55, 0.45, 100, 0.05)])
def test_exact_power_matches_the_scipy_double_sum(p1, p2, n, alpha):
    assert PW.exact_power(p1, p2, n, alpha) == pytest.approx(_scipy_power(p1, p2, n, alpha), abs=1e-9)


def test_exact_power_against_fair_matches_scipy():
    n, p0, p1, alpha = 80, 0.5, 0.62, 0.05
    zc = scipy_stats.norm.ppf(1 - alpha / 2)
    x = np.arange(n + 1)
    f = scipy_stats.binom.pmf(x, n, p1)
    rej = np.abs(x / n - p0) >= zc * math.sqrt(p0 * (1 - p0) / n) - 1e-12
    assert PW.exact_power(p1, 2 * p0 - p1, n, alpha, "gegen den Fairwert") == pytest.approx(float((f * rej).sum()), abs=1e-9)


@pytest.mark.parametrize("p1, p2", [(0.55, 0.5), (0.6, 0.4), (0.75, 0.7), (0.9, 0.8), (0.45, 0.35), (0.3, 0.2)])
@pytest.mark.parametrize("power", [0.8, 0.9])
def test_the_formula_hits_the_target_power_within_the_discreteness_of_the_normal_approximation(p1, p2, power):
    """Bei N aus der Formel liegt die exakte Teststärke nahe dem Ziel (Normalnäherung; ab N rund 50 innerhalb von 4 Pp)."""
    n = int(PW.whole_competitions(PW.n_two_groups(p1, p2, 0.05, power)))
    assert n >= 50
    assert abs(PW.exact_power(p1, p2, n, 0.05) - power) < 0.04


def test_simulated_power_with_an_independent_generator_agrees():
    """numpy.random.default_rng, binomiale Zufallszahlen: ein von der Demo unabhängiger Würfelweg."""
    rng = np.random.default_rng(12345)
    p1, p2, n, alpha, reps = 0.6, 0.45, 120, 0.05, 40000
    x1 = rng.binomial(n, p1, reps)
    x2 = rng.binomial(n, p2, reps)
    rate = float(np.mean(PW.rejects_two_groups(x1, x2, n, alpha)))
    assert rate == pytest.approx(PW.exact_power(p1, p2, n, alpha), abs=4 * math.sqrt(0.25 / reps))


def test_the_demo_generator_agrees_with_the_exact_power_within_four_standard_errors():
    for seed in (1, 2, 3):
        sim = PW.simulate_power(0.62, 0.5, 150, 0.05, 20000, seed)
        exact = PW.exact_power(0.62, 0.5, 150, 0.05)
        assert abs(sim.rate - exact) <= 4 * sim.se


def test_binomial_draws_follow_the_binomial_distribution():
    u = np.linspace(0.0005, 0.9995, 1000)
    draws = PW.binomial_draws(30, 0.3, u)
    assert draws.min() >= 0 and draws.max() <= 30
    assert np.mean(draws) == pytest.approx(30 * 0.3, abs=0.2)
    assert np.var(draws) == pytest.approx(30 * 0.3 * 0.7, rel=0.05)


# --- 2. Hin- und Rückspiel --------------------------------------------------------------------------------------------------------------------------------

def _duel_loops(lam_home, lam_away, elo, strong_home_second_leg, rule, extra_time, pen_strong=0.5, kmax=14):
    """Vierfachschleife über alle Torzahlen (scipy.stats.poisson). Mannschaften S (stark) und W; die Verlängerung als Schleife über beide Torzahlen."""
    fs, fw = 10 ** (elo / 800), 10 ** (-elo / 800)
    pm = lambda lam: scipy_stats.poisson.pmf(np.arange(kmax + 1), lam)    # noqa: E731
    if strong_home_second_leg:
        p_s1, p_w1 = pm(lam_away * fs), pm(lam_home * fw)       # Hinspiel: S auswärts, W zu Hause
        p_s2, p_w2 = pm(lam_home * fs), pm(lam_away * fw)       # Rückspiel: S zu Hause, W auswärts
        host_rate, guest_rate = lam_home * fs / 3, lam_away * fw / 3
    else:
        p_s1, p_w1 = pm(lam_home * fs), pm(lam_away * fw)       # Hinspiel: S zu Hause
        p_s2, p_w2 = pm(lam_away * fs), pm(lam_home * fw)       # Rückspiel: S auswärts, W zu Hause
        host_rate, guest_rate = lam_home * fw / 3, lam_away * fs / 3   # Verlängerung im Stadion von W
    ph, pg = pm(host_rate), pm(guest_rate)
    total = 0.0
    for gs1, gw1, gs2, gw2 in itertools.product(range(kmax + 1), repeat=4):
        w = p_s1[gs1] * p_w1[gw1] * p_s2[gs2] * p_w2[gw2]
        d = gs1 + gs2 - gw1 - gw2
        away_s = gs1 if strong_home_second_leg else gs2
        away_w = gw2 if strong_home_second_leg else gw1
        if d > 0:
            total += w
        elif d == 0:
            if rule and away_s != away_w:
                total += w * (away_s > away_w)
                continue
            if not extra_time:
                total += w * pen_strong
                continue
            for h, g in itertools.product(range(kmax + 1), repeat=2):
                q = ph[h] * pg[g]
                s_goals, w_goals = (h, g) if strong_home_second_leg else (g, h)
                if s_goals > w_goals:
                    total += w * q
                elif s_goals == w_goals:
                    if rule and h == g and g >= 1:
                        guest_is_s = not strong_home_second_leg
                        total += w * q * (1.0 if guest_is_s else 0.0)
                    else:
                        total += w * q * pen_strong
    return total


@pytest.mark.parametrize("home", [True, False])
@pytest.mark.parametrize("rule", [False, True])
@pytest.mark.parametrize("extra_time", [False, True])
@pytest.mark.parametrize("lam_home, lam_away, elo", [(1.0, 0.6, 0), (0.9, 0.5, 100), (0.7, 0.7, 200)])
def test_duel_matches_the_four_fold_loop(lam_home, lam_away, elo, extra_time, rule, home):
    got = D.advance_chance(lam_home, lam_away, float(elo), home, rule, extra_time).p_strong
    assert got == pytest.approx(_duel_loops(lam_home, lam_away, elo, home, rule, extra_time), abs=1e-9)


def test_duel_without_rule_and_extra_time_is_the_skellam_closed_form():
    """Ohne Regel und ohne Verlängerung: P(S weiter) = P(D > 0) + P(D = 0)/2 mit D ~ Skellam (Summe der Raten beider Spiele)."""
    for lam_home, lam_away, elo in ((1.5, 1.1, 0), (1.3, 0.9, 150), (2.0, 0.5, 300)):
        fs, fw = 10 ** (elo / 800), 10 ** (-elo / 800)
        mu1 = (lam_home + lam_away) * fs
        mu2 = (lam_home + lam_away) * fw
        want = scipy_stats.skellam.sf(0, mu1, mu2) + 0.5 * scipy_stats.skellam.pmf(0, mu1, mu2)
        for home in (True, False):
            assert D.advance_chance(lam_home, lam_away, float(elo), home, False, False).p_strong == pytest.approx(want, abs=1e-10)


# --- 3. Elfmeterschießen ----------------------------------------------------------------------------------------------------------------------------------

def _shootout_enumeration(p_first, p_second, d, order, shots):
    """Aufzählung aller 2^(2·shots) Schussfolgen (Treffer oder nicht) mit eigener Zählung; Sudden Death als Summe über 600 Paare."""
    seq = []
    for pair in range(shots):
        a_first = order == "ABAB" or pair % 2 == 0
        seq += [0, 1] if a_first else [1, 0]
    p_win = p_tie = 0.0
    for outcome in itertools.product([0, 1], repeat=len(seq)):
        goals = [0, 0]
        prob = 1.0
        for team, hit in zip(seq, outcome):
            p = (p_first, p_second)[team]
            if goals[team] < goals[1 - team]:
                p = max(0.0, p - d)
            prob *= p if hit else 1 - p
            goals[team] += hit
        if goals[0] > goals[1]:
            p_win += prob
        elif goals[0] == goals[1]:
            p_tie += prob
    sd = 0.0
    alive = 1.0
    for pair in range(shots, shots + 600):
        a_first = order == "ABAB" or pair % 2 == 0
        # Paar: erster Schütze auf Gleichstand, zweiter im Rückstand, wenn der erste traf
        pf, ps = (p_first, p_second) if a_first else (p_second, p_first)
        df = pf * (1 - max(0.0, ps - d))          # der erste Schütze gewinnt das Paar
        ds = (1 - pf) * ps                        # der zweite
        win_a = df if a_first else ds
        sd += alive * win_a
        alive *= 1 - df - ds
    return p_win + p_tie * sd


@pytest.mark.parametrize("order", ["ABAB", "ABBA"])
@pytest.mark.parametrize("shots", [1, 2, 3, 5])
@pytest.mark.parametrize("p_first, p_second, d", [(0.75, 0.75, 0.05), (0.8, 0.6, 0.1), (0.6, 0.7, 0.2), (0.5, 0.5, 0.0), (0.9, 0.85, 0.25)])
def test_shootout_matches_the_enumeration_of_all_shot_sequences(order, shots, p_first, p_second, d):
    assert SO.first_team_win(p_first, p_second, d, order, shots) == pytest.approx(_shootout_enumeration(p_first, p_second, d, order, shots), abs=1e-9)


def test_shootout_reference_values_of_the_neighbour_readme():
    """Referenzwerte (README der Nachbar-Demo reihenfolge-demo, gelesen 2026-11-18, dort mit dem eigenen Code berechnet): p = 0,75, 5 Schüsse, d = 0,05 / 0,10 / 0,15 / 0,25."""
    abab = [SO.first_team_win(0.75, 0.75, d, "ABAB", 5) for d in (0.05, 0.10, 0.15, 0.25)]
    abba = [SO.first_team_win(0.75, 0.75, d, "ABBA", 5) for d in (0.05, 0.10, 0.15, 0.25)]
    assert [round(100 * x, 1) for x in abab] == [54.5, 58.3, 61.5, 66.7]
    assert [round(100 * x, 1) for x in abba] == [50.7, 51.5, 52.5, 55.0]
    assert SO.first_team_win(0.8, 0.6, 0.0, "ABAB", 5) == pytest.approx(0.799473664, abs=1e-9)


# --- 4. Auftrittsreihenfolge ------------------------------------------------------------------------------------------------------------------------------

def _scores_loop(q_pos, e_pos, sigma, drift, anchor):
    """Wertung je Auftritt in reinem Python (Listen, `statistics.fmean`), ein Wettbewerb nach dem anderen."""
    import statistics

    out = []
    for q_row, e_row in zip(q_pos, e_pos):
        n = len(q_row)
        prev = 0.0
        ratings = []
        for k in range(n):
            notes = [q_row[k] + sigma * e + drift * (k - (n - 1) / 2) + anchor * prev for e in e_row[k]]
            prev = statistics.fmean(notes)
            ratings.append(prev)
        out.append(ratings)
    return np.array(out)


@pytest.mark.parametrize("drift, anchor", [(0.0, 0.0), (0.05, 0.0), (0.0, 0.3), (0.1, -0.4)])
def test_panel_scores_match_the_python_loop(drift, anchor):
    q, e, keys = J.draw_competitions(5, 3, 60, 7)
    order = J.order_with_strongest_at(q, keys, 2)
    q_pos = np.take_along_axis(q, order, axis=1)
    e_pos = np.take_along_axis(e, order[:, :, None], axis=1)
    got = J.panel_scores(q_pos, e_pos, 0.5, drift, anchor)
    assert np.allclose(got, _scores_loop(q_pos.tolist(), e_pos.tolist(), 0.5, drift, anchor), atol=1e-12)


def test_order_with_strongest_at_matches_a_list_construction():
    q, _, keys = J.draw_competitions(6, 2, 50, 3)
    for position in (0, 2, 5):
        order = J.order_with_strongest_at(q, keys, position)
        for r in range(50):
            best = int(np.argmax(q[r]))
            others = sorted((i for i in range(6) if i != best), key=lambda i: (keys[r, i], i))
            want = others[:position] + [best] + others[position:]
            assert list(order[r]) == want


@pytest.mark.parametrize("beta", [0.0, 0.2, -0.3])
@pytest.mark.parametrize("position", [0, 1])
def test_two_performers_match_the_exact_integral(beta, position):
    """n = 2, γ = 0: D = |q1 - q2| (Halbnormal mit Skala √2) gegen das Panel-Rauschen τ = σ·√(2/J); die Qualitätsbeste auf Platz p hat den Vorsprung β·(Platz-Unterschied)."""
    sigma, judges = 0.5, 5
    tau = sigma * math.sqrt(2.0 / judges)
    offset = beta * (1 if position == 1 else -1) * 1.0           # Platz 1 (Index 1) bekommt +β gegenüber dem Platz davor, Platz 0 das Negative
    density = lambda x: 2 * scipy_stats.norm.pdf(x, scale=math.sqrt(2.0))    # noqa: E731
    want, _ = scipy_integrate.quad(lambda x: density(x) * scipy_stats.norm.cdf((x + offset) / tau), 0, np.inf)
    runs = 60000
    got = J.strongest_wins(2, judges, sigma, beta, 0.0, runs, 99, positions=[position])[0]
    assert abs(got.mean() - want) <= 4 * math.sqrt(want * (1 - want) / runs)


def test_independent_generator_agrees_with_the_vectorised_simulation():
    """Unabhängiger Lauf mit numpy.random.default_rng und Schleife je Wettbewerb gegen die vektorisierte Simulation (Drift und Angleichung)."""
    rng = np.random.default_rng(2024)
    n, judges, sigma, beta, gamma, runs = 6, 4, 0.5, 0.06, 0.2, 40000
    wins = 0
    for _ in range(runs):
        q = rng.normal(size=n)
        best = int(np.argmax(q))
        others = [i for i in range(n) if i != best]
        rng.shuffle(others)
        order = others[:3] + [best] + others[3:]                  # die Beste auf Platz 3 (von 0 gezählt)
        prev = 0.0
        ratings = []
        for k, person in enumerate(order):
            notes = q[person] + sigma * rng.normal(size=judges) + beta * (k - (n - 1) / 2) + gamma * prev
            prev = float(notes.mean())
            ratings.append(prev)
        wins += int(np.argmax(ratings)) == 3
    got = J.strongest_wins(n, judges, sigma, beta, gamma, 40000, 5, positions=[3])[0].mean()
    se = math.sqrt(0.25 / runs) * math.sqrt(2)
    assert abs(wins / runs - got) <= 4 * se


# --- 5. Spielplan und Form --------------------------------------------------------------------------------------------------------------------------------

def _elo_probs(ra, rb, draw_max):
    e = 1 / (1 + 10 ** (-(ra - rb) / 400))
    d = min(draw_max, 2 * min(e, 1 - e))
    return e - d / 2, d, 1 - e - d / 2           # Sieg, Remis, Niederlage


def _start_pi_loop(ratings, m, rho, draw_max, iters=300):
    """Fixpunkt des Mittelfeld-Modells mit Schleifen in reinem Python."""
    n = len(ratings)
    states = (-1, 0, 1)
    pi = [[1 / 3] * 3 for _ in range(n)]
    for _ in range(iters):
        new = []
        for i in range(n):
            q = [[0.0] * 3 for _ in range(3)]
            for s in range(3):
                for j in range(n):
                    if j == i:
                        continue
                    for t in range(3):
                        w, d, l = _elo_probs(ratings[i] + m * states[s], ratings[j] + m * states[t], draw_max)
                        weight = pi[j][t] / (n - 1)
                        q[s][2] += weight * w
                        q[s][1] += weight * d
                        q[s][0] += weight * l
            row = [sum(pi[i][s] * ((rho if s == k else 0.0) + (1 - rho) * q[s][k]) for s in range(3)) for k in range(3)]
            total = sum(row)
            new.append([x / total for x in row])
        pi = new
    return pi


def _exact_titles_n4(ratings, perm, m, rho, draw_max, pi):
    """Vollständige Aufzählung der Ergebnisbäume einer Liga mit vier Mannschaften (einfache Runde): exakte Titelchance der Mannschaft 0 (Gleichstand zu gleichen Teilen).

    Zustand: (Zustände der 4 Mannschaften, Punkte der 4 Mannschaften); jede Partie verzweigt in Sieg, Remis, Niederlage, danach hält jede Mannschaft ihren Zustand
    mit Wahrscheinlichkeit ρ oder setzt ihn auf ihr Ergebnis."""
    states = (-1, 0, 1)
    dist = {}
    for s in itertools.product(range(3), repeat=4):
        pr = 1.0
        for team in range(4):
            pr *= pi[team][s[team]]
        dist[(s, (0, 0, 0, 0))] = pr
    for rnd in circle_rounds(4):
        pairs = [(perm[a], perm[b]) for a, b in rnd]
        for lo, hi in pairs:
            lo, hi = min(lo, hi), max(lo, hi)
            nxt = {}
            for (s, pts), pr in dist.items():
                w, d, l = _elo_probs(ratings[lo] + m * states[s[lo]], ratings[hi] + m * states[s[hi]], draw_max)
                for res, pres in ((1, w), (0, d), (-1, l)):
                    if pres == 0.0:
                        continue
                    new_pts = list(pts)
                    new_pts[lo] += (3, 1, 0)[(1, 0, -1).index(res)]
                    new_pts[hi] += (0, 1, 3)[(1, 0, -1).index(res)]
                    for keep_lo in (True, False):
                        for keep_hi in (True, False):
                            pk = (rho if keep_lo else 1 - rho) * (rho if keep_hi else 1 - rho)
                            if pk == 0.0:
                                continue
                            ns = list(s)
                            if not keep_lo:
                                ns[lo] = res + 1
                            if not keep_hi:
                                ns[hi] = -res + 1
                            key = (tuple(ns), tuple(new_pts))
                            nxt[key] = nxt.get(key, 0.0) + pr * pres * pk
            dist = nxt
        # Jede Mannschaft spielt je Spieltag genau einmal: die Partien eines Spieltags beeinflussen einander nicht, die Reihenfolge der Verarbeitung ist gleichgültig
    total = 0.0
    for (s, pts), pr in dist.items():
        top = max(pts)
        leaders = [i for i in range(4) if pts[i] == top]
        total += pr * ((1 / len(leaders)) if 0 in leaders else 0.0)
    return total


@pytest.mark.parametrize("perm", [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_simulated_title_chance_matches_the_exact_enumeration_for_four_teams(perm):
    ratings = elo_ladder(4, 300.0)
    m, rho, dmax = 100.0, 0.5, 0.25
    pi = _start_pi_loop(list(ratings), m, rho, dmax)
    want = _exact_titles_n4(list(ratings), perm, m, rho, dmax, pi)
    runs = 100000
    run = PL.simulate_plans(ratings, np.array([perm]), runs, 77, m, rho, dmax)
    got = float(run.share[0].mean())
    # Standardfehler eines Anteils; Läufe unabhängig, der Anteil je Lauf liegt in [0, 1]
    se = float(run.share[0].std(ddof=1) / math.sqrt(runs))
    assert abs(got - want) <= 4 * se, (got, want, se)


def test_start_distribution_matches_the_loop_fixed_point():
    ratings = [150.0, 50.0, -50.0, -150.0]
    for m, rho in ((50.0, 0.5), (100.0, 0.0), (25.0, 0.9)):
        want = np.array(_start_pi_loop(ratings, m, rho, 0.25))
        assert np.allclose(PL.form_start_distribution(np.array(ratings), m, rho, 0.25), want, atol=1e-9)


def test_exact_title_chances_sum_to_one_over_the_teams_with_the_enumeration():
    """Gegenprobe für die Aufzählung selbst: die Titelchancen der vier Mannschaften (Mannschaft i als „Mannschaft 0“ umnummeriert) summieren sich zu eins."""
    ratings = [150.0, 50.0, -50.0, -150.0]
    m, rho, dmax = 50.0, 0.5, 0.25
    pi = _start_pi_loop(ratings, m, rho, dmax)
    perm = (0, 1, 2, 3)
    total = 0.0
    for team in range(4):
        order = [team] + [i for i in range(4) if i != team]
        # Mannschaft `team` wird zur Mannschaft 0, indem alle Listen und die Zuordnung umnummeriert werden
        r2 = [ratings[i] for i in order]
        pi2 = [pi[i] for i in order]
        inv = {old: new for new, old in enumerate(order)}
        total += _exact_titles_n4(r2, tuple(inv[p] for p in perm), m, rho, dmax, pi2)
    assert total == pytest.approx(1.0, abs=1e-9)


# --- 6. Rauschgrenze --------------------------------------------------------------------------------------------------------------------------------------

def _top_exact(n, elo_step, draw_max=0.0):
    """Aufzählung aller 2^(Paare) bzw. 3^(Paare) Ergebnisse einer Liga mit einer Partie je Paar: exakte Chance der Stärksten an der Spitze (Gleichstand zu gleichen Teilen)."""
    ratings = [-elo_step * i for i in range(n)]
    pairs = list(itertools.combinations(range(n), 2))
    total = 0.0
    for outcome in itertools.product((1, 0, -1), repeat=len(pairs)):
        prob = 1.0
        pts = [0] * n
        for (i, j), res in zip(pairs, outcome):
            w, d, l = _elo_probs(ratings[i], ratings[j], draw_max)
            prob *= {1: w, 0: d, -1: l}[res]
            pts[i] += {1: 3, 0: 1, -1: 0}[res]
            pts[j] += {1: 0, 0: 1, -1: 3}[res]
        if prob == 0.0:
            continue
        top = max(pts)
        leaders = [k for k in range(n) if pts[k] == top]
        total += prob * ((1 / len(leaders)) if 0 in leaders else 0.0)
    return total


@pytest.mark.parametrize("n, step, draw_max", [(3, 100.0, 0.0), (4, NO.ELO_STEP, 0.0), (3, 60.0, 0.3), (4, 0.0, 0.0)])
def test_noise_floor_matches_the_enumeration(n, step, draw_max):
    want = _top_exact(n, step, draw_max)
    res = NO.top_chance(n, 1, step, 60000, 11, draw_max)
    assert abs(res.p_top - want) <= 4 * res.se + 1e-9


def test_noise_floor_of_equal_teams_is_one_over_n():
    for n in (2, 3, 5):
        assert NO.top_chance(n, 1, 0.0, 40000, 3).p_top == pytest.approx(1 / n, abs=0.012)


def _plan_classes_n4():
    """Die 24 Zuordnungen von vier Mannschaften ergeben nur sechs verschiedene Spielpläne (Folge der Spieltage als Mengen von Paaren): je eine Vertreterin."""
    classes = {}
    for perm in itertools.permutations(range(4)):
        key = tuple(frozenset(frozenset((perm[a], perm[b])) for a, b in rnd) for rnd in circle_rounds(4))
        classes.setdefault(key, perm)
    return list(classes.values())


def test_four_teams_have_six_distinct_plans():
    assert len(_plan_classes_n4()) == 6


def test_the_lage_difference_of_the_simulation_matches_the_exact_one_for_four_teams():
    """Exakter Lagenunterschied (größte minus kleinste Titelchance der sechs Spielpläne, Aufzählung) gegen die Simulation `plan_lage` (24 Zuordnungen, 40.000 Läufe, drei Zufalls-Seeds, Mittel)."""
    ratings = list(elo_ladder(4, 300.0))
    m, rho, dmax = 100.0, 0.0, 0.25
    pi = _start_pi_loop(ratings, m, rho, dmax)
    values = [_exact_titles_n4(ratings, perm, m, rho, dmax, pi) for perm in _plan_classes_n4()]
    exact = max(values) - min(values)
    sims = [PL.plan_lage(4, 300.0, 40000, seed, m, rho, dmax, 24).delta for seed in (1, 2, 3)]
    assert 100 * exact == pytest.approx(2.24, abs=0.005)                  # README: exakt 2,24 Pp
    assert 100 * float(np.mean(sims)) == pytest.approx(2.16, abs=0.15)     # README: Simulation 2,16 Pp (2,01 / 2,26 / 2,20)
    assert abs(float(np.mean(sims)) - exact) < 0.004, (exact, sims)
