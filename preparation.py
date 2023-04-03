import datetime
import os
import time
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')


def fetching(end_date=datetime.date.today().year):
    start_time = time.time()

    dfs = []
    for year in range(1968, end_date):
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        r = requests.get(url)
        data = r.content.decode('utf-8')
        df = pd.read_csv(StringIO(data))
        dfs.append(df)

    all_matches = pd.concat(dfs, ignore_index=True)

    all_matches.to_csv('data/all_matches.csv', index=False)

    print(f"{fetching.__name__.capitalize()} ✓ in {round(time.time() - start_time, 2)}s")


def mixing_players(data):
    start_time = time.time()

    l1 = ['p1', 'p1_hand', 'p1_height', 'p1_age', 'p1_rank', 'p1_ace', 'p1_double_faults', 'p1_break_points_faced',
          'p1_break_points_saved', 'p1_serves_won', 'p1_serves_total', ]
    l2 = ['p2', 'p2_hand', 'p2_height', 'p2_age', 'p2_rank', 'p2_ace', 'p2_double_faults', 'p2_break_points_faced',
          'p2_break_points_saved', 'p2_serves_won', 'p2_serves_total']

    data['random'] = np.random.randint(2, size=len(data.index))

    for i in data.index:
        if not data['random'][i]:
            for j, k in zip(l1, l2):
                data[j][i], data[k][i] = data[k][i], data[j][i]

    data['winner'] = data['winner'] == data['p1']
    data.drop(columns=['random'], inplace=True)
    data.to_csv('data/mixed.csv', index=False)

    print(f"{mixing_players.__name__.capitalize()} ✓ in {round(time.time() - start_time, 2)}s")

    return data


def filtering(data):
    data['tourney_date'] = pd.to_datetime(data['tourney_date'], format='%Y%m%d')
    data["w_serves_won"] = data["w_1stWon"] + data["w_2ndWon"]
    data["l_serves_won"] = data["l_1stWon"] + data["l_2ndWon"]
    tmp = data[
        ['surface', 'tourney_date', 'minutes', 'winner_name', 'winner_hand', 'winner_ht', 'winner_age', 'winner_rank',
         'w_ace', 'w_df', 'w_bpFaced', 'w_bpSaved', 'w_serves_won', 'w_svpt', 'loser_name', 'loser_hand', 'loser_ht',
         'loser_age', 'loser_rank', 'l_ace', 'l_df', 'l_bpFaced', 'l_bpSaved', 'l_serves_won', 'l_svpt']]

    tmp.columns = ['surface', 'tourney_date', 'minutes', 'p1', 'p1_hand', 'p1_height', 'p1_age', 'p1_rank', 'p1_ace',
                   'p1_double_faults', 'p1_break_points_faced', 'p1_break_points_saved', 'p1_serves_won',
                   'p1_serves_total', 'p2', 'p2_hand', 'p2_height', 'p2_age', 'p2_rank', 'p2_ace', 'p2_double_faults',
                   'p2_break_points_faced', 'p2_break_points_saved', 'p2_serves_won', 'p2_serves_total']

    tmp['winner'] = tmp['p1']

    return tmp


def frames_initialization(data):
    unique_player = np.unique(data[["p1", "p2"]])
    unique_surface = data['surface'].unique()

    player_index = ["win", "total", "last5", "left_win", "left_total", "right_win",
                    "right_total", "serves_won", "serves_total",
                    "break_points_saved", "break_points_faced"]
    player = pd.DataFrame(0, index=player_index, columns=unique_player)
    player.loc["elo"] = 1500
    player.loc["height"] = 0
    player.loc["hand"] = 0

    surface_index = pd.MultiIndex.from_product([unique_surface, ["win", "total"]])
    surface = pd.DataFrame(0, index=surface_index, columns=unique_player)

    versus_index = pd.MultiIndex.from_product([unique_player, ["won", "lost"]])
    versus = pd.DataFrame(0.0, index=versus_index, columns=unique_player)

    return player, surface, versus


def elo(elo1, elo2, n1, n2, result):
    def k(n):
        return 250 / (pow((5 + n), 0.4))

    def pi(first, second):
        return pow(1 + pow(10, (second - first) / 400), -1)

    e1 = elo1 + k(n1) * (result - pi(elo1, elo2))
    e2 = elo2 + k(n2) * ((1 - result) - pi(elo2, elo1))

    return e1, e2


def featuring(data):
    data["p1_break"] = 0.0
    data["p2_break"] = 0.0
    data["p1_serve"] = 0.0
    data["p2_serve"] = 0.0
    data["p1_elo"] = 0.0
    data["p2_elo"] = 0.0
    data["p1_hand_form"] = 0.0
    data["p2_hand_form"] = 0.0
    data["p1_win_h2h"] = 0.0
    data["p2_win_h2h"] = 0.0
    data["p1_win_last5"] = 0.0
    data["p2_win_last5"] = 0.0
    data["p1_win_overall"] = 0.0
    data["p2_win_overall"] = 0.0
    data["p1_surface_overall"] = 0.0
    data["p2_surface_overall"] = 0.0

    data['p1_height'] = data['p1_height'].fillna(value=round(data['p1_height'].mean()))
    data['p2_height'] = data['p2_height'].fillna(value=round(data['p2_height'].mean()))

    data = data[(data['p1_hand'] != 'U') & (data['p2_hand'] != 'U')]

    return data.dropna()


def fill_params(frame, index, df_player, df_surface, df_versus):
    p1_name = frame["p1"][index]
    p2_name = frame["p2"][index]
    surface = frame["surface"][index]
    p1_hand = frame["p1_hand"][index]
    p2_hand = frame["p2_hand"][index]

    df_player[p1_name]['height'] = frame['p1_height'][index]
    df_player[p2_name]['height'] = frame['p2_height'][index]

    if df_player[p1_name]['hand'] == 0:
        if p1_hand == 'L':
            df_player[p1_name]['hand'] = 1
        elif p1_hand == 'R':
            df_player[p1_name]['hand'] = 2

    if df_player[p2_name]['hand'] == 0:
        if p2_hand == 'L':
            df_player[p2_name]['hand'] = 1
        elif p2_hand == 'R':
            df_player[p2_name]['hand'] = 2

    if df_player[p1_name]["total"] > 0:
        frame["p1_win_overall"][index] = (df_player[p1_name]["win"] / df_player[p1_name]["total"])

    if df_player[p2_name]["total"] > 0:
        frame["p2_win_overall"][index] = (df_player[p2_name]["win"] / df_player[p2_name]["total"])

    if df_player[p1_name]["serves_total"] > 0:
        frame["p1_serve"][index] = df_player[p1_name]["serves_won"] / df_player[p1_name]["serves_total"]

    if df_player[p2_name]["serves_total"] > 0:
        frame["p2_serve"][index] = df_player[p2_name]["serves_won"] / df_player[p2_name]["serves_total"]

    if df_player[p1_name]["break_points_faced"] > 0:
        frame["p1_break"][index] = df_player[p1_name]["break_points_saved"] / df_player[p1_name]["break_points_faced"]

    if df_player[p2_name]["break_points_faced"] > 0:
        frame["p2_break"][index] = df_player[p2_name]["break_points_saved"] / df_player[p2_name]["break_points_faced"]

    frame['p1_win_h2h'][index] = df_versus[p1_name][p2_name]['won']
    frame['p2_win_h2h'][index] = df_versus[p2_name][p1_name]['won']

    frame["p1_win_last5"][index] = df_player[p1_name]["last5"]
    frame["p2_win_last5"][index] = df_player[p2_name]["last5"]

    if df_surface[p1_name][surface]["total"] > 0:
        frame["p1_surface_overall"][index] = (
                df_surface[p1_name][surface]["win"] / df_surface[p1_name][surface]["total"])

    if df_surface[p2_name][surface]["total"] > 0:
        frame["p2_surface_overall"][index] = (
                df_surface[p2_name][surface]["win"] / df_surface[p2_name][surface]["total"])

    frame["p1_elo"][index] = df_player[p1_name]["elo"]
    frame["p2_elo"][index] = df_player[p2_name]["elo"]

    if p2_hand == "R" and df_player[p1_name]["right_total"] > 0:
        frame["p1_hand_form"][index] = (
                df_player[p1_name]["right_win"] / df_player[p1_name]["right_total"]
        )
    elif p2_hand == "L" and df_player[p1_name]["left_total"] > 0:
        frame["p1_hand_form"][index] = (
                df_player[p1_name]["left_win"] / df_player[p1_name]["left_total"]
        )

    if p1_hand == "R" and df_player[p2_name]["right_total"] > 0:
        frame["p2_hand_form"][index] = (
                df_player[p2_name]["right_win"] / df_player[p2_name]["right_total"]
        )
    elif p1_hand == "L" and df_player[p2_name]["left_total"] > 0:
        frame["p2_hand_form"][index] = (
                df_player[p2_name]["left_win"] / df_player[p2_name]["left_total"]
        )


def calc_params(frame, index, df_player, df_surface, df_versus):
    p1_name = frame["p1"][index]
    p2_name = frame["p2"][index]
    surface = frame["surface"][index]
    p1_hand = frame["p1_hand"][index]
    p2_hand = frame["p2_hand"][index]
    winner = frame["winner"][index]

    df_player[p1_name]["elo"], df_player[p2_name]["elo"] = elo(
        df_player[p1_name]["elo"],
        df_player[p2_name]["elo"],
        df_player[p1_name]["total"],
        df_player[p2_name]["total"],
        winner,
    )

    df_player[p1_name]["total"] += 1
    df_player[p2_name]["total"] += 1

    df_surface.loc[(surface, "total"), p1_name] += 1
    df_surface.loc[(surface, "total"), p2_name] += 1

    df_player[p1_name]["serves_won"] += frame["p1_serves_won"][index]
    df_player[p2_name]["serves_won"] += frame["p2_serves_won"][index]

    df_player[p1_name]["serves_total"] += frame["p1_serves_total"][index]
    df_player[p2_name]["serves_total"] += frame["p2_serves_total"][index]

    df_player[p1_name]["break_points_saved"] += frame["p1_break_points_saved"][index]
    df_player[p2_name]["break_points_saved"] += frame["p2_break_points_saved"][index]

    df_player[p1_name]["break_points_faced"] += frame["p1_break_points_faced"][index]
    df_player[p2_name]["break_points_faced"] += frame["p2_break_points_faced"][index]

    if p2_hand == "R":
        df_player[p1_name]["right_total"] += 1
    elif p2_hand == "L":
        df_player[p1_name]["left_total"] += 1

    if p1_hand == "R":
        df_player[p2_name]["right_total"] += 1
    elif p1_hand == "L":
        df_player[p2_name]["left_total"] += 1

    if winner:
        df_player[p1_name]["win"] += 1
        df_surface.loc[(surface, "win"), p1_name] += 1
        df_versus[p1_name][p2_name]["won"] += 1
        df_versus[p2_name][p1_name]["lost"] += 1

        if p2_hand == "R":
            df_player[p1_name]["right_win"] += 1
        elif p2_hand == "L":
            df_player[p1_name]["left_win"] += 1

        if df_player[p1_name]["last5"] < 5:
            df_player[p1_name]["last5"] += 1
        if df_player[p2_name]["last5"] > 0:
            df_player[p2_name]["last5"] -= 1

    else:
        df_player[p2_name]["win"] += 1
        df_surface.loc[(surface, "win"), p2_name] += 1
        df_versus[p2_name][p1_name]["won"] += 1
        df_versus[p1_name][p2_name]["lost"] += 1

        if p1_hand == "R":
            df_player[p2_name]["right_win"] += 1
        elif p1_hand == "L":
            df_player[p2_name]["left_win"] += 1

        if df_player[p1_name]["last5"] > 0:
            df_player[p1_name]["last5"] -= 1
        if df_player[p2_name]["last5"] < 5:
            df_player[p2_name]["last5"] += 1


def calculations(data):
    start_time = time.time()

    df_player, df_surface, df_versus = frames_initialization(data)
    for index in data.index:
        fill_params(data, index, df_player, df_surface, df_versus)
        calc_params(data, index, df_player, df_surface, df_versus)

    print(f"{calculations.__name__.capitalize()} ✓ in {round(time.time() - start_time, 2)}s")

    return data


def resulting(data):
    start_time = time.time()

    result = data[['p1', 'p2', 'winner', 'minutes']]

    result["diff_height"] = data["p1_height"] - data["p2_height"]
    result["diff_break_points"] = data["p1_break"] - data["p2_break"]
    result["diff_serve_points"] = data["p1_serve"] - data["p2_serve"]
    result["diff_elo"] = data["p1_elo"] - data["p2_elo"]
    result["diff_hand"] = data["p1_hand_form"] - data["p2_hand_form"]
    result["diff_win_h2h"] = data["p1_win_h2h"] - data["p2_win_h2h"]
    result["diff_win_last5"] = data["p1_win_last5"] - data["p2_win_last5"]
    result["diff_win_overall"] = data["p1_win_overall"] - data["p2_win_overall"]
    result["diff_surface_overall"] = data["p1_surface_overall"] - data["p2_surface_overall"]

    result.to_csv('data/result.csv', index=False)

    print(f"{resulting.__name__.capitalize()} ✓ in {round(time.time() - start_time, 2)}s")

    return result


def preprocessing(filename='all_matches.csv'):
    data = pd.read_csv("data/" + filename)
    filtered_data = filtering(data)

    if not os.path.isfile("data/mixed.csv"):
        mixed_data = mixing_players(filtered_data)
    else:
        mixed_data = pd.read_csv('data/mixed.csv')

    featured_frame = featuring(mixed_data)
    calculated_frame = calculations(featured_frame)
    resulting(calculated_frame)


if __name__ == "__main__":
    for i in ["data", "models"]:
        if not os.path.exists(i):
            os.mkdir(i)

    if not os.path.isfile("data/" + "all_matches.csv"):
        fetching()

    preprocessing()
