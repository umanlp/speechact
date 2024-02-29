import argparse
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import logging
import datetime
import math

def get_label_within_legis(df, col, label):
    # label column must be a string, not a list of labels
    df["month_year"] = pd.to_datetime(df.date) + pd.offsets.MonthBegin(-1)
    coarse_date = df.groupby(["month_year"]).filter(lambda x: len(x) > 10).groupby(["month_year"])[
        col].value_counts(normalize=True)
    coarse_date_raw = \
    df.groupby(["month_year"]).filter(lambda x: len(x) > 10).groupby(["month_year"])[
        col].value_counts()
    print(coarse_date_raw)
    coarse_date_all = pd.concat([coarse_date, coarse_date_raw], axis=1)
    coarse_date_all.reset_index(inplace=True)
    conflict_date = coarse_date_all[["month_year", "proportion"]][coarse_date_all[col] == label]

    return conflict_date


def plot_conflict_within_legis(p):

    in_gov_parties = party_df[(party_df["legislaturperiode"] == p) & (party_df["in_government"] == True)]
    conflict_date_in_gov = get_label_within_legis(in_gov_parties, "coarse_category", "conflict")
    ax = conflict_date_in_gov.plot(x="month_year", y="proportion", label="Fractions in government", color="b")
    not_in_gov_parties = party_df[(party_df["legislaturperiode"] == p) & (party_df["in_government"] == False)]
    conflict_date_not_in_gov = get_label_within_legis(not_in_gov_parties, "coarse_category", "conflict")
    conflict_date_not_in_gov.plot(ax=ax, x="month_year", y="proportion", label="Fractions not in government", color="r")
    plt.ylim(0, 0.7)
    plt.savefig("analyses/predictions_for_analysis/" + "_within_legis_" + str(p) + ".png")
    #plt.show()


def plot_consol_covid_large_timeframe():
    in_gov_parties = single_labels[(single_labels["legislaturperiode"] == 19)
                                   & (single_labels["date"] >= datetime.date(year=2020, month=1, day=1)) # pre-covid
                                   & (single_labels["in_government"] == True)]
    conflict_date_in_gov = get_label_within_legis(in_gov_parties, "single_kondra_label", "consolidating")
    ax = conflict_date_in_gov.plot(x="month_year", y="proportion", label="Fractions in government", color="b")
    not_in_gov_parties = single_labels[(single_labels["legislaturperiode"] == 19)
                                   & (single_labels["date"] >= datetime.date(year=2020, month=1, day=1)) # pre-covid
                                   & (single_labels["in_government"] == False)]
    conflict_date_not_in_gov = get_label_within_legis(not_in_gov_parties, "single_kondra_label", "consolidating")
    conflict_date_not_in_gov.plot(ax=ax, x="month_year", y="proportion", label="Fractions not in government", color="r")
    plt.ylim(0, 0.6)
    plt.savefig("analyses/predictions_for_analysis/" + "consol_within_legis_" + str(19) + ".png")
    #plt.show()


def plot_consol_covid():
    covid = single_labels[(single_labels["date"] < datetime.date(year=2020, month=7, day=1))
                            & (single_labels["date"] >= datetime.date(year=2020, month=1, day=1))]
    covid["month_year"] = pd.to_datetime(covid.date) + pd.offsets.MonthBegin(-1)

    covid_pds = covid[covid["party_name"] == "pds"]
    covid_gruene = covid[covid["party_name"] == "gruene"]
    covid_afd = covid[covid["party_name"] == "afd"]
    covid_fdp = covid[covid["party_name"] == "fdp"]

    print("pds:")
    consol_covid_pds = get_label_within_legis(covid_pds, "single_kondra_label", "consolidating")
    print("-----\ngruene:")
    consol_covid_gruene = get_label_within_legis(covid_gruene, "single_kondra_label", "consolidating")
    print("-----\nafd:")
    consol_covid_afd = get_label_within_legis(covid_afd, "single_kondra_label", "consolidating")
    print("-----\nfdp:")
    consol_covid_fdp = get_label_within_legis(covid_fdp, "single_kondra_label", "consolidating")

    ax = consol_covid_pds.plot(x="month_year", y="proportion", label="PDS/Linke", color="r")
    consol_covid_gruene.plot(ax=ax, x="month_year", y="proportion", label="Gruene", color="g", linestyle="dotted")
    consol_covid_afd.plot(ax=ax, x="month_year", y="proportion", label="AfD", color="b", linestyle="dashed")
    consol_covid_fdp.plot(ax=ax, x="month_year", y="proportion", label="FDP", color="y", linestyle="dashdot")
    plt.ylim(0, 0.4)
    plt.savefig("analyses/predictions_for_analysis/" + "consol_covid_" + ".png")
    plt.show()
    print("")


if __name__ == '__main__':

    rcParams.update({'figure.autolayout': True})
    ftsize = 15

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_party_pickles') # "analyses/predictions_for_analysis/"
    args = parser.parse_args()

    logging.basicConfig(filename="analyses/predictions_for_analysis/all_parties_26_02_2024.log",
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    party_names = ["afd", "cdu", "fdp", "gruene", "pds", "spd"]

    party_dfs = []
    for name in party_names:
        path_to_pickle = args.path_to_party_pickles + name + ".pkl"
        party_df = pd.read_pickle(path_to_pickle)
        party_dfs.append(party_df)

    all_party_df_with_empties_all_legis = pd.concat(party_dfs)
    all_party_df_with_empties_all_legis["year"] = all_party_df_with_empties_all_legis["date"].apply(lambda x: x.year)
    min_year = 2003
    all_party_df_with_empties = all_party_df_with_empties_all_legis[all_party_df_with_empties_all_legis["year"] >= min_year]
    logging.info("CAUTION: Dropped all utterances before the year {}.".format(min_year))

    all_party_df_with_empties.reset_index(inplace=True)
    logging.info("Initial dataframe length: {} rows".format(len(all_party_df_with_empties)))

    # dropping things:
    party_df_full = all_party_df_with_empties[all_party_df_with_empties["predicted_labels"].apply(lambda x: len(x) > 0)]
    logging.info("Dropped {} rows without any predicted speech act labels.".format((len(all_party_df_with_empties)-len(party_df_full))))

    drop_labels = ["Evaluation", "Macro", "Question-All", "Expressive"]
    drop_combinations = list(combinations(drop_labels, 2)) # this only generates labels of length 2
    drop_ordered_lists = [list(el) for el in drop_combinations]
    drop_ordered_lists_rev = [sub[::-1] for sub in drop_ordered_lists] # this reversing only works for labels of length 2
    drop_labels_as_lists = [[x] for x in drop_labels]
    to_drop = drop_labels_as_lists + drop_ordered_lists + drop_ordered_lists_rev
    party_df = party_df_full[~party_df_full["predicted_labels"].isin(to_drop)]
    logging.info("Dropped {} rows that correspond to the following speech acts or combinations thereof: {}"
          .format((len(party_df_full)-len(party_df)), drop_labels))

    logging.info("Size of Relevant subset: {}".format(len(party_df)))
    logging.info("Number of utterances per party:\n{}".format(party_df["party_name"].value_counts(normalize=False).to_string()))

    #party_df["year"] = party_df["date"].apply(lambda x: x.year)
    legis_counts = party_df["legislaturperiode"].value_counts(normalize=False)
    legis_props = party_df["legislaturperiode"].value_counts(normalize=True)
    logging.info(pd.concat([legis_counts, legis_props], axis=1).sort_index().to_string())
    legislaturperioden = []
    legislaturperioden_unique_kondra_labels = []

    in_gov_conflict_props = []
    not_in_gov_conflict_props = []

    in_gov_consol_props = []
    not_in_gov_consol_props = []

    mask = party_df.predicted_labels.apply(lambda x: len(x) == 1)
    single_labels = party_df[mask]
    logging.info("Analyzing consolidating on a subset of the data. Subset size: {} utterances.".format(len(single_labels)))
    logging.info(
        "Number of utterances per party:\n{}".format(single_labels["party_name"].value_counts(normalize=False).to_string()))
    single_legis_counts = single_labels["legislaturperiode"].value_counts(normalize=False)
    single_legis_props = single_labels["legislaturperiode"].value_counts(normalize=True)
    logging.info(pd.concat([single_legis_counts, single_legis_props], axis=1).sort_index().to_string())
    single_labels["single_kondra_label"] = single_labels.kondratenko_labels.apply(lambda x: x[0])

    for p in party_df["legislaturperiode"].unique():
        in_gov_parties = party_df[(party_df["legislaturperiode"] == p) & (party_df["in_government"] == True)]
        not_in_gov_parties = party_df[(party_df["legislaturperiode"] == p) & (party_df["in_government"] == False)]
        legislaturperioden.append(p)

        # 3 outcomes: conflict, coop or both
        in_gov_conflict = in_gov_parties["coarse_category"].value_counts(normalize=True)["conflict"]
        not_in_gov_conflict = not_in_gov_parties["coarse_category"].value_counts(normalize=True)["conflict"]
        in_gov_conflict_props.append(in_gov_conflict)
        not_in_gov_conflict_props.append(not_in_gov_conflict)

    for p in single_labels["legislaturperiode"].unique():
        in_gov_parties = single_labels[
            (single_labels["legislaturperiode"] == p) & (single_labels["in_government"] == True)]
        not_in_gov_parties = single_labels[
            (single_labels["legislaturperiode"] == p) & (single_labels["in_government"] == False)]
        legislaturperioden_unique_kondra_labels.append(p)
        in_gov_consol = in_gov_parties["single_kondra_label"].value_counts(normalize=True)["consolidating"]
        not_in_gov_consol = not_in_gov_parties["single_kondra_label"].value_counts(normalize=True)["consolidating"]
        in_gov_consol_props.append(in_gov_consol)
        not_in_gov_consol_props.append(not_in_gov_consol)

    assert len(legislaturperioden) == len(legislaturperioden_unique_kondra_labels)

    conflict_df = pd.DataFrame({"legislaturperiode": legislaturperioden, "in_gov_conflict": in_gov_conflict_props,
                                "not_in_gov_conflict": not_in_gov_conflict_props, "in_gov_consol": in_gov_consol_props,
                                "not_in_gov_consol": not_in_gov_consol_props})
    conflict_df.sort_values(by=["legislaturperiode"], inplace=True)
    conflict_df.reset_index(inplace=True, drop=True)

    ax = conflict_df.plot(kind='bar', width=0.9, x='legislaturperiode',
                     y=['in_gov_conflict', 'not_in_gov_conflict', 'in_gov_consol', 'not_in_gov_consol'],
                     color=["#AE3636", "#AE3636", "#6DB6FF", "#6DB6FF"],
                     )

    bars = ax.patches
    patterns = ['////', '', '\\\\', '']
    hatches = []
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)

    ax.legend(['Conflict (in gov)', 'Conflict (not in gov)', 'Consolidating (in gov)', 'Consolidating (not in gov)'])

    plt.ylim(0, 0.6)
    plt.yticks(fontsize=ftsize)
    plt.xticks(fontsize=ftsize)
    plt.xlabel("Legislative period", fontsize=ftsize)
    plt.ylabel("Proportion", fontsize=ftsize)
    plt.savefig(args.path_to_party_pickles + "conflict_consol_per_legis_all_parties.png")
    plt.show()

    logging.info(
        "Conflict and consolidating speech acts by legislative period:\n{}".format(conflict_df.to_string()))

    print("")