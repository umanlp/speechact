import argparse
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import logging
import numpy as np

def get_color_coded_years(party_df):

    unique_years = []
    colors = []
    cusp_coords = []
    #for y in party_df["year"].unique():
    for y in range(min_year, 2024):

        in_gov_set = set(party_df["in_government"][party_df["year"] == y].tolist())
        #unique_years.append(y)

        if (y == 2013 or y == 2017) and (party_df["party_name"].unique()[0] == "fdp"):
            colors.append("k")  # cusp years
            cusp_coords.append(y)
        else:
            if len(in_gov_set) == 0:
                colors.append("grey") # party not in parliament
            elif len(in_gov_set) > 1:
                colors.append("k")  # cusp years
                cusp_coords.append(y)
            else:  # == 1
                in_gov_bool = in_gov_set.pop()
                if in_gov_bool:
                    colors.append("b")
                else:
                    colors.append("r")

    #return unique_years, colors, cusp_coords
    return colors, cusp_coords


def annotate_min_max(conflict_year, ax):
    ymin_index = conflict_year["proportion"].idxmin()
    ymin_value = conflict_year.loc[ymin_index]["proportion"]
    xmin_value = int(conflict_year.loc[ymin_index]["year"])
    ax.annotate('Min: {}'.format(round(ymin_value, 3)), xy=(xmin_value, ymin_value),
                xytext=(xmin_value, ymin_value - 0.05),
                arrowprops=dict(width=0.3, headwidth=0.4, color='k', shrink=0.05),
                fontsize=ftsize)

    ymax_index = conflict_year["proportion"].idxmax()
    ymax_value = conflict_year.loc[ymax_index]["proportion"]
    xmax_value = int(conflict_year.loc[ymax_index]["year"])
    ax.annotate('Max: {}'.format(round(ymax_value, 3)), xy=(xmax_value, ymax_value),
                xytext=(xmax_value, ymax_value + 0.05),
                arrowprops=dict(width=0.3, headwidth=0.4, color='k', shrink=0.05),
                fontsize=ftsize)

    return


def fill_missing_years(df):

    for y in range(min_year, 2024):
        if len(df[df["year"] == y]) == 0:
            empty_df = pd.DataFrame([[y, np.nan]], columns=df.columns)
            df = pd.concat([empty_df, df], ignore_index=True)
            # fill in missing years with None so they don't get plotted

    return df

if __name__ == '__main__':

    rcParams.update({'figure.autolayout': True})
    ftsize = 15

    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_party_pickle')

    args = parser.parse_args()
    party_pickle = args.path_to_party_pickle

    parent_folder, party_name = "/".join(party_pickle.split("/")[:-1]), party_pickle.split("/")[-1].split(".")[0]
    party_pickle_folder = parent_folder + "/" + party_name + "_results/"

    if not os.path.exists(party_pickle_folder):
        os.makedirs(party_pickle_folder)

    logging.basicConfig(filename=party_pickle_folder + party_name + ".log",
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    party_df_with_empties_all_legis = pd.read_pickle(party_pickle)
    party_df_with_empties_all_legis.reset_index(inplace=True)
    party_df_with_empties_all_legis["year"] = party_df_with_empties_all_legis["date"].apply(lambda x: x.year)
    #allowed_periods = [15, 16, 17, 18, 19, 20]
    #party_df_with_empties = party_df_with_empties_all_legis[
        #party_df_with_empties_all_legis["legislaturperiode"].isin(allowed_periods)]
    min_year = 2003
    party_df_with_empties = party_df_with_empties_all_legis[party_df_with_empties_all_legis["year"] >= min_year]
    logging.info("CAUTION: Dropped all utterances before the year 2002.")

    logging.info("Initial dataframe length: {} rows".format(len(party_df_with_empties)))

    # dropping things:
    party_df_full = party_df_with_empties[party_df_with_empties["predicted_labels"].apply(lambda x: len(x) > 0)]
    logging.info("Dropped {} rows without any predicted speech act labels.".format((len(party_df_with_empties)-len(party_df_full))))

    drop_labels = ["Evaluation", "Macro", "Question-All", "Expressive"]
    drop_combinations = list(combinations(drop_labels, 2)) # this only generates labels of length 2
    drop_ordered_lists = [list(el) for el in drop_combinations]
    drop_ordered_lists_rev = [sub[::-1] for sub in drop_ordered_lists] # this reversing only works for labels of length 2
    drop_labels_as_lists = [[x] for x in drop_labels]
    to_drop = drop_labels_as_lists + drop_ordered_lists + drop_ordered_lists_rev
    party_df = party_df_full[~party_df_full["predicted_labels"].isin(to_drop)]
    logging.info("Dropped {} rows that correspond to the following speech acts or combinations thereof: {}"
          .format((len(party_df_full)-len(party_df)), drop_labels))

    logging.info(party_df["party_name"].value_counts().to_string())
    logging.info(party_df["in_government"].value_counts().to_string())

    if len(party_df["in_government"].value_counts()) > 1:
        in_gov_party_df = party_df[party_df["in_government"] == True]
        not_in_gov_party_df = party_df[party_df["in_government"] != True]
        logging.info("\n------- Party in government: -------")
        coarse_raw = in_gov_party_df["coarse_category"].value_counts()
        coarse_normalized = in_gov_party_df["coarse_category"].value_counts(normalize=True)
        coarse = pd.concat([coarse_raw, coarse_normalized], axis=1)
        logging.info(coarse.to_string())
        kondratenko_raw = in_gov_party_df["kondratenko_labels"].apply(tuple).value_counts()
        kondratenko_normalized = in_gov_party_df["kondratenko_labels"].apply(tuple).value_counts(normalize=True)
        kondratenko = pd.concat([kondratenko_raw, kondratenko_normalized], axis=1)
        logging.info(kondratenko.to_string())
        labels_raw = in_gov_party_df["predicted_labels"].apply(tuple).value_counts()
        labels_normalized = in_gov_party_df["predicted_labels"].apply(tuple).value_counts(normalize=True)
        labels = pd.concat([labels_raw, labels_normalized], axis=1)
        logging.info(labels.to_string())
        logging.info("\n------- Party not in government: -------")
        coarse_raw = not_in_gov_party_df["coarse_category"].value_counts()
        coarse_normalized = not_in_gov_party_df["coarse_category"].value_counts(normalize=True)
        coarse = pd.concat([coarse_raw, coarse_normalized], axis=1)
        logging.info(coarse.to_string())
        kondratenko_raw = not_in_gov_party_df["kondratenko_labels"].apply(tuple).value_counts()
        kondratenko_normalized = not_in_gov_party_df["kondratenko_labels"].apply(tuple).value_counts(normalize=True)
        kondratenko = pd.concat([kondratenko_raw, kondratenko_normalized], axis=1)
        logging.info(kondratenko.to_string())
        labels_raw = not_in_gov_party_df["predicted_labels"].apply(tuple).value_counts()
        labels_normalized = not_in_gov_party_df["predicted_labels"].apply(tuple).value_counts(normalize=True)
        labels = pd.concat([labels_raw, labels_normalized], axis=1)
        logging.info(labels.to_string())

    else:
        coarse_raw = party_df["coarse_category"].value_counts()
        coarse_normalized = party_df["coarse_category"].value_counts(normalize=True)
        coarse = pd.concat([coarse_raw, coarse_normalized], axis=1)
        logging.info(coarse.to_string())
        kondratenko_raw = party_df["kondratenko_labels"].apply(tuple).value_counts()
        kondratenko_normalized = party_df["kondratenko_labels"].apply(tuple).value_counts(normalize=True)
        kondratenko = pd.concat([kondratenko_raw, kondratenko_normalized], axis=1)
        logging.info(kondratenko.to_string())
        labels_raw = party_df["predicted_labels"].apply(tuple).value_counts()
        labels_normalized = party_df["predicted_labels"].apply(tuple).value_counts(normalize=True)
        labels = pd.concat([labels_raw, labels_normalized], axis=1)
        logging.info(labels.to_string())

    logging.info("Data by year:")
    logging.info("Coarse by year:")
    coarse_year_raw = party_df.groupby(["year"])["coarse_category"].value_counts()
    coarse_year_normalized = party_df.groupby(["year"])["coarse_category"].value_counts(normalize=True)
    coarse_year = pd.concat([coarse_year_raw, coarse_year_normalized], axis=1)
    coarse_year.reset_index(inplace=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logging.info(coarse_year.to_string())
    conflict_year = coarse_year[["year", "proportion"]][coarse_year["coarse_category"] == "conflict"]
    conflict_year = fill_missing_years(conflict_year)

    #x = conflict_year["year"].tolist()
    #y = conflict_year["proportion"].tolist()
    #ax = plt.plot(x, y, label='Conflict', color="#661100")
    conflict_year.sort_values("year", inplace=True)
    ax = conflict_year.plot(x='year', y='proportion', label='Conflict', color="#AE3636")
    plt.ylim(0, 0.6)
    plt.yticks(fontsize=ftsize)
    plt.xlabel("")
    plt.ylabel("Proportion", fontsize=ftsize)
    plt.legend(fontsize=ftsize, loc='upper left')

    # color coded years on x-axis
    colors, cusp_coords = get_color_coded_years(party_df)
    #plt.xticks(unique_years, rotation=90)
    plt.xticks(range(min_year, 2024), rotation=90, fontsize=ftsize)
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)

    # draw vertical line at cusp years
    for xc in cusp_coords:
        plt.axvline(x=xc, linestyle="dotted", color="k", linewidth=0.5)

    # annotate ymin and ymax
    annotate_min_max(conflict_year, ax)

    plt.savefig(party_pickle_folder + party_name + "_conflict.png")
    plt.show()

    mask = party_df.predicted_labels.apply(lambda x: len(x) == 1)
    single_labels = party_df[mask]
    logging.info("Comparing Demand and Request on a subset of the data. Subset size: {} utterances.".format(len(single_labels)))
    single_labels["single_label"] = single_labels.predicted_labels.apply(lambda x: x[0])
    labels_raw_year = single_labels.groupby(["year"])["single_label"].value_counts()
    labels_normalized_year = single_labels.groupby(["year"])["single_label"].value_counts(normalize=True)
    labels_year = pd.concat([labels_raw_year, labels_normalized_year], axis=1)
    labels_year.reset_index(inplace=True)
    request_year = labels_year[["year", "proportion"]][labels_year["single_label"] == "Request"]
    request_year = fill_missing_years(request_year)
    request_year.sort_values("year", inplace=True)
    demand_year = labels_year[["year", "proportion"]][labels_year["single_label"] == "Demand"]
    demand_year = fill_missing_years(demand_year)
    demand_year.sort_values("year", inplace=True)
    ax = request_year.plot(x='year', y='proportion', label="Request", color="#44AA99")
    demand_year.plot(ax=ax, x='year', y='proportion', label="Demand", color="#AA4499", linestyle='dashed')
    plt.ylim(0, 0.3)
    plt.yticks(fontsize=ftsize)
    plt.xlabel("")
    plt.ylabel("Proportion", fontsize=ftsize)
    plt.legend(fontsize=ftsize, loc=0)

    # color coded years on x-axis
    plt.xticks(range(min_year, 2024), rotation=90, fontsize=ftsize)
    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)

    # draw vertical line at cusp years
    for xc in cusp_coords:
        plt.axvline(x=xc, linestyle="dotted", color="k", linewidth=0.5)

    plt.savefig(party_pickle_folder + party_name + "_request_demand.png")
    plt.show()

    print("")