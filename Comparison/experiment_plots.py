from PREDICT import PREDICT
from PREDICT.Metrics import Accuracy, CalibrationSlope, CoxSnellR2, CITL, AUROC, AUPRC, F1Score, Precision, Recall, Sensitivity, Specificity, OE
from sklearn.linear_model import LogisticRegression

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import itertools
import os
from sklearn.calibration import calibration_curve
import matplotlib.cm as cm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import binom
import matplotlib.patches as mpatches


def plot_patients_per_month(df, model_type:str, gender:str='', resultsloc:str='./'):
    """
    Plots the number of people per month.
    Args:
        df (pd.DataFrame) : DataFrame of patient data.
        model_type (str) : String of model name e.g. 'qrisk'.
        gender (str) : If using the qrisk model pick between the male and female model e.g. "female". Defaults to ''.

    """
    df['date'] = pd.to_datetime(df['date'])

    # Create a 'month' column for grouping
    df['month'] = df['date'].dt.to_period('M')

    # Count number of patients per month
    monthly_patient_count = df.groupby('month').size()

    # Convert PeriodIndex to datetime for plotting
    monthly_patient_count.index = monthly_patient_count.index.to_timestamp()

    plt.figure(figsize=(10, 6))
    monthly_patient_count.plot(kind='bar')
    plt.title("Number of Patients per Month")
    plt.xlabel("Month")
    plt.ylabel("Patient Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(resultsloc, f"{model_type}_{gender}_num_patients_per_month.png"), dpi=600, bbox_inches='tight')
    plt.show()
        

def plot_count_of_patients_over_threshold_risk(threshold=0.1, model_type='qrisk2', gender='', fileloc='./'):
    """
    Plot the number of people per month who have over x% risk of the outcome.

    Args:
        threshold (float) : Risk threshold value. Defaults to 0.1.
        model_type (str) : String of model name e.g. 'qrisk'. Defaults to 'qrisk2'.
        gender (str) : If using the qrisk model pick between the male and female model e.g. "female". Defaults to ''.

    """
    def count_over_threshold(df, date_col, threshold=0.1):
        df[date_col] = pd.to_datetime(df[date_col])
        # filter rows
        filtered = df[df['prediction'] > threshold] # new_prediction
        # group by month
        monthly = filtered.groupby(pd.Grouper(key=date_col, freq='M')).size()
        return monthly
    
    plt.figure(figsize=(10,6))

    if gender != '':
        gender = gender+'_'

    if os.path.exists(os.path.join(fileloc, f"probs_and_outcomes/Baseline_{model_type}_{gender}predictions_and_outcomes.csv")):
        baseline = pd.read_csv(os.path.join(fileloc, f"probs_and_outcomes/Baseline_{model_type}_{gender}predictions_and_outcomes.csv"))
        counts1 = count_over_threshold(baseline, 'date', threshold=threshold)
        counts1.plot(label='Baseline')
    else:
        print(f"File 'probs_and_outcomes/Baseline_{model_type}_{gender}predictions_and_outcomes.csv' for baseline predictions and outcomes does not exist.")

    if os.path.exists(os.path.join(fileloc, f"probs_and_outcomes/Regular Testing_{model_type}_{gender}predictions_and_outcomes.csv")):
        regular = pd.read_csv(os.path.join(fileloc, f"probs_and_outcomes/Regular Testing_{model_type}_{gender}predictions_and_outcomes.csv"))
        counts2 = count_over_threshold(regular, 'date', threshold=threshold)
        counts2.plot(label='Regular Testing', alpha=0.6, linestyle='--')
    else:
        print("File for regular testing predictions and outcomes does not exist.")

    if os.path.exists(os.path.join(fileloc, f"probs_and_outcomes/Static Threshold_{model_type}_{gender}predictions_and_outcomes.csv")):
        static = pd.read_csv(os.path.join(fileloc, f"probs_and_outcomes/Static Threshold_{model_type}_{gender}predictions_and_outcomes.csv"))
        counts3 = count_over_threshold(static, 'date', threshold=threshold)
        counts3.plot(label='Static Threshold', alpha=0.6, linestyle=':')
    else:
        print("File for static threshold predictions and outcomes does not exist.")

    if os.path.exists(os.path.join(fileloc, f"probs_and_outcomes/SPC_{model_type}_{gender}predictions_and_outcomes.csv")):
        spc = pd.read_csv(os.path.join(fileloc, f"probs_and_outcomes/SPC_{model_type}_{gender}predictions_and_outcomes.csv"))
        counts4 = count_over_threshold(spc, 'date', threshold=threshold)
        counts4.plot(label='SPC', alpha=0.6, linestyle='-.')
    else:
        print("File for SPC predictions and outcomes does not exist.")

    if os.path.exists(os.path.join(fileloc, f"probs_and_outcomes/Bayesian_{model_type}_{gender}predictions_and_outcomes.csv")):
        bayesian = pd.read_csv(os.path.join(fileloc, f"probs_and_outcomes/Bayesian_{model_type}_{gender}predictions_and_outcomes.csv"))
        counts5 = count_over_threshold(bayesian, 'date', threshold=threshold)
        counts5.plot(label='Bayesian', alpha=0.6)
    else:
        print("File for bayesian predictions and outcomes does not exist.")
    if os.path.exists(os.path.join(fileloc, f"probs_and_outcomes/Baseline_{model_type}_{gender}predictions_and_outcomes.csv")):
        # plot the true number of people who have a heart attack or stroke
        baseline_pos_out = baseline[baseline['outcome'] == 1]
        monthly_true = baseline_pos_out.groupby(pd.Grouper(key='date', freq='M')).size()
        monthly_true.plot(label='True Positive Outcome', alpha=0.3)

    plt.xlabel('Date')
    plt.ylim(-50, None)
    plt.ylabel(f'Number of people with >{int(threshold*100)}% probability')
    plt.title(f'Count of People with Over {int(threshold*100)}% Risk of Outcome')
    plt.legend()
    plt.savefig(os.path.join(fileloc, f"{model_type}_{gender}count_over_{int(threshold*100)}%_risk.png"), dpi=600, bbox_inches='tight')
    plt.show()

def plot_method_comparison_metrics(metrics_df, recalthreshold, model_updates, model_type, gender='', fileloc='./'):
    """
    Plot the metric comparison graphs with each line showing a different PREDICT method.

    Args:
        metrics_df (str) : csv file name where performance metrics for each method are saved.
        recalthreshold (float) : static threshold method AUROC threshold.
        model_updates (str) : csv file name where dates of model updates with method names are stored.
        model_type (str) : name of the model used e.g. QRISK2.
        gender (str) : if using the QRISK model, define whether to use male or female. Defaults to ''.

    """
    if gender != '':
        gender = gender+'_'
    metrics_df = pd.read_csv(metrics_df)
    model_updates = pd.read_csv(model_updates)
    metrics_df["Time"] = pd.to_datetime(metrics_df["Time"])
    sns.set(font_scale=1.2)
    metric_choices = ["Accuracy", "AUROC", "Precision", "CalibrationSlope", "OE", "CITL", "AUPRC", "F1Score", "Sensitivity", "Specificity", "CoxSnellR2"]

    for metric_choice in metric_choices:

        metrics_df["Method"] = metrics_df["Method"].replace({"Static Threshold": f"Static Threshold"})

        fig, ax = plt.subplots(figsize=(14, 7))

        sns.lineplot(
            data=metrics_df,
            x="Time",
            y=metric_choice,
            hue="Method",
            ci=None,
            style="Method",
            ax=ax
        )

        
        model_updates["date"] = pd.to_datetime(model_updates["date"])

        for method in ['Regular Testing', "Static Threshold", "SPC", "Bayesian"]:
            if method in model_updates['method'].values:
                if method == "Regular Testing":
                    marker = 'o'
                    colour = 'orange'
                    markerbias = -0.02
                elif method == "Static Threshold":
                    marker = '|'
                    colour = 'green'
                    markerbias = 0.0
                elif method == "SPC":
                    marker = '^'
                    colour = 'red'
                    markerbias = 0.02
                elif method == "Bayesian":
                    marker = 'D'
                    colour = 'purple'
                    markerbias = 0.04
                subset = model_updates[model_updates["method"] == method]
                ax.scatter(
                    subset["date"],
                    [metrics_df[metric_choice].min() + markerbias]*len(subset),
                    label=f"{method} update",
                    marker=marker,
                    color=colour,
                    alpha=0.4,
                    s=40
                )

        ax.set_title(metric_choice)
        ax.set_xlabel("Date")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(fileloc, f"{model_type}_{gender}{metric_choice}.png"), dpi=600, bbox_inches='tight')
        plt.show()


def plot_calibration_yearly(model, method_list = ['Baseline', 'Regular Testing', 'Static Threshold', 'SPC', 'Bayesian'], gender='', fileloc='./'):
    """Plots the calibration slope each year as one plot with each line as a different PREDICT method.

    Args:
        model (str): Name of the model used e.g. 'qrisk'.
        method_list (list): List of the methods to plot the yearly calibration slope of.
        gender (str): If using a model separated by gender include a string e.g. 'female'. Defaults to ''.
    """
    if gender != '':
        gender = gender+'_'
    for method in method_list:
        if os.path.exists(os.path.join(fileloc, f"probs_and_outcomes/{method}_{model}_{gender}predictions_and_outcomes.csv")):
            continue
        else:
            print(f"'{os.path.join(fileloc, f'probs_and_outcomes/{method}_{model}_{gender}predictions_and_outcomes.csv')}' file does not exist... \nYearly calibration plotting cancelled.\nmethod_list should only contain methods which you have the ...predictions_and_outcomes.csv file for.")
            return
        

        
    method = 'Baseline'
    df = pd.read_csv(os.path.join(fileloc, f'probs_and_outcomes/{method}_{model}_{gender}predictions_and_outcomes.csv'))

    # convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Plot a calibration slope for each year:

    df['year'] = df['date'].dt.year

    def bin_stats(preds, y, n_bins=10):
        # decile bins (quantile); duplicates='drop' handles ties
        bins = pd.qcut(preds, q=n_bins, duplicates='drop')
        g = pd.DataFrame({'pred': preds, 'y': y, 'bin': bins}).groupby('bin')
        mean_pred = g['pred'].mean()
        obs = g['y'].mean()
        count = g['y'].count()
        se = np.sqrt(obs * (1 - obs) / count)
        ci_low = (obs - 1.96 * se).clip(0, 1)
        ci_high = (obs + 1.96 * se).clip(0, 1)
        return pd.DataFrame({
            'mean_pred': mean_pred,
            'obs': obs,
            'n': count,
            'ci_low': ci_low,
            'ci_high': ci_high
        }).dropna().reset_index(drop=True)

    def compute_cslope(preds, y):
        eps = 1e-6
        p = np.clip(preds, eps, 1 - eps)
        logit_p = np.log(p / (1 - p)).reshape(-1, 1)
        try:
            lr = LogisticRegression(solver='lbfgs', fit_intercept=True, max_iter=200)
            lr.fit(logit_p, y)
            return float(lr.coef_[0][0])
        except Exception:
            return np.nan

    # Settings
    n_bins = 10
    lowess_frac = None  # set to None to disable LOWESS

    years = sorted(df['year'].unique())
    palette = sns.color_palette('tab10', 5)

    for year in years:
        fig, ax = plt.subplots(figsize=(9, 6))
        legend_handles = []
        legend_labels = []
        plotted_any = False
        for i, method in enumerate(method_list):
            df = pd.read_csv(os.path.join(fileloc, f'probs_and_outcomes/{method}_{model}_{gender}predictions_and_outcomes.csv'))
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            year_df = df[df['year'] == year]

            if year_df.empty:
                continue

            preds = year_df['prediction'].values
            y_true = year_df['outcome'].values

            stats = bin_stats(preds, y_true, n_bins=n_bins)
            if stats.empty:
                continue

            color = palette[i]

            # Connect binned points with a line
            ax.plot(stats['mean_pred'], stats['obs'],
                    linewidth=1.5, linestyle='-', zorder=3)

            # # Draw the points
            # ax.plot(stats['mean_pred'], stats['obs'],
            #         color=color, marker=None, markersize=4, linestyle='None', zorder=4)

            # 4) Optional LOWESS smooth (continuous curve from full data)
            if lowess_frac is not None and len(preds) >= 10:
                smooth = lowess(y_true, preds, frac=lowess_frac, return_sorted=True)
                ax.plot(smooth[:, 0], smooth[:, 1], color=color, linewidth=1.0, alpha=0.8, linestyle='--', zorder=1)

            # 5) Compute calibration slope and add legend entry
            cslope = compute_cslope(preds, y_true)
            label = f"{method} (slope={cslope:.2f})" if np.isfinite(cslope) else f"{method} (slope=NA)"
            ln, = ax.plot([], [], color=color, linewidth=1.5)  # invisible handle for legend
            legend_handles.append(ln)
            legend_labels.append(label)

            plotted_any = True

            if not plotted_any:
                plt.close(fig)
                continue

        # perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Perfect')

        ax.set_title(f'Calibration in {year}')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed fraction of positives')
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 0.3)
        ax.grid(True, linestyle=':', alpha=0.6)

        ax.legend(legend_handles, legend_labels, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        plt.savefig(os.path.join(fileloc, f"probs_and_outcomes/{model}_{gender}cslope_{year}.png"), dpi=600, bbox_inches='tight')

def plot_predictor_distributions(df, predictors, plot_type, model_name, fileloc='./'):
    """Plots the distributions of the predictors, can choose from using a violin plot, a stacked bar chart
    or a percentage stacked barchart. One bar is plotted for each month.

    Args:
        df (pd.DataFrame): Dataframe where predictors are columns and rows are individual visits.
        predictors (list): List of the predictors to plot.
        plot_type (str): What type of plot to draw, either 'violin', 'stacked_bar' or 'stacked_perc'.
        model_name (str): Name of the model e.g. 'qrisk2_female', used to name the saved plots.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    if plot_type == 'violin':
        for predictor in predictors:
            print(f"Plotting distribution per year for {predictor}")  
            df['month_year_label'] = df['date'].dt.strftime('%b %Y')          # 'Jan 2020'

            # make it a categorical with chronological order
            order = sorted(df['month_year_label'].unique(), key=lambda x: pd.to_datetime(x, format='%b %Y'))
            df['month_year_label'] = pd.Categorical(df['month_year_label'], categories=order, ordered=True)

            plt.violinplot(x='month_year_label', y=f'{predictor}', data=df)#, inner='box', cut=0)
            plt.title(f'Distribution of {predictor} by month and year')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(fileloc, f"predictor_distributions/{predictor}_{model_name}_violinplot.png"), dpi=600, bbox_inches='tight')

    if plot_type == 'stacked_bar':
        for bin_pred in predictors:
            print(f"Printing stacked bar charts for {bin_pred}")
            
            # Create month-year label
            df['month_year_label'] = df['date'].dt.strftime('%b %Y')
            
            # Ensure chronological order
            order = sorted(df['month_year_label'].unique(), 
                        key=lambda x: pd.to_datetime(x, format='%b %Y'))
            df['month_year_label'] = pd.Categorical(df['month_year_label'], categories=order, ordered=True)
            
            # Aggregate counts of 0s and 1s per month
            counts = df.groupby(['month_year_label', bin_pred]).size().unstack(fill_value=0)
            
            # Plot stacked bar chart
            counts.plot(kind='bar', stacked=True, 
                        color=['darkblue', 'lightblue'], figsize=(10,6))
            
            # Add legend
            top_bar = mpatches.Patch(color='darkblue', label=f'{bin_pred} = 0')
            bottom_bar = mpatches.Patch(color='lightblue', label=f'{bin_pred} = 1')
            plt.legend(handles=[top_bar, bottom_bar])
            
            plt.title(f"Distribution of {bin_pred} over time")
            plt.xlabel("Month-Year")
            plt.ylabel("Count")
            
            # Save before showing
            plt.savefig(os.path.join(fileloc, f"predictor_distributions/{bin_pred}_{model_name}_stacked_barplot.png"), 
                        dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()

    if plot_type == 'stacked_perc':
        for bin_pred in predictors:
            print(f"Printing stacked percentage bar charts for {bin_pred}")
            
            # Create month-year label
            df['month_year_label'] = df['date'].dt.strftime('%b %Y')
            
            # Ensure chronological order
            order = sorted(df['month_year_label'].unique(), 
                        key=lambda x: pd.to_datetime(x, format='%b %Y'))
            df['month_year_label'] = pd.Categorical(df['month_year_label'], categories=order, ordered=True)
            
            # Aggregate counts of 0s and 1s per month
            counts = df.groupby(['month_year_label', bin_pred]).size().unstack(fill_value=0)
            
            # Convert to percentages
            percentages = counts.div(counts.sum(axis=1), axis=0) * 100
            
            # Plot stacked percentage bar chart
            percentages.plot(kind='bar', stacked=True, 
                            color=['darkblue', 'lightblue'], figsize=(10,6))
            
            # Add legend
            top_bar = mpatches.Patch(color='darkblue', label=f'{bin_pred} = 0')
            bottom_bar = mpatches.Patch(color='lightblue', label=f'{bin_pred} = 1')
            plt.legend(handles=[top_bar, bottom_bar])
            
            plt.title(f"Percentage distribution of {bin_pred} over time")
            plt.xlabel("Month-Year")
            plt.ylabel("Percentage")
            
            # Save before showing
            plt.savefig(os.path.join(fileloc, f"predictor_distributions/{bin_pred}_{model_name}_stacked_percentage_barplot.png"), 
                        dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()