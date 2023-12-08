#!/usr/bin/env python3
'''
T. A-N authored this Python script
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot
from sklearn.preprocessing import QuantileTransformer


class NormalityChecker:
    def __init__(self, df):
        self.df = df

    def _plot_distribution(self, ax, data, title):        

        sns.distplot(data,
                     fit=norm,
                     hist_kws=dict(edgecolor="black",
                                   linewidth=2,
                                   color='blue'),
                     kde_kws={'linestyle': '--',
                              'linewidth': 2,
                              "color": "darkgreen",
                              "label": "KDE"},
                     ax=ax)
        ax.grid(True, linestyle='--')

        (mu, sigma) = norm.fit(data)
        ax.legend(['Normal dist. ($\mu=$ {:.2f}; $\sigma=$ {:.2f})'.format(mu, sigma)],
                  loc='best')
        ax.set_title(title)

    def _qq_plot(self, ax, data):        

        probplot(data, plot=ax, rvalue=True)
        ax.grid(True, linestyle='--')

    def check_normality(self, cname):
        fig, ((ax1, ax2)) = plt.subplots(1, 2,
                                        figsize=(12, 6),
                                        dpi=80)

        self._plot_distribution(ax1, self.df[cname], '{} distribution'.format(cname))
        self._qq_plot(ax2, self.df[cname])        

        fig.tight_layout()


class LogTransformChecker:
    def __init__(self, df):
        self.df = df

    def _plot_distribution(self, ax, data, title, log_transform=False):
        if log_transform:
            data = np.log1p(data)

        sns.distplot(data,
                     fit=norm,
                     hist_kws=dict(edgecolor="black",
                                   linewidth=2,
                                   color='blue'),
                     kde_kws={'linestyle': '--',
                              'linewidth': 2,
                              "color": "darkgreen",
                              "label": "KDE"},
                     ax=ax)
        ax.grid(True, linestyle='--')

        (mu, sigma) = norm.fit(data)
        ax.legend(['Normal dist. ($\mu=$ {:.2f}; $\sigma=$ {:.2f})'.format(mu, sigma)],
                  loc='best')
        ax.set_title(title)

    def _qq_plot(self, ax, data, log_transform=False):
        if log_transform:
            data = np.log1p(data)

        probplot(data, plot=ax, rvalue=True)
        ax.grid(True, linestyle='--')

    def check_normality(self, cname, return_log_transform=True):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                    figsize=(12, 12),
                                                    dpi=80)

        self._plot_distribution(ax1, self.df[cname], '{} distribution'.format(cname))
        self._qq_plot(ax2, self.df[cname])

        if return_log_transform:
            self.df[cname] = np.log1p(self.df[cname])
            self._plot_distribution(ax3, self.df[cname], '{} distribution in log scale'.format(cname), log_transform=True)
            self._qq_plot(ax4, self.df[cname], log_transform=True)

        fig.tight_layout()

        if return_log_transform:
            return self.df[cname]

# Example usage:
# normality_checker = LogTransformChecker(your_dataframe)
# transformed_column = normality_checker.check_normality('your_column_name')



class QuantileTransformChecker:
    def __init__(self, df):
        self.df = df

    def _plot_distribution(self, ax, data, title):
        sns.distplot(data,
                     fit=norm,
                     hist_kws=dict(edgecolor="black",
                                   linewidth=2,
                                   color='blue'),
                     kde_kws={'linestyle': '--',
                              'linewidth': 2,
                              "color": "darkgreen",
                              "label": "KDE"},
                     ax=ax)
        ax.grid(True, linestyle='--')

        (mu, sigma) = norm.fit(data)
        ax.legend(['Normal dist. ($\mu=$ {:.2f}; $\sigma=$ {:.2f})'.format(mu, sigma)],
                  loc='best')
        ax.set_title(title)

    def _qq_plot(self, ax, data):
        probplot(data, plot=ax, rvalue=True)
        ax.grid(True, linestyle='--')

    def _quantile_transform(self, data):
        quantile = QuantileTransformer(output_distribution='normal')
        dataQt = quantile.fit_transform(data)
        return pd.DataFrame(dataQt, columns=data.columns)

    def check_normality(self, cname, return_qt_transform=True):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                    figsize=(12, 12),
                                                    dpi=80)

        # 1st Plot
        self._plot_distribution(ax1, self.df[cname], '{} distribution'.format(cname))

        # 2nd Plot
        self._qq_plot(ax2, self.df[cname])

        # Quantile transformation
        self.df = self._quantile_transform(self.df)

        # 3rd Plot
        self._plot_distribution(ax3, self.df[cname], '{} distribution after quantile transformation'.format(cname))

        # 4th Plot
        self._qq_plot(ax4, self.df[cname])

        fig.tight_layout()

        if return_qt_transform:
            return self.df[cname]

# Example usage:
# normality_checker = QuantileTransformChecker(your_dataframe)
# transformed_column = normality_checker.check_normality('your_column_name')
