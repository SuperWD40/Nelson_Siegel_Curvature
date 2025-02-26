import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import tempfile
import matplotlib.pyplot as plt

class NelsonSiegelFitter:
    def __init__(self, maturities, bounds=[(0, 1), (-1, 1), (-1, 1), (0, 2)], x0=[0.01, 0.01, 0.01, 0.5], method='Nelder-Mead'):
        """
        Initialise un estimateur de Nelson-Siegel.

        Paramètres :
        - maturities (np.array) : Maturités annualisées correspondant aux colonnes des rendements.
        - bounds (list) : Contraintes pour les paramètres [(min, max), ...].
        - x0 (list) : Valeurs initiales pour [Intercept, Slope, Curvature, Lambda].
        - method (str) : Méthode d'optimisation. Par défaut 'Nelder-Mead'.
        """
        self.maturities = maturities
        self.bounds = bounds
        self.x0 = x0
        self.method = method
        self.ns_params_df = None
        self.ns_curve_df = None

    def _ns_function(self, P, T):
        """Calcule la courbe Nelson-Siegel pour un jeu de paramètres P et des maturités T."""
        return (P[0] +
                P[1] * ((1 - np.exp(-T / P[3])) / (T / P[3])) +
                P[2] * (((1 - np.exp(-T / P[3])) / (T / P[3])) - np.exp(-T / P[3])))

    def _objective_function(self, P, T, Y):
        """Fonction objectif pour minimiser l'erreur entre la courbe ajustée et les rendements observés."""
        return np.sum((Y - self._ns_function(P, T)) ** 2)

    def fit_parameters(self, yields):
        """
        Ajuste la courbe de Nelson-Siegel sur une série temporelle de rendements obligataires.

        Paramètre :
        - yields (pd.DataFrame) : DataFrame des rendements obligataires avec dates en index et maturités en colonnes.

        Stocke en mémoire :
        - ns_params_df (pd.DataFrame) : Paramètres ajustés Nelson-Siegel et résidus.
        """
        self.ns_params_df = pd.DataFrame(columns=['Intercept', 'Slope', 'Curvature', 'Lambda', 'Residuals'])

        for date, yield_curve in tqdm(yields.iterrows(), desc="Fitting Nelson-Siegel"):
            res = minimize(self._objective_function, 
                           x0=self.x0, 
                           args=(self.maturities.round(4), np.array(yield_curve) / 100),  
                           method=self.method, 
                           bounds=self.bounds)

            self.ns_params_df.loc[date] = np.append(res.x, res.fun)
        return self.ns_params_df

    def generate_curve(self, curve_maturities):
        """
        Génère les rendements ajustés sur une grille de maturités spécifiée.

        Paramètre :
        - curve_maturities (np.array) : Maturités pour lesquelles calculer la courbe Nelson-Siegel.

        Stocke en mémoire :
        - ns_curve_df (pd.DataFrame) : Rendements ajustés sur les maturités définies.
        """
        if self.ns_params_df is None:
            raise ValueError("Les paramètres Nelson-Siegel doivent être ajustés avant de générer la courbe.")

        self.ns_curve_df = pd.DataFrame(index=self.ns_params_df.index, columns=curve_maturities)

        for date, params in self.ns_params_df.iterrows():
            self.ns_curve_df.loc[date] = self._ns_function(params[:4], curve_maturities)
        return self.ns_curve_df

def plot_ns_histo(ns_data, show=True):
    """
    Plot the historical evolution of Nelson-Siegel parameters with a 5-week moving average.

    Parameters:
    - ns_data: DataFrame containing weekly Nelson-Siegel parameters ('Intercept', 'Slope', 'Curvature', 'Lambda')
    - show: Boolean flag to display the plot or save it to a file

    Returns:
    - None if show=True (displays the plot)
    - Temporary file containing the saved image if show=False
    """

    # Define colors for different Nelson-Siegel parameters
    colors = {'Intercept': 'blue', 'Slope': 'red', 'Curvature': 'green'}

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the 5-week rolling average of Intercept, Slope, and Curvature
    for col in ['Intercept', 'Slope', 'Curvature']:
        ax1.plot(ns_data.index,
                 ns_data[col].rolling(5).mean(), 
                 label=col, 
                 color=colors[col],
                 linewidth=1)

    # Configure primary y-axis (β parameters)
    ax1.set_ylabel('β', color='black', rotation=0, fontsize=12, labelpad=10)
    ax1.tick_params(axis='y', colors='black')

    # Create a secondary y-axis for Lambda
    ax2 = ax1.twinx()
    ax2.plot(ns_data.index,
             ns_data['Lambda'].rolling(5).mean(),
             label='Lambda',
             color='green',
             linestyle=':',
             linewidth=1.25)

    # Configure secondary y-axis (λ parameter)
    ax2.set_ylabel('λ', color='green', rotation=0, fontsize=12, labelpad=10)
    ax2.tick_params(axis='y', colors='green')

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set the plot title
    plt.title("Nelson-Siegel Parameters (5-week Moving Average)")

    # Show or save the plot
    if show:
        plt.show()
    else:
        # Save the plot to a temporary file
        ns_histo_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(ns_histo_file.name, format='png', bbox_inches='tight')
        plt.close()
        return ns_histo_file

def plot_ns_comps(yields, mat_yields, ns, mat_ns, ns_param, show=True):
    """
    Plot observed bond yields and the Nelson-Siegel model curve.

    Parameters:
    - yields: array of observed bond yields.
    - mat_yields: array of maturities corresponding to observed yields.
    - ns: array of Nelson-Siegel estimated yields.
    - mat_ns: array of maturities corresponding to the Nelson-Siegel curve.
    - ns_param: list of estimated Nelson-Siegel parameters [Intercept, Slope, Curvature, Lambda, Residuals].
    - show: boolean, if True displays the plot, otherwise saves it as an image.

    Returns:
    - None if show=True.
    - Temporary file object containing the saved plot if show=False.
    """

    # Create the main figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12,6))

    # Scatter plot of observed bond yields
    ax1.scatter(x=mat_yields, y=np.array(yields)/100, color='blue', marker='+', label="Observed Yields")
    ax1.set_ylabel('Yields', color='blue')
    ax1.tick_params(axis='y', colors='blue')

    # Create a secondary y-axis for the Nelson-Siegel curve
    ax2 = ax1.twinx()
    ax2.scatter(x=mat_ns, y=ns, color='red', marker='.', label="Nelson-Siegel")
    ax2.set_ylabel('Nelson-Siegel', color='red')
    ax2.tick_params(axis='y', colors='red')

    # Format Nelson-Siegel parameters as text for legend
    params_text = (f"Intercept: {ns_param[0]:.4f}\n"
                   f"Slope: {ns_param[1]:.4f}\n"
                   f"Curvature: {ns_param[2]:.4f}\n"
                   f"Lambda: {ns_param[3]:.4f}\n"
                   f"Residuals: {ns_param[4]:.4f}")
    
    # Create an invisible dummy legend entry for displaying parameters
    dummy_line = plt.Line2D([], [], color="white", label=params_text)

    # Collect legend handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2 + [dummy_line]
    labels = labels1 + labels2 + [params_text]

    # Set legend outside the plot
    ax1.legend(handles, labels, bbox_to_anchor=(1.07, 1), loc='upper left')

    # Set plot title with the last available date in the dataset
    plt.title(f"TBond Yield Curve as of {ns.index[-1].strftime('%Y-%m-%d')}")
    plt.grid(alpha=0.5, linestyle='--')
    
    # Show or save the plot
    if show:
        plt.show()
    else:
        ns_comps_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(ns_comps_file.name, format='png', bbox_inches='tight')
        plt.close()
        return ns_comps_file
