import matplotlib.pyplot as plt
import numpy as np

def comparar_metricas():
    """
    Generates a bar chart comparing the metrics of three models: Boolean, Selected, and LSI.

    This function does not take any arguments. It uses predefined dictionaries for the metrics of each model.
    The metrics are: Precision, Recall, F, F1, R-Prec., and Fallout.

    The function creates a bar chart with three sets of bars, one for each model, and displays it using matplotlib.
    The bars are colored differently for each model to distinguish them visually.

    Note: This function does not return any value. It directly displays the chart using plt.show().
    """
    modelo_booleano_metricas = {
        'Precision': 0.1617,
        'Recall': 0.3567,
        'Fb': 0.1461,
        'F1': 0.1117,
        'R-Prec.': 0.1648,
        'Fallout': 0.1115
    }

    modelo_escogido_metricas = {
        'Precision': 0.1569,
        'Recall': 0.3644,
        'Fb': 0.1455,
        'F1': 0.1106,
        'R-Prec.': 0.1588,
        'Fallout': 0.1137
    }

    modelo_lsi_metricas = {
        'Precision': 0.1840,
        'Recall': 0.1493,
        'Fb': 0.1192,
        'F1': 0.1070,
        'R-Prec.': 0.1834,
        'Fallout': 0.0004
    }

    fig, ax = plt.subplots()

    metrics = ['Precision', 'Recall', 'Fb', 'F1', 'R-Prec.', 'Fallout']
    modelo_1_values = [modelo_booleano_metricas[metric] for metric in metrics]
    modelo_2_values = [modelo_escogido_metricas[metric] for metric in metrics]
    modelo_3_values = [modelo_lsi_metricas[metric] for metric in metrics]

    n = len(metrics)
    width = 0.25
    r = np.arange(n)

    modelo_1_bars = ax.bar(r, modelo_1_values, color='lightblue', width=width, edgecolor='black', label='Modelo booleano')
    modelo_2_bars = ax.bar(r + width, modelo_2_values, color='lightgreen', width=width, edgecolor='black', label='Modelo extendido')
    modelo_3_bars = ax.bar(r + 2*width, modelo_3_values, color='red', width=width, edgecolor='black', label='Modelo lsi')

    ax.legend()
    ax.set_title('Comparación de Métricas entre Modelos')

    ax.set_xticks(r + width / 2)
    ax.set_xticklabels(metrics)

    ax.set_ylabel('Valores')
    ax.set_xlabel('Métricas')

    for bar in modelo_1_bars:
        bar.set_color('lightblue')

    for bar in modelo_2_bars:
        bar.set_color('lightgreen')

    for bar in modelo_3_bars:
        bar.set_color('red')

    plt.show()

comparar_metricas()
