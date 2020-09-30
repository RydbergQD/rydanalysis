import streamlit as st
import plotly.graph_objects as go

from rydanalysis.data_structure.ryd_data import *


def plot_variables(variables, tmstp):
    # variables['index_2'] = np.random.randint(0, 2, size=len(variables))
    if variables.shape[1] == 1:
        var = variables[variables.index == tmstp]
        variables_scatter_plot(variables.iloc[:, 0], var)
    elif variables.shape[1] == 2:
        var = variables[variables.index == tmstp]
        var = var.set_index(variables.columns[0])
        variables = variables.set_index(variables.columns[0]).squeeze()
        variables_scatter_plot(variables, var, xlabel=variables.index.name)
    else:
        plot_variables_parallel_categories(variables, tmstp)


def variables_scatter_plot(variables, var, xlabel='timestamp'):
    fig = go.Figure(
        data=[
            go.Scatter(
                x=variables.index,
                y=variables,
                mode='markers',
                marker=dict(size=10, color='grey'),
                name='Other data'
            ),
            go.Scatter(
                x=var.index,
                y=var.iloc[:, 0],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='This shot'
            )
        ],
        layout=go.Layout(width=300, height=300,
                         xaxis=dict(title_text=xlabel),
                         yaxis_title_text=variables.name,
                         title='Overview of variables'
                         ),
    )
    st.sidebar.plotly_chart(fig, )


def plot_variables_parallel_categories(variables: pd.DataFrame, tmstp):
    chosen = get_same_variables(variables, tmstp)
    color = np.zeros(len(variables), dtype='uint8')
    color[chosen] = 1
    colorscale = [[0, 'gray'], [1, 'red']]
    dimensions = [dict(values=variables[label], label=label) for label in variables.columns]
    fig = go.Figure(
        data=go.Parcats(
            dimensions=dimensions, line={'colorscale': colorscale, 'cmin': 0,
                                         'cmax': 1, 'color': color, 'shape': 'hspline'}
        ),
        layout=go.Layout(width=300, height=300)
    )
    st.sidebar.plotly_chart(fig, )


def get_same_variables(variables: pd.DataFrame, tmstp) -> pd.DataFrame:
    """
    Returns all variables with values that are equal as tmstp.
    Args:
        variables: Dataframe with variables and tmstp as index
        tmstp: Timestamp where variables are compared to.

    Returns:
        variables with values that are equal as tmstp
    """
    var = variables[variables.index == tmstp].iloc[0]
    return (variables == var).all(axis=1).values
