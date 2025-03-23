# -*- coding: utf-8 -*-
import dash
import gunicorn
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import plotly.graph_objects as go
from dash import dash_table
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import io
import base64
import gunicorn

# Load data
df_real = pd.read_csv("df_real.csv", index_col=0, parse_dates=True)
df_total = pd.read_csv("df_total.csv", index_col=0, parse_dates=True)
df_2019 = pd.read_csv("df_2019.csv", index_col=0, parse_dates=True)
df_FeatureSelectionP2 = pd.read_csv("df_FeatureSelectionP2.csv",index_col=0, parse_dates=True)

# Set cut-off date
test_cutoff_date = '2019-01-01'

# Split the dataset
df_data = df_total.loc[df_total.index < test_cutoff_date].dropna()  # 2017-2018 data
df_2019_raw = df_total.loc[df_total.index >= test_cutoff_date]  # 2019 data

# Concatenate and sort
df_raw = pd.concat([df_data, df_real]).sort_index()

# Define date range and columns
start_date = df_raw.index.min()
end_date = df_raw.index.max()

# Only for 2017-2018 EDA
start_date_data = df_data.index.min()
end_date_data = df_data.index.max()

#Features List for the Feature Selection Tab
features_list = df_FeatureSelectionP2.columns.tolist()[1:]

# Define the successful features
successful_features = ['Power-1', 'Power-2', 'Temperature [C]', 'WeekDay^2', 'Sin_hour']


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

def generate_graph(df, selected_columns, start_date, end_date, graph_type="plot"):
    """Generates graph based on selected features, date range, and graph type."""
    filtered_df = df.loc[start_date:end_date, selected_columns] if selected_columns else df.loc[start_date:end_date]

    data = []
    for col in selected_columns:
        if graph_type == 'scatter':
            data.append(go.Scatter(x=filtered_df.index, y=filtered_df[col], mode='markers', name=col))
        elif graph_type == 'boxplot':
            data.append(go.Box(y=filtered_df[col], name=col))
        else:
            data.append(go.Scatter(x=filtered_df.index, y=filtered_df[col], mode='lines', name=col))

    layout = go.Layout(title={'text': 'Selected Data Over Time','x': 0.5,'xanchor': 'center', 'y': 0.85,'yanchor': 'top'}, xaxis_title='Date', yaxis_title='Value', template='plotly_white')
    return go.Figure(data=data, layout=layout)

# Load pre-trained models
models = {}
for model_name in ['RF_model', 'GB_model', 'LR_model', 'NN_model', 'DT_model']:
    with open(f'{model_name}.pkl', 'rb') as file:
        models[model_name] = pickle.load(file)

#Load dataset
Z = df_2019.values
Y_actual = Z[:, 0]  # Actual power consumption
X_features = Z[:, [1, 2, 3, 4, 5]]  # Predictor features

# App Layout
app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'padding': '20px','minHeight': '200vh' }, children=[
    html.H1('Civil Pav Energy Forecast Tool [kWh]', style={'textAlign': 'center', 'color': '#004b6b'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H2("Raw Data", style={'color': '#343a40'}),
                html.P('Select Raw Features to Display:', style={'color': '#495057'}),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': col, 'value': col} for col in df_raw.columns],
                    value=[df_raw.columns[0]], multi=True,
                    style={'backgroundColor': '#ffffff', 'color': '#000000'}
                ),
                html.P("Define Date Interval:", style={'color': '#495057', 'margin-top': '20px'}),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=start_date,
                    max_date_allowed=end_date,
                    start_date=start_date,
                    end_date=end_date,
                    display_format='YYYY-MM-DD',
                    style={'color': '#000000'}
                ),
                dcc.Graph(id='graph')
            ])
        ]),

        dcc.Tab(label='Exploratory Data Analysis', value='tab-2', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H2("Exploratory Data Analysis (EDA)", style={'color': '#343a40'}),
                html.P('Select Features to Display:', style={'color': '#495057'}),
                dcc.Dropdown(
                    id='eda-column-dropdown',
                    options=[{'label': col, 'value': col} for col in df_data.columns],  # 2017-2018 data only
                    value=[df_data.columns[0]], multi=True,
                    style={'backgroundColor': '#ffffff', 'color': '#000000'}
                ),
                html.P("Select Graph Type:", style={'color': '#495057', 'margin-top': '20px'}),
                dcc.Dropdown(
                    id='graph-type-dropdown',
                    options=[
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Box Plot', 'value': 'boxplot'},
                        {'label': 'Line Plot', 'value': 'plot'}
                    ],
                    value='plot'
                ),
                dcc.Graph(id='eda-graph')
            ])
        ]),

        dcc.Tab(label='Feature Selection', value='tab-3', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H2("Feature Selection", style={'color': '#343a40'}),

                html.P("Select Features:", style={'color': '#495057'}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in features_list],
                    multi=True,
                    style={'backgroundColor': '#ffffff', 'color': '#000000'}
                ),

                html.Label("Select Successful Features:", style={'color': '#495057', 'margin-top': '20px'}),
                dcc.Checklist(
                    id='success-checkbox',
                    options=[{'label': '', 'value': 'checked'}],  # Empty label, acts as a small square
                    inline=True
                ),
                html.Pre(id='ols-summary', style={'white-space': 'pre-wrap'}),

                html.P("Select Feature Selection Method:", style={'color': '#495057', 'margin-top': '20px'}),
                dcc.Dropdown(
                    id='method-dropdown',
                    options=[
                        {'label': 'Select kBest', 'value': 'kbest'},
                        {'label': 'Wrapper Method: Recursive Feature Elimination (RFE)', 'value': 'rfe'},
                        {'label': 'Ensemble Method: Random Forest Regressor', 'value': 'rf'}
                    ],
                    value='kbest'
                ),

                html.Button("Run Selection", id='run-button', n_clicks=0, style={'color': '#495057', 'margin-top': '10px'}),

                html.H3("Feature Scores"),
                dash_table.DataTable(
                    id='feature-table',
                    columns=[
                        {"name": "Feature", "id": "Feature"},
                        {"name": "Score", "id": "Score"}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': '#004b6b',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_data={'backgroundColor': '#e3f2fd', 'color': 'black'},
                    style_cell={'textAlign': 'center'},
                )
            ])
        ]),

        dcc.Tab(label='Forecasting Models', value='tab-4', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H2("Forecasting Models", style={'color': '#343a40'}),
                html.P(" Using the «Successful features selection», select a forecasting model to predict the 2019 power consumption:", style={'color': '#495057'}),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Random Forest', 'value': 'RF_model'},
                        {'label': 'Gradient Boosting', 'value': 'GB_model'},
                        {'label': 'Linear Regression', 'value': 'LR_model'},
                        {'label': 'Neural Networks', 'value': 'NN_model'},
                        {'label': 'Decision Tree Regressor', 'value': 'DT_model'},
                    ],
                    value='RF_model',  # Default model
                    style={'backgroundColor': '#ffffff', 'color': '#000000'}

                ),
                dcc.Graph(id='forecast-line-graph'),
                dcc.Graph(id='forecast-scatter-graph')
            ])
        ]),

        dcc.Tab(label='Metrics', value='tab-5', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H2("Model Performance Metrics", style={'color': '#343a40'}),
                html.P("Performance metrics for the selected forecasting model:", style={'color': '#495057'}),

                # Table with Model Metrics
                dash_table.DataTable(
                    id='metrics-table',
                    columns=[
                        {"name": "Metrics", "id": "Metrics"},
                        {"name": "Model", "id": "Model"},
                        {"name": "ASHRAE", "id": "ASHRAE"},
                        {"name": "IPMVP", "id": "IPMVP"},
                    ],
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': '#004b6b',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_data={'backgroundColor': '#e3f2fd', 'color': 'black'},
                    style_cell={'textAlign': 'center'},
                ),

                # Professional-looking note
                html.P(
                    "*ASHRAE: American Society of Heating, Refrigerating and Air-Conditioning Engineers. "
                    "IPMVP: International Performance Measurement and Verification Protocol.",
                    style={'fontSize': '12px', 'color': '#6c757d'}
                )
            ])
        ])
    ])
])
# Callback for updating Raw Data graph dynamically
@app.callback(
    Output('graph', 'figure'),
    [Input('column-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)

def update_graph(selected_columns, start_date, end_date):
    """Updates the graph based on selected columns and date range."""
    if not selected_columns:
        selected_columns = [df_raw.columns[0]]
    return generate_graph(df_raw, selected_columns, start_date, end_date)

# Callback for updating EDA graph dynamically
@app.callback(
    Output('eda-graph', 'figure'),
    [Input('eda-column-dropdown', 'value'),
     Input('graph-type-dropdown', 'value')]
)
def update_eda_graph(selected_columns, graph_type):
    """Updates the EDA graph based on selected columns and graph type."""
    if not selected_columns:
        selected_columns = [df_data.columns[0]]
    return generate_graph(df_data, selected_columns, start_date_data, end_date_data, graph_type)

# Callback for Feature Selection
@app.callback(
    Output('feature-table', 'data'),
    [Input('run-button', 'n_clicks')],
    [dash.dependencies.State('feature-dropdown', 'value'),
     dash.dependencies.State('method-dropdown', 'value')]
)
def update_feature_selection(n_clicks, selected_features, method):
    if not selected_features or n_clicks == 0:
        return []

    # Ensure that we are working with the correct data
    Z = df_FeatureSelectionP2.values
    selected_indices = [df_FeatureSelectionP2.columns.get_loc(col) for col in selected_features]
    X = Z[:, selected_indices]  # Select only the columns corresponding to selected features
    Y = Z[:, 0]  # Target variable (e.g., power consumption)

    # Initialize scores list
    scores = []

    if method == 'kbest':
        try:
            selector = SelectKBest(score_func=f_regression, k='all')  # Use 'all' to get scores for all features
            selector.fit(X, Y)  # Ensure X and Y are accessible here
            scores = selector.scores_

            # Create a DataFrame with Feature names and Scores
            feature_scores = pd.DataFrame({
                'Feature': selected_features,
                'Score': scores
            })

            # Sort the DataFrame by Score (descending order)
            feature_scores = feature_scores.sort_values(by='Score', ascending=False)

            return feature_scores.to_dict(orient='records')  # Return this to Dash

        except Exception as e:
            print(f"Error in SelectKBest: {e}")
            return []  # Return empty list if error occurs

    elif method == 'rfe':
        model = LinearRegression()
        selector = RFE(model, n_features_to_select=1)
        selector.fit(X, Y)  # Ensure X and Y are accessible here
        scores = selector.ranking_
        # Create a DataFrame with Feature names and Scores
        feature_scores = pd.DataFrame({
            'Feature': selected_features,
            'Score': scores
        })

        # Sort the DataFrame by Ranking (ascending order)
        feature_scores = feature_scores.sort_values(by='Score', ascending=True)

        return feature_scores.to_dict(orient='records')  # Return this to Dash

    elif method == 'rf':
        model = RandomForestRegressor()
        model.fit(X, Y)  # Ensure X and Y are accessible here
        scores = model.feature_importances_

        # Create a DataFrame with Feature names and Feature Importances
        feature_scores = pd.DataFrame({
            'Feature': selected_features,
            'Score': scores
        })

        # Sort the DataFrame by Score (descending order)
        feature_scores = feature_scores.sort_values(by='Score', ascending=False)

        return feature_scores.to_dict(orient='records')  # Return this to Dash



@app.callback(
    [Output('feature-dropdown', 'value'),
     Output('ols-summary', 'children')],
    [Input('success-checkbox', 'value')]
)
def successful_feature_selection(checkbox_value):
    if checkbox_value:
        # Select successful features
        selected_features = successful_features

        # Extract data
        Z = df_FeatureSelectionP2[selected_features].values
        y = df_FeatureSelectionP2.iloc[:, 0].values  # Assuming first column is target variable
        X = add_constant(Z)  # Add constant for OLS

        # Fit OLS model
        model = OLS(y, X).fit()
        summary_text = model.summary().as_text()

        return selected_features, summary_text
    else:
        # If unchecked, clear selection and OLS summary
        return [], ""

#Calling for Forecasting Models
@app.callback(
    [Output('forecast-line-graph', 'figure'),
     Output('forecast-scatter-graph', 'figure')],
    [Input('model-dropdown', 'value')]
)
def update_forecast_graph(selected_model):
    model_labels = {
        'RF_model': 'Random Forest',
        'GB_model': 'Gradient Boosting',
        'LR_model': 'Linear Regression',
        'NN_model': 'Neural Networks',
        'DT_model': 'Decision Tree Regressor'
    }

    model_title = model_labels[selected_model]
    model = models[selected_model]

    Y_pred = model.predict(X_features)
    x_axis = df_2019.index  # Use actual timestamps or index range

    # Line plot: Actual vs. Predicted
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=x_axis, y=Y_actual, mode='lines', name='Actual Values'))
    line_fig.add_trace(go.Scatter(x=x_axis, y=Y_pred, mode='lines', name='Predicted Values', line=dict(color='red')))
    line_fig.update_layout(
        title={
        'text': f'{model_title}: Actual vs Predicted',
        'x': 0.5,
        'xanchor': 'center',
        'y': 0.85,
        'yanchor': 'top'
        },
        xaxis_title='Time',
        yaxis_title='Values',
        template='plotly_white'
    )

    # Scatter plot: Actual vs. Predicted
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(x=Y_actual, y=Y_pred, mode='markers', name='Predicted vs Actual',
                                     marker=dict(color='blue', opacity=0.5)))
    scatter_fig.update_layout(
        title={
            'text': f'{model_title}: Scatter Plot',
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.85,
            'yanchor': 'top'
        },
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        template='plotly_white'
    )

    return line_fig, scatter_fig


# Callback to update the Metrics table
@app.callback(
    Output('metrics-table', 'data'),
    [Input('model-dropdown', 'value')]
)
def update_metrics(selected_model):
    """Computes and updates error metrics for the selected model."""

    if selected_model not in models:
        return []

    # Get predictions
    model = models[selected_model]
    Y_pred = model.predict(X_features)

    # Compute error metrics
    MAE = metrics.mean_absolute_error(Y_actual, Y_pred)
    MBE = np.mean(Y_actual - Y_pred)
    MSE = metrics.mean_squared_error(Y_actual, Y_pred)
    RMSE = np.sqrt(MSE)
    cvRMSE = (RMSE / np.mean(Y_actual)) * 100  # Convert to %
    NMBE = (MBE / np.mean(Y_actual)) * 100  # Convert to %

    # Table Data Format
    metrics_data = [
        {"Metrics": "MAE (Mean Absolute Error)", "Model": round(MAE, 3), "ASHRAE": "", "IPMVP": ""},
        {"Metrics": "MBE (Mean Bias Error)", "Model": round(MBE, 3), "ASHRAE": "", "IPMVP": ""},
        {"Metrics": "MSE (Mean Squared Error)", "Model": round(MSE, 3), "ASHRAE": "", "IPMVP": ""},
        {"Metrics": "RMSE (Root Mean Squared Error)", "Model": round(RMSE, 3), "ASHRAE": "", "IPMVP": ""},
        {"Metrics": "cvRMSE (Coefficient of Variation RMSE)", "Model": f"{round(cvRMSE, 2)}%", "ASHRAE": "≤30%", "IPMVP": "≤20%"},
        {"Metrics": "NMBE (Normalized Mean Bias Error)", "Model": f"{round(NMBE, 2)}%", "ASHRAE": "±10%", "IPMVP": "±5%"},
    ]

    return metrics_data



if __name__ == '__main__':
    app.run(debug=False)
