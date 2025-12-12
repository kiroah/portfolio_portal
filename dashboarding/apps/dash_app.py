from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd
import datetime
import plotly.graph_objects as go
import numpy as np


# IMPORTANT: Keep the trailing slash
app = Dash(__name__, url_base_pathname='/dash/') 

# 2. Expose the server variable (CRITICAL for deployment)
server = app.server 

# 3. Your Data & Plotly Express Code
source_df = pd.read_csv("../data/ap.csv")
main_df = source_df.iloc[:50000].copy()
main_df.assign(day=1)[['year','period','day']]
main_df['date'] = pd.to_datetime(main_df.assign(day=1).rename(columns={'period':'month'})[['year','month','day']])

DEFAULT_METRIC_1 = 'Electricity per KWH'

def get_visibilities(index,traces_per_set,num_of_sets):
    visibility_list = [[True if y == index else False for x in range(0,traces_per_set)] for y in range(0,num_of_sets)]
    return [item for sublist in visibility_list for item in sublist]

# Create initial figure
fig = go.Figure()

# Get unique categories from the DataFrame
metrics = main_df['name'].sort_values().unique()
metrics_size = len(metrics)
default_metric_index = int(np.where(metrics ==DEFAULT_METRIC_1)[0][0])

# Add a trace for each category
for metric in metrics:
    # Filter data for the current category
    filtered_df = main_df[(main_df['name'] == metric) & (main_df['location type'] == 'National')]
    
    # Add a trace for the current category
    default_visible = True if metric == DEFAULT_METRIC_1 else False
    fig.add_trace(go.Scatter(x=filtered_df['date'], 
                             y=filtered_df['value'], 
                             mode='lines', 
                             name=metric,
                             visible=default_visible,
                             hovertemplate='%{y:$,.3f}'))
    if (len(filtered_df) > 0):
        coefficients = np.polyfit(filtered_df['date'].astype('int64'), filtered_df['value'], 1)
        poly1 = np.poly1d(coefficients)
        y_fit = poly1(filtered_df['date'].astype('int64'))
        fig.add_trace(go.Scatter(x=filtered_df['date'], 
                             y=y_fit, mode='lines', 
                             line=dict(dash='dash'),
                             visible=default_visible,
                             name='Regression Line'))
    else:
        fig.add_trace(go.Scatter(x=filtered_df['date'], 
                             y=[0] * len(filtered_df), mode='lines', 
                             line=dict(dash='dash'),
                             visible=default_visible,
                             name='Regression Line'))
    

# Create dropdown buttons
metric_dropdown = []

for index, metric in enumerate(metrics):
    metric_dropdown.append(
        {'label': metric, 'method': 'update', 'args': [{'visible': get_visibilities(index,2,metrics_size)}]}
    )
metric_menu = {
        'buttons': metric_dropdown,
        'direction': 'down',
        'showactive': True,
        'active': 0,
        'x': 0.5,
        'xanchor': 'center',
        'y': 1.3,
        'yanchor': 'top'
        
    }

date_dropdown = []
dates = main_df['date'].sort_values(ascending=False).unique()

for d, d_s in zip(dates, np.datetime_as_string(dates,unit='M')): 
    date_dropdown.append(
        {
                'label': d_s,
                'method': 'relayout',
                'args': [{'xaxis.range': [d - np.timedelta64(3*365,'D'), d]}]
            }
    )    
date_menu = {
        'buttons': date_dropdown,
        'direction': 'down',
        'showactive': True,
        'yanchor': 'top',
        'y': 1.5,
        'xanchor': 'center',
            'x': 0.5,
}
    
    
# Update layout to include the dropdown menu
fig.update_layout(
     height=600,
    updatemenus=[metric_menu, date_menu],
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    legend=dict(
        yanchor="bottom",  # Anchor to the bottom
        y=-0.5,  # Position below the plot
        xanchor="center",  # Center horizontally
        x=0.5  # Position at the center
    ),

)


# 4. Define the Layout (This replaces fig.show())
app.layout = html.Div([
    html.H1("[Demo app] US Bureau of labor statistics - average price data'"),
    dcc.Graph(figure=fig)
])

# 5. Run the app (Local testing only)
if __name__ == '__main__':
    app.run_server(debug=True)