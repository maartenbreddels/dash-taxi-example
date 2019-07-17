import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

import vaex
import numpy as np
# df = vaex.open('/Users/maartenbreddels/datasets/nytaxi/nyc_taxi2015.hdf5')
df = vaex.open('s3://vaex/taxi/yellow_taxi_2015_f32s.hdf5?anon=true')

df = df[:10_000_000]  # comment this line to get all the data

labels = {
    'week': list((map(str, range(1, 53)))),
    'hour': list((map(str, range(24)))),
    'month': ['All', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
}

df['pu_day_of_week'] = df.pickup_datetime.dt.dayofweek
df['pu_hour'] = df.pickup_datetime.dt.hour
df['pu_month'] = df.pickup_datetime.dt.month - 1

df.categorize(column='pu_hour', labels=labels['hour'], check=False)
df.categorize(column='pu_day_of_week', labels=labels['day_of_week'], check=False)

x = str(df.pickup_longitude)
y = str(df.pickup_latitude)
limits = df.limits([x, y], '96%')

app.layout = html.Div(children=[
    html.H1(children='Hello Dash & Vaex'),
    dcc.Dropdown(id='month',
        options=[{'label': k, 'value': i} for i, k in enumerate(labels['month'])],
        value=0
    ),
    html.Div([
        dcc.Graph(
            id='my-graph',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [9, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                ],
                'layout': {
                    'title': 'Dash+Vaex: 150 million taxi rides.',
                    'xaxis': {'label': x, 'range': limits[0]},
                    'yaxis': {'label': y, 'range': limits[1]}
                }
            }
        )],
        style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}
    ),
    html.Div([
        dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            ),
        dcc.Graph(
            id='my-bar',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [9, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                ],
                'layout': {
                    'title': 'Dash+Vaex: 150 million taxi rides.',
                    'xaxis': {'label': x, 'range': limits[0]},
                    'yaxis': {'label': y, 'range': limits[1]}
                }
            }
        )],
        style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'padding': '0 20'}
    ),

])

@app.callback(
    [Output(component_id='my-graph', component_property='figure'),
     Output(component_id='my-bar', component_property='figure')],
    [Input(component_id='month', component_property='value'),
     Input(component_id='my-graph', component_property='relayoutData'),
     Input('yaxis-type', 'value'),]
)
def update_output_div(month_value, relayoutData, yaxis_type):
    print(relayoutData)
    if relayoutData is not None and 'xaxis.range[0]' in relayoutData:# is None or relayoutData.get('autosize', False) or :
        d = relayoutData
        user_limits = [[d['xaxis.range[0]'], d['xaxis.range[1]']], [d['yaxis.range[0]'], d['yaxis.range[1]']]]
    else:
        user_limits = limits
    limits_x, limits_y = user_limits
    if month_value == 0:
        dff = df
    else:
        dff = df[df.pu_month==month_value-1]
    shape = 256
    count_all = dff.count(binby=[x, y, dff.pu_hour], limits=[user_limits[0], user_limits[1], None], shape=shape, edges=True)
    count = count_all.sum(axis=2)[2:-1, 2:-1]
    count_hours_zoom = count_all[2:-1, 2:-1].sum(axis=(0,1))[2:-1]
    count_hours_all = count_all.sum(axis=(0,1))[2:-1]
    
    total_count = count.sum()
    z = np.log1p(count).T.tolist()
    data = {'z': z, 'x': dff.bin_centers(x, limits_x, shape=shape), 'y': dff.bin_centers(y, limits_y, shape=shape), 'type': 'heatmap'}
    month = labels['month'][month_value]
    figure_heat = {
            'data': [data],
            'layout': {
                'title': f'Taxi pickups for {month} (total {total_count:,})',
                'xaxis': {'label': x},
                'yaxis': {'label': y}
            }
        }

    figure_bar = {
            'data': [
                {'x': labels['hour'], 'y': count_hours_zoom.tolist(), 'type':'bar', 'name': 'Zoomed region'},
                {'x': labels['hour'], 'y': count_hours_all.tolist(),  'type':'bar', 'name': 'Full region'},
            ],
            'layout': {
                'title': f'Pickup hours in zoomed region',
                'xaxis': {'label': 'Pickup hour'},
                'yaxis': {'label': 'counts', 'type': 'linear' if yaxis_type == 'Linear' else 'log'}
            }
        }
    return figure_heat, figure_bar

if __name__ == '__main__':
    app.run_server(debug=True)
