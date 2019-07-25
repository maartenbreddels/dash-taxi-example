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
shape = 256

# if we don't fill the store, dash will call the callbacks with None as data
count_all = df.count(binby=[x, y, df.pu_hour],
                    limits=[limits[0], limits[1], None],
                    shape=shape, edges=True,
                    progress=True)
count = count_all.sum(axis=2)[2:-1, 2:-1]
count_hours_zoom = count_all[2:-1, 2:-1].sum(axis=(0,1))[2:-1]
count_hours_all = count_all.sum(axis=(0,1))[2:-1]


app.layout = html.Div(children=[
    dcc.Store(id='limits', data=limits),
    dcc.Store(id='data-heatmap', data=count.T.tolist()),
    dcc.Store(id='data-bar', data=[count_hours_zoom, count_hours_all]),
    html.H1(children='Hello Dash & Vaex'),
    dcc.Dropdown(id='month',
        options=[{'label': k, 'value': i} for i, k in enumerate(labels['month'])],
        value=0
    ),
    html.Div([
        dcc.Graph(
            id='heatmap',
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
            id='bar-chart',
        )],
        style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'padding': '0 20'}
    ),

])


@app.callback(
    Output(component_id='limits', component_property='data'),
    [Input(component_id='heatmap', component_property='relayoutData')]
)
def update_limits(relayoutData):
    print('update_limits', relayoutData)
    if relayoutData is not None and 'xaxis.range[0]' in relayoutData:
        d = relayoutData
        user_limits = [[d['xaxis.range[0]'], d['xaxis.range[1]']], [d['yaxis.range[0]'], d['yaxis.range[1]']]]
    else:
        user_limits = limits
    return user_limits


@app.callback(
    [Output(component_id='data-heatmap', component_property='data'),
     Output(component_id='data-bar', component_property='data')],
    [Input(component_id='month', component_property='value'),
     Input(component_id='limits', component_property='data')]
)
def update_data(month_value, user_limits):
    print('updating data: limits', user_limits)
    limits_x, limits_y = user_limits
    if month_value == 0:
        dff = df
    else:
        dff = df[df.pu_month==month_value-1]
    count_all = dff.count(binby=[x, y, dff.pu_hour], limits=[user_limits[0], user_limits[1], None], shape=shape, edges=True)
    count = count_all.sum(axis=2)[2:-1, 2:-1]
    count_hours_zoom = count_all[2:-1, 2:-1].sum(axis=(0,1))[2:-1]
    count_hours_all = count_all.sum(axis=(0,1))[2:-1]
    
    return count.T.tolist(), [count_hours_zoom, count_hours_all]


@app.callback(
    Output(component_id='heatmap', component_property='figure'),
    [Input(component_id='month', component_property='value'),
     Input(component_id='data-heatmap', component_property='data'),
     Input(component_id='limits', component_property='data')]
)
def update_figure_2d(month_value, data_heatmap, limits):
    print('update_figure_2d')
    counts = np.array(data_heatmap)
    limits_x, limits_y = limits
    z = np.log1p(counts).tolist()
    data = {'z': z, 'x': df.bin_centers(x, limits_x, shape=shape), 'y': df.bin_centers(y, limits_y, shape=shape), 'type': 'heatmap'}
    total_count = counts.sum()
    month = labels['month'][month_value]
    return {
            'data': [data],
            'layout': {
                'title': f'Taxi pickups for {month} (total {total_count:,})',
                'xaxis': {'label': x},
                'yaxis': {'label': y}
            }
        }


@app.callback(
    Output(component_id='bar-chart', component_property='figure'),
    [Input(component_id='data-bar', component_property='data'),
     Input('yaxis-type', 'value')]
)
def update_figure_bar(data_bar, yaxis_type):
    print('update_figure_bar')
    count_hours_zoom, count_hours_all = data_bar
    return {
            'data': [
                {'x': labels['hour'], 'y': count_hours_zoom, 'type':'bar', 'name': 'Zoomed region'},
                {'x': labels['hour'], 'y': count_hours_all,  'type':'bar', 'name': 'Full region'},
            ],
            'layout': {
                'title': f'Pickup hours in zoomed region',
                'xaxis': {'label': 'Pickup hour'},
                'yaxis': {'label': 'counts', 'type': 'linear' if yaxis_type == 'Linear' else 'log'}
            }
        }

if __name__ == '__main__':
    app.run_server(debug=True)
