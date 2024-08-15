import pandas as pd
import numpy as np
import panel as pn
import bokeh.plotting as bp
from bokeh.models import ColumnDataSource, HoverTool,Div,Span,Label,ResetTool, HTMLTemplateFormatter,NumberFormatter, InlineStyleSheet, Legend
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.layouts import column
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
from bokeh.models.widgets import DataTable, TableColumn
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from bokeh.palettes import Category20
import math



# Enable the Panel extension
pn.extension(raw_css=[
    """
    @import url('https://fonts.googleapis.com/css2?family=Open Sans:wght@400;700&display=swap');
    body{
        background-color: #F0F0F0;
    }
    .bk-root .bk-text-title {
        font-family: 'Open Sans' !important;
    }
    .bk-root .bk-plot-title {
        font-family: 'Open Sans' !important;
    }
    .bk-root .bk-axis-label {
        font-family: 'Open Sans' !important;
    }
    .bk-root .bk-axis-major-label {
        font-family: 'Open Sans' !important;
    }
    .bk-root .bk-legend-label {
        font-family: 'Open Sans' !important;
    }
    h1 {
    font-family:'Open Sans'  !important;
    }

    .bk-tab {
      font-size: 16px;
      background-color: #CCCCCC; 
    }
    

    .bk-tab.bk-active {
        background-color:: #FFFFFF !important; 
        border-bottom: 1px solid red;

    }    
    """
])

stylesheet = InlineStyleSheet(css=
    """
    .slick-header-columns {
        font-family: 'Open Sans' !important;
        font-weight: bold;
    }
    """
)



source_df = pd.read_csv("ap.csv")
#main_df = source_df.iloc[:50000].copy()
main_df = source_df.copy()

main_df.assign(day=1)[['year','period','day']]
main_df['date'] = pd.to_datetime(main_df.assign(day=1).rename(columns={'period':'month'})[['year','month','day']])
# Change few fields to categorical to reduce memory consumption
main_df['name'] = main_df['name'].astype('category')
#main_df['location'] = main_df['location'].astype('category')
#main_df['location type'] = main_df['location type'].astype('category')

def format_percent_change(percentage):
    if math.isnan(percentage):
        return "N/A"
    elif percentage < 0:
        return f"▼{abs(percentage) * 100.0:.1f}%"
    elif percentage > 0:
        return f"▲{percentage * 100.0:.1f}%"
    else:
        return f"{percentage * 100.0:.1f}%"

    

#create MoM change and YoY change fields
main_df['last month date'] = main_df['date'] - pd.DateOffset(months=1)
main_df['last month value'] = pd.merge(main_df,main_df,how='left',left_on=['name','location','last month date'],right_on=['name','location','date'])['value_y']
main_df['mom change'] = (main_df['value'] / main_df['last month value']) - 1.0
main_df['mom change text'] = main_df['mom change'].map(lambda x: f'<div style="color:darkgreen; text-align:center">▲{x*100.0:.1f}%</div>' 
                                                       if x > 0 else f'<div style="color:darkred; text-align:center">▼{abs(x)*100.0:.1f}%</div>')


main_df['last year date'] = main_df['date'] - pd.DateOffset(years=1)
main_df['last year value'] = pd.merge(main_df,main_df,how='left',left_on=['name','location','last year date'],right_on=['name','location','date'])['value_y']
main_df['yoy change'] = (main_df['value'] / main_df['last year value']) - 1.0
main_df['yoy change text'] = main_df['yoy change'].map(lambda x: f'<div style="color:darkgreen; text-align:center">▲{x*100.0:.1f}%</div>' 
                                                       if x > 0 else f'<div style="color:darkred; text-align:center">▼{abs(x)*100.0:.1f}%</div>')


DEFAULT_METRIC_1 = 'Electricity per KWH'
DEFAULT_METRIC_2 = 'Gasoline, unleaded regular, per gallon/3.785 liters'
DEFAULT_LOCTYPE = 'Regional'

# Get unique categories from the DataFrame
metrics = list(main_df['name'].sort_values().unique())
metrics_size = len(metrics)
default_metric1_index = metrics.index(DEFAULT_METRIC_1)
default_metric2_index =  metrics.index(DEFAULT_METRIC_2)


dates = list(main_df['date'].dt.strftime('%Y-%m').sort_values(ascending=False).unique())



palette20 = Category20[20]
palette10 = Category10[10]


# Initial data
initial_date_s = dates[0]
initial_date = pd.to_datetime(dates[0], format='%Y-%m')


"""
Dashboard 1 (Summary) 
Shows current value & over time graph for 2 KPIs and top movers. 

User can pick a date (as of) and two metrics to display on the graphs. 
For each metric, there will be a scorecard (displaying as of price + YoY change) and a line graph that shows change over time + regression line
In addition, there will be two tables showing metrics that increased or decreased the most year over year, respectively

Main UI elements: 
    db1_metric_dropdown1 - Dropwown menu to choose the 1st metric
    db1_metric_dropdown2 - Dropwown menu to choose the 2nd metric 
    db1_date_dropdown  - Dropwown menu to choose the as of date
    db1_plot1 - Line graph showing the price change over time for the 1st metric
    db1_scorecard_text1 - Scorecard showing the value for the 1st metric as of date and relative %-change year over year
    db1_plot1 - Line graph showing the price change over time for the 2nd metric
    db1_scorecard_text1 - Scorecard showing the value for the 2nd as of date and relative %-change year over year
    db1_table1 - Top 5 metrics that increased the most YoY for the chosen date
    db1_table2 - Top 5 metrics that decreased the most YoY for the chosen date

"""


def generate_scorecard_text(name, current_value,previous_value, date):
    """ Generates scorecard text for the scorecard element 
    In:
        name (string)- name of the metric to generate for, used for display
        current_value (float) - Current value for the metric to display
        previous_value (float) - Used to calculate the relative difference (year over year change)
        date (datetime) - Date to display
    Out:
        html text for the scorecard
    """
    
    #We will be adding triangle symbols to indicate positive or negative YoY change, with different coloring
    if ((current_value is not None) and (previous_value is not None)):
        if (current_value > previous_value):
            yoy_color = 'darkgreen'
            sign = '▲'
        elif (current_value < previous_value):
            yoy_color = 'darkred'
            sign = '▼'
        else:
            yoy_color = 'darkgrey'
            sign = ''
    else:
        yoy_color='darkgrey'
        sign =''
    return f"""
    <div style="border: 1px solid black; padding: 10px;text-align: center;line-height:1.2; width:200px">
         <span style="font-size: 16px;"> {name} - {date.strftime('%Y-%m')} </span>
         </br>
         <span style="font-size: 36px; font-weight: bold;">${"N/A" if current_value is None else "{:.3f}".format(current_value)} </span>
         </br>
         <span style="font-size: 18px;font-weight: bold; color: {yoy_color};">{sign}{"N/A" if previous_value is None else "{:.1f}".format(abs(current_value / previous_value - 1.0)*100.0)}% </span>
         </br>
         <span style="font-size: 14px; color: grey;">YoY change</span>
    </div>
    """


def add_regression_line(source, plot, x_col, y_col):
    """ Adds regression line to the plot
    In:
        source (ColumnDataSource)- Source data
        plot (figure) - plot to add the regression line
        x_col (string) - column name to be used to x-axis 
        y_col (string) - column name to be used to y-axis
    Out:
        No returns, but updates plot with the regression line
    """

    x = source.data[x_col]
    y = source.data[y_col]

    # Make sure there's data
    if len(x) > 1:
        # Fit a linear regression model
        coefficients = np.polyfit(x.astype('float64'), y, 1)
        poly1 = np.poly1d(coefficients)
        # Create regression line
        y_fit = poly1(x.astype('float64'))

        plot.line([x[0],x[-1]], [y_fit[0],y_fit[-1]], 
                  line_width=2, color='red', legend_label='Regression Line', line_dash='dashed')


def update_db1_charts(source, plot, scorecard_text, selected_date, selected_metric, ):
    """ Function to update data for the charts based on chosen date and metric
    In:
        source (ColumnDataSource) - Data source for the charts, to be updated
        plot (figure) - plot to update
        scorecard_text(div) - scorecard element to update
        selected_date (datetime) - date to filter the data by
        selected_metric (string) - metric to filter the data by
    Out:
        None, but plot and scorecard_text will be updated 
    """
    
    #Update date by first extracting from the main dataframe
    new_filtered_df = main_df[(main_df['name'] == selected_metric) & \
                         (main_df['location type'] == 'National') &
                         (main_df['date'] >= selected_date - relativedelta(years=3)) & 
                         (main_df['date'] <= selected_date)].sort_values(by='date',ascending=True)
    
    #Update scorecard text by first getting the current and previous value, and pass to the generate_scorecard_text function. 
    if (len(new_filtered_df)==0):
        current_value = None
        previous_value = None
    else:
        current_value = new_filtered_df['value'].values[0]
        previous_value_df = new_filtered_df.loc[new_filtered_df['date'] == (selected_date - relativedelta(years=1)),'value']
        if (len(previous_value_df)==0):
            previous_value = None
        else:
            previous_value = previous_value_df.values[0]
    scorecard_text.text = generate_scorecard_text(selected_metric, current_value,previous_value,selected_date)

    
    #Update plot
    #If new_filtered_df is empty (i.e. user didn't choose metric + date with data), then show all data instead 
    if (len(new_filtered_df)==0):
        new_filtered_df = main_df[(main_df['name'] == selected_metric) & \
                         (main_df['location type'] == 'National') &
                         (main_df['date'] <= selected_date)].sort_values(by='date',ascending=True)
    source.data = ColumnDataSource.from_df(new_filtered_df)
    plot.title = f'{selected_metric} over Time'
    plot.renderers = plot.renderers[:1]  # Remove old regression line, keep original line
    add_regression_line(source, plot,'date','value')

    
def update_db1_table(selected_date, source):
    """ 
    In:
        selected_date - Date to show the data for
        source (ColumnDataSource) - source to update
    Out:
        None, but updates the data for the source will be updated
    """

    new_filtered_df = main_df[(main_df['location type'] == 'National') &
                     (~main_df['yoy change'].isna()) &
                     (main_df['date'] == selected_date)].sort_values(by='yoy change',ascending=False).head(5)

    source.data = ColumnDataSource.from_df(new_filtered_df)


    

# Initialization of all elements

# Dataframes
db1_initial_df1 = main_df[(main_df['name'] == DEFAULT_METRIC_1) & 
                 (main_df['location type'] == 'National') &
                 (main_df['date'] >= initial_date - relativedelta(years=3)) & 
                 (main_df['date'] <= initial_date)].sort_values(by='date',ascending=True)
db1_initial_df2 = main_df[(main_df['name'] == DEFAULT_METRIC_2) & 
                 (main_df['location type'] == 'National') &
                 (main_df['date'] >= initial_date - relativedelta(years=3)) & 
                 (main_df['date'] <= initial_date)].sort_values(by='date',ascending=True)

db1_initial_df3 = main_df[(main_df['location type'] == 'National') &
                 (~main_df['yoy change'].isna()) &
                 (main_df['date'] == initial_date)].sort_values(by='yoy change',ascending=False).head(5)

db1_initial_df4 = main_df[(main_df['location type'] == 'National') &
                 (~main_df['yoy change'].isna()) &
                 (main_df['date'] <= initial_date)].sort_values(by='yoy change',ascending=True).head(5)


# Scorecard
db1_current_value1 = db1_initial_df1['value'].values[-1]
db1_previous_value1 = db1_initial_df1.loc[db1_initial_df1['date'] == (initial_date - relativedelta(years=1)),'value'].values[0]
db1_scorecard_text1 = Div(text=generate_scorecard_text(DEFAULT_METRIC_1, db1_current_value1,db1_previous_value1,initial_date)  , min_width=220, height=100)

db1_current_value2 = db1_initial_df2['value'].values[-1]
db1_previous_value2 = db1_initial_df2.loc[db1_initial_df2['date'] == (initial_date - relativedelta(years=1)),'value'].values[0]
db1_scorecard_text2 = Div(text=generate_scorecard_text(DEFAULT_METRIC_2, db1_current_value2,db1_previous_value2,initial_date)  , min_width=220, height=100)
    

    
#Plots    
db1_source1 = ColumnDataSource(db1_initial_df1)
db1_source2 = ColumnDataSource(db1_initial_df2)

db1_plot1 = figure(x_axis_type='datetime', title=f'{DEFAULT_METRIC_1} over Time', aspect_ratio=2,min_height=200, min_width=400)
db1_plot1.line('date', 'value', source=db1_source1, line_width=2, color=palette10[0])
db1_plot1.add_tools(HoverTool(tooltips=[('Date', '@date{%F}'), ('Value', '@value')], formatters={'@date': 'datetime'}))

db1_plot2 = figure(x_axis_type='datetime', title=f'{DEFAULT_METRIC_2} over Time', aspect_ratio=2, min_height=200, min_width=400)
db1_plot2.line('date', 'value', source=db1_source2, line_width=2, color=palette10[1])
db1_plot2.add_tools(HoverTool(tooltips=[('Date', '@date{%F}'), ('Value', '@value')], formatters={'@date': 'datetime'}))

add_regression_line(db1_source1, db1_plot1,'date','value')
add_regression_line(db1_source2, db1_plot2,'date','value')



#UI Tables
db1_source3 = ColumnDataSource(db1_initial_df3)
db1_source4 = ColumnDataSource(db1_initial_df4)

db1_table_columns = [
    TableColumn(field="name", title="Metric"),
    TableColumn(field="value", title="Value",formatter=NumberFormatter(format="$0,0.000",text_align="right")),
    TableColumn(field="yoy change text", title="YoY Change", formatter=HTMLTemplateFormatter(template="<%= value %>")),
    TableColumn(field="mom change text", title="MoM Change", formatter=HTMLTemplateFormatter(template="<%= value %>"))
]

db1_table1 = DataTable(source=db1_source3, columns=db1_table_columns, width=400, height=280, stylesheets=[stylesheet], index_position=None,)
db1_table2 = DataTable(source=db1_source4, columns=db1_table_columns, width=400, height=280, stylesheets=[stylesheet], index_position=None,)




# Create dropdown widgets
db1_metric_dropdown1 = pn.widgets.Select(name='Metric 1',options=metrics, value=DEFAULT_METRIC_1)
db1_metric_dropdown2 = pn.widgets.Select(name='Metric 2',options=metrics, value=DEFAULT_METRIC_2)
db1_date_dropdown = pn.widgets.Select(name='As of',options=dates, value=initial_date_s,width=150)



def db1_updates(event):
    """ Triggering function when the user inputs are updated 
    In:
        event - not used
    Out:
        None, but the elements will be updated
    """

    widget_name = event.obj.name
    selected_date =pd.to_datetime(db1_date_dropdown.value, format='%Y-%m')
    
    # Updates for Metric 1 (if metric 1 or date updated by user)
    if (event.obj == db1_metric_dropdown1 or event.obj == db1_date_dropdown):
        selected_metric = db1_metric_dropdown1.value
        update_db1_charts(db1_source1, db1_plot1,db1_scorecard_text1,selected_date,selected_metric)

    # Updates for Metric 2  (if metric 2 or date updated by user)
    if (event.obj  == db1_metric_dropdown2 or event.obj == db1_date_dropdown):
        selected_metric = db1_metric_dropdown2.value
        update_db1_charts(db1_source2, db1_plot2,db1_scorecard_text2,selected_date,selected_metric)

    # Updates for UI tables  (if date updated by user)
    if (event.obj == db1_date_dropdown):
        update_db1_table(db1_source3)
        update_db1_table(db1_source4)
		
		
		
"""
Dashboard 2 (Detail) 
Shows the breakdown of metric by different location (regional or local)

User can pick a date (as of), metric and breaking down the data by regional or local
There will be a bar graph showing the average price as of the date chosen + reference line for U.S. Average
There will also be a line graph for showing the price change over time for the region/local

Main UI elements: 
    db2_metric_dropdown1 - Dropwown menu to choose the 1st metric
    db2_loctype_dropdown2 - Dropwown menu to choose between Regional or Local, which changes the data breakdown method 
    db2_date_dropdown  - Dropwown menu to choose the as of date
    db2_bar1 - Bar showing the average price for the chosen date & metric, in addition to horizontal reference line for national average price
    db2_line1 - Line graph showing the price change over time for the chosen metric broken down by region or local 

"""

# Initialization of all elements

#Dataframe
db2_initial_df1 = main_df[(main_df['name'] == DEFAULT_METRIC_1) & 
                 (main_df['location type'] == DEFAULT_LOCTYPE) &
                 (main_df['date'] == initial_date)].sort_values(by='value',ascending=False)

db2_initial_df2 = main_df[(main_df['name'] == DEFAULT_METRIC_1) & 
                 ((main_df['location type'] == DEFAULT_LOCTYPE) + (main_df['location type'] == 'National')) &
                 (main_df['date'] >= initial_date - relativedelta(years=3)) & 
                 (main_df['date'] <= initial_date)].sort_values(by='date',ascending=True)




#Bar graph
db2_source1 = ColumnDataSource(db2_initial_df1)
db2_bar1 = figure(x_range=db2_initial_df1['location'], title=f'{DEFAULT_METRIC_1} as of {initial_date.strftime("%Y-%m")}', 
           x_axis_label='Location', y_axis_label='Values',  aspect_ratio=1.5,min_height=200, min_width=400)
db2_bar1.vbar(x='location', top='value', width=0.5, color="blue",source=db2_source1)
db2_bar1.xaxis.major_label_orientation = -0.5
db2_bar1.add_tools(HoverTool(tooltips=[('Location', '@location'), ('Value', '@value')]))

#Add reference line to bar graph for national average
db2_national_value = main_df.loc[(main_df['name'] == DEFAULT_METRIC_1) & 
                 (main_df['location type'] == 'National') &
                 (main_df['date'] == initial_date),'value'].values[0]
reference_line = Span(location=db2_national_value, dimension='width', line_color='red', line_width=2, line_dash='dashed')
db2_bar1.add_layout(reference_line)
label = Label(x=len(db2_initial_df1), y=db2_national_value, text='National average', text_font_size='10pt', text_color='red', text_baseline='bottom', text_align ='right')
db2_bar1.add_layout(label)

db2_plot1_legend_list = []
#Line graph
db2_plot1 = figure(x_axis_type='datetime', title=f'{DEFAULT_METRIC_1} over time', aspect_ratio=1.5,min_height=200, min_width=400)
locations = db2_initial_df2['location'].unique()
#Add line for each location
for i, location in enumerate(locations):
    location_df = db2_initial_df2[db2_initial_df2['location'] == location]
    db2_source2 = ColumnDataSource(location_df)
    l = db2_plot1.line('date', 'value', source=db2_source2, line_width=2, color=palette20[i % len(palette20)])
    db2_plot1_legend_list.append((location,[l]))

db2_plot1.add_tools(HoverTool(tooltips=[('Location', '@location'), ('Date', '@date{%F}'), ('Value', '@value')], formatters={'@date': 'datetime'}))
db2_plot1_legend = Legend(items=db2_plot1_legend_list)
db2_plot1_legend.label_text_font_size = '8pt'
db2_plot1_legend.spacing = 0
db2_plot1.add_layout(db2_plot1_legend, 'right')

#dropdown 
db2_metric_dropdown1 = pn.widgets.Select(name='Metric 1',options=metrics, value=DEFAULT_METRIC_1)
db2_date_dropdown = pn.widgets.Select(name='As of',options=dates, value=initial_date_s,width=150)
db2_loctype_dropdown = pn.widgets.Select(name='Regional/Local',options=["Regional","Local"], value=DEFAULT_LOCTYPE,width=150)
        
        
        
def db2_updates(event):
    """ 
    In:
        event - not used
    Out:
        None, but the elements will be updated
    """
    
    #Get values from dropdowns
    selected_date =pd.to_datetime(db2_date_dropdown.value, format='%Y-%m')
    selected_metric = db2_metric_dropdown1.value
    selected_loctype = db2_loctype_dropdown.value

    #Get relevant data, to be used for bar graph + line graph
    db2_filtered_df1 = main_df[(main_df['name'] == selected_metric) & 
                     (main_df['location type'] == selected_loctype) &
                     (main_df['date'] == selected_date)].sort_values(by='value',ascending=False)
    
    db2_filtered_df2 = main_df[(main_df['name'] == selected_metric) & 
                     ((main_df['location type'] == selected_loctype) + (main_df['location type'] == 'National')) &
                     (main_df['date'] >= selected_date - relativedelta(years=3)) & 
                     (main_df['date'] <= selected_date)].sort_values(by='date',ascending=True)
    
    #Update bar graph
    db2_source1.data = ColumnDataSource.from_df(db2_filtered_df1)
    #Update axis info
    db2_bar1.x_range.factors = db2_filtered_df1['location']
    db2_bar1.y_range.start = 0
    db2_bar1.title = f'{selected_metric} as of {initial_date.strftime("%Y-%m")}'

    #Update reference line for U.S. Average
    db2_bar1.renderers = db2_bar1.renderers[:2]  # Remove old regression line, keep original line
    db2_national_value = main_df.loc[(main_df['name'] == selected_metric) & 
                     (main_df['location type'] == 'National') &
                     (main_df['date'] == selected_date),'value'].values[0]
    reference_line.location = db2_national_value
    label.x = len(db2_filtered_df1)
    label.y = db2_national_value
    
    
    #For line graph, clear all lines and recreate them
    db2_plot1.renderers=[]
    db2_plot1.legend.items = []
    db2_plot1.legend.clear()
    db2_plot1_legend_list = []
    locations = db2_filtered_df2['location'].unique()
    for i, location in enumerate(locations):
        location_filtered_df = db2_filtered_df2[db2_filtered_df2['location'] == location]
        db2_source2 = ColumnDataSource(location_filtered_df)
        l=db2_plot1.line('date', 'value', source=db2_source2, line_width=2, color=palette20[i % len(palette20)])
        db2_plot1_legend_list.append((location,[l]))
    #Update x-axis and y-axis range based on new values
    db2_plot1.x_range.update()
    db2_plot1.y_range.update()
    db2_plot1.title=f'{selected_metric} over time'
    db2_plot1_legend = Legend(items=db2_plot1_legend_list)
    db2_plot1_legend.label_text_font_size = '8pt'
    db2_plot1_legend.spacing = 0
    db2_plot1.add_layout(db2_plot1_legend, 'right')
    


"""
Dashboard 3 (Comparison) 
Allows the user to pick two metrics to see how the compare against each other

User can pick a date range and two metrics to compare
There will be a line graph showing the metrics over time
There will also be a scatter plot using the first metric as x-axis and second metrics as y-axis, in addition to a regression line to see the correlation

Main UI elements: 
    db3_date_from_dropdown - Dropwown menu to choose the date from 
    db3_date_to_dropdown - Dropwown menu to choose the date to
    db3_metric_dropdown1 - Dropwown menu to choose the 1st metric
    db3_metric_dropdown2 - Dropwown menu to choose the 2nd metric

"""


# Initialization of all elements
initial_date_to = initial_date
initial_date_to_s = initial_date_s
initial_date_from =  initial_date - relativedelta(years=3)
initial_date_from_s = initial_date_from.strftime('%Y-%m')

#dropdown menus
db3_date_from_dropdown = pn.widgets.Select(name='Date from',options=dates, value=initial_date_from_s,width=150)
db3_date_to_dropdown = pn.widgets.Select(name='Date to',options=dates, value=initial_date_to_s,width=150)
db3_metric_dropdown1 = pn.widgets.Select(name='Metric 1',options=metrics, value=DEFAULT_METRIC_1)
db3_metric_dropdown2 = pn.widgets.Select(name='Metric 2',options=metrics, value=DEFAULT_METRIC_2)


#Dataframes
db3_initial_df1 = main_df[(main_df['name'].isin([DEFAULT_METRIC_1,DEFAULT_METRIC_2])) & 
                 (main_df['location type'] == 'National') &
                 (main_df['date'] >= initial_date_from) & 
                 (main_df['date'] <= initial_date_to)].sort_values(by='date',ascending=True)
db3_initial_df2 = pd.pivot_table(db3_initial_df1,index='date',values='value',columns='name').reset_index()


#Line graph
db3_plot1 = figure(x_axis_type='datetime', title='Metric Values Over Time',  aspect_ratio=1.5,min_height=200, min_width=400)
for i, metric in enumerate([db3_metric_dropdown1.value, db3_metric_dropdown2.value]):
    metric_filtered_df = db3_initial_df1[db3_initial_df1['name'] == metric]
    source = ColumnDataSource(metric_filtered_df)
    db3_plot1.line('date', 'value', source=source, legend_label=metric, line_width=2, color=palette10[i % len(palette10)])
db3_plot1.add_tools(HoverTool(tooltips=[('Metric','@name'),('Date', '@date{%F}'), ('Value', '@value')], formatters={'@date': 'datetime'}))

#Scatter plot
db3_source1 = ColumnDataSource(db3_initial_df2)
db3_scatter1 = figure(title="Scatter Plot", x_axis_label=db3_metric_dropdown1.value, y_axis_label=db3_metric_dropdown2.value,  aspect_ratio=1.5,min_height=200, min_width=400)
scatter = db3_scatter1.scatter(x=db3_metric_dropdown1.value, y=db3_metric_dropdown2.value, source=db3_source1)
db3_scatter1.add_tools(HoverTool(tooltips=[('Metric 1',db3_metric_dropdown1.value),('Metric 2',db3_metric_dropdown2.value),('Date', '@date{%F}'), ('Value', '@value')], formatters={'@date': 'datetime'}))
#Add regression line to scatter plot
add_regression_line(db3_source1, db3_scatter1,db3_metric_dropdown1.value,db3_metric_dropdown2.value)



# Define the callback function
def db3_updates(event):
    """ 
    In:
        event - not used
    Out:
        None, but the elements will be updated
    """

    #Get (and convert) selected dates & metrics
    selected_date_from =pd.to_datetime(db3_date_from_dropdown.value, format='%Y-%m')
    selected_date_to =pd.to_datetime(db3_date_to_dropdown.value, format='%Y-%m')
    selected_metric1 = db3_metric_dropdown1.value
    selected_metric2 = db3_metric_dropdown2.value
    

    #Get data based on chosen metrics and dates
    db3_filtered_df1 = main_df[(main_df['name'].isin([selected_metric1,selected_metric2])) & 
                     (main_df['location type'] == 'National') &
                     (main_df['date'] >= selected_date_from) & 
                     (main_df['date'] <= selected_date_to)].sort_values(by='date',ascending=True)
    db3_filtered_df2 = pd.pivot_table(db3_filtered_df1,index='date',values='value',columns='name').reset_index().sort_values(by=selected_metric1)

    #Update line graph
    #Remove all renders first and recreate
    db3_plot1.renderers=[]
    db3_plot1.legend.items = []
    db3_plot1.legend.clear()    
    for i, metric in enumerate([selected_metric1, selected_metric2]):
        metric_filtered_df = db3_filtered_df1[db3_filtered_df1['name'] == metric]
        source = ColumnDataSource(metric_filtered_df)
        db3_plot1.line('date', 'value', source=source, legend_label=metric, line_width=2, color=palette10[i % len(palette10)])
    db3_plot1.add_tools(HoverTool(tooltips=[('Metric','@name'),('Date', '@date{%F}'), ('Value', '@value')], formatters={'@date': 'datetime'}))

    #Update scatter plot
    db3_scatter1.xaxis.axis_label = selected_metric1
    db3_scatter1.yaxis.axis_label = selected_metric2
    scatter.glyph.x = selected_metric1
    scatter.glyph.y = selected_metric2
    db3_source1.data = db3_filtered_df2
    db3_scatter1.renderers = [scatter]
    add_regression_line(db3_source1, db3_scatter1,selected_metric1,selected_metric2)
    
db1_metric_dropdown1.param.watch(db1_updates, 'value')
db1_date_dropdown.param.watch(db1_updates, 'value')
db1_metric_dropdown2.param.watch(db1_updates, 'value')
db1_plot1.legend.location = "bottom_right"


db2_metric_dropdown1.param.watch(db2_updates, 'value')
db2_date_dropdown.param.watch(db2_updates, 'value')
db2_loctype_dropdown.param.watch(db2_updates, 'value')

db3_metric_dropdown1.param.watch(db3_updates, 'value')
db3_metric_dropdown2.param.watch(db3_updates, 'value')
db3_date_from_dropdown.param.watch(db3_updates, 'value')
db3_date_to_dropdown.param.watch(db3_updates, 'value')



# Create a Panel layout
dashboard1 = pn.Column(
    Div(text="<h1>U.S. Average price data - Summary</h1> *Data based on national average </br>"),
    pn.Row(
        db1_date_dropdown,
        db1_metric_dropdown1,
        db1_metric_dropdown2,
    ),
    pn.Spacer(height=80),
    pn.Row(
        pn.Column(
            pn.Row(
                db1_scorecard_text1,
                db1_plot1,
            ),
        pn.Spacer(height=40),

            pn.Row(
                db1_scorecard_text2,
                db1_plot2,
            ),
        ),
        pn.Spacer(width=30),
        pn.Column(
                Div(text="Top Year-Over-Year Increases",styles={'text-align':'center','color':'darkgreen'}),
                db1_table1,
                pn.Spacer(height=100),
                Div(text="Top Year-Over-Year Decreases",styles={'text-align':'center','color':'darkred'}),
                db1_table2,
        ),
    )
)

dashboard2 = pn.Column(
    Div(text="<h1>U.S. Average price data - Detail</h1>"),
    pn.Row(
        db2_date_dropdown,
        db2_metric_dropdown1,
        db2_loctype_dropdown,
    ),
    pn.Spacer(height=80),
    pn.Row(
        db2_bar1,
        pn.Spacer(width=30),
        db2_plot1
    ),
)

dashboard3 = pn.Column(
    Div(text="<h1>U.S. Average price data - Comparison</h1> *Data based on national average </br>"),
     pn.Row(
        db3_date_from_dropdown,
        db3_date_to_dropdown,
        db3_metric_dropdown1,
        db3_metric_dropdown2,
    ),
    pn.Spacer(height=80),
    pn.Row(
        db3_plot1,
        pn.Spacer(width=30),
        db3_scatter1
    ),
)


main = pn.Tabs(
    ('Summary', dashboard1),
    ('Detail', dashboard2),
    ('Comparison', dashboard3),
    
)

template = pn.template.MaterialTemplate(title='[Demo app] US Bureau of labor statistics - average price data')
template.main.append(main)

template.servable()
