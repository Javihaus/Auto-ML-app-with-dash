import flask
import pandas as pd
import numpy as np
import scipy
from scipy.cluster import hierarchy as hc
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Group, Scheme
from dash.dependencies import Input, Output
import dash_daq as daq
from sklearn.svm import LinearSVC
from sklearn.inspection import partial_dependence
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, RobustScaler
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import statsmodels.api as sm
import xgboost as xgb
from xgboost import XGBRegressor 
import gc
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from datetime import timedelta

external_stylesheets = [dbc.themes.LITERA]
server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)
colors = {'background': '#F1EEE6'}

#Load  data 
def load_csv(path):
    return pd.read_csv(path, 
                       engine='c', 
                       parse_dates=True, 
                       infer_datetime_format=True, 
                       low_memory=False)

def min_max(df):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    df_scaled = min_max_scaler.fit_transform(df[[item for item in df.drop(columns='date').columns]])
    df_scaled= pd.DataFrame(df_scaled)
    df_scaled.columns = df.drop(columns=['date']).columns
    df = df_scaled.merge(df['date'], left_index=True, right_index=True)
    df = df.mask(df==0).fillna(method='backfill')
    return df

def fix_missing(df):
    df = df.fillna(method='ffill')
    df = df.fillna(method='backfill')
    return df

path='gimnasio.csv'
# we store original data set as df_original
df_original = load_csv(path)
df_original = df_original.mask(df_original==0).fillna(method='backfill')
df_original['date']=pd.to_datetime(df_original['date'])
df_original = df_original.iloc[:2451]
df = df_original
# preparing dataframe for classification
df = fix_missing(df)
df = min_max(df)
df['date']=pd.to_datetime(df['date'])
col = df.drop(columns='date').columns
for i in col:
    df[i] = df[i].map('{:,.3f}'.format)
    df
df[col] = df[col].astype('float64')

source =pd.DataFrame(columns=['Features', 'Explanation', 'Range'])
source['Features']=df.columns
source['Explanation'][0]='Twitter Intensity analysys in word "economia" (daily data). Source: Twitter'
source['Explanation'][1]='Twitter Polarization analysys in word "economia" (daily data). Source: Twitter'
source['Explanation'][2]='Twitter Subjectivity analysys  in word "economia"(daily data). Source: Twitter'
source['Explanation'][3]='Twitter Intensity analysys in word "gimnasio"(daily data). Source: Twitter'
source['Explanation'][4]='Twitter Polarization analysys in word "gimnasio" (daily data). Source: Twitter'
source['Explanation'][5]='Google trends word "gimnasio". Source: Google (monthly data)'
source['Explanation'][6]='Google trends word "sanitas". Source: Google (monthly data)'
source['Explanation'][7]='Google trends word "adeslas". Source: Google (monthly data)'
source['Explanation'][8]='Google trends word "decathlon". Source: Google (monthly data)'
source['Explanation'][9]='IBEX35 close. Source: Yahoo Finance (daily data)'
source['Explanation'][10]='Growth of fitness industry in spain (year data). Source: Statista 2020'
source['Explanation'][11]='date'
source['Range'][0]='-1 to +1'
source['Range'][1]='-1 to +1'
source['Range'][2]='-1 to +1'
source['Range'][3]='-1 to +1'
source['Range'][4]='-1 to +1'
source['Range'][5]='0 - 100%'
source['Range'][6]='0 - 100%'
source['Range'][7]='0 - 100%'
source['Range'][8]='0 - 100%'
source['Range'][9]='thousands'
source['Range'][10]='0 to 1'
source['Range'][11]='Date Y/M/D'

available_indicators = df.columns.unique()
variable_list = df.drop(columns=['date']).columns.unique().tolist()
dfa = {'Logistic Regression': 
                [LogisticRegression(fit_intercept=True,max_iter=1000, random_state=0)],
       'Linear Support Vector': 
               [LinearSVC(random_state=0, tol=1e-5)],
       #'Random Forest Classifier': 
       #         [RandomForestClassifier(n_estimators=300,
       #                      max_depth=10,
       #                      criterion='entropy',
       #                      bootstrap=True,
       #                      n_jobs=-1,
       #                      random_state=0,
       #                      oob_score=True)],
       #'Gradient Boosting Classifier':
       #         [GradientBoostingClassifier(random_state=0)]
        }
dfa = pd.DataFrame(dfa, columns = ['Logistic Regression',
                                   'Linear Support Vector',
                                   #'Random Forest Classifier',
                                   #'Gradient Boosting Classifier'
                                  ])
algorithm = dfa.columns.unique().tolist()



#dash/html code
app.layout = html.Div([
    dcc.Tabs(id="tabs-with-classes",
             parent_className='custom-tabs',
             className='custom-tabs-container',
             children=[
    dcc.Tab(label='DATA SOURCE',
            value='tab-1',
            className='custom-tab',
            selected_className='custom-tab--selected',
            children=[
    html.H3('Fitness industry in Spain',
            style={
                'font-family': 'Open Sans',
                'padding':'25px', 
                'font-weight': '600', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'5%',
            
        }),
    html.P('Copyright Javier Marín (2021). MIT license. ' ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'3%',            
        }),
    html.P('We have used open data sources to describe the fitness industry in Spain in 2020. We want to know more about the growing perspectives of this industry according users insights about gyms. We have selected only a few features related with consumers attitudes and motivations regarding this industry. Below you can see each feature explanation and source.  ' ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'3%',            
        }),
                
    dash_table.DataTable(id='table1',
           columns=[{"name": i, "id": i} for i in source.columns],
           data=source.to_dict('records'),
           fixed_rows={'headers': True},
           style_table={'width':'70%',
                         'margin-left':'15%',
                         'margin-bottom':'3%'},
           style_cell={'textAlign': 'center',
                        'font-family': 'Open Sans',
                        'height':'auto',
                        'font_size': '15px'
                        },
                ),
    dash_table.DataTable( id='table2',
            columns=[{"name": i, "id": i, 'format': Format(precision=2)} for i in df.columns],
            data=df.to_dict('dict'),
            page_size=10,
            fixed_rows={'headers': True},
            style_table={'width':'90%',
                         'margin-left':'5%'},
            style_cell={ 'minWidth': 105, 
                        'maxWidth': 200, 
                        'height':'auto',
                        'width': 200,
                        'overflow': 'hidden',
                        'margin-left': '1%',
                        'font_size': '15px',
                        'font-family': 'Open Sans',
                        },
                )
        ]),
    dcc.Tab(label='DATA CLASSIFICATION',
                value='tab-2',
                className='custom-tab',
                selected_className='custom-tab--selected',
                children=[
    html.Div([
        html.H3('Introduction to classification algorithms',
            style={
                'font-family': 'Open Sans',
                'padding':'25px', 
                'font-weight': '600', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'5%',
            
        }),
        html.P('We start with a dataset that includes several time-series of different "variables" or "features". With this data set we ara going to do two analysis: "predict" the value of this features in the future (look at the time series tab) or, know more about how this features are correlated between them. The idea is we have a special interest feature and we wonder if we could predict this feature (we will call "target" feature) as a combination of the rest of features. , so we could predict this "target" by knowing the rest of features evolution. This problem becomes a classification problem and gives us lots of insights from our data' ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
            
        }),
        html.Img(src='assets/Introduction.png', style={
                'display': 'block',
                'margin-left': '11%', 
                'margin-right': 'auto', 
                'width': '75%', 
                'margin-bottom':'1%', 
        }),
        html.P('To make it simplier, we will divide the target feature in two parts: high values (values over average value) and low values (values below average value). So we have a binary classification problem.' ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
            
        }),
        html.H3('1. General overview ',
            style={
                'font-family': 'Open Sans',
                'padding':'25px', 
                'font-weight': '600', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'2%',
            
        }), 
        html.P('First of all, we need to have a general view of variables in dataset. Following you can see a plot with the comparion between variables pairs to start having an overview. As all this vars are time dependent, you will also see time-dependency plots of the choosen variables. ' ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
            
        }),
        html.Div([ html.Div([
            html.Div([
                html.P('Select variable for axis x in the plot:' ,
            style={
                'width': '125%',
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'5px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'30%',
            
        }),
                ], className="two columns"),
            
        html.Div([
            html.P('Select variable for axis y in the plot:' ,
            style={
                'width': '125%',
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'5px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'135%',
            
                })
                ], className="two columns")
            ], className="row")
        ]),
              
        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=df.columns[1] 
        )],
            style={
                'width': '25%', 
                'display': 'inline-block',
                'font-family': 'Open Sans', 
                'padding': '25px',
                'background-color': '#F1EEE6', 
                'margin-left': '50px',
                
            }),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=df.columns[2]
        )], 
            style={
                'width': '25%', 
                'float': 'right', 
                'display': 'inline-block', 
                'font-family': 'Open Sans', 
                'padding': '25px', 
                'background-color': '#F1EEE6',
                'margin-right': '100px' })
        ]),
        html.Div(children=[
        dcc.Graph(id='indicator-graphic')
        ], 
            style={
                'width':'84%', 
                'height': '60%', 
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE', 
                'border-radius': '5px', 
                'background-color': '#F1EEE6' , 
                'margin-left': '75px', 
                'margin-bottom': '70px'
         }),
        
        html.P('We have represented pairs of variables in a contour plot. We have seen how variables are correlated by pairs. All conclusions from this visual exploration are needed to understant a bit more the dataset and the relation betwen variables.  We can include in this comparison the feature we are going to select as "target" and plot aginst other features to see partial dependences in the data set. Rememember that or problem is that we want to know the dependece of all features together with the target variable. If we plot the relation of a feature with our target its a partial dependence because give us only partial information. ' ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
            
        }), 
        html.H3('2. Data correlation' ,
            
            style={
                'font-family': 'Open Sans', 
                'padding':'15px', 
                'font-weight': '600',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'5%',
            
        }),
        html.P('Choose the features you consider are the most interesting, or remove the ones you think are not going to be useful. We will start plotting its correlation' ,
            
            style={
                'font-family': 'Open Sans', 
                'font-size':'20px',
                'padding':'15px', 
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'3%',            
                  }
               ),    
        html.Div([
        dcc.Dropdown(
                id="variable_list",
                options=[{"label": i, "value": i} for i in variable_list],
                value=variable_list[:],
                multi=True,            
                )],
            style={
                'width': '30%', 
                'display': 'inline-block', 
                'font-family': 'Open Sans',
                'padding': '25px', 
                'background-color': '#F1EEE6',
                'margin-left': '1%'
                    }
                ),        
        html.Div([
        dcc.Graph(id='correlation'        
        )],
            style={
                'width': '60%', 
                'display': 'inline-block', 
                'box-shadow': '10px 5px 12px #DBCBBE',
                'border-radius': '5px', 
                'background-color': '#F1EEE6',
                'margin-left': '25px',
                'right': '50px',
                'margin-bottom': '1%'
                    }
                ),                   
        html.H3('3. Features importance ',            
            style={ 
                'font-family': 'Open Sans',
                'padding':'25px', 
                'font-weight': '600', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'3%',
                    }
            ),           
            html.P('To predict a target feature as a combination of the rest of features, select target and we will calculate the general variables importance to predict this target in a general classification.' ,            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
                    }
                ),
    html.P('First of all, select the target feature: ' ,            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'6%',
                'margin-top':'1%',
                    }
                ),  
        html.Div([
            dcc.Dropdown(
                id='target',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=df.columns[1]        
                )],
            style={
               'width': '40%', 
               'display': 'inline-block', 
               'font-family': 'Open Sans',
               'background-color': '#F1EEE6',
               'padding': '25px', 
               'margin-left': '50px'
                    }
                ),
    
        html.P('We have a N-dimensional space (we have 11 features including target). A 1-dimensional space is a line, 2-dimensional space is a plane (for example a 2D plot) and a 3-dimensional space is similar to a cube (a 3D plot). But for N > 3, we do not have direct techniques to visually explore this data. And this is very important for us because would give us an idea if data can be easily divided in two parts for our data visualization (remember, target-High and target-Low) ' ,            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'6%',
                'margin-top':'1%',
                    }
                ),
         html.Img(src='assets/Manifold.png', style={
                'display': 'block',
                'margin-left': 'auto', 
                'margin-right': 'auto', 
                'width': '80%', 
                'margin-bottom':'5%', 
        }),
        html.P('There are special algorithms that performs what is called "dimensionality reduction". In our case we would like to have all data from our 11 dimensional space embeded into a 2 dimensional space. So we could visualize interesting information from data as for example if is separable (classificable). For doing that we are going to use a very interesting algorithm: "t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING" that takes a set of points in a high-dimensional space and find a "faithful" representation of these points in a lower-dimensional space. t-sne algorithm is very useful in Machine Learning and also for quantum physics, genetics, thermodynamics, and even gambling. How does it work? Maths behind are hard, but intuitively we can say we create a new low dimensionality representation of this high-dimensionality data by calculating distances betwen points and measuring how much information we learn with our new representation having already known about the data. Algortihm minimizes this amount of information difference in order to be minimum. Ideally the information amount should be the same, but in practice the excercise of creating the new representation forces us to make several assumptions. The idea is that we do not have to gain information from original data and new representation has to be very close in information value, otherwise we are creating a new dataset. Alorithms task is to get another representation having very similar information than original dataset. ' ,            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'6%',
                'margin-top':'1%',
                'margin-bottom':'2%',
                    }
                ),
        html.Div([
        dcc.Graph(id='tsne'        
                )], 
            style={
                'width': '80%', 
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE', 
                'border-radius': '5px',
                'background-color': '#F1EEE6', 
                'margin-left': '10%',
                'margin-bottom': '1%'
                    }
                ),
        html.P('Now you have a dataset visual representation for a binary classification on your target variable in HIGH and LOW values. If the points are not clearly separated, binary classification accuracy is not going to be interesting. If you see t-sne algorithm separates HIGH points from LOW points quite clearly, you will perform this classification with success. Also youcan see how algorithm sometimes tends to group points into clusters and separate them. These are hidden patterns in your data set ¡ ',            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
                    }
                ),
        html.P('All features in dataset has not the same importance when doing a binary classification on your target. To know wich features has more importance is interesting because it can tell you from wich variables your target depends more. But also having this information is imoportant cause alows us eliminate variables having residual importance when performing the classification algorithms. We can save computational resources ¡. Let check in our dataset wich variables are more relevant to perform our classification task:',            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
                    }
                ),
        html.Div([
        dcc.Graph(id='feature_importance'        
                )], 
            style={
                'width': '80%', 
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE', 
                'border-radius': '5px',
                'background-color': '#F1EEE6', 
                'margin-left': '10%',
                'margin-bottom': '1%'
                    }
                ),  
    html.P('We can see now wich variable influences more the classification. We can decide now remove some of these variables. But, ¿how can be sure we are removing not too much variables to keep the classification accuracy? If we delete too much variables, we are reducing original information in our dataset. But there is an optimal number variables we can reduce to keep all information dataset. Lets check ',            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
                    }
                ),
     html.P('In the vertical axis we can see the percentage from the original data and in the horizantal axis we see the number of features. With all features (100%) we have of course the original data (we keel 100% infomration) . But we can see that removing some features we still keep more than 90% of our original data information. The ideal minimum features number would have a correspodence betwen 90 and 95% of original data.',            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
                    }
                ),
    html.Div([
        dcc.Graph(id='variance'        
                )], 
            style={
                'width': '80%', 
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE', 
                'border-radius': '5px',
                'background-color': '#F1EEE6', 
                'margin-left': '10%',
                'margin-bottom': '2%'
                    }
                ),  
      html.P('Another interesting calculation to do in our data is what we have done with features but grouping features in clusters. The dataset information is build by grouping these clusters. Inside a cluster, we can see features close to each other. So we use this property when we remove features (the idea is not remove all features from the same cluster ¡)',
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
                    }
                ),
    html.Img(src='assets/Dendogram.png', style={
                'display': 'block',
                'margin-left': 'auto', 
                'margin-right': 'auto', 
                'width': '70%', 
                'margin-bottom':'2%', 
        }),
    html.Div([
        dcc.Graph(id='dendogram'        
                )], 
            style={
                'width': '80%', 
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE', 
                'border-radius': '5px',
                'background-color': '#F1EEE6', 
                'margin-left': '10%',
                'margin-bottom': '2%'
                    }
                ),  
      html.Div([ html.H3('4.  Classification' ,            
            style={
                'display': 'inline-block', 
                'font-family': 'Open Sans', 
                'padding':'25px', 
                'font-weight': '600' ,
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'5%',
                    }
                ),
        html.P(' We want to answer the following question: With selected target, ¿wich accuracy could we get predicting the target value combining the rest of variables? But first we have to choose the variables you want to work with, together with the target variable. Remember we have shown above wich features are more important for classification, and also wich ones offers similar information to classification algorithms' ,           
            style={ 
                'display': 'inline-block', 
                'font-size':'20px',
                'font-family': 'Open Sans', 
                'padding':'25px', 
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'0.5%', 
                    }
               ),
        html.P('With all informtion you have from above, now select the variables you want to use for running classification (you have to select as well the target feature ¡): ' ,            
            style={ 
                'font-family': 'Open Sans',
                'font-size':'20px',
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'4%',
                'margin-top':'1%',
                    }
                ),        
        html.Div([
        dcc.Dropdown(
                id="features_list",
                options=[{"label": i, "value": i} for i in variable_list],
                value=variable_list[:],
                multi=True,            
                )],
            style={
                'width': '95%', 
                #'display': 'inline-block', 
                'font-family': 'Open Sans',
                'padding': '25px', 
                'background-color': '#F1EEE6',
                'margin-left': '2%'
                    }
                ),
        html.P('Now we will split target values in only two: from targets minimum value to mean value - we will label it as LOW-, and from mean value to maximum - label as HIGH -. Our classification problem will classify betwen this two sets (LOW and HIGH). We want to check if the rest of the features can tell us if target will belong to set LOW or to set HIGH. Also we want to know the accuracy of this classification.' ,           
            style={ 
                'display': 'inline-block', 
                'font-family': 'Open Sans', 
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'0.5%', 
                    }
               ),
        html.P('We will start with some basic linear algorithms. Choose an algorithm from the list : ' ,
            style={ 
                'display': 'inline-block', 
                'font-family': 'Open Sans', 
                'font-size':'20px',
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'4%',
                'margin-bottom':'3%',
                    }
               )
               ]), 
        html.Div([
            dcc.Dropdown(
                id='algorithm',
                options=[{'label': i, 'value': i} for i in algorithm],
                value='Logistic Regression'        
                )
                 ],
            style={
                'width': '40%', 
                'display': 'inline-block',
                'font-family': 'Open Sans', 
                'background-color': '#F1EEE6', 
                'margin-left': '30%',
                'margin-bottom' : '5%'
                    }
                ),
        html.Div([
        dcc.Graph(id='classification'        
        )],            
            style={
                  'width': '88%', 
                  'display': 'inline-block', 
                  'box-shadow': '10px 5px 12px #DBCBBE',
                  'border-radius': '5px',
                  'background-color': '#F1EEE6', 
                  'margin-left': '6%',
                  'margin-bottom':'3%'
                    }
                ),
     html.P('We are going to look why and how the algorithm has concluded this. First we look at the features that have been more important for each algorithm decission. ' ,
            style={ 
                'display': 'inline-block', 
                'font-family': 'Open Sans', 
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
                    }
               ),
    html.Div([dcc.Graph(id='coefficients'       
                )],            
            style={
                  'width': '88%', 
                  'display': 'inline-block', 
                  'box-shadow': '10px 5px 12px #DBCBBE',
                  'border-radius': '5px',
                  'background-color': '#F1EEE6', 
                  'margin-left': '6%',
                  'margin-bottom':'3%'
                    }
                ),
        html.P('We have seen how much classification depends on each feature. If we want to know how each individual feature affects the result, we need to know how "partially" depends the clasification from this feature (why is called "partial dependence"). In the following plot we show this partial dependence. You have to select a feature that you have choosen for target classification and you will se how how target classification depends on this single feature.' ,
            style={ 
                'display': 'inline-block', 
                'font-family': 'Open Sans', 
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-bottom':'2%',
                    }
               ),
        html.Div([
            dcc.Dropdown(
                id='variable',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=df.columns[2]        
                )],
            style={
               'width': '40%', 
               'display': 'inline-block', 
               'font-family': 'Open Sans',
               'background-color': '#F1EEE6',
               'margin-left': '50px', 
               'margin-bottom': '34%'
                    }
                ),
        html.Div([
            dcc.Graph(id='partial_dependence'       
                )],            
            style={
                  'width': '47%', 
                  'display': 'inline-block', 
                  'box-shadow': '10px 5px 12px #DBCBBE',
                  'border-radius': '5px',
                  'background-color': '#F1EEE6', 
                  'margin-left': '2%',
                  #'margin-top':'15%',
                  'margin-bottom':'2%'
                    }
                ),
        ]),
        dcc.Tab(label='TIME SERIES ANALISYS',
                value='tab-3',
                className='custom-tab',
                selected_className='custom-tab--selected',
                children=[
        html.H3('Decoding timeseries',
            style={
                'font-family': 'Open Sans',
                'padding':'25px', 
                'font-weight': '600', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'5%',
            
        }),
        html.P('Time series forecasting have become the "holy grail" for most Machine Learning scientists. Trends estimation demand is pulling scientists to deal with more and more complex models. Uncertainty, in its metaforic meaning, have become one of the most used words in business strategy. Uncertainty, in its mathematical definition, have allways been there, but assuming high levels of it forces us to use more sophisticated models to forecast time series. Apart from forecasting as output, time series "encodes" a lot of information that can be very useful.' ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'3%',            
        }),
        html.P('Select a feature to start visualizing time series: ' ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'2%',            
        }),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column2',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=df.columns[6] 
        )],
            style={
                'width': '30%', 
                'display': 'inline-block',
                'font-family': 'Open Sans', 
                'background-color': '#F1EEE6', 
                'margin-left': '35%',
                'margin-bottom':'1%'
                
            }),
        html.Div([
        dcc.Graph(id='y-time-series2'
        
        )],
            style={
                'width': '80%',
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE',
                'border-radius': '5px', 
                'background-color': '#F1EEE6', 
                'margin-left':'10%'
              }),
        html.Div([
        dcc.Graph(id='y-time-series21'
        
        )],
            style={
                'width': '40%',
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE',
                'border-radius': '5px', 
                'background-color': '#F1EEE6', 
                'margin-left':'10%'
              }),
        html.Div([
        dcc.Graph(id='y-time-series22'
        
        )],
            style={
                'width': '40%',
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE',
                'border-radius': '5px', 
                'background-color': '#F1EEE6', 
                'margin-right':'5%'
              }),  
                    
        html.P('One of the most important aspects of time series is its stationarity. Time series that shows similar patterns over the time are called stationaries. In Business Intelligence applications the most of time series are stationary. For example, product sales has similar trends every week, month or year. A given product sales can increase during the weekend (from Friday to Monday) because people has more free time and go shoping massively. The same happens every month (people spend more money at the begining of the month for example). An also happens during the year, depending on products, where some seasons concetrate a high amount of sales. And this is happening over and over again. We call this starionarity. In the plots below we can see the same feture you have selected for analysis but with weekly and monthly values. All three plots shows differente stationary behaviours.  ' ,           
            style={ 
                'display': 'inline-block', 
                'font-size':'20px',
                'font-family': 'Open Sans', 
                'padding':'25px', 
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'0.5%', 
                    }
               ),
        html.Div([
        dcc.Graph(id='autocorrelation'
        
        )],
            style={
                'width': '80%',
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE',
                'border-radius': '5px', 
                'background-color': '#F1EEE6', 
                'margin-left':'10%'
              }), 
        html.P('We can plot variable autocorrelation with different time lags or time steps. Plot starts in autocorrelation value of 1, corresponding to time a lag of 0. Then drops down to 0 as time lags are growing. In a feature with no autocorrelation, plot will be oscilating around 0 with very small peaks (with a peak value very near 0). If this peaks are high or the plot does not rapidally drops to near zero, time series has autocorrelation. Where peaks are very high will show that in this time lag there is a high autocorrelation. High peaks can be found at any point, but usually in time series like product sales, this peaks or high values appears in time lags of 7 (week), 30 (month), 120 (quarter) or 360 (year). It means every 7-30-120-360 the same patterns appears over and over again. ',
               style={
                'display': 'inline-block', 
                'font-size':'20px',
                'font-family': 'Open Sans', 
                'padding':'25px', 
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'0.5%', 
                    }
               ),
        html.P('¿Why is important autocorrelation? Because when is high, is not possible to predict feature values with ordinary statistical methods. If we do, we can get important errors forecasting features. Autocorrelation is also useful because its presence tells you important things about the variable and potential problems with your model. The autocorrelation function can be used basically to detect non-randomness in data and identify cyclical patterns if present. The good thing is that we can filter this patterns to get an "adjusted" timeseries and to perform better forecasting. ',
               style={
                'display': 'inline-block', 
                'font-size':'20px',
                'font-family': 'Open Sans', 
                'padding':'25px', 
                'font-weight': '400',
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                    }
               ), 
        html.H5('Can we predict all time series?',
            style={
                'font-family': 'Open Sans',
                'padding':'25px', 
                'font-weight': '600', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',
            
        }),
        html.P(' We could make predictions in all time series but the error we could get in some of them makes prediction unpractical. Stationary series (with high autocorrelation) needs important adjustmanets to get a simple result. Non-stationary series (with no autocorrelation) can be random, with high noise or chaotic (from chaos theory). Random series has some hints, as well as very noisy series. Chaotic time series are my favourites. This series has "long term memory", encoding very interesting information inside. Generally, chaotic time series shows a great dependence on the initial conditions. Systems very sensitive to initial conditions tends to amplify any small change all over the time, so will make its forecast less predictable. This is an essential aspect of deterministic chaos, small differences in the initial values of a systems can grow into very large differences with time. The reason is that these differences has an exponential growth with time.'  ,
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',            
        }),        
        html.P(' There are two very powerful technologies that performs the task of time series forecasting with outstaning results: Probabilistoc Programming and Neural Networs. Probabilistic Programming models are a higher abstract of human thought in contrast to human brain neuronal structure modeled by Neural Networks. Probabilistic Programming therefore models human intuition in terms of incorporating what we are certain or uncertain about. It models how we go about looking for correlation in data and explore what we do not know and refine details in an areas of interest. They are a more intuitive approach to usage in data analysis. Neural Networks are a system made up of a number of simple, highly interconnected processing elements, which process information by their dynamic state response to external inputs. Maths behind both are difficult and computational resources needed are high. In this app we are going to use a simpler but very promising algorithm call XGBoost. It is easier to explain and requires less computational resources.',
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',            
        }),   
        html.P('To understant XGBoost we have to understand decission trees (DT). As we can see in the figure below, DT are a very interesting algortithms for classification. But, can we use then for time series regression (or forecast)? The response is yes. In the same figure we can see how a DT can be applied to a time series regression. We have as well splits, leafs and nodes. ',
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',            
        }),
        html.Img(src='assets/Trees.png', style={
                'display': 'block',
                'margin-left': '11%', 
                'margin-right': 'auto', 
                'width': '75%', 
                'margin-bottom':'1%', 
        }),
        html.P(' XGBoost belongs to a class of machine learning models called ensemble methods. It creates a sequence of models such that these models attempt to correct the mistakes of the models before them in the sequence. The first model is built on training data or the original dataset, the second model improves the first model, the third model improves the second, and so on. Now imagine our forcasting problem and how Decission Trees works. Model draws a first DT regressor (splits the series for a certain point as we have seen in the picture above). This regressor is able to make a prediction for some points but not for others. Then, algorithm performs another DT regressor focused on the points that have not been predicted in the first regressor. Then a third RT regressor performs another calculation with points that have not been predicted in in the second regressor, and so on. This process continues and we have a combined final regressor which predicts all the data points correctly.',
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',            
        }),
        html.Img(src='assets/Split.png', style={
                'display': 'block',
                'margin-left': '11%', 
                'margin-right': 'auto', 
                'width': '75%', 
                'margin-bottom':'1%', 
        }),
        html.Div([dcc.Graph(id='xgboost'        
        )],
            style={
                'width': '80%',
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE',
                'border-radius': '5px', 
                'background-color': '#F1EEE6', 
                'margin-left':'10%'
              }),
        html.P(' In the above graph we see how XGBoost regressor have fitted our feature points. In order to use less computational resources, we have used monthly data. We also have splitted the series in two sections: 70% of points for training the algortihm, and 30% for testing it. In the graph we heve plotted training points section. As accuracy metric, we use Mean Squared Error. Basically it means: when MSE is close to 0, we have done a good regression; higher MSE values, less accurate regression. ',
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',            
        }),
        html.Div([ dcc.Graph(id='split'        
        )],
            style={
                'width': '80%',
                'display': 'inline-block',
                'box-shadow': '10px 5px 12px #DBCBBE',
                'border-radius': '5px', 
                'background-color': '#F1EEE6', 
                'margin-left':'10%'
              }),
        html.P(' You may wonder ¿at wich point algorithm decides first split?, ¿and rest of splits?. XGBoost uses a computing objective (also called regularized objective) consisting in measuring the difference between the prediction y<sup>t</sup> and target y<sup>p</sup> values, and penalizing the complexity of the model by selecting a model employing simple and predictive functions. In above graph we see wich split has XGBoost algorithm choosen first. ¿Is this information useful? Could be, as it means this point in the series (or split) is the most important point for performing the regression, so we should take care what happened in this point (something important happened here). Algorithm wants this objective having a value smaller as possible. ',
            style={
                'font-family': 'Open Sans',
                'font-size':'20px',
                'padding':'25px', 
                'font-weight': '400', 
                'background-color': '#F1EEE6',
                'margin-left':'2%',
                'margin-top':'1%',            
        }),
        
# closing html    
    ]),        
    ]),
    ]
    ) 


#end of html programming

#Begining of callback functions
@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value')])

def update_graph(xaxis_column, yaxis_column):   
    fig = px.density_contour(df_original, 
                     x=xaxis_column,
                     y=yaxis_column,
                     template='plotly_white')
    fig.update_traces(contours_coloring="fill", contours_showlabels = True)
    fig.update_layout({'paper_bgcolor': '#ffffff',
                  'plot_bgcolor': '#ffffff'})
    return fig


@app.callback(
    Output('correlation', 'figure'),
    [Input('variable_list', 'value')])

def update_graph(variable_list):   
    corr = df[variable_list].corr()
    fig = px.imshow(corr, template='plotly_white')
    fig.update_layout({'paper_bgcolor': '#ffffff',
                  'plot_bgcolor': '#ffffff'})   
    return fig


@app.callback(
    Output('tsne', 'figure'),
    [Input('target', 'value'),
    Input('variable_list', 'value')])

def update_graph_tsne(target, variable_list):
    df['date'] = pd.to_datetime(df['date'])
    df.set_axis(df['date'], inplace=True)
    df0 = df.resample('10d').mean()
    df1=df0[variable_list]
    X = df1.drop(columns=[target])
    X = X.fillna(method='ffill')
    X = X.fillna(method='backfill')
    
    c_target = df1[target]
    c_target_ = pd.cut(c_target, bins=[df1[target].min(),df1[target].mean(),
                                        df1[target].max()],labels=['LOW','HIGH'])
    y = c_target_    
    y = y.fillna(method='ffill')
    y = y.fillna(method='backfill')   
    
    tsne1 = TSNE(n_components=2, random_state=0, learning_rate=500, perplexity=50, n_jobs=-1)
    projections = tsne1.fit_transform(X)
    fig = px.scatter(projections, x=0, y=1, color=y, labels={
                     "0": "New dimension 1",
                     "1": "New dimension 2",
                    "color" : target
                 },
                title= f"Low dimension (2-D) visualization of dataset classified by {target}") 
   
    fig.update_layout({'paper_bgcolor': '#ffffff',
                  'plot_bgcolor': '#ffffff'})
    return fig


@app.callback(
    Output('feature_importance', 'figure'),
    [Input('target', 'value'),
    Input('variable_list', 'value')])

def update_graph(target, variable_list):
    # One Hot Encodes all labels before Machine Learning
    df1 = df[variable_list]
    one_hot_cols = df1.columns.unique().tolist()
    one_hot_cols.remove(target)
    dataset_bin_enc = pd.get_dummies(df1, columns=one_hot_cols)
    df_r = df1.apply(LabelEncoder().fit_transform)
    clf = RandomForestClassifier()
    clf.fit(df_r.dropna().drop(columns=[target]), df_r[target].dropna())   
    importance = clf.feature_importances_
    importance = pd.DataFrame(importance, index=df_r.drop((target), axis=1).columns, columns=["Feature Importance"])
    importance = importance.reset_index()
    importance.rename(columns={list(importance)[0] : "Feature"}, inplace = True)
    importance.sort_values(by='Feature Importance', ascending=True)    
    fig = px.bar(importance, x='Feature', y='Feature Importance',
                 color=clf.feature_importances_, 
                 color_continuous_scale='reds',
                 title=f'Features importance for {target} classification'
                 )
    fig.update_layout({'paper_bgcolor': '#ffffff',
                  'plot_bgcolor': '#ffffff'})
    return fig

@app.callback(
    Output('variance', 'figure'),
    [Input('variable_list', 'value'),
    Input('target', 'value')])

def update_graph_variance(variable_list, target):
    df1=df[variable_list]
    X = df1.drop(columns=[target])
    X = X.fillna(method='ffill')
    X = X.fillna(method='backfill')
    pca_c = PCA(n_components=len(X.columns))
    pca_cc = pca_c.fit_transform(X)
    co = pd.DataFrame(pca_c.explained_variance_ratio_*100)
    fig= px.line(x=co.index, y=np.cumsum(co[0]),
       labels={'x':'Number of features',
               'y': f'Cumulative Variance for {target} prediction'})
    return fig


@app.callback(
    Output('dendogram', 'figure'),
    [Input('variable_list', 'value')])

def update_graph11(variable_list):
    df_den = df[variable_list].dropna()
    if 'date' in df_den.columns:
        df_den = df_den.drop(columns=['date'])
    corr = np.round(scipy.stats.spearmanr(df_den).correlation, 8)
    fig = ff.create_dendrogram(corr, 
                           orientation='left',
                           labels=df_den.columns, 
                            )
    # width 1024 px for 80% container's width. If container's width 
    # is 100%, width is 1250 px (in a laptop)
    fig.update_layout(width=1024, height=len(df_den.columns)*50) 
    return fig

@app.callback(
    Output('classification', 'figure'),
    [Input('target', 'value'),
     Input('algorithm', 'value'),
     Input('features_list', 'value')])

def update_graph2(target, algorithm, features_list):
    df1=df[features_list]
    X = df1.drop(columns=[target])
    c_target = df1[target]
    c_target_ = pd.cut(c_target, bins=[df1[target].min(),df1[target].mean(),
                                        df1[target].max()],labels=['LOW','HIGH'])
    y = c_target_    
    X = X.fillna(method='ffill')
    X =  X.fillna(method='backfill')
    y = y.fillna(method='ffill')
    y =  y.fillna(method='backfill')    
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=0)    
    X_test = preprocessing.scale(X_test)
    X_train = preprocessing.scale(X_train)
    clf = dfa[algorithm].values[0]    
    clf.fit(X_train,y_train)
    y_pred= clf.predict(X_test)    
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Original values'], 
                                   colnames=['Predicted values'])    
    fig = px.imshow(confusion_matrix, 
                    title=f' "{algorithm}" algorithm predicts "{target}" as rest of variables combination, with an accuracy of {metrics.accuracy_score(y_test, y_pred)*100:.2f} %')
    fig.update_layout({'paper_bgcolor': '#ffffff','plot_bgcolor': '#ffffff'})
    return fig

@app.callback(
    Output('coefficients', 'figure'),
    [Input('target', 'value'),
     Input('algorithm', 'value'),
     Input('features_list', 'value')])

def update_graph3(target, algorithm, features_list):
    df1 = df[features_list].dropna()
    X = df1.drop(columns=[target])
    X = X.fillna(method='ffill')
    X =  X.fillna(method='backfill')
    c_target = df1[target]
    c_target_ = pd.cut(c_target,
                       bins=[df1[target].min(),df1[target].mean(),
                                        df1[target].max()],
                       labels=[f'{target} LOW', f'{target} HIGH'])
    y = c_target_   
    y = y.fillna(method='ffill')
    y = y.fillna(method='backfill') 
  
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = dfa[algorithm].values[0]
    X_train = preprocessing.scale(X_train)
    classifier = clf.fit(X_train,y_train)
    
    def plot_feature_contrib(feature_names, contrib,selected_ft=0,crop_zeros=False,zero_element=0.):
        assert len(contrib) == len(feature_names), "feature_names and participation should have the same length"
        assert selected_ft < len(feature_names), "selected_features out of range"
        if selected_ft == 0:
            selected_ft = len(contrib)
        nms = feature_names.copy()
        sorted_idx = np.argsort(np.abs(contrib))[::-1][:selected_ft]
        contrib = np.array([contrib[i] for i in sorted_idx])
        nms = [nms[i] for i in sorted_idx] 
        if crop_zeros:
            contrib = list(filter(zero_element.__ne__, contrib))
            selected_ft = len(contrib)
            nms = nms[:selected_ft]   
        fig = px.bar(x=nms, y=contrib, 
                     labels={'x':'Feature', 'y':f'Feature contribution to {target} classification'}, 
                     color=contrib, 
                     color_continuous_scale='RdBu')
        fig.update_layout( title= f'Feature coefficients used by {algorithm} to classify {target}')
        #fig.add_annotation( text=f'For {target} = LOW ', x=nms[len(nms)-3], y=-0.75, arrowhead=1, showarrow=False)
        #fig.add_annotation( text=f'For {target} = HIGH', x=nms[len(nms)-3], y=0.75, arrowhead=1, showarrow=False)
        fig.update_layout({'paper_bgcolor': '#ffffff',
                           'plot_bgcolor': '#ffffff'})
        return fig
    
    def global_explanation_plot(feature_names, clf, selected_ft=0):
        if hasattr(clf, 'coef_'):
            coefs = clf.coef_
            coefs
        else:
            coefs = clf.feature_importances_
            coefs
        coefs = coefs.reshape(1,len(X.columns))
        
        if coefs.shape[0] == 1:            
            coefs = coefs[0]
        return plot_feature_contrib(feature_names=feature_names, contrib=coefs, selected_ft=selected_ft, crop_zeros=True) 
    
    return global_explanation_plot(X.columns, classifier, selected_ft=0)

@app.callback(
    Output('partial_dependence', 'figure'),
    [Input('target', 'value'),
     Input('algorithm', 'value'),
     Input('features_list', 'value'),
     Input('variable', 'value')])

def update_graph4(target, algorithm, features_list, variable):
    df2 = df[features_list].dropna()
    X = df2.drop(columns=[target])
    pardep = partial_dependence(dfa[algorithm].values[0], 
                                X, 
                                [variable],
                                percentiles=(0.05, 0.95),
                                )               
    fig = px.line(x = pardep[1][0], 
                  y = pardep[0][0], 
                  labels = { 'x': variable, 'y':'Partial dependence'}, 
                  title=f' "{target}" classification with {algorithm}')
    #fig.add_annotation( text=f'{target} = HIGH ', x=df2[variable].mean(), y=pardep[0][0].mean()*1.15, arrowhead=1, showarrow=False)
    #fig.add_annotation( text=f'{target} = LOW', x=df2[variable].mean(), y=pardep[0][0].mean()*0.35, arrowhead=1, showarrow=False)
    return fig

@app.callback(
    Output('y-time-series2', 'figure'),
    [Input('yaxis-column2', 'value')])

def update_y_timeseries2(yaxis_column2):  
    fig = px.scatter(df_original, x='date', y=yaxis_column2, 
                     template='plotly_white',
                    title=f'Changes of {yaxis_column2} over time (daily data)')
    fig.update_traces(marker=dict(size=5, color='#0f4c81'))
    fig.update_layout({'paper_bgcolor': '#ffffff','plot_bgcolor': '#ffffff'})
    return fig

@app.callback(
    Output('y-time-series21', 'figure'),
    [Input('yaxis-column2', 'value')])

def update_y_timeseries21(yaxis_column2):  
    data = df_original.set_index('date').resample('w').mean().reset_index()
    fig = px.line(data, x='date', y=yaxis_column2, 
                     template='plotly_white',
                    title=f'Changes of {yaxis_column2} over time (weekly data)')
    fig.update_layout({'paper_bgcolor': '#ffffff','plot_bgcolor': '#ffffff'})
    return fig

@app.callback(
    Output('y-time-series22', 'figure'),
    [Input('yaxis-column2', 'value')])

def update_y_timeseries22(yaxis_column2):  
    data = df_original.set_index('date').resample('m').mean().reset_index()
    fig = px.line(data, x='date', y=yaxis_column2, 
                     template='plotly_white',
                    title=f'Changes of {yaxis_column2} over time (monthly data)')
    fig.update_layout({'paper_bgcolor': '#ffffff','plot_bgcolor': '#ffffff'})
    return fig

@app.callback(
    Output('autocorrelation', 'figure'),
    [Input('yaxis-column2', 'value')])

def autocorrelation(yaxis_column2):      
    
    d = {'lag': [0,7,30,60, 90, 120, 200, 250, 300, 340, 360, 380, 400, 450, 500], 
         'autocorr': [df_original[yaxis_column2].autocorr(lag=0),
                     df_original[yaxis_column2].autocorr(lag=7),
                     df_original[yaxis_column2].autocorr(lag=30),
                     df_original[yaxis_column2].autocorr(lag=60),
                     df_original[yaxis_column2].autocorr(lag=90),
                     df_original[yaxis_column2].autocorr(lag=120),
                     df_original[yaxis_column2].autocorr(lag=200),
                     df_original[yaxis_column2].autocorr(lag=250),
                     df_original[yaxis_column2].autocorr(lag=300),
                     df_original[yaxis_column2].autocorr(lag=340),
                     df_original[yaxis_column2].autocorr(lag=360),
                     df_original[yaxis_column2].autocorr(lag=380),
                     df_original[yaxis_column2].autocorr(lag=400),
                     df_original[yaxis_column2].autocorr(lag=450),
                     df_original[yaxis_column2].autocorr(lag=500),
                     ]}
    corr = pd.DataFrame(data=d)
    fig = px.line(corr, x='lag', y='autocorr',
                 labels={ 'lag': 'lag (time difference in days)', 'autocorr':'Autocorrelation'},
                 template='plotly_white',
                 title=f'Autocorrelation analysis of {yaxis_column2}')
    fig.update_layout({'paper_bgcolor': '#ffffff','plot_bgcolor': '#ffffff'})
    fig.add_annotation( text='Week', x=7, y=0, arrowhead=2, showarrow=True)
    fig.add_annotation( text='Month', x=30, y=0, arrowhead=2, showarrow=True)
    fig.add_annotation( text='Quarter', x=90, y=0, arrowhead=2, showarrow=True)
    fig.add_annotation( text='Half year', x=180, y=0, arrowhead=2, showarrow=True)
    fig.add_annotation( text='Year', x=360, y=0, arrowhead=2, showarrow=True)
    return fig 

@app.callback(
    Output('xgboost', 'figure'),
    [Input('yaxis-column2', 'value')])

def xgboost(yaxis_column2):
    target=yaxis_column2
    df_raw = df_original.filter([target], axis=1)
    df_raw = df_raw.iloc[:-15]
    transformer = RobustScaler().fit(df_raw)
    dfn = pd.DataFrame(transformer.transform(df_raw), columns=[target])
    df_raw2 = pd.DataFrame(df_original['date'])
    df_raw2['date'] = pd.to_datetime(df_raw2['date'])
    dfn = dfn.join(df_raw2)
    dfn.set_axis(dfn['date'], inplace=True)
    dfn = dfn.drop(columns='date')
    dfn = dfn.resample('m').mean()
    y_data = dfn[target].values
    X_train, X_test, y_train, y_test = train_test_split(np.arange(0,len(dfn)), y_data, test_size=0.33, random_state=42)
    X = X_train.reshape((-1,1))
    y = y_train.reshape((-1,1))
    xgb_model = XGBRegressor(n_estimators=25,
                         learning_rate=0.5, # Etha=Range[0-1], default=0.3
                         max_depth=6, # default = 6
                         objective='reg:squarederror',  # default=reg:squarederror
                         booster='gbtree', # default = 'gbtree'.'gblinear' uses linear functions
                         random_state=42,
                         n_jobs=-1)
    xgb_model.fit(X, y,
            #eval_set=[(X_train.reshape((-1,1)), y_train.reshape((-1,1))),
            #          (X_test.reshape((-1,1)), y_test.reshape((-1,1)))],
            #eval_metric='logloss',
            #verbose=False
            )
    y_pred = xgb_model.predict(X_test.reshape((-1,1)))
    mse = mean_squared_error(y_test, y_pred)
    #start plot
    trace1 = go.Scatter(
    x = df.index,
    y = y_test,
    mode = 'markers',
    name = 'Train Data used for XGBoost algorithm')
    trace2 = go.Scatter(
    x = df.index,
    y = y_pred,
    mode = 'lines',
    name = 'Prediction with XGBoost algorithm')
    layout = go.Layout(
    title = f"Mean Squared Error, mse= {mse}",
    xaxis = {'title' : "Time delta in months"},
    yaxis = {'title' : f"Feature = {target} (scaled)"})
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.update_layout({'paper_bgcolor': '#ffffff','plot_bgcolor': '#ffffff'})
    return fig


@app.callback(
    Output('split', 'figure'),
    [Input('yaxis-column2', 'value')])

def split(yaxis_column2):
    target=yaxis_column2
    df_raw = df_original.filter([target], axis=1)
    df_raw = df_raw.iloc[:-15]
    transformer = RobustScaler().fit(df_raw)
    dfn = pd.DataFrame(transformer.transform(df_raw), columns=[target])
    df_raw2 = pd.DataFrame(df_original['date'])
    df_raw2['date'] = pd.to_datetime(df_raw2['date'])
    dfn = dfn.join(df_raw2)
    dfn.set_axis(dfn['date'], inplace=True)
    dfn = dfn.drop(columns='date')
    dfn = dfn.resample('m').mean()
    y_data = dfn[target].values
    X_train, X_test, y_train, y_test = train_test_split(np.arange(0,len(dfn)), y_data, test_size=0.33, random_state=42)
    X=X_train.reshape((-1,1))
    y=y_train.reshape((-1,1))
    xgb_model = XGBRegressor(n_estimators=25,
                         learning_rate=0.5, # Etha=Range[0-1], default=0.3
                         max_depth=6, # default = 6
                         objective='reg:squarederror',  # default=reg:squarederror
                         booster='gbtree', # default = 'gbtree'.'gblinear' uses linear functions
                         random_state=42,
                         n_jobs=-1)
    xgb_model.fit(X, y,
            #eval_set=[(X_train.reshape((-1,1)), y_train.reshape((-1,1))),
            #          (X_test.reshape((-1,1)), y_test.reshape((-1,1)))],
            #eval_metric='logloss',
            #verbose=False
            )
    booster = xgb_model.get_booster()    
    df_tree = booster.trees_to_dataframe()
    df_tree = df_tree.dropna()
    split_max= df_tree.Gain
    #start plot
    trace1 = go.Scatter(
        x = X_train, 
        y = y_train, 
        mode = 'markers', 
        name = 'Test Data')
    layout = go.Layout(
        title = f"First Split performed by XGBoost algorithm",
        xaxis = {'title' : "Delta time in months"},
        yaxis = {'title' : f"Feature = {target} (scaled)"})
    fig = go.Figure(data=[trace1], layout=layout)
    fig.update_layout({'paper_bgcolor': '#ffffff','plot_bgcolor': '#ffffff'})
    #fig.add_vline(x=(split_max[np.array(split_max.index)[0]]),
    #         line_width=2, 
    #         line_color='crimson', 
    #         line_dash='dash', 
    #         annotation_text="Split 1", 
    #        annotation_position="top", 
    #        annotation_font_size=20)
    fig.add_shape(type="line",
    x0=split_max[np.array(split_max.index)[0]], 
    y0=y_train.min(), 
    x1=split_max[np.array(split_max.index)[0]], 
    y1=y_train.max(),
    line=dict(
        color="Crimson",
        width=3,
        dash="dash")
    )
    return fig  

# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, use_reloader=False)
