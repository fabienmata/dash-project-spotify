# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd

#fetch the data
my_playlist = pd.read_csv("https://raw.githubusercontent.com/fabienmata/dash-project-spotify/master/daily%20mix%20combined%20playlist.csv")
print(my_playlist)


# %%
#descriptive plots
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import chart_studio
import chart_studio.plotly as py

#set a name for the columns we are going to use
variables = ['danceability','energy','loudness','speechiness','instrumentalness','liveness','valence','tempo']

#plot with 8 subplots
fig = make_subplots(2,4, subplot_titles=('danceability','energy',
'loudness','speechiness','instrumentalness','liveness','valence','tempo'))

#set the plot indexes in the list of sublots
num = [[i,j] for i in range(1,3) for j in range(1,5)]
zipped = [list(t) for t in zip(num, variables)]

#one plot per variable
for i in zipped:
   fig.add_trace(
       go.Histogram(x=my_playlist[i[1]] ,
       opacity= .3,
       histnorm = 'probability',
       nbinsx = 20), i[0][0], i[0][1])

#set the size of the nime of the plots
for i in fig['layout']['annotations']:
   i['font'] = dict(size=10)
fig.update_layout(
   showlegend=False)
fig.layout.font.family='Rockwell'
fig.layout.font.size = 10
fig.show()


# %%
#plot with dropdowns to adjust the x, y axis and the color 

fig2 = go.Figure()
fig2.add_scatter(x=my_playlist[variables[0]], 
        y=my_playlist[variables[1]], 
        mode='markers',
        text= my_playlist['artist'],
        hovertext= my_playlist['track_name'],
        hovertemplate='artist:%{text} <br>track:%{hovertext}',
        marker= go.scatter.Marker(
        size = 10,
        color= my_playlist[variables[2]],
        colorscale='Reds', 
        showscale= False, 
        colorbar= dict(title= 'Loudness') , 
        opacity= .5))

#set button objects which are used later
buttons1 = [dict(method = "restyle",
                 args = [{'x': [my_playlist[variables[k]]],
                          }], 
                 label = variables[k])   for k in range(0, len(variables))]

buttons2 = [dict(method = "restyle",
                 args = [{
                          'y': [my_playlist[variables[k]]],
                          }],
                 label = variables[k])   for k in range(1, len(variables))]

buttons3 = [dict(method = "restyle",
                 args = [{
                          'marker.color': [my_playlist[variables[k]]],
                          }],
                 label = variables[k])   for k in range(0, len(variables))]
button_layer_1_height = 1.15

#actually put the buttons
fig2.update_layout(title_text='',
                  font_family = 'Rockwell',
                  title_x=0.4,
                  width=825,
                  height=450,
                  updatemenus=[dict(active=0,
                                    buttons=buttons1,
                                    x=0.07,
                                    y=button_layer_1_height,
                                    xanchor='left',
                                    yanchor='top'),
                              
                               dict(buttons=buttons2,
                                    x=0.37,
                                    y=button_layer_1_height,
                                    xanchor='left',
                                    yanchor='top'),

                                dict(buttons=buttons3,
                                    x=0.67,
                                    y=button_layer_1_height,
                                    xanchor='left',
                                    yanchor='top')
                              ])

#put annotation for each button
fig2.add_annotation(
            x=0,
            y=1.13,
            xref='paper',
            yref='paper',
            showarrow=False,
            xanchor='left',
            text="X axis")
fig2.add_annotation(
            x=0.30,
            y=1.13,
            showarrow=False,
            xref='paper',
            yref='paper',
            xanchor='left',
            #yanchor='top',
            text="Y axis")
fig2.add_annotation(
            x=0.60,
            y=1.13,
            showarrow=False,
            xref='paper',
            yref='paper',
            xanchor='left',
            #yanchor='top',
            text="Color")

fig2.show()


# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#preprocess the data 
min_max_scaler = MinMaxScaler()
vita = min_max_scaler.fit_transform(mon_playlist[variables])

#the PCA 
pca = PCA()
pca.fit(vita)

#Principal components
fig3 = go.Figure()
fig3.add_trace(go.Bar(
    y=pca.explained_variance_ratio_,
    opacity=.5,
    marker_color = "darkred"))
fig3.show()


# %%
#plot of the actual PCA
import numpy as np 

#only take 2 components
pca.n_components = 2
X_reduced = pca.fit_transform(vita)
df_X_reduced = pd.DataFrame(X_reduced)#index=vita.index"""
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=df_X_reduced[0], y=df_X_reduced[1],
    name='sin',
    mode='markers',
    text= my_playlist['artist'],
    hovertext= my_playlist['genres'],
    hovertemplate='artist:%{text} <br>genres:%{hovertext}',
    textfont= dict(family = 'Rockwell'),
    marker= go.scatter.Marker(
        size = 10,
        color= my_playlist['loudness'],
        colorscale='Reds', 
        showscale= True, 
        #colorbar= dict(title= 'Loudness') , 
        opacity= .5)
    ))

#add the lines  
for i, variable in enumerate(variables):
    fig5.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    fig5.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="auto",
        yanchor="auto",
        text=variable,
    )
fig5.update_layout(title_text='',
                  font_family = 'Rockwell',
                  title_x=0.4,
                  width=825,
                  updatemenus=[dict(active=0,
                    buttons=buttons3,
                    x=0.07,
                    y=1.15,
                    xanchor='left',
                    yanchor='top')
                              ])
           
fig5.add_annotation(
            x=0,
            y=1.13,
            xref='paper',
            yref='paper',
            showarrow=False,
            xanchor='left',
            text="Color")
fig5.show()


# %%
print ('Explained Variance Ratio = ', sum(pca.explained_variance_ratio_[: 3]))


# %%
###KMeans

import numpy as np
from sklearn.cluster import KMeans
import dash

#function to loop over the number of cluster
def cluster(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_reduced)
    Z = kmeans.predict(X_reduced)
    return kmeans, Z

#set the limit of the loop
max_clusters = 20
inertias = np.zeros(max_clusters)

for i in list(range(1, max_clusters)):
    kmeans, Z = cluster(i)
    inertias[i] = kmeans.inertia_

#elbow method plot
fig6 = go.Figure()
fig6.add_trace(go.Scatter(
            x=list(range(1, max_clusters)),
            y=inertias[1:]
    ))
fig6.update_layout(
    font_family= 'Rockwell')
fig6.show()


# %%



# %%
#set the number of cluster
n_clusters = 3
model, Z = cluster(n_clusters)

#plot the clusters in the PCA axis
fig7 = go.Figure()

#Previous PCA plot
fig7.add_trace(go.Scatter(
    x=df_X_reduced[0], 
    y=df_X_reduced[1],
    name= '',
    mode='markers',
    text= my_playlist['artist'],
    hovertext= my_playlist['genres'],
    hovertemplate='artist:%{text} <br>genres:%{hovertext}',
    textfont= dict(family = 'Rockwell'),
    marker= go.scatter.Marker(
        size = 10, 
        color= Z,
        colorscale= "blugrn", 
        showscale= False, 
        opacity= .5), 
    showlegend=True
    ))

#add the centroïds 
fig7.add_trace(go.Scatter(
    x=model.cluster_centers_[:, 0],
    y=model.cluster_centers_[:, 1],
    name='',
    mode='markers',
    marker=go.scatter.Marker(
        symbol='x',
        size=12,
        color= 'darkred',
        opacity= .5),
    showlegend=False))
fig7.update_layout(
    font_family= 'Rockwell')
fig7.show()


# %%



