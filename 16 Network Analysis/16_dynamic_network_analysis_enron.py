import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from igraph import Graph, mean as ig_mean
import chart_studio.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import ruptures as rpt


d = pd.read_csv("dsilt-ml-code/16 Network Analysis/enron_emails_network.csv")
d.timestamp = pd.to_datetime(d.timestamp)


def filter_by_date(df, date_string, before_or_after):
    """
    Filter dataframe to records occurring after a date, 
    or before a date.
    """
    filter_date = pd.to_datetime(date_string)
    if before_or_after == "before":
        return df[df['timestamp'] < filter_date].copy()
    elif before_or_after == "after":
        return df[df['timestamp'] >= filter_date + pd.DateOffset(days=1)].copy()
    else:
        return print("Invalid argument.")


def reset_node_ids(df):
    """
    Every time the dataframe used to create the graph is filtered,
    the node IDs need to be reset to start from 0.  This function
    updates the from_id and to_id columns.
    """
    le = LabelEncoder()
    all_node_names = list(set(df['from_name'].values.tolist() + df['to_name'].values.tolist()))
    le.fit(all_node_names)
    df['from_id'] = le.transform(df['from_name'])
    df['to_id'] = le.transform(df['to_name'])
    return df, le


def scale_array_for_plotting(np_arr, range_min=1, range_max=4):
    scaler = MinMaxScaler(feature_range=(range_min, range_max))
    return np.round(scaler.fit_transform(np_arr), 2)



def df_to_edgelist(df, weight_type):
    """
    Get edge list with node names
    Assumes each node ID has 1 unique node name (happens to be true for this dataset)
    
    Edge weight_type == 'freq' gives an average nbr of emails sent per month in time range
    Edge weight_type == 'count' gives the count of emails sent
    Edge weights are scaled to the same scale
    """
    if weight_type == 'freq':
        try:
            nbr_months = int((max(df.timestamp) - min(df.timestamp))/np.timedelta64(1, 'M'))
            edge_list_df = pd.DataFrame(df.groupby(['from_id', 'to_id', 'from_name', 'to_name']).apply(lambda x: len(x)/nbr_months)).reset_index().rename(columns={0: 'weight'})
        except: # this is executed when the timedelta is 0
            edge_list_df = pd.DataFrame(df.groupby(['from_id', 'to_id', 'from_name', 'to_name']).apply(len)).reset_index().rename(columns={0: 'weight'})
        edge_list_df['weight'] = scale_array_for_plotting(edge_list_df['weight'].values.reshape(-1, 1))
    elif weight_type == 'count':
        edge_list_df = pd.DataFrame(df.groupby(['from_id', 'to_id', 'from_name', 'to_name']).apply(len)).reset_index().rename(columns={0: 'weight'})
        edge_list_df['weight'] = scale_array_for_plotting(edge_list_df['weight'].values.reshape(-1, 1))
    return edge_list_df


def edge_list_df_to_igraph(edge_list_df, node_id_mapper):
    """
    Convert dataframe with edge details to igraph graph object
    """
    nodes = list(set(edge_list_df.from_id.values.tolist() + edge_list_df.to_id.values.tolist()))
    #node_names = list(set(edge_list_df.from_name.values.tolist() + edge_list_df.to_name.values.tolist()))
    edges = list(zip(edge_list_df.from_id, edge_list_df.to_id))
    weights = list(edge_list_df.weight.values)
    g = Graph()
    g.add_vertices(len(nodes))
    g.add_edges(edges)
    g.es['weight'] = weights
    g.vs['label'] = list(node_id_mapper.inverse_transform(np.array(range(len(g.vs)))))
    g.vs['community'] = 0  # Set original community the same for all nodes
    return g, edges


def plot_network_w_plotly_3d(g, edgelist):
    layout = g.layout('kk', dim=3)

    # Coordinates of nodes
    Xn=[layout[k][0] for k in range(len(g.vs))]
    Yn=[layout[k][1] for k in range(len(g.vs))]
    Zn=[layout[k][2] for k in range(len(g.vs))]
    # Coordinates of edge ends
    Xe=[]
    Ye=[]
    Ze=[]
    for e in edgelist:
        Xe+=[layout[e[0]][0], layout[e[1]][0], None]
        Ye+=[layout[e[0]][1], layout[e[1]][1], None]
        Ze+=[layout[e[0]][2], layout[e[1]][2], None]

    # Plot edges
    trace1 = go.Scatter3d(x=Xe, y=Ye, z=Ze,
                        mode='lines',
                        line=dict(color='rgb(125,125,125)', width=2),
                        hoverinfo='none'
                        )

    # Plot nodes
    trace2 = go.Scatter3d(x=Xn, y=Yn, z=Zn,
                        mode='markers',
                        name='employees',
                        marker=dict(symbol='circle',
                                    size=8,
                                    color=g.vs['community'],
                                    colorscale='spectral',
                                    line=dict(color='rgb(50,50,50)', width=0.5)
                                    ),
                        text=[g.vs['label'][n] + ", community: " + str(g.vs['community'][n]) for n in range(len(g.vs))],
                        hoverinfo='text'
                        )
                    
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = go.Layout(
        title="Enron Emails Network",
        width=1000,
        height=1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
            ),
        margin=dict(
            t=50
            )
        )

    data = [trace1, trace2]
    return go.Figure(data=data, layout=layout)


# Explore complete network for entire time period

'''
dsub, node_id_mapper = reset_node_ids(d.copy())
edge_list_df = df_to_edgelist(dsub, weight_type='freq')
g, edgelist = edge_list_df_to_igraph(edge_list_df, node_id_mapper)
fig = plot_network_w_plotly_3d(g, edgelist)
fig.show()

communities = g.community_infomap()
g.vs['community'] = communities.membership

fig = plot_network_w_plotly_3d(g, edgelist)
fig.show()
'''

# Explore Phillip K Allen's network for entire time period

'''
dsub, node_id_mapper = reset_node_ids(d[(d['from_name']=='phillip.allen') | (d['to_name']=='phillip.allen')].copy())
edge_list_df = df_to_edgelist(dsub, weight_type='freq')
g, edgelist = edge_list_df_to_igraph(edge_list_df, node_id_mapper)
fig = plot_network_w_plotly_3d(g, edgelist)
fig.show()
'''

# Explore community membership of Phillip K Allen's community before and after his resignation

dsub, node_id_mapper = reset_node_ids(d.copy())
edge_list_df = df_to_edgelist(dsub, weight_type='freq')
g, edgelist = edge_list_df_to_igraph(edge_list_df, node_id_mapper)

communities = g.community_infomap()
g.vs['community'] = communities.membership

comm_of_interest = g.vs[node_id_mapper.transform(['phillip.allen'])[0]]['community']
nodes_in_orig_comm = [n for n in g.vs if n['community'] == comm_of_interest]
node_names_in_orig_comm = [n['label'] for n in g.vs if n['community'] == comm_of_interest]
# Filter dataset to emails after Phillip K Allen left Enron
final_email_timestamp = max(d[(d['from_name']=='phillip.allen') | (d['to_name']=='phillip.allen')]['timestamp'])
print(final_email_timestamp)  # cheat and hardcode this date
dsub_after_pallen = filter_by_date(dsub, '2001-07-25', 'after')
# Redo community detection to get communities after he left
dsub_after_pallen, node_id_mapper_after = reset_node_ids(dsub_after_pallen.copy())
edge_list_df_after = df_to_edgelist(dsub_after_pallen, weight_type='freq')
g_after, edgelist_after = edge_list_df_to_igraph(edge_list_df_after, node_id_mapper_after)
communities_after = g_after.community_infomap()
g_after.vs['community'] = communities_after.membership

# Find out where previous community members went
print("Where members of Phillip Allen's community went after he left:", 
      [n['community'] for n in g_after.vs if n['label'] in node_names_in_orig_comm])
g_after.vs['label']  # make sure orig community members are truly gone


# Explore the network's topological features over time

# Inspect the 3 metrics for the current graph
print("Avg Degree:", ig_mean(g.degree()))
print("Diameter:", g.diameter())
print("Nbr communities:", len(communities))

# Break up dataset by week and track weekly topology
dsub = d.set_index(d['timestamp']).copy()
time_keeper = []
avg_degree_over_time = []
diameter_over_time = []
nbr_communities_over_time = []
for group_name, df_group in dsub.groupby(pd.Grouper(freq='W')):
    try:
        dsub_group, node_id_mapper_group = reset_node_ids(df_group.copy())
        edge_list_df_group = df_to_edgelist(dsub_group, weight_type='freq')
        g_group, edgelist_group = edge_list_df_to_igraph(edge_list_df_group, node_id_mapper_group)
        avg_degree_over_time.append(ig_mean(g_group.degree()))
        diameter_over_time.append(g_group.diameter())
        nbr_communities_over_time.append(len(g_group.community_infomap()))
        time_keeper.append(group_name)  # only append to time if all other code works
    except: # executes when there are no emails sent in any given week
        next

plt.plot(time_keeper, avg_degree_over_time)
plt.title('Enron Emails Network: Average Degree Centrality by Week')
plt.xlabel('Time')
plt.ylabel('Avg Degree Centrality')
plt.show()

plt.plot(time_keeper, diameter_over_time)
plt.title('Enron Emails Network: Diameter by Week')
plt.xlabel('Time')
plt.ylabel('Network Diameter')
plt.show()

plt.plot(time_keeper, nbr_communities_over_time)
plt.title('Enron Emails Network: Infomap Community Count by Week')
plt.xlabel('Time')
plt.ylabel('Number of Communities')
plt.show()

# Change point detection
weeks_to_ignore = 4

algo = rpt.Pelt(model="rbf").fit(np.array(avg_degree_over_time))
result = algo.predict(pen=4)
print("Change points in diameter: ", [time_keeper[i] for i in result[:len(result)-1]])
rpt.display(np.array(avg_degree_over_time), [weeks_to_ignore], result)
plt.title("Change Points in Avg Degree Centrality Over Time")
plt.tight_layout()
plt.show()

algo = rpt.Pelt(model="rbf").fit(np.array(diameter_over_time))
result = algo.predict(pen=4)
print("Change points in diameter: ", [time_keeper[i] for i in result[:len(result)-1]])
rpt.display(np.array(diameter_over_time), [weeks_to_ignore], result)
plt.title("Change Points in Diameter Over Time")
plt.tight_layout()
plt.show()

algo = rpt.Pelt(model="rbf").fit(np.array(nbr_communities_over_time))
result = algo.predict(pen=4)
print("Change points in nbr communities: ", [time_keeper[i] for i in result[:len(result)-1]])
rpt.display(np.array(nbr_communities_over_time), [weeks_to_ignore], result)
plt.title("Change Points in Nbr Communities Over Time")
plt.tight_layout()
plt.show()








