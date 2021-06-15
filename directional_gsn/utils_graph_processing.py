import torch
import numpy as np
# from torch_geometric.utils import remove_self_loops, to_undirected
import networkx as nx

import graph_tool as gt
import graph_tool.topology as gt_topology

def to_undirected(edge_index):
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    return edge_index

def automorphism_orbits(edge_list, print_msgs=True, **kwargs):
    
    ##### vertex automorphism orbits ##### 

    directed=kwargs['directed'] if 'directed' in kwargs else False
    
    graph = gt.Graph(directed=directed)
    graph.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph)
    gt.stats.remove_parallel_edges(graph)

    # compute the vertex automorphism group
    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v
    
    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role
        
    orbit_membership_list = [[],[]]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse = True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i,vertex in enumerate(orbit_membership_list[0])}


    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit]+[vertex]
    
    aut_count = len(aut_group)
    
    if print_msgs:
        print('Orbit partition of given substructure: {}'.format(orbit_partition)) 
        print('Number of orbits: {}'.format(len(orbit_partition)))
        print('Automorphism count: {}'.format(aut_count))

    return graph, orbit_partition, orbit_membership, aut_count

def induced_edge_automorphism_orbits(edge_list, **kwargs):
    
    ##### induced edge automorphism orbits (according to the vertex automorphism group) #####
    
    directed=kwargs['directed'] if 'directed' in kwargs else False
    directed_orbits=kwargs['directed_orbits'] if 'directed_orbits' in kwargs else False

    graph, orbit_partition, orbit_membership, aut_count = automorphism_orbits(edge_list=edge_list,
                                                                              directed=directed,
                                                                              print_msgs=False)
    edge_orbit_partition = dict()
    edge_orbit_membership = dict()
    edge_orbits2inds = dict()
    ind = 0
    
    if not directed:
        edge_list = to_undirected(torch.tensor(graph.get_edges()).transpose(1,0)).transpose(1,0).tolist()

    # infer edge automorphisms from the vertex automorphisms
    for i,edge in enumerate(edge_list):
        if directed_orbits:
            edge_orbit = (orbit_membership[edge[0]], orbit_membership[edge[1]])
        else:
            edge_orbit = frozenset([orbit_membership[edge[0]], orbit_membership[edge[1]]])
        if edge_orbit not in edge_orbits2inds:
            edge_orbits2inds[edge_orbit] = ind
            ind_edge_orbit = ind
            ind += 1
        else:
            ind_edge_orbit = edge_orbits2inds[edge_orbit]

        if ind_edge_orbit not in edge_orbit_partition:
            edge_orbit_partition[ind_edge_orbit] = [tuple(edge)]
        else:
            edge_orbit_partition[ind_edge_orbit] += [tuple(edge)] 

        edge_orbit_membership[i] = ind_edge_orbit

    print('Edge orbit partition of given substructure: {}'.format(edge_orbit_partition)) 
    print('Number of edge orbits: {}'.format(len(edge_orbit_partition)))
    print('Graph (vertex) automorphism count: {}'.format(aut_count))
    
    return graph, edge_orbit_partition, edge_orbit_membership, aut_count


def subgraph_isomorphism_vertex_counts(edge_index, **kwargs):
    
    ##### vertex structural identifiers #####
    
    subgraph_dict, induced, num_nodes = kwargs['subgraph_dict'], kwargs['induced'], kwargs['num_nodes']
    directed = kwargs['directed'] if 'directed' in kwargs else False
    
    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index.transpose(1,0).cpu().numpy()))
    gt.stats.remove_self_loops(G_gt)
    gt.stats.remove_parallel_edges(G_gt)  
       
    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=induced, subgraph=True, generator=True)
    
    ## num_nodes should be explicitly set for the following edge case: 
    ## when there is an isolated vertex whose index is larger
    ## than the maximum available index in the edge_index
    
    counts = np.zeros((num_nodes, len(subgraph_dict['orbit_partition'])))
    for sub_iso_curr in sub_iso:
        for i,node in enumerate(sub_iso_curr):
            # increase the count for each orbit
            counts[node, subgraph_dict['orbit_membership'][i]] +=1
    counts = counts/subgraph_dict['aut_count']
        
    counts = torch.tensor(counts)
    
    return counts


def subgraph_isomorphism_edge_counts(edge_index, **kwargs):
    
    ##### edge structural identifiers #####
    
    subgraph_dict, induced = kwargs['subgraph_dict'], kwargs['induced']
    directed = kwargs['directed'] if 'directed' in kwargs else False
    
    edge_index = edge_index.transpose(1,0).cpu().numpy()
    edge_dict = {}
    for i, edge in enumerate(edge_index):         
        edge_dict[tuple(edge)] = i
        
    if not directed:
        subgraph_edges = to_undirected(torch.tensor(subgraph_dict['subgraph'].get_edges().tolist()).transpose(1,0)).transpose(1,0).tolist()

    
    G_gt = gt.Graph(directed=directed)
    G_gt.add_edge_list(list(edge_index))
    gt.stats.remove_self_loops(G_gt)
    gt.stats.remove_parallel_edges(G_gt)  
       
    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=induced, subgraph=True, generator=True)
    
            
    counts = np.zeros((edge_index.shape[0], len(subgraph_dict['orbit_partition'])))
    
    for sub_iso_curr in sub_iso:
        mapping = sub_iso_curr.get_array()
#         import pdb;pdb.set_trace()
        for i,edge in enumerate(subgraph_edges): 
            
            # for every edge in the graph H, find the edge in the subgraph G_S to which it is mapped
            # (by finding where its endpoints are matched). 
            # Then, increase the count of the matched edge w.r.t. the corresponding orbit
            # Repeat for the reverse edge (the one with the opposite direction)
            
            edge_orbit = subgraph_dict['orbit_membership'][i]
            mapped_edge = tuple([mapping[edge[0]], mapping[edge[1]]])
            counts[edge_dict[mapped_edge], edge_orbit] += 1
            
    counts = counts/subgraph_dict['aut_count']
    
    counts = torch.tensor(counts)
    
    return counts

 
    
    
    
#----------------------- line graph edge automorphism: deprecated
    


def edge_automorphism_orbits(edge_list, **kwargs):
    
    ##### edge automorphism orbits according to the line graph #####
    
    directed=kwargs['directed'] if 'directed' in kwargs else False

    graph_nx = nx.from_edgelist(edge_list)
    graph = gt.Graph(directed=directed)
    graph.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph)
    gt.stats.remove_parallel_edges(graph)
    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)
    aut_count = len(aut_group)
    
    ##### compute line graph vertex automorphism orbits #####

    graph_nx_line = nx.line_graph(graph_nx)
    mapping = {node: i for i,node in enumerate(graph_nx_line.nodes)}
    inverse_mapping = {i: node for i,node in enumerate(graph_nx_line.nodes)}

    graph_nx_line = nx.relabel_nodes(graph_nx_line, mapping)
    line_graph = gt.Graph(directed=directed)
    line_graph.add_edge_list(list(graph_nx_line.edges))

    gt.stats.remove_self_loops(line_graph)
    gt.stats.remove_parallel_edges(line_graph)  

    aut_group_edges = gt_topology.subgraph_isomorphism(line_graph, line_graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in line_graph.get_vertices():
        orbit_membership[v] = v

    for aut in aut_group_edges:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role

    orbit_membership_list = [[],[]]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse = True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i,vertex in enumerate(orbit_membership_list[0])}

    orbit_partition= {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [inverse_mapping[vertex]] if orbit not in orbit_partition else orbit_partition[orbit]+[inverse_mapping[vertex]]    

    ##### transfer line graph vertex automorphism orbits to original edges #####

    orbit_membership_new = {}
    for i,edge in enumerate(graph.get_edges()): 
        mapped_edge = mapping[tuple(edge)] if tuple(edge) in mapping else mapping[tuple([edge[1],edge[0]])]
        orbit_membership_new[i] = orbit_membership[mapped_edge]

    print('Edge orbit partition of given substructure: {}'.format(orbit_partition)) 
    print('Number of edge orbits: {}'.format(len(orbit_partition)))
    print('Graph (vertex) automorphism count: {}'.format(aut_count))
    
    return graph, orbit_partition, orbit_membership_new, aut_count