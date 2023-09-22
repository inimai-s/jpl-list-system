import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.spatial as spatial
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import dgl
from dgl.dataloading import GraphDataLoader

import networkx as nx

import random
import re
import pickle
from operator import itemgetter
#import utility.acronym_finder as af

        
if torch.cuda.is_available(): device = torch.device("cuda:0" ) 
else: device = torch.device("cpu") 
 
def process_text(text):
    parentheses = re.findall('\(.*?\)',text)
    for p in parentheses:
        text = text.replace(p, '')
        
    sentences = text.split('. ')
    text = '. '.join([s for s in sentences if '@' not in s and '.org' not in s and '.gov' not in s and '.com' not in s and '.mil' not in s and '.edu' not in s])
    return text

### TODO: Prepare Solicitaion-Phrase Matches ###
def get_solicitation_phrases(solicitations_df, s_idx, s_idxx, inverse_guide, pvc_file='phrase_embs_test3.pkl'):
    phrase_vector_collection = pickle.load(open(pvc_file, 'rb'))
    sol_phrase_edges = []

    s_desc = solicitations_df.Description.values[s_idx]
    
    for k in phrase_vector_collection.keys():
        if k in s_desc.lower(): 
            print('Phrase Match Found in Solicitation:', k)
            sol_phrase_edges.append([int(inverse_guide[k]), s_idxx])
    return sol_phrase_edges
##############

#### TODO: Prepare Scientist-Phrases Dictionary #####
def scientist_phrase_map(dataset):
    g = dataset[0]
    u, v = g.edges()

    adj = torch.zeros((dataset.used_nodes, dataset.used_nodes))
    adj[u.long(), v.long()] = 1.
    adj_neg = 1 - adj - torch.eye(dataset.used_nodes)

    scientist_phrase_dict = {}
    for sci in dataset.scientists:
        sci = int(sci)
        phrases = []
        nodes_assc_w_sci = np.where(adj[:, sci].numpy()==1.)[0]
        papers_assc_w_sci = [p for p in nodes_assc_w_sci if str(p) in dataset.papers]

        for n in nodes_assc_w_sci:
            if n in np.where(g.ndata['label'] == dataset.entity_types['Phrase'])[0]:
                phrases.append(n)

        for pap in papers_assc_w_sci:
            nodes_assc_w_pap = np.where(adj[:, pap].numpy()==1.)[0]
            for n in nodes_assc_w_pap:
                if n in np.where(g.ndata['label'] == dataset.entity_types['Phrase'])[0]:
                    phrases.append(n)

        scientist_phrase_dict[int(sci)] = phrases
    
    return scientist_phrase_dict
############################################################    

def load_model(save_name):
    model = f"./params/{save_name}/gnn.pt"
    out_model = f"./params/{save_name}/outlayer.pt"
    train_graph = f'./params/{save_name}/train_graph.pt'
    train_edge_features = f'./params/{save_name}/train_edge_features.pt'
    subgraph = f'./params/{save_name}/solicitation_subgraph.pt'

    gnn = torch.load(model).cpu()
    pred = torch.load(out_model).cpu()
    train_g = torch.load(train_graph)
    train_efeat = torch.load(train_edge_features)
    solicitation_subgraph = torch.load(subgraph)
    
    return gnn, pred, train_g, train_efeat, solicitation_subgraph

def plt_GV_curves(metrics):
    for k1, v1 in metrics.items():
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel(k1)
        plt.title(f'Graph Validation {k1}')
        for k, v in v1.items():
            plt.plot(v, label=k)
        plt.legend()
        plt.savefig(f'results/{k1}_curves_graphvalidation.png', dpi=100)
        plt.show()
        
def iterate_dataloader(dataloader, model, criterion, optimizer, train=True):
    epoch_losses, epoch_accuracies  = [], []
    num_correct, num_tests = 0, 0
    
    for batched_graph, labels in dataloader:
        pred = model(batched_graph.to(device), batched_graph.ndata['attr'].float().to(device))
        labels = labels.float()
        loss = criterion(pred, labels.to(device))

        if train:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        num_correct += (torch.round(pred) == labels.to(device)).sum().item()
        num_tests += len(labels)
        accuracy = num_correct / num_tests
        epoch_accuracies.append(accuracy)
        epoch_losses.append(loss.item())

    return sum(epoch_losses)/len(epoch_losses), sum(epoch_accuracies)/len(epoch_accuracies)
        
def infer_subgraph_scores(sci,
                            model,
                            g,
                            s_id,
                            sol_phrase_edges,
                            phrases):
    
    scientist_phrase_edges = [[s_id, sci]]

    for phrase_idx in phrases: scientist_phrase_edges.append([phrase_idx, sci])

    sg_edges = scientist_phrase_edges + sol_phrase_edges
    sg = dgl.graph(tuple(zip(*sg_edges)), num_nodes=g.num_nodes())
    sg = construct_graph(sg, g.ndata['feat'], g.ndata['label'])

    dataloader = GraphDataLoader([sg], sampler=SubsetRandomSampler(torch.arange(1)), batch_size=1, drop_last=False)
    it = iter(dataloader)
    batch = next(it)
    return model(batch, batch.ndata['attr'].float())

def construct_graph(ds, node_feat, node_label):
    u, v = ds.edges()
    unq_nodes = list(set(list(u.numpy()) + list(v.numpy())))
    reindex = dict(zip(unq_nodes, range(len(unq_nodes))))
    u = [reindex[i] for i in u.numpy()]
    v = [reindex[i] for i in v.numpy()]
    attr = node_feat[unq_nodes]
    label = node_label[unq_nodes]
    
    g = dgl.graph((u, v), num_nodes=len(unq_nodes))
    g.ndata['label'] = label
    g.ndata['attr']  = attr
    
    return dgl.add_self_loop(g)

def create_graph_dataset(positive_subgraphs, negative_subgraphs, node_feat, node_label): 
    dataset = []

    for ds in positive_subgraphs:
        g = construct_graph(ds, node_feat, node_label)
        dataset.append((g, torch.ones(1).squeeze(0)))

    for ds in negative_subgraphs:
        g = construct_graph(ds, node_feat, node_label)
        dataset.append((g, torch.zeros(1).squeeze(0)))
        
    return dataset

def phrase_cxf(doc):
    phrase_vector_collection = pickle.load(open('phrase_embs.pkl', 'rb'))
    matched_phrases = []
    for k in phrase_vector_collection.keys():
        if k in doc.lower(): 
            matched_phrases.append(k)
            
    return matched_phrases

def get_phrase_connectivity(graph_file,
                           df,
                           doc,
                           s_final_phrase_vector_collection,
                           save=0,
                           pvc_file='phrase_embs.pkl',
                           npm_file='phrase_node_matches.pkl'
                           ):
    phrase_vector_collection = pickle.load(open(pvc_file, 'rb'))
    dataset = pickle.load(open(graph_file, "rb"))
    dataset.process_phrases(pvc_file=pvc_file,
                            npm_file=npm_file)

    inverse_guide = {v: k for k, v in dataset.guide.items()}
    phrase_weights = {}
    phrase_matches = {}
    
    for n in range(df.shape[0]):
        name = df.names.values[n]
        score = df['Graph Validated Score'].values[n]
        print('Scientist:', name, 'Score:', score)
        phrase_matches[name] = {'scientist': [], 'solicitation': [], 'related_phrases': []}

        name_id = inverse_guide[name]

        u, v = dataset[0].edges()
        # Get all edges related to scientist
        sci_edge_ids = np.where(v == int(name_id))
        related_node_ids = u[sci_edge_ids]
        related_node_ids = torch.cat([related_node_ids, torch.tensor([int(name_id)])])

        # Get all edges related to scientist and 1 step out (i.e -> get phrases related to paper related to scientist)
        scientist_phrases = []
        for rn_id in related_node_ids[:]:
            all_edges_ids = np.where(v == rn_id)
            all_src_ids = u[all_edges_ids]

            for s_id in all_src_ids:
                if dataset[0].ndata['label'][s_id] == 4:
                    scientist_phrases.append((str(s_id.numpy()),  dataset.guide[str(s_id.numpy())]))

        network_phrases = list(s_final_phrase_vector_collection.keys()) + ['solicitation', name]
        connected_phrases = [('solicitation', k) for k in s_final_phrase_vector_collection.keys()] + [('solicitation', name)]
        node_colors = ['blue']*len(s_final_phrase_vector_collection.keys()) + ['red', 'green']

        for p in s_final_phrase_vector_collection:
            phrase_matches[name]['solicitation'].append(p)
                
            for sp in set(scientist_phrases):
                p_vec = s_final_phrase_vector_collection[p]
                sp_vec = phrase_vector_collection[sp[1]]
                cosine_sim = 1 - spatial.distance.cosine(p_vec, sp_vec)
                if cosine_sim > 0.70: 
                    if p not in phrase_matches[name]['solicitation']: phrase_matches[name]['solicitation'].append(p)
                    if sp[1] not in network_phrases:
                        network_phrases.append(sp[1])
                        node_colors.append('blue')
                    connected_phrases.append((p, sp[1]))
                    connected_phrases.append((sp[1], p))
                    connected_phrases.append((name, sp[1]))
                    
                    phrase_matches[name]['scientist'].append(sp[1])
                    phrase_matches[name]['related_phrases'].append([sp[1], p])
                        
                    if sp[1] not in phrase_weights.keys():
                        phrase_weights[sp[1]] = []
                    phrase_weights[sp[1]].append(cosine_sim*score)
                    print(p, sp[1], cosine_sim)
                    
                if sp[1] in doc.lower():
                    if sp[1] not in network_phrases:
                        network_phrases.append(sp[1])
                        node_colors.append('blue')
                    connected_phrases.append(('solicitation', sp[1]))
                    
                    phrase_matches[name]['solicitation'].append(sp[1])

        plt.figure()
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["figure.autolayout"] = True

        G = nx.DiGraph()
        G.add_nodes_from(network_phrases)
        G.add_edges_from(connected_phrases)
        pos = nx.spring_layout(G, scale=8)  # double distance between all nodes
        nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=4000)
        
        if save:
            savename = '_'.join(name.split(' '))
            plt.savefig(f'results/3DWinds_phraseCorrelationGraphs/{savename}.png')
        plt.show()
        
    for k in phrase_weights.keys():
        phrase_weights[k] = sum(phrase_weights[k])/df.shape[0]
        
    return phrase_matches, phrase_weights
        
def scale_features(g):
    papers = np.where(g.ndata['label']==3)[0]
    scaled_feats = []
    for n, i in enumerate(g.ndata['feat']):
        if n in papers:
            scaled_feats.append(torch.stack([(i[0]*0.75) + (i[1]*0.25),
                                             torch.zeros(768),
                                             torch.zeros(768)]))
        else: 
            scaled_feats.append(i)

    return torch.stack(scaled_feats)
    
def get_paper_title(paper):
    paper = paper.replace("“", '"').replace("”", '"')
    return re.findall('"([^"]*)"', paper)

def compute_accuracy(out, lbs):
    matches = 0
    total = len(lbs)
    softmax_outputs = torch.nn.Softmax()(out)
    
    for i in range(total):
        if softmax_outputs[i][lbs[i]] > 0.5: 
            matches += 1

    return matches/total

def infer(data, 
          dataset,
          gnn, 
          pred, 
          train_g, 
          train_efeat, 
          solicitation_subgraph,
          edge_id=0):
    
    train_g.ndata['feat'][-1] = data
    h = gnn(train_g, train_g.ndata['feat'], train_efeat)
    test_s_score, _ = pred(solicitation_subgraph, h)
    
    test_s_outputs = torch.nn.Softmax(dim=1)(test_s_score)
    # Reshape so that it's organized by solicitation
    #     solicitations_df = pd.read_csv('./data/solicitations.csv')
    test_s_outputs = test_s_outputs.view(23, len(dataset.scientists), -1)
    o = test_s_outputs[-1]
    
    # Attach id's of scientists so we know who's scores belong to who
    results = torch.cat([o, torch.tensor([int(s) for s in dataset.scientists]).unsqueeze(1)], dim=1)
    # Sort results
    sorted_results = sorted(results.detach().numpy(), key=itemgetter(int(edge_id)), reverse=True)

    scientist_names = []
    df_results = pd.DataFrame(sorted_results)
    for i in range(df_results.shape[0]):
        scientist_names.append(dataset.guide[str(int(df_results[len(dataset.edge_types.keys())+1].values[i]))])
    df_results['names'] = scientist_names
    df_results['idx'] = df_results[len(dataset.edge_types.keys())-1]
    df_results = df_results[['names'] + [int(edge_id)]]
    df_results.columns = ['names','score']

    # Preview results
    return df_results #[df_results['score']>0.50]

def create_test_graph(num_nodes_graph,
                     obj_src,
                     obj_dst):
    # Let's say your objective source are solicitations and 
    # your objective destination are scientists.
    # (Match best scientist to a solicitation):
    
    # Then you want your data to look as follows
    #    u=[[SOL1, SOL1, ... SOL1], [SOL2, SOL2, ... SOL2], ...]
    #    v=[[sci1, sci2, ... sciN], [sci1, sci2, ... sciN], ...]
    
    # So you iterate over your objectives to generate the v array, then copy your src across all objectives
    
    # Solicitation test creation
    test_s_u, test_s_v = [], []

    for i in obj_src: 
        test_s_u += [int(i)]*len(obj_dst)
        test_s_v += [int(k) for k in obj_dst]

    test_s_u, test_s_v = torch.tensor(test_s_u).int(), torch.tensor(test_s_v).int()

    return dgl.graph((test_s_u, test_s_v), num_nodes=num_nodes_graph)
    
def infer_v2(data, 
          gnn, 
          pred, 
          train_g, 
          train_efeat, 
          solicitation_subgraph,
          obj_dst,
          edge_id=0,
          num_sources=23):
    
    train_g.ndata['feat'][-1] = data
    h = gnn(train_g, train_g.ndata['feat'], train_efeat)
    test_s_score, _ = pred(solicitation_subgraph, h)
    
    test_s_outputs = torch.nn.Softmax(dim=1)(test_s_score)
    # Reshape so that it's organized by solicitation
    test_s_outputs = test_s_outputs.view(num_sources, len(obj_dst), -1)
    o = test_s_outputs[-1]
    
    # Attach id's of objective_dst so we know who's scores belong to who
    results = torch.cat([o, torch.tensor([int(s) for s in obj_dst]).unsqueeze(1)], dim=1)
    
    df_results = pd.DataFrame(results.detach().numpy())
    df_results = df_results[[df_results.columns[-1]]+ [int(edge_id)]]
    
    df_results.columns = ['ids','score']
    df_results.ids = df_results.ids.astype(int)
    return df_results

def process_recommendations(df_results, dataset, edge_id=0):
#     dst_names = [dataset.guide[str(int(df_results.ids.values[i]))] for i in range(df_results.shape[0])]
    
    df_results['names'] = df_results['ids'].astype(int) #dst_names
#     print(df_results)
    df_results = df_results[['names', 'score']]
    df_results.columns = ['names','score']
    
    return df_results 
    
    return df_results 

def train(model,
          pred,
          optimizer,
          train_g,
          train_features,
          train_labels,
          test_labels,
          train_pos_g,
          train_neg_g,
          test_pos_g,
          test_neg_g, 
          epochs=10000,
          save_name="KP"
         ):
    
    losses = []
    test_losses = []

    accuracies = []
    test_accuracies = []

    max_test_accuracy = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    labels = torch.cat([train_labels, torch.tensor([max(train_labels).numpy()+1]*train_neg_g.num_edges())])
    test_labels = torch.cat([test_labels, torch.tensor([max(train_labels).numpy()+1]*test_neg_g.num_edges())])
    
    for e in range(epochs):
        # forward
        h = model(train_g.to(device), train_g.ndata['feat'].to(device), train_features.to(device))
        pos_score, _ = pred(train_pos_g.to(device), h)
        neg_score, _ = pred(train_neg_g.to(device), h)
        outputs = torch.cat([pos_score, neg_score])
        
        loss = criterion(outputs, labels.long().to(device))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            accuracy = compute_accuracy(outputs, labels)
            
            losses.append(loss.item())
            accuracies.append(accuracy)
            
            if test_pos_g.num_edges() > 0:
                model.eval(), pred.eval()
                h = model(train_g.to(device), train_g.ndata['feat'].to(device), train_features.to(device))
                test_pos_score, _ = pred(test_pos_g.to(device), h)
                test_neg_score, _ = pred(test_neg_g.to(device), h)
                test_outputs = torch.cat([test_pos_score, test_neg_score])

                test_loss = criterion(test_outputs, test_labels.long().to(device))  
                test_losses.append(test_loss.item())
                test_accuracy = compute_accuracy(test_outputs, test_labels)
                test_accuracies.append(test_accuracy)

                if test_accuracy > max_test_accuracy:
                    max_test_accuracy = test_accuracy
                    torch.save(model, f'./params/{save_name}/gnn_BEST.pt')
                    torch.save(h, f'./params/{save_name}/hidden_BEST.pt')
                    torch.save(pred, f'./params/{save_name}/outlayer_BEST.pt')

            torch.save(model, f'./params/{save_name}/gnn.pt')
            torch.save(h, f'./params/{save_name}/hidden.pt')
            torch.save(pred, f'./params/{save_name}/outlayer.pt')

            model.train(), pred.train()
            
        if e % 50 == 0: 
            print('Epoch {}/{}, Train loss: {}, Train accuracy: {}/1.0'.format(e+1, epochs, loss, accuracy))
            if test_pos_g.num_edges() > 0:
                print('Test loss: {}, Test accuracy: {}/1.0'.format(test_loss, test_accuracy))
            
    plt.figure(dpi=100)
    plt.title("Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 2)
    if test_pos_g.num_edges() > 0: plt.plot(list(range(0, len(test_losses)*5, 5)), test_losses, label="Test Loss")
    plt.plot(list(range(0, len(test_losses)*5, 5)), losses, label="Train Loss")
    plt.legend()
    plt.savefig(f'results/loss_curves_edgeprediction.png', dpi=100)
    plt.show()

    plt.figure(dpi=100)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    if test_pos_g.num_edges() > 0: plt.plot(list(range(0, len(test_losses)*5, 5)), test_accuracies, label="Test Accuracy")
    plt.plot(list(range(0, len(test_losses)*5, 5)), accuracies, label="Train Accuracy")
    plt.legend()
    plt.savefig(f'results/accuracy_curves_edgeprediction.png', dpi=100)
    plt.show()
   
def train_pd(model,
          pred,
          optimizer,
          train_g,
          train_features,
          train_labels,
          test_labels,
          train_pos_g,
          test_pos_g,
          epochs=10000,
          save_name="KP"
         ):
    
    losses = []
    test_losses = []

    accuracies = []
    test_accuracies = []

    max_test_accuracy = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    labels = torch.cat([train_labels])
    test_labels = torch.cat([test_labels])
    
    for e in range(epochs):
        # forward
        h = model(train_g.to(device), train_g.ndata['feat'].to(device), train_features.to(device))
        pos_score, _ = pred(train_pos_g.to(device), h)
        outputs = torch.cat([pos_score])
        
        loss = criterion(outputs, labels.long().to(device))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            accuracy = compute_accuracy(outputs, labels)
            
            losses.append(loss.item())
            accuracies.append(accuracy)
            
            if test_pos_g.num_edges() > 0:
                model.eval(), pred.eval()
                h = model(train_g.to(device), train_g.ndata['feat'].to(device), train_features.to(device))
                test_pos_score, _ = pred(test_pos_g.to(device), h)
                test_outputs = torch.cat([test_pos_score])

                test_loss = criterion(test_outputs, test_labels.long().to(device))  
                test_losses.append(test_loss.item())
                test_accuracy = compute_accuracy(test_outputs, test_labels)
                test_accuracies.append(test_accuracy)

                if test_accuracy > max_test_accuracy:
                    max_test_accuracy = test_accuracy
                    torch.save(model, f'./params/{save_name}/gnn_BEST.pt')
                    torch.save(h, f'./params/{save_name}/hidden_BEST.pt')
                    torch.save(pred, f'./params/{save_name}/outlayer_BEST.pt')

            torch.save(model, f'./params/{save_name}/gnn.pt')
            torch.save(h, f'./params/{save_name}/hidden.pt')
            torch.save(pred, f'./params/{save_name}/outlayer.pt')

            model.train(), pred.train()
            
        if e % 50 == 0: 
            print('Epoch {}/{}, Train loss: {}, Train accuracy: {}/1.0'.format(e+1, epochs, loss, accuracy))
            if test_pos_g.num_edges() > 0:
                print('Test loss: {}, Test accuracy: {}/1.0'.format(test_loss, test_accuracy))
            
    plt.figure(dpi=100)
    plt.title("Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 2)
    if test_pos_g.num_edges() > 0: plt.plot(list(range(0, len(test_losses)*5, 5)), test_losses, label="Test Loss")
    plt.plot(list(range(0, len(test_losses)*5, 5)), losses, label="Train Loss")
    plt.legend()
    plt.savefig(f'results/loss_curves_edgeprediction.png', dpi=100)
    plt.show()

    plt.figure(dpi=100)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    if test_pos_g.num_edges() > 0: plt.plot(list(range(0, len(test_losses)*5, 5)), test_accuracies, label="Test Accuracy")
    plt.plot(list(range(0, len(test_losses)*5, 5)), accuracies, label="Train Accuracy")
    plt.legend()
    plt.savefig(f'results/accuracy_curves_edgeprediction.png', dpi=100)
    plt.show()
    
    
def display_test_data(dataset, train_data_dict):
    g = dataset[0]
    df = pd.DataFrame()
    u, v = g.edges()
    neg_u, neg_v = train_data_dict['neg_u'], train_data_dict['neg_v']
    test_src = [dataset.guide[str(int(i))] for i in u[train_data_dict['test_samples']]] + [dataset.guide[str(int(i))] for i in neg_u[train_data_dict['test_samples']]]
    test_dst = [dataset.guide[str(int(i))] for i in v[train_data_dict['test_samples']]] + [dataset.guide[str(int(i))] for i in neg_v[train_data_dict['test_samples']]]

    inverse_edge_types = {v: k for k, v in dataset.edge_types.items()}
    inverse_edge_types['4'] = "NO_EDGE"
    
    df['ground truth'] = [inverse_edge_types[str(i)] for i in torch.cat([train_data_dict['test_labels'], torch.tensor([4]*train_data_dict['test_neg_g'].num_edges())]).numpy()]
    df['Scientist'] = test_dst
    df['Project/Group/Paper'] = test_src
    
    return df

import transformers
import string
from collections import Counter, defaultdict
#import pke
import torch
import re
import json
import pandas as pd


def load_phrases():
    """Load the projects_phrases.csv and convert the json phrases to a dictionary

    Returns:
        DataFrame, dict
    """
    projects = pd.read_csv('data/projects_phrases.csv')
    all_phrases = {}
    for idx, row in projects.iterrows():
        if type(row['phrases']) == str:
            all_phrases[idx] = json.loads(row['phrases'])
    return projects, all_phrases
    

def get_attention_mask(text):
    with torch.no_grad():
        tokens = tokenizer.encode(text, return_tensors='pt')
        # model_output = model(tokens, output_hidden_states=False, output_attentions=True)
        # attentions = model_output.attentions # layers, [batch_size(1), num_heads, seqlen, seqlen]

        output = model(tokens)
        #output[0]: pooled 768 dim outputs for each token - shape [batch x seqlen x 768]
        #output[1]: final 768 dim output for document - shape [batch x 768]
        #output[2]: intermediate embedded outputs at each layer - (13 x [batch x seqlen x 768])
            # 1st is plain input embeddings, next 12 are output layers
        #output[3]: intermediate attentions at each layer (12 x [batch x heads x seqlen x seqlen])
        attentions = output[3]
        return attentions
    
    
def get_embeddings(text):
    """Given a block of text, encode the text using a transformer model

    Args:
        text (string): Text to be encoded

    Returns:
        Tensor: tensorized output 
    """
    with torch.no_grad():
        tokens = tokenizer.encode(text, return_tensors='pt')
        output = model(tokens)
    
    hidden_states = output[2] #(13 x [batch x seqlen x 768])
    embeddings = torch.stack(hidden_states) #[13 x batch x seqlen x 768]
    embeddings = torch.squeeze(embeddings) #[13 x seqlen x 768]
    embeddings = embeddings.permute(1, 0, 2) #[seqlen x 13 x 768]
    return embeddings
    
    
def get_phrases(text, max_gram=False):
    """Given a block of text, return a list of phrases extracted by frequency.
    Phrases are contiguous sets of words that appear with frequency greater than 1
    in a document 

    This returns phrases which are termed "silver labels", which can be used for 
    language model training. UCPhrase paper says that these phrases are better than
    distant supervision labels (i.e., phrases with a corresponding wikipedia entry)
    
    Args:
        text (string): block of text to extract phrases from
        max_gram (bool, optional): if max_gram is set to True, the algorithm
        will perform a filtering step where phrases that are subsets of other phrases
        will removed, so only maximal phrases are kept. Defaults to False.

    Returns:
        dict: a dictionary indexed by phrases, with stored values giving the 
            locations of the phrase in the document
    """
    phrase2cnt = Counter()
    phrase2instances = defaultdict(list)

    # Use tokenizer to get positions of words
    tokens = tokenizer.tokenize(text, add_special_tokens=False)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    widxs = [i for i, token in enumerate(tokens) if token.startswith(GPT_TOKEN)]

    sent = ' '.join(tokens)
    sent_dict = {'ids': ids, 'widxs': widxs}
    tokens = sent.lower().split()

    num_words = len(widxs)
    widxs.append(len(tokens))
    
    # Go through all possible lengths of phrases and scan through the text tokens
    # to count up phrases of the type
    for n in range(MINGRAMS, MAXGRAMS + 2):
        for i_word in range(num_words - n + 1):
            l_idx = widxs[i_word]
            r_idx = widxs[i_word + n] - 1
            ngram = tuple(tokens[l_idx: r_idx+1])
            ngram = tuple(''.join(ngram).split(GPT_TOKEN.lower())[1:])
            if is_valid_ngram(ngram):
                phrase = ' '.join(ngram)
                phrase2cnt[phrase] += 1
                phrase2instances[phrase].append([l_idx, r_idx])
                
    phrases = [phrase for phrase, count in phrase2cnt.items() if count >= MINCOUNT]
    phrases = sorted(phrases, key=lambda p: len(p), reverse=True)
    
    # Only keep phrases that are not subphrases of others
    # This is not always desirable, since sometimes subphrases themselves are different
    if max_gram:
        for p in phrases:
            has_longer_pattern = False
            for cp in cleaned_phrases:
                if p in cp:
                    has_longer_pattern = True
                    break
            if not has_longer_pattern and len(p.split()) <= MAXGRAMS:
                cleaned_phrases.append(p)
        phrase2instances = {p: phrase2instances[p] for p in cleaned_phrases}
    else:
        phrase2instances = {p: phrase2instances[p] for p in phrases}
        
    return phrase2instances




def get_phrases_docs(docs, max_gram=True):
    """Perform the same procedure as get_phrases(), but this time on a list of
    text documents. This way, we collect possible phrases from a corpus, so even if there are
    shorter documents where phrases simply do not appear enough, we can still count them
    as phrases since they were seen in other documents

    Args:
        docs (list): List of dictionaries, each dict must contain a 'text' field
        max_gram (bool, optional): Whether to remove instances of phrases that are local subphrases
        of another phrase. E.g., in a sentence "spectral image data" will not be count as
        "spectral image", "image data", and "spectral image data" for phrases. 
        Note this implementation is currently different than in get_phrases(), which 
        does not check if phrases are local max phrases, but global ones. Defaults to True.

    Returns:
        list, Counter: 
            list: list of documents with their text and phrase2instances which are token indexes
            after the document has been tokenized
            Counter: count of all potential phrases that were seen in the corpus
    """
    phrase2cnt = Counter()
    
    # Collect potential phrases for each document
    for doc in docs:
        text = doc['text']
        phrase2instances = defaultdict(list)
        
        # Use tokenizer to get positions of words
        tokens = tokenizer.tokenize(text, add_special_tokens=False)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        widxs = [i for i, token in enumerate(tokens) if token.startswith(GPT_TOKEN)]

        sent = ' '.join(tokens)
        sent_dict = {'ids': ids, 'widxs': widxs}
        tokens = sent.lower().split()

        num_words = len(widxs)
        widxs.append(len(tokens))
        
        # Go through all possible lengths of phrases and scan through the text tokens
        # to count up phrases of the type
        for n in range(MINGRAMS, MAXGRAMS + 2):
            for i_word in range(num_words - n + 1):
                l_idx = widxs[i_word]
                r_idx = widxs[i_word + n] - 1
                ngram = tuple(tokens[l_idx: r_idx+1])
                ngram = tuple(''.join(ngram).split(GPT_TOKEN.lower())[1:])
                if is_valid_ngram(ngram):
                    phrase = ' '.join(ngram)
                    phrase2cnt[phrase] += 1
                    phrase2instances[phrase].append([l_idx, r_idx])
                    
        doc['phrase2instances'] = phrase2instances
        # docs.append(doc)

    # Check which phrases have enough collective occurrences
    phrases = [phrase for phrase, count in phrase2cnt.items() if count >= MINCOUNT]
    phrases = sorted(phrases, key=lambda p: len(p), reverse=True)

    for doc in docs:
        valid_phrases = {}
        for phrase in doc['phrase2instances']:
            if phrase2cnt[phrase] >= MINCOUNT:
                valid_phrases[phrase] = doc['phrase2instances'][phrase]
        doc['phrase2instances'] = valid_phrases
        
        if max_gram:
            instances = []
            doc_phrases = []
            remove_instances = []
            # Convert phrase instances to two lists so we can easily compare positions
            for phrase in valid_phrases:
                for instance in valid_phrases[phrase]:
                    instances.append(instance)
                    doc_phrases.append(phrase)
            # Check which phrases are subsets of another phrase within the same document
            for i in range(len(instances)):
                for j in range(i+1, len(instances)):
                    if instances[j][0] <= instances[i][0] and instances[i][1] <= instances[j][1]:
                        remove_instances.append(i)

            # Finally filter out those subphrases and decrease their counts in phrase2cnt
            valid_phrases = defaultdict(list)
            for i in range(len(instances)):
                instance = instances[i]
                phrase = doc_phrases[i]
                if i in remove_instances:
                    phrase2cnt[phrase] -= 1
                else:
                    valid_phrases[phrase].append(instance)
            
            # !NOTE: we could perform one more filtering to further prune
            # phrases which fell below the min count if we wanted to, but it is also possible
            # that phrases that could otherwise be subphrases may also themselves be important
            # as abbreviations to the max phrase
            
            doc['phrase2instances'] = valid_phrases        

    return docs, phrase2cnt
    

def topic_rank_phrases(text, n_best=10):
    """Use PKE to generate key phrase candidates using TopicRank

    Args:
        text (string): Text input
        n_best (int, optional): How many best phrases to return. Defaults to 10.

    Returns:
        list: List of phrases and their graph rank score
    """
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=text, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    return extractor.get_n_best(n=n_best)


def is_valid_ngram(ngram: list):
    """Used by get_phrases() to check whether ngram is valid phrase"""
    for token in ngram:
        if not token or token in STPWRDS or token.isdigit():
            return False
    charset = set(''.join(ngram))
    #Check if any characters in ngram are punctuation
    #wouldn't this skip over any words with punctuation in?
    if not charset or (charset & PUNCS):
        return False
    if ngram[0].startswith('-') or ngram[-1].endswith('-'):
        return False
    return True

from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd 

def pre_process(text):
    
    # lowercase
    text=text.lower()
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        if score < 0.150:
            break
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

# put the common code into several methods
def get_keywords(idx):

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs_test[idx]]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    
    return keywords

def print_results(idx,keywords):
     # now print the results
    print("\n=====Title=====")
    print(docs_title[idx])
    print("\n=====Body=====")
    print(docs_body[idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])
