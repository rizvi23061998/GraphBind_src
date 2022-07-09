import pickle
from xmlrpc.client import Boolean
import pandas as pd
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm,trange
import random
import torch
from torch_geometric.data import InMemoryDataset, Data
import prettytable as pt
import math
import argparse
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import subprocess
from transformers import BertModel, BertTokenizer
from pympler import asizeof



def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--ligand", dest="ligand", help="A ligand type. It can be chosen from DNA,RNA,CA,MG,MN,ATP,HEME.")
    parser.add_argument("--psepos", dest="psepos", default='SC',
                        help="Pseudo position of residues. SC, CA, C stand for centroid of side chain, alpha-C atom and centroid of residue, respectively.")
    parser.add_argument("--features", dest="features", default='PSSM,HMM,SS,AF',
                        help="Feature groups. Multiple features should be separated by commas. You can combine features from PSSM, HMM, SS(secondary structure) and AF(atom features).")
    parser.add_argument("--context_radius", dest="context_radius",type=int, help="Radius of structure context.")
    parser.add_argument("--trans_anno", dest="trans_anno",type=bool, default=True,
                        help="Transfer binding annotations for DNA-(RNA-)binding protein training data sets or not.")
    parser.add_argument("--tvseed", dest='tvseed',type=int, default=1995, help='The random seed used to separate the validation set from training set.')
    parser.add_argument("--fsteps", dest='fsteps',type=int, default=128, help='Batch size during featurization with LM model.')
    parser.add_argument("--tasks", dest='tasks', default="1,2,3,4", help='Tasks needed to be done. ')
    return parser.parse_args()

def checkargs(args):
    if args.ligand not in ['DNA','RNA','CA','MN','MG','ATP','HEME']:
        print('ERROR: ligand "{}" is not supported by GraphBind!'.format(args.ligand))
        raise ValueError
    if args.psepos not in ['SC','CA','C']:
        print('ERROR: pseudo position of a residue "{}" is not supported by GraphBind!'.format(args.psepos))
        raise ValueError
    features = args.features.strip().split(',')
    for feature in features:
        if feature not in ['PSSM','HMM','SS','AF','LM']:
            print('ERROR: feature "{}" is not supported by GraphBind!'.format(feature))
            raise ValueError
    if args.context_radius<=0:
        print('ERROR: radius of structure context should be positive!')
        raise ValueError
    tasks = args.tasks.strip().split(',')
    for task in tasks:
        if int(task) not in [1,2,3,4]:
            print('ERROR: task {} is not supported by GraphBind-LM!'.format(task))
            raise ValueError    
    return

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    return gpu_memory_map


class NeighResidue3DPoint(InMemoryDataset):
    def __init__(self,root, dataset,transform=None, pre_transform=None):
        super(NeighResidue3DPoint, self).__init__(root, transform, pre_transform)

        if dataset == 'train':
            self.data,self.slices = torch.load(self.processed_paths[0])
        elif dataset == 'valid':
            self.data,self.slices = torch.load(self.processed_paths[1])
        elif dataset == 'test':
            self.data,self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        splits = ['train', 'valid', 'test']
        return ['{}_data.pkl'.format(s) for s in splits]

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def _download(self):
        pass

    def process(self):

        seq_data_dict = {}
        for s, dataset in enumerate(['train','valid','test']):
            data_list = []
            with open(self.raw_dir+'/{}_data.pkl'.format(dataset),'rb') as f:
                [data_dict,seqlist] = pickle.load(f)
            for seq in tqdm(seqlist):
                seq_data_list = []
                seq_data = data_dict[seq]
                for res_data in seq_data:
                    node_feas = res_data['node_feas']
                    node_feas = torch.tensor(node_feas,dtype=torch.float32)
                    pos = torch.tensor(res_data['pos'],dtype=torch.float32)
                    label = torch.tensor([res_data['label']],dtype=torch.float32)
                    data = Data(x=node_feas,pos=pos, y=label)
                    seq_data_list.append(data)
                data_list.extend(seq_data_list)
                seq_data_dict[seq] = seq_data_list
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[s])
        torch.save(seq_data_dict, root_dir + '/processed/seq_data_dict.pt')

def Create_NeighResidue3DPoint(psepos,dist,feature_dir,raw_dir,seqanno,feature_combine,train_list,valid_list,test_list):
    with open(feature_dir + '/' + ligand + '_psepos_{}.pkl'.format(psepos), 'rb') as f:
        residue_psepos = pickle.load(f)
    with open(feature_dir+'/'+ligand+'_residue_features_{}.pkl'.format(feature_combine),'rb') as f:
        residue_feas = pickle.load(f)
    
    # print(residue_psepos[train_list[0]])

    for s, (dataset, seqlist) in enumerate(zip(['train', 'valid', 'test'],
                                             [train_list,valid_list, test_list])):
        print("Calculating neighbourhood for ", dataset, " dataset.")
        data_dict = {}
        count = 0
        total_len = len(seqlist)
        batch_no = 1
        batch_size = 80
        for seq in tqdm(seqlist):
            # print("Calculating neighbourhood for ", seq, " ...")
            seq_data = []
            feas = residue_feas[seq]
            pos = residue_psepos[seq]
            label = np.array(list(map(int, list(seqanno[seq]['anno']))))

            # print("Size of list: ", asizeof.asizeof(data_dict)/1024.0/1024.0)
            for i in range(len(label)):
                res_psepos = pos[i]
                res_dist = np.sqrt(np.sum((pos-res_psepos)**2,axis=1))
                neigh_index = np.where(res_dist<dist)[0]
                res_atom_id = np.arange(len(neigh_index))
                id_dict = dict(list(zip(neigh_index,res_atom_id)))
                res_pos = pos[neigh_index]-res_psepos
                res_feas = feas[neigh_index]

                res_label = label[i]
                res_data = {'node_feas': res_feas.astype('float32'),
                            'pos': res_pos.astype('float32'),
                            'label': res_label.astype('float32'),
                            'neigh_index':neigh_index.astype('int32')}
                seq_data.append(res_data)
            
            data_dict[seq] = seq_data
            count += 1
            if (batch_no*batch_size )== count or count == total_len:
                print("Size of list: ", asizeof.asizeof(data_dict)/1024.0/1024.0)
                with open(raw_dir + '/{}_data_batch{}.pkl'.format(dataset,batch_no), 'wb') as f:
                    pickle.dump([data_dict, seqlist], f)
                    data_dict = {}
                batch_no += 1
                print("Size of list: ", asizeof.asizeof(data_dict)/1024.0/1024.0)
                


        # with open(raw_dir + '/{}_data.pkl'.format(dataset), 'wb') as f:
        #     pickle.dump([data_dict, seqlist], f)

    return

def def_atom_features():
    A = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,3,0]}
    V = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,1,0], 'CG1':[0,3,0], 'CG2':[0,3,0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,1,1] }
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,1], 'CG':[0,2,1], 'CD':[0,2,1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,1,0], 'CD1':[0,3,0], 'CD2':[0,3,0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'CG1':[0,2,0], 'CG2':[0,3,0], 'CD1':[0,3,0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,2,0], 'CD':[0,2,0], 'NE':[0,1,0], 'CZ':[1,0,0], 'NH1':[0,2,0], 'NH2':[0,2,0] }
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[-1,0,0], 'OD1':[-1,0,0], 'OD2':[-1,0,0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[-1,0,0], 'OE1':[-1,0,0], 'OE2':[-1,0,0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'OG':[0,1,0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'OG1':[0,1,0], 'CG2':[0,3,0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'SG':[-1,1,0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,0,0], 'OD1':[0,0,0], 'ND2':[0,2,0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,0,0], 'OE1':[0,0,0], 'NE2':[0,2,0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'ND1':[-1,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'NE2':[-1,1,1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,2,0], 'CE':[0,2,0], 'NZ':[0,3,1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,0,1], 'OH':[-1,1,0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'SD':[0,0,0], 'CE':[0,3,0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,0,1], 'NE1':[0,1,1], 'CE2':[0,0,1], 'CE3':[0,1,1], 'CZ2':[0,1,1], 'CZ3':[0,1,1], 'CH2':[0,1,1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                   'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0]/2+0.5,i_fea[1]/3,i_fea[2]]

    return atom_features

def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict ={'GLY':'G','ALA':'A','VAL':'V','ILE':'I','LEU':'L','PHE':'F','PRO':'P','MET':'M','TRP':'W','CYS':'C',
               'SER':'S','THR':'T','ASN':'N','GLN':'Q','TYR':'Y','HIS':'H','ASP':'D','GLU':'E','LYS':'K','ARG':'R'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path,'r')
    pdb_res = pd.DataFrame(columns=['ID','atom','res','res_id','xyz','B_factor'])
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H':1,'C':12,'O':16,'N':14,'S':32,'FE':56,'P':31,'BR':80,'F':19,'CO':59,'V':51,
                            'I':127,'CL':35.5,'CA':40,'B':10.8,'ZN':65.5,'MG':24.3,'NA':23,'HG':200.6,'MN':55,
                            'K':39.1,'AP':31,'AC':227,'AL':27,'W':183.9,'SE':79,'NI':58.7}

    while True:
        line = pdb_file.readline()
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count+=1
            res_pdb_id = int(line[22:26])
            if res_pdb_id != before_res_pdb_id:
                res_count +=1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N','CA','C','O','H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5,0.5,0.5]
            tmps = pd.Series(
                {'ID': atom_count, 'atom':line[12:16].strip(),'atom_type':atom_type, 'res': res, 'res_id': int(line[22:26]),
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),'occupancy':float(line[54:60]),
                 'B_factor': float(line[60:66]),'mass':Relative_atomic_mass[atom_type],'is_sidechain':is_sidechain,
                 'charge':atom_fea[0],'num_H':atom_fea[1],'ring':atom_fea[2]})
            if len(res_id_list) == 0:
                res_id_list.append(int(line[22:26]))
            elif res_id_list[-1] != int(line[22:26]):
                res_id_list.append(int(line[22:26]))
            pdb_res = pdb_res.append(tmps,ignore_index=True)
        if line.startswith('TER'):
            break

    return pdb_res,res_id_list

def cal_PDBDF(seqlist, PDB_chain_dir, PDB_DF_dir):

    if not os.path.exists(PDB_DF_dir):
        os.mkdir(PDB_DF_dir)

    for seq_id in tqdm(seqlist):
        # print(seq_id)
        file_path = PDB_chain_dir + '/{}.pdb'.format(seq_id)
        with open(file_path, 'r') as f:
            text = f.readlines()
        if len(text) == 1:
            print('ERROR: PDB {} is empty.'.format(seq_id))
        if not os.path.exists(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id)):
            try:
                pdb_DF, res_id_list = get_pdb_DF(file_path)
                with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'wb') as f:
                    pickle.dump({'pdb_DF': pdb_DF, 'res_id_list': res_id_list}, f)
            except KeyError:
                print('ERROR: UNK in ', seq_id)
                raise KeyError


    return

def cal_Psepos(seqlist,PDB_DF_dir,Dataset_dir,psepos,ligand,seqanno):

    seq_CA_pos = {}
    seq_centroid = {}
    seq_sidechain_centroid = {}  

    for seq_id in tqdm(seqlist):
        
        with open(PDB_DF_dir+'/{}.csv.pkl'.format(seq_id),'rb') as f:
            tmp = pickle.load(f)
        pdb_res_i,res_id_list = tmp['pdb_DF'],tmp['res_id_list']

        res_CA_pos = []
        res_centroid = []
        res_sidechain_centroid = []
        res_types = []
        for res_id in res_id_list:
            res_type = pdb_res_i[pdb_res_i['res_id'] == res_id]['res'].values[0]
            res_types.append(res_type)

            res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
            xyz = np.array(res_atom_df['xyz'].tolist())
            masses = np.array(res_atom_df['mass'].tolist()).reshape(-1,1)
            centroid = np.sum(masses*xyz,axis=0)/np.sum(masses)
            res_sidechain_atom_df = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['is_sidechain'] == 1)]

            try:
                CA = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['atom'] == 'CA')]['xyz'].values[0]
            except IndexError:
                print('IndexError: no CA in seq:{} res_id:{}'.format(seq_id,res_id))
                CA = centroid

            res_CA_pos.append(CA)
            res_centroid.append(centroid)

            if len(res_sidechain_atom_df) == 0:
                res_sidechain_centroid.append(centroid)
            else:
                xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
                masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
                sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
                res_sidechain_centroid.append(sidechain_centroid)

        if ''.join(res_types) != seqanno[seq_id]['seq']:
            print(seq_id)
            print(''.join(res_types))
            print(seqanno[seq_id]['seq'])
            return
        res_CA_pos = np.array(res_CA_pos)
        res_centroid = np.array(res_centroid)
        res_sidechain_centroid = np.array(res_sidechain_centroid)
        seq_CA_pos[seq_id] = res_CA_pos
        seq_centroid[seq_id] = res_centroid
        seq_sidechain_centroid[seq_id] = res_sidechain_centroid

    if psepos =='CA':
        with open(Dataset_dir + '/'+ligand+'_psepos_'+psepos+'.pkl', 'wb') as f:
            pickle.dump(seq_CA_pos, f)
    elif psepos == 'C':
        with open(Dataset_dir + '/'+ligand+'_psepos_'+psepos+'.pkl', 'wb') as f:
            pickle.dump(seq_centroid, f)
    elif psepos == 'SC':
        with open(Dataset_dir + '/'+ligand+'_psepos_'+psepos+'.pkl', 'wb') as f:
            pickle.dump(seq_sidechain_centroid, f)

    return

def get_features_T5_XL_Uniref50(seqlist,seqanno,feature_dir,ligand,model_path, batch_size):
    print("Loading model .. ..")
    # Initialization
    print("Model path:", model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False )
    model = T5EncoderModel.from_pretrained(model_path)
    print("Model loaded.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    model = model.eval()
    
    print(get_gpu_memory_map())
    seqlist = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqlist]
    
    
    features = {}
    seq_count = len(seqlist)
    print("# of sequences: ",seq_count)

    for i in tqdm(range(int(seq_count/batch_size)+1)):
        low = int(batch_size)*i
        high = min(int(batch_size)*(i+1), seq_count)

        print("Range:", low, ":", high)
        seq_batch_i = []
        for seqid in seqlist[low:high]:
            seq_batch_i.append(seqanno[seqid]["seq"])

        ids = tokenizer.batch_encode_plus(seq_batch_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        print(get_gpu_memory_map())

        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)                
        embedding = embedding.last_hidden_state.cpu().numpy()

        
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            features[seqlist[low + seq_num]] = seq_emd

    with open(feature_dir + "/"+ ligand + "_residue_features_T5.pkl", "wb") as fp:
        pickle.dump(features, fp)

    return

def get_features_protbert_bfd(seqlist,seqanno,feature_dir,ligand,model_path, batch_size):
    print("Loading model .. ..")
    # Initialization
    print("Model path:", model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False )
    model = BertModel.from_pretrained(model_path)
    print("Model loaded.")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    model = model.eval()
    
    print(get_gpu_memory_map())
    
    
    features = {}
    seq_count = len(seqlist)
    print("# of sequences: ",seq_count)

    for i in tqdm(range(int(seq_count/batch_size)+1)):
        low = int(batch_size)*i
        high = min(int(batch_size)*(i+1), seq_count)

        # print("Range:", low, ":", high)
        seq_batch_i = []
        for seqid in seqlist[low:high]:
            seq_batch_i.append(seqanno[seqid]["seq"])

        seq_batch_i = [" ".join(sequence) for sequence in seq_batch_i]
        seq_batch_i = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_batch_i]
        # print(seq_batch_i[0])
        ids = tokenizer.batch_encode_plus(seq_batch_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # print(get_gpu_memory_map())

        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        # print((embedding.shape))
        
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            # print(seq_len)
            seq_emd = embedding[seq_num][1:seq_len-1]
            features[seqlist[low + seq_num]] = seq_emd
            # print(seq_emd.shape)
        
        del embedding
        del input_ids
        del attention_mask

    with open(feature_dir + "/"+ ligand + "_residue_features_protbert_bfd.pkl", "wb") as fp:
        pickle.dump(features, fp)
        print("Feature Dumped to ",(feature_dir + "/"+ ligand + "_residue_features_protbert_bfd.pkl"))
        print("Length of features: ", len(features))

    return

def tv_split(train_list,seed):
    random.seed(seed)
    random.shuffle(train_list)
    valid_list = train_list[:int(len(train_list)*0.2)]
    train_list = train_list[int(len(train_list)*0.2):]
    return train_list,valid_list

def StatisticsSampleNum(train_list,valid_list,test_list,seqanno):
    def sub(seqlist,seqanno):
        pos_num_all = 0
        res_num_all = 0
        for seqid in seqlist:
            anno = list(map(int,list(seqanno[seqid]['anno'])))
            pos_num = sum(anno)
            res_num = len(anno)
            pos_num_all += pos_num
            res_num_all += res_num
        neg_num_all = res_num_all - pos_num_all
        pnratio = pos_num_all/float(neg_num_all)
        return len(seqlist), res_num_all, pos_num_all,neg_num_all,pnratio

    tb = pt.PrettyTable()
    tb.field_names = ['Dataset','NumSeq', 'NumRes', 'NumPos', 'NumNeg', 'PNratio']
    tb.float_format = '0.3'

    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list+valid_list,seqanno)
    tb.add_row(['train+valid',seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list,seqanno)
    tb.add_row(['train',seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(valid_list,seqanno)
    tb.add_row(['valid',seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(test_list,seqanno)
    tb.add_row(['test',seq_num, res_num, pos_num, neg_num, pnratio])
    print(tb)
    return


if __name__ == '__main__':
    
    args = parse_args()
    checkargs(args)
    
    ligand = 'P'+args.ligand if args.ligand != 'HEME' else 'PHEM'
    psepos = args.psepos
    trans_anno = args.trans_anno
    dist = args.context_radius
    fsteps = args.fsteps
    tasks = args.tasks.strip().split(',')
    print("Tasks: ", tasks)

    feature_list = []
    feature_combine = ''
    if 'PSSM' in args.features:
        feature_list.append('PSSM')
        feature_combine += 'P'
    if 'HMM' in args.features:
        feature_list.append('HMM')
        feature_combine += 'H'
    if 'SS' in args.features:
        feature_list.append('SS')
        feature_combine += 'S'
    if 'AF' in args.features:
        feature_list.append('AF')
        feature_combine += 'A'
    if 'LM' in args.features:
        feature_list.append('LM')
        feature_combine += 'L'

    trainingset_dict = {'PDNA':'DNA-573_Train.txt',
                        'PRNA': 'RNA-495_Train.txt',
                        'PMN':'MN-440_Train.txt',
                        'PCA':'CA-1022_Train.txt',
                        'PMG':'MG-1194_Train.txt',
                        'PATP':'ATP-388_Train.txt',
                        'PHEM':'HEM-175_Train.txt'
                        }

    testset_dict = {'PDNA':'DNA-129_Test.txt',
                    'PRNA':'RNA-117_Test.txt',
                    'PMN':'MN-144_Test.txt',
                    'PCA':'CA-515_Test.txt',
                    'PMG':'MG-651_Test.txt',
                    'PATP':'ATP-41_Test.txt',
                    'PHEM':'HEM-96_Test.txt'
                    }


    Dataset_dir = os.path.abspath('..')+'/Datasets'+'/'+ligand
    PDB_chain_dir = Dataset_dir+'/PDB'
    feature_dir = Dataset_dir 
    # model_path = '../LM/prot_t5_xl_uniref50/'
    # model_path = "Rostlab/prot_t5_xl_uniref50"
    model_path = "../LM/prot_bert_bfd/"
    trainset_anno = Dataset_dir + '/{}'.format(trainingset_dict[ligand])
    testset_anno = Dataset_dir+'/{}'.format(testset_dict[ligand])


    seqanno = {}
    train_list = []
    test_list = []


    if ligand in ['PDNA','PRNA']:
        with open(trainset_anno, 'r') as f:
            train_text = f.readlines()
        if trans_anno:
            for i in range(0,len(train_text),4):
                query_id = train_text[i].strip()[1:]
                if query_id[-1].islower():
                    query_id+=query_id[-1]
                query_seq = train_text[i+1].strip()
                query_anno = train_text[i+2].strip()
                train_list.append(query_id)
                seqanno[query_id] = {'seq':query_seq,'anno':query_anno}
        else:
            for i in range(0,len(train_text),4):
                query_id = train_text[i].strip()[1:]
                if query_id[-1].islower():
                    query_id += query_id[-1]
                query_seq = train_text[i+1].strip()
                query_anno = train_text[i+3].strip()
                train_list.append(query_id)
                seqanno[query_id] = {'seq':query_seq,'anno':query_anno}
        with open(testset_anno, 'r') as f:
            test_text = f.readlines()
        for i in range(0, len(test_text), 3):
            query_id = test_text[i].strip()[1:]
            if query_id[-1].islower():
                query_id += query_id[-1]
            query_seq = test_text[i + 1].strip()
            query_anno = test_text[i + 2].strip()
            test_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
    else:
        with open(trainset_anno, 'r') as f:
            train_text = f.readlines()
        for i in range(0, len(train_text), 3):
            query_id = train_text[i].strip()[1:]
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}

        with open(testset_anno,'r') as f:
            test_text = f.readlines()
        for i in range(0, len(test_text), 3):
            query_id = test_text[i].strip()[1:]
            query_seq = test_text[i + 1].strip()
            query_anno = test_text[i + 2].strip()
            test_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}

    train_list, valid_list = tv_split(train_list,args.tvseed)
    StatisticsSampleNum(train_list,valid_list,test_list,seqanno)

    PDB_DF_dir = Dataset_dir+'/PDB_DF'
    seqlist = train_list + valid_list + test_list

    if '1' in tasks:
        print('1.Extract the PDB information.')
        cal_PDBDF(seqlist, PDB_chain_dir, PDB_DF_dir)
    if '2' in tasks: 
        print('2.calculate the pseudo positions.')
        cal_Psepos(seqlist,PDB_DF_dir,Dataset_dir,psepos,ligand,seqanno)
    if '3' in tasks:
        print('3.calculate the residue features.')
        if 'AF' in feature_list:
            atomfea = True
            feature_list.remove('AF')
        else:
            atomfea = False

        # cal_PSSM(ligand, seqlist, Dataset_dir+'/feature/PSSM', Dataset_dir)
        # cal_HMM(ligand, seqlist, Dataset_dir+'/feature/HMM', Dataset_dir)
        # cal_DSSP(ligand, seqlist, Dataset_dir+'/feature/SS', Dataset_dir)

        # PDBResidueFeature(seqlist, PDB_DF_dir, Dataset_dir, ligand, feature_list,feature_combine,atomfea)
        feature_combine = "protbert_bfd"
        # get_features_T5_XL_Uniref50(seqlist,seqanno,feature_dir,ligand,model_path, fsteps)
        get_features_protbert_bfd(seqlist,seqanno,feature_dir,ligand,model_path, fsteps)

    if '4' in tasks:
        root_dir = Dataset_dir + '/' + ligand + '_{}_dist{}_{}'.format(psepos, dist, feature_combine)
        raw_dir = root_dir + '/raw'
        if os.path.exists(raw_dir):
            shutil.rmtree(root_dir)
        os.makedirs(raw_dir)
        feature_combine = "protbert_bfd"
        print('4.Calculate the neighborhood of residues. Save to {}.'.format(root_dir))
        Create_NeighResidue3DPoint(psepos,dist,Dataset_dir,raw_dir,seqanno,feature_combine,train_list,valid_list,test_list)
        _ = NeighResidue3DPoint(root=root_dir,dataset='train')
