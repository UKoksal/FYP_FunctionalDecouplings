import graphviz
import matplotlib.pyplot as plt
import networkx as nx
import pydotplus
import numpy as np
import itertools
from networkx.algorithms.approximation import vertex_cover, min_weighted_dominating_set
from collections import defaultdict
import pickle

def to_binary(inword, bits):
    pad = '{0:0' + str(bits) +'b}'
    res = [int(i) for i in list(pad.format(inword))]
    return res
def hash_list(lst):
    hashstr = ''
    for i in range(len(lst)-1):
        hashstr+=str(lst[i])
    return hashstr
def hash_whole_list(lst):
    hashstr = ''
    for i in range(len(lst)):
        hashstr+=str(lst[i])
    return hashstr

def hamming_dist(lst1, lst2):
    dist = 0
    for i in range(len(lst1)):
        if(lst1[i] != lst2[i]):
            dist+=1
    return dist

class label(Exception): pass

def to_decimal_gray(lst): #a general gray code implementation should be used instead of for 2 and 3 bits only
    if(len(lst)==2):
        if(lst == [0,0]):
            return 0
        elif(lst == [0,1]):
            return 1
        elif(lst == [1,0]):
            return 3
        elif(lst == [1,1]):
            return 2

    if(len(lst)==3):
        if(lst == [0,0,0]):
            return 0
        elif(lst == [0,0,1]):
            return 1
        elif(lst == [0,1,0]):
            return 3
        elif(lst == [0,1,1]):
            return 2
        elif(lst == [1,0,0]):
            return 7
        elif(lst == [1,0,1]):
            return 6
        elif(lst == [1,1,0]):
            return 4
        elif(lst == [1,1,1]):
            return 5
    return 0

def to_decimal(lst):
    sum = 0
    idx = len(lst)-1
    for i in lst:
        sum += i * 2**idx
        idx-=1
    return sum

def generate_bfn(in_num):
    pad = '{0:0' + str(2**in_num) +'b}'
    res = list(map(lambda n : [int(i) for i in list(pad.format(n))], range(2**(2**in_num))))

    if in_num >1:
        res = res[1:-1]
        in_len = 2**in_num
        div = 2
        for _ in range(in_num):
            arr = [0]*in_len
            for j in range(in_len):
                if j%div >= div/2 :
                    arr[j] = 1
            div *= 2
            res.remove(arr)
            arr = list(map(lambda n : (n+1)%2, arr))
            res.remove(arr)
    return res


def check_sets(s1, s2, clq):
    if(len(set(clq).intersection(s1))==0):
        return False

    if(len(set(clq).intersection(s2))==0):
        return False
    return True

class Node:
    def __init__(self, label, funcs, graph):
        self.label = label
        self.output = 0
        self.funcs = funcs
        self.g = graph
        self.fidx = 0
        self.funlen = len(funcs)-1
        self.fun = self.funcs[0]
        self.funid = (self.label, self.fidx) #to indicate which function is in the functional decoupling
        self.allfunids = [(self.label, idx) for idx,val in enumerate(self.funcs)]
        self.decoupling = []
        self.decoupled_set = []


    def compute(self, predvals):
        self.output = self.fun[to_decimal(predvals)]
        return self.output
    def next_fun(self):
        if self.fidx < self.funlen:
            self.fidx = self.fidx + 1
            self.fun = self.funcs[self.fidx]
            self.funid = (self.label, self.fidx)
            return True
        else:
            self.fidx = 0
            self.fun = self.funcs[self.fidx]
            self.funid = (self.label, self.fidx)
            return False
    def refresh(self):
        self.fidx = 0
        self.funlen = len(self.funcs)-1
        self.fun = self.funcs[0]
        self.funid = (self.label, self.fidx)
        self.allfunids = [(self.label, idx) for idx,val in enumerate(self.funcs)]


class Topology:
    def __init__(self, filename):
        g = graphviz.Source.from_file(filename)
        pydotgraph = pydotplus.graph_from_dot_data(g.source)
        self.topology = nx.nx_pydot.from_pydot(pydotgraph)
        self.output_nodes = [u for u, deg in self.topology.out_degree() if not deg]
        self.input_nodes = [u for u, deg in self.topology.in_degree() if not deg]
        in_num = max([deg for u,deg in self.topology.in_degree()]) #to find which boolean funcs are being considered
        bfns = list(map(lambda n : generate_bfn(n), range(in_num+1))) #generate binary funs up to max inputs

        self.nodemap = {} #dict from label to node
        constfn = [[0,1]]
        for node in self.input_nodes:
            self.nodemap[node] = Node(node, constfn, self.topology)
        for node in self.output_nodes:
            self.nodemap[node] = Node(node, constfn, self.topology)

        for u, deg in self.topology.in_degree():
            if deg and self.topology.out_degree(u):
                # self.possible_fns[u] = bfns[deg+1] # assign a function to every mid node
                self.nodemap[u] = Node(u, bfns[deg], self.topology)
        self.nodeorder = list(nx.topological_sort(self.topology))
        self.nodeorder = [item for item in self.nodeorder if item not in self.input_nodes]
        self.nodeorder = [item for item in self.nodeorder if item not in self.output_nodes]
        self.funlist = list(map(lambda n : self.nodemap[n], self.nodeorder))
        self.iteration_count = 1
        adjsize = 0
        for fun in self.funlist:
            self.iteration_count = self.iteration_count * len(fun.funcs)
            adjsize += len(fun.funcs)
            print(len(fun.funcs))
        self.adjindices = [el.allfunids for el in self.funlist]
        self.adjindices = list(itertools.chain(*self.adjindices))
        self.adjdict = {val:idx for idx,val in enumerate(self.adjindices)}
        self.separateidx = []
        for fun in self.funlist:
            dictindices = [self.adjdict[el] for el in fun.allfunids]
            self.separateidx.append(dictindices)
            edges = itertools.combinations(dictindices, 2)
        self.funcombs = []
        self.continuity_metric = 1

    def configure_metric(self, cont_metric):
        if(cont_metric == 'single'):
            self.continuity_metric = 0
        else:
            self.continuity_metric = 1



    def run(self, inlst): #for a given input inlst compute the output for the topology with the current functions assigned
        i = 0
        for label in self.input_nodes:
            self.nodemap[label].output = inlst[i]
            i = i+1

        for label in self.nodeorder:
            node = self.nodemap[label]
            preds = self.topology.predecessors(label)
            predvals = list(map(lambda n: self.nodemap[n].output, preds))
            node.compute(predvals)

        for label in self.output_nodes:
            self.nodemap[label].output = self.nodemap[list(self.topology.predecessors(label))[0]].output

    def is_single_cont(self, inputs, results, k):
        inp_combs = self.inp_combinas
        a = 0
        b = 0
        for comb in inp_combs:
            if results[comb[0]] != results[comb[1]]:
                a+=1
            else:
                b+=1
        if(a/(a+b) < 1/k):
            return True
        return False


    def is_Lipschitz(self, inputs, results, k):
        inp_idx = list(range(len(inputs)))
        inp_combs = list(itertools.combinations(inp_idx, 2))
        for comb in inp_combs:
            sum = abs(to_decimal(results[comb[0]]) - to_decimal(results[comb[1]]))
            suminp = abs(to_decimal(inputs[comb[0]]) - to_decimal(inputs[comb[1]]))
            if sum > k*suminp:
                return False
        return True


    def add_nodes_to_adj(self):
        ids = [el.funid for el in self.funlist]
        dictindices = [self.adjdict[el] for el in ids]
        self.funcombs.append(dictindices)


    def add_nodes_to_funcombs(self): #make a list of all fun combs which are lip cont
        ids = [el.fidx for el in self.funlist]
        self.funcombs.append(ids)

    def iterate(self, const_k):
        bitnum = len(self.input_nodes)
        maxin = 2**bitnum
        inputwords = [to_binary(el, bitnum) for el in range(maxin)]
        inp_idx = list(range(len(inputwords)))
        self.inp_combinas = list(itertools.combinations(inp_idx, 2))
        succ = 0
        nextfunidx = 0
        for k in range(self.iteration_count):
            results = []
            for inlst in inputwords:
                self.run(inlst)
                outword = [self.nodemap[u].output for u in self.output_nodes]
                results.append(outword)
            cont = False
            if(self.continuity_metric):
                cont = self.is_Lipschitz(inputwords, results, const_k)
            else:
                cont = self.is_single_cont(inputwords, results, const_k)
            if(cont):
                # self.add_nodes_to_adj()
                self.add_nodes_to_funcombs()
                succ+=1
            while(nextfunidx < len(self.funlist) and not self.funlist[nextfunidx].next_fun()):
                nextfunidx = nextfunidx+1
            nextfunidx = 0
        print ("Total functions meeting metric criteria: ", succ, " out of ", self.iteration_count);
        return succ



    def find_decoupling(self):

        combo = self.funcombs

        fcols = []
        for i in range(len(combo[0])):
            fcols.append([x[i] for x in combo])

        funlengths = [len(x) for x in fcols]


        sz = len(combo[0])
        for i in range(sz-1, -1, -1):
            idxdict = {}
            funmap = defaultdict(list)
            set2map = defaultdict(list)
            idxmap = []
            keymap = []
            ctr = 0
            score = 0
            bestscore = 0
            bestix = []
            bestedge = []
            for el in combo:
                set2map[el[-1]].append(hash_list(el))
                idxdict[hash_list(el)] = el[:-1]
            set2keys = list(set2map.keys())
            for j in range(len(set2keys), 0, -1):
                edges = itertools.combinations(set2keys, j)
                for edge in edges:
                    hashes = [set(set2map[x]) for x in edge]
                    ix = list(set.intersection(*hashes))
                    if(j*len(ix) > bestscore):
                        bestscore = j*len(ix)
                        bestix = ix
                        bestedge = edge

            self.funlist[i].decoupling = list(bestedge)
            combo = [idxdict[x] for x in bestix]

        cardin = 1
        for node in self.nodeorder:
            print(self.nodemap[node].label, self.nodemap[node].decoupling)
            cardin*=len(self.nodemap[node].decoupling)
        print('cardinality = ', cardin)


    def find_composite(self):
        bitnum = len(self.input_nodes)
        maxin = 2**bitnum
        inputwords = [to_binary(el, bitnum) for el in range(maxin)]
        succ = 0
        nextfunidx = 0

        composite_funs = []
        itrcount = 1
        for node in self.nodeorder:
            funindices = self.nodemap[node].decoupling
            itrcount *= len(funindices)
            # print(funindices)
            decfuns = [self.nodemap[node].funcs[x] for x in funindices]
            # print(decfuns)
            self.nodemap[node].funcs = decfuns
            self.nodemap[node].refresh()

        composite_funlst = []
        comp_lst = []
        for k in range(itrcount):
            composite_fun = defaultdict(list)
            results = []
            comp_res = []
            for inlst in inputwords:
                self.run(inlst)
                outword = [self.nodemap[u].output for u in self.output_nodes]
                results.append(outword)
                comp_res = comp_res+outword
                composite_fun[hash_whole_list(inlst)].append(outword)

            composite_funs.append(composite_fun)
            composite_funlst.append(results)
            comp_lst.append(comp_res)
            while(nextfunidx < len(self.funlist) and not self.funlist[nextfunidx].next_fun()):
                nextfunidx = nextfunidx+1
            nextfunidx = 0

        if(not self.continuity_metric):
            file_name = "binarytree.pkl"
            open_file = open(file_name, "wb")
            pickle.dump(comp_lst, open_file)
            open_file.close()


        sumvc = 0
        keys = []
        for inlst in inputwords:
            keys.append(hash_whole_list(inlst))
        for i in range(len(results[0])):
            f = defaultdict(list)
            for fund in composite_funs:
                for key in fund:
                    f[key].append(fund[key][0][i])
            vc = 1
            while(vc<itrcount):
                fcombs = list(itertools.combinations(keys, vc))
                pad = '{0:0' + str(vc) +'b}'
                testset = list(map(lambda n : [int(i) for i in list(pad.format(n))], range((2**vc))))
                vcset = []

                for comb in fcombs:
                    resout = []
                    for j in range(itrcount):
                        resin = []
                        for k in comb:
                            resin.append(f[k][j])
                        resout.append(resin)
                    vcset.append(resout)


                all_seen = False
                for vset in vcset:
                    all_seen = True
                    for test in testset:
                        if test not in vset:
                            all_seen = False
                    if(all_seen):
                        vc+=1
                        break
                if(not all_seen):
                    break

            sumvc+=(vc-1)

        print('vc = ', sumvc)









if __name__ == "__main__":

    g = graphviz.Digraph('G', filename='topology.gv')



    # direct
    g.edge('f2', 'q')
    g.edge('f1', 'f3')
    g.edge('a1', 'f3')
    g.edge('f3', 's1')

    g.edge('f1', 'f2')
    g.edge('a1', 'f2')



    g.edge('a0', 'f0')
    g.edge('c', 'f0')
    g.edge('f0', 's0')

    g.edge('a0', 'f1')
    g.edge('c', 'f1')














    g.view()
    print(g.source)



    Top = Topology('topology.gv')
    # Top.configure_metric('single')    #use if analysing a tree topology
    Top.iterate(2)
    print('FINDING DECOUPLING')
    Top.find_decoupling()
    Top.find_composite()
