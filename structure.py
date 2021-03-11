from test_structure.test_structure import test_struct
import numpy as np

class structure():
    def __init__(self, model_type, n, X, EI_req=1*10**6):
        """
        model_type (str): beam, truss, unit or equiv_cant
        n (int): number of non-basic nodes
        X (array - float): locations of nodes and thickness of members
        EI_req (float): minimum EI target
        """
        if model_type == "unit" or model_type == "equiv_cant":
            self.mystruct = test_struct(num_units=1)
        else:
            self.mystruct = test_struct()
        
        node_locs = np.array(X[0:3*n]).reshape(n, 3)
        self.mystruct.make_nodes(node_locs)
        self.mystruct.make_mem_ps(X[3*n:])
        self.mystruct.make_struct()
        if model_type == "truss":
            self.mystruct.release_moments()

        self.mystruct.record_struct_info()
        if model_type == "equiv_cant":
            EI = self.mystruct.get_equiv_EI()
        else:
            EI = self.mystruct.get_EI(100)
        
        if self.mystruct.min_l<0.01:
            self.score = 10**9
        elif np.isnan(EI):
            self.score = 10**10
        else:
            self.score = self.mystruct.mass+10**7*max(1/EI-1/EI_req, 0)
