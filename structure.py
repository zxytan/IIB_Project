from test_structure.test_structure import test_struct
import numpy as np

class structure():
    def __init__(self, model_type, n, X, EI_req=1*10**7):
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
        self.mass = self.mystruct.mass
        if model_type == "equiv_cant":
            self.EI = self.mystruct.get_cant_EI()
        else:
            self.EI = self.mystruct.get_EI(100)
        
        if self.mystruct.min_l<0.01:
            self.score = 10**9
        elif np.isnan(self.EI):
            self.score = 10**10
        else:
            self.score = self.mass+10**7*max(1/self.EI-1/EI_req, 0)
        
        self.equiv_EI = self.mystruct.get_equiv_beam_EI(100)
        self.equiv_score = self.mass+10**7*max(1/self.equiv_EI-1/EI_req, 0)
