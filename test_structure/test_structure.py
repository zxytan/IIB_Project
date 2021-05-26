# Units for this model are m and N

# Import 'FEModel3D' and 'Visualization' from 'PyNite'
from PyNite import FEModel3D
from PyNite import Visualization
import numpy as np

#make node class for adding non basic nodes
class my_node():
    def __init__(self, name, x, y, z):
        self.name=name
        self.x = x
        self.y = y
        self.z = z

#make member properties class
class member_props():
    """
    gives member properties given diameter d (m), Young's modulus E (Pa) and Poisson ratio v
    default generic material E = 1GPa, v = 0.3
    """
    def __init__(self, d, E=1*10**9, v=0.3):
        self.E = E
        self.G = E/(2*(1+v))
        self.Iy = np.pi*(d/2)**4/4
        self.Iz = np.pi*(d/2)**4/4
        self.J = np.pi*(d/2)**4/2
        self.A = np.pi*(d/2)**2
        

class test_struct:
    def __init__(self, num_units=8, unit_len=1):
        self.num_units = num_units
        self.unit_len = unit_len
        self.L = num_units*unit_len
        self.num_nodes = 0
        self.non_basic_nodes = []

    def make_nodes(self, node_locs):
        self.num_nodes = len(node_locs)
        for i, node in enumerate(node_locs):
            self.non_basic_nodes.append(my_node(chr(97+i), node[0], node[1], node[2]))

    def make_mem_ps(self, member_ds, E=1*10**9, p=1, v=0.3):
        """
        member diameters list in order shown here: https://photos.app.goo.gl/GNUjZherzaSmKiod9
        default generic material E = 1GPa, v = 0.3, unit density
        """
        self.num_members = int((8+self.num_nodes*8+(self.num_nodes*(self.num_nodes-1)/2)))
        if len(member_ds) != self.num_members:
            raise NameError(f'wrong number of member diameters supplied! must be {self.num_members}')
        self.member_ds = member_ds
        self.E = E
        self.p = p
        self.v = v
        self.mem_p = []
        for d in self.member_ds:
            if d<0.001:
                d = 0
            self.mem_p.append(member_props(d, self.E, self.v))

    # Create a new model
    def make_struct(self):
        """
        function to create pynite struct model of test structure
        """
        self.struct = FEModel3D()
        #define nodes
        for i in range(self.num_units+1):
            self.struct.AddNode(f'{i*4+1}', i*self.unit_len, 0, 0)
            self.struct.AddNode(f'{i*4+2}', i*self.unit_len, 0, self.unit_len)
            self.struct.AddNode(f'{i*4+3}', i*self.unit_len, self.unit_len, self.unit_len)
            self.struct.AddNode(f'{i*4+4}', i*self.unit_len, self.unit_len, 0)

        non_basic_node_names = []

        for i in range(self.num_units):
            for node in self.non_basic_nodes:
                self.struct.AddNode(node.name+str(i+1), i*self.unit_len+node.x, node.y, node.z)
                non_basic_node_names.append(node.name+str(i+1))

        base_node_names = np.array(range(1, (self.num_units+1)*4+1)).astype('str')
        self.base_node_names = base_node_names.reshape(-1, 4)
                
        for i, node_group in enumerate(self.base_node_names):
            #connect basic nodes in one plane
            for n in [0, 1, 2, 3]:
                if n == 3:
                    m = 0
                else:
                    m = n+1
                if(self.mem_p[n].A!=0.):
                    self.struct.AddMember(node_group[n]+'-'+node_group[m], node_group[n], node_group[m], \
                        self.mem_p[n].E, self.mem_p[n].G, self.mem_p[n].Iy, self.mem_p[n].Iz, self.mem_p[n].J, self.mem_p[n].A)

        for i in range(self.num_units):
            non_basic_node_group = non_basic_node_names[i*self.num_nodes:(i+1)*self.num_nodes]
            #connect non basic nodes in a unit
            idx = -int(self.num_nodes*(self.num_nodes-1)/2)
            for n, node in enumerate(non_basic_node_group):
                for m, node_2 in enumerate(non_basic_node_group):
                    if n < m:
                        if(self.mem_p[idx].A!=0.):
                            self.struct.AddMember(node+'-'+node_2, node, node_2, \
                                self.mem_p[idx].E, self.mem_p[idx].G, self.mem_p[idx].Iy, self.mem_p[idx].Iz, self.mem_p[idx].J, self.mem_p[idx].A)
                        idx += 1


            base_node_group = base_node_names[i*4:i*4+8]
            #connect base nodes to appropriate non basic nodes in a unit
            for i, node in enumerate(non_basic_node_group):
                for n, base_node in enumerate(base_node_group):
                    idx = 8+i*8+n
                    if(self.mem_p[idx].A!=0.):
                        self.struct.AddMember(base_node+'-'+node, base_node, node, \
                            self.mem_p[idx].E, self.mem_p[idx].G, self.mem_p[idx].Iy, self.mem_p[idx].Iz, self.mem_p[idx].J, self.mem_p[idx].A)


        #connect base nodes across units
        for n, node_group in enumerate(self.base_node_names.T):
            for i in range(self.num_units):
                if(self.mem_p[n+4].A!=0.):
                    self.struct.AddMember(node_group[i]+'-'+node_group[i+1], node_group[i], node_group[i+1], \
                        self.mem_p[n+4].E, self.mem_p[n+4].G, self.mem_p[n+4].Iy, self.mem_p[n+4].Iz, self.mem_p[n+4].J, self.mem_p[n+4].A)    

        self.struct.DefineSupport('1', True, True, True, False, False, False)
        self.struct.DefineSupport('2', True, True, True, False, False, False)
        self.struct.DefineSupport('3', True, True, True, False, False, False)
        self.struct.DefineSupport('4', True, True, True, False, False, False)

    def record_struct_info(self):
        Ls = []
        member_masses = []
        for member in self.struct.Members:
            L = member.L()
            Ls.append(L)
            A = member.A
            member_masses.append(self.p*L*A)
        self.min_l = min(Ls)
        self.mass = sum(member_masses)

    def release_moments(self):
        for member in self.struct.Members:
            self.struct.DefineReleases(member.Name, False, False, False, False, True, True, \
                           False, False, False, False, True, True)

    def get_EI(self, tip_load):
        self.struct.ClearLoads()
        load_nodes = self.base_node_names[-1]
        for node in load_nodes:
            self.struct.AddNodeLoad(node, 'FY', -tip_load/4)
        self.struct.Analyze()
        deflections = []
        for i in range(4):
            deflections.append(abs(self.struct.GetNode(load_nodes[i]).DY['Combo 1']))
        deflection = np.mean(deflections)
        return(tip_load*self.L**3/(3*abs(deflection)))


    def get_cant_EI(self):
        area = self.mass/(self.L*self.p)
        r = np.sqrt(area/np.pi)
        I = np.pi*r**4/4
        return(self.E*I)

    def get_equiv_beam_EI(self, tip_load):
        member_mass = self.mass/(self.num_units*8+4)
        member_area = member_mass/(self.unit_len/self.p)
        member_diameter = 2*np.sqrt(member_area/np.pi)
        equiv_beam = test_struct(unit_len=self.unit_len)
        equiv_beam.make_nodes([])
        equiv_beam.make_mem_ps([member_diameter]*8)
        equiv_beam.make_struct()
        equiv_beam.record_struct_info()
        return(equiv_beam.get_EI(tip_load))


# node_locs = np.array([[17, 9, 2]])/20
# member_ds = [0.04]*10 + [0]*5 + [0.01]

# s = test_struct(num_units=1)
# s.make_nodes(node_locs)
# s.make_mem_ps(member_ds)
# s.make_struct()
# s.record_struct_info()
# print(s.mass)
# print(s.get_EI(100))
# print(s.get_cant_EI())
# print(s.get_equiv_beam_EI())
# Visualization.RenderModel(s.struct, text_height=0.05, render_loads=True, deformed_shape=True, deformed_scale=1)
