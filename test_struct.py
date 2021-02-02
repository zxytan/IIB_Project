# Units for this model are mm

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
    def __init__(self, E, G, Iy, Iz, J, A):
        self.E = E
        self.G = G
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.A = A

# Create a new model
def make_truss(num_units, non_basic_nodes, mem_p):
    """
    function to create pynite truss model of test struct
    inputs:
    num_units = number of base units to make up struct
    non_basic_nodes = additional nodes w/ relative positions and names
    mem_p = list of member properties, order seen here: https://photos.app.goo.gl/GNUjZherzaSmKiod9
    """

    num_non_basic_nodes = len(non_basic_nodes)    
    num_members = (num_units*2+1)*4+num_units*(num_non_basic_nodes*8+(num_non_basic_nodes*(num_non_basic_nodes-1)/2))
    if len(mem_p) != num_members:
        print(f'wrong number of member properties supplied! must be {num_members}')
        return

    truss = FEModel3D()  
    #define nodes
    for i in range(num_units+1):
        truss.AddNode(f'{i*4+1}', i*20, 0, 0)
        truss.AddNode(f'{i*4+2}', i*20, 0, 20)
        truss.AddNode(f'{i*4+3}', i*20, 20, 20)
        truss.AddNode(f'{i*4+4}', i*20, 20, 0)

    non_basic_node_names = []
    for i in range(num_units):
        for node in non_basic_nodes:
            truss.AddNode(node.name+str(i+1), i*20+node.x, node.y, node.z)
            non_basic_node_names.append(node.name+str(i+1))

    base_node_names = np.array(range(1, (num_units+1)*4+1))
    base_node_names = base_node_names.reshape(-1, 4).astype('str')

    print(base_node_names)
    
    for i, node_group in enumerate(base_node_names):
        #connect basic nodes in one plane
        truss.AddMember(node_group[0]+'-'+node_group[1], node_group[0], node_group[1], mem_p[0].E, mem_p[0].G, mem_p[0].Iy, mem_p[0].Iz, mem_p[0].J, mem_p[0].A)
        truss.AddMember(node_group[1]+'-'+node_group[2], node_group[1], node_group[2], mem_p[1].E, mem_p[1].G, mem_p[1].Iy, mem_p[1].Iz, mem_p[1].J, mem_p[1].A)
        truss.AddMember(node_group[2]+'-'+node_group[3], node_group[2], node_group[3], mem_p[2].E, mem_p[2].G, mem_p[2].Iy, mem_p[2].Iz, mem_p[2].J, mem_p[2].A)
        truss.AddMember(node_group[3]+'-'+node_group[0], node_group[3], node_group[0], mem_p[3].E, mem_p[3].G, mem_p[3].Iy, mem_p[3].Iz, mem_p[3].J, mem_p[3].A)


    for i in range(num_units):
        non_basic_node_group = non_basic_node_names[i*num_non_basic_nodes:(i+1)*num_non_basic_nodes]
        #connect non basic nodes in a unit
        idx = -int(num_non_basic_nodes*(num_non_basic_nodes-1)/2)
        for n, node in enumerate(non_basic_node_group):
            for m, node_2 in enumerate(non_basic_node_group):
                if n < m:
                    truss.AddMember(node+'-'+node_2, node, node_2, mem_p[idx].E, mem_p[idx].G, mem_p[idx].Iy, mem_p[idx].Iz, mem_p[idx].J, mem_p[idx].A)
                    idx += 1


        base_node_group = base_node_names.flatten()[i*4:i*4+8]
        #connect base nodes to appropriate non basic nodes in a unit
        for i, node in enumerate(non_basic_node_group):
            for n, base_node in enumerate(base_node_group):
                truss.AddMember(base_node+'-'+node, base_node, node, mem_p[8+i*8+n].E, mem_p[8+i*8+n].G, mem_p[8+i*8+n].Iy, mem_p[8+i*8+n].Iz, mem_p[8+i*8+n].J, mem_p[8+i*8+n].A)


    #connect base nodes across units
    for n, node_group in enumerate(base_node_names.T):
        for i in range(num_units):
            truss.AddMember(node_group[i]+'-'+node_group[i+1], node_group[i], node_group[i+1], mem_p[n+4].E, mem_p[n+4].G, mem_p[n+4].Iy, mem_p[n+4].Iz, mem_p[n+4].J, mem_p[n+4].A)    

    truss.DefineSupport('1', True, True, True, True, True, True)
    truss.DefineSupport('2', True, True, True, True, True, True)
    truss.DefineSupport('3', True, True, True, True, True, True)
    truss.DefineSupport('4', True, True, True, True, True, True)

    

    return(truss)

node_a = my_node('a', 3, 5, 8)
node_b = my_node('b', 19, 18, 18)
members_p = member_props(9999,100,100,100,100,100)
node_c = my_node('c', 10, 20, 15)
truss = make_truss(1, [node_a], [members_p]*20)

Visualization.RenderModel(truss, text_height=1, render_loads=False)


