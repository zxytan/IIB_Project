# Units for this model are mm

# Import 'FEModel3D' and 'Visualization' from 'PyNite'
from PyNite import FEModel3D
from PyNite import Visualization

class my_node():
    def __init__(self, name, x, y, z):
        self.name=name
        self.x = x
        self.y = y
        self.z = z

# Create a new model


def make_unit(origin, non_basic_nodes):

    unit_truss = FEModel3D()
    #define nodes
    unit_truss.AddNode('1', origin[0], origin[1], origin[2])
    unit_truss.AddNode('2', origin[0], origin[1], origin[2]+20)
    unit_truss.AddNode('3', origin[0], origin[1]+20, origin[2]+20)
    unit_truss.AddNode('4', origin[0], origin[1]+20, origin[2])
    unit_truss.AddNode('5', origin[0]+20, origin[1], origin[2])
    unit_truss.AddNode('6', origin[0]+20, origin[1], origin[2]+20)
    unit_truss.AddNode('7', origin[0]+20, origin[1]+20, origin[2]+20)
    unit_truss.AddNode('8', origin[0]+20, origin[1]+20, origin[2])
    
    for node in non_basic_nodes:
        unit_truss.AddNode(node.name, origin[0]+node.x, origin[1]+node.y, origin[2]+node.z)

    #make members
    E= 9999999
    G = 100
    Iy = 100
    Iz = 100
    J=100
    A=100
    unit_truss.AddMember('12', '1', '2', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('14', '1', '4', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('15', '1', '5', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('23', '2', '3', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('26', '2', '6', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('34', '3', '4', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('37', '3', '7', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('48', '4', '8', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('56', '5', '6', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('58', '5', '8', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('67', '6', '7', E, G, Iy, Iz, J, A)
    unit_truss.AddMember('78', '7', '8', E, G, Iy, Iz, J, A)

    for base_node in ['1', '2', '3', '4', '5', '6', '7', '8']:
        for node in non_basic_nodes:
            unit_truss.AddMember(base_node+node.name, base_node, node.name, E, G, Iy, Iz, J, A)
    for i, node in enumerate(non_basic_nodes):
        for n, node_2 in enumerate(non_basic_nodes):
            if i < n:
                unit_truss.AddMember(node.name+node_2.name, node.name, node_2.name, E, G, Iy, Iz, J, A)

    unit_truss.DefineSupport('5', True, True, True, True, True, True)
    unit_truss.DefineSupport('6', True, True, True, True, True, True)
    unit_truss.DefineSupport('7', True, True, True, True, True, True)
    unit_truss.DefineSupport('8', True, True, True, True, True, True)

    return(unit_truss)


# # Add nodal loads
# truss.AddNodeLoad('A', 'FX', 10)
# truss.AddNodeLoad('A', 'FY', 60)
# truss.AddNodeLoad('A', 'FZ', 20)

# # Analyze the model
# truss.Analyze()

# # Print results
# print('Member BC calculated axial force: ' + str(truss.GetMember('BC').MaxAxial()))
# print('Member BC expected axial force: 32.7 Tension')
# print('Member BD calculated axial force: ' + str(truss.GetMember('BD').MaxAxial()))
# print('Member BD expected axial force: 45.2 Tension')
# print('Member BE calculated axial force: ' + str(truss.GetMember('BE').MaxAxial()))
# print('Member BE expected axial force: 112.1 Compression')

# # Render the model for viewing. The text height will be set to 50 mm.
# # Because the members in this example are nearly rigid, there will be virtually no deformation. The deformed shape won't be rendered.
# # The program has created a default load case 'Case 1' and a default load combo 'Combo 1' since we didn't specify any. We'll display 'Case 1'.
node_a = my_node('a', 3, 5, 8)
node_b = my_node('b', 19, 18, 18)
node_c = my_node('c', 10, 20, 15)
unit = make_unit([0,0,0], [node_a, node_b, node_c])

Visualization.RenderModel(unit, text_height=0.05, render_loads=True, case='Case 1')


