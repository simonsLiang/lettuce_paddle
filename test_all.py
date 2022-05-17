import paddle
import lettuce as lt
device = "gpu"  
dtype = paddle.float32

lattice = lt.Lattice( lt.D2Q9, device, dtype)
flow =  lt.Obstacle2D(256,256, reynolds_number=10, mach_number=0.05,lattice=lattice,char_length_lu=10.1)
collision =  lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming =  lt.StandardStreaming(lattice)
simulation =  lt.Simulation(flow=flow, lattice=lattice,  collision=collision, streaming=streaming)
mlups = simulation.step(num_steps=1000)
print("Obstacle2D Performance in MLUPS:", mlups)

lattice = lt.Lattice( lt.D2Q9, device, dtype)
flow =  lt.CouetteFlow2D(256,reynolds_number=10, mach_number=0.05,lattice=lattice)
collision =  lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming =  lt.StandardStreaming(lattice)
simulation =  lt.Simulation(flow=flow, lattice=lattice,  collision=collision, streaming=streaming)
mlups = simulation.step(num_steps=1000)
print("CouetteFlow2D Performance in MLUPS:", mlups)

lattice = lt.Lattice( lt.D2Q9, device, dtype)
flow =  lt.DecayingTurbulence(256, reynolds_number=10, mach_number=0.05,lattice=lattice)
collision =  lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming =  lt.StandardStreaming(lattice)
simulation =  lt.Simulation(flow=flow, lattice=lattice,  collision=collision, streaming=streaming)
mlups = simulation.step(num_steps=1000)
print("DecayingTurbulence Performance in MLUPS:", mlups)

lattice = lt.Lattice( lt.D2Q9, device, dtype)
flow =  lt.PoiseuilleFlow2D(256, reynolds_number=10, mach_number=0.05,lattice=lattice)
collision =  lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming =  lt.StandardStreaming(lattice)
simulation =  lt.Simulation(flow=flow, lattice=lattice,  collision=collision, streaming=streaming)
mlups = simulation.step(num_steps=1000)
print("PoiseuilleFlow2D Performance in MLUPS:", mlups)