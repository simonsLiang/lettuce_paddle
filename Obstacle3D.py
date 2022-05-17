import sys
sys.path.append('./')
import lettuce as lt
import paddle
import numpy as np
import matplotlib.pyplot as plt


paddle.set_device('gpu')

def EnergyReporter(lattice, flow, interval=1, starting_iteration=0, out=sys.stdout):
    from lettuce.observables import IncompressibleKineticEnergy
    return lt.ObservableReporter(IncompressibleKineticEnergy(lattice, flow), interval=interval, out=out)
print("start")
device = ''  # replace with device("cpu"), if no GPU is available
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=paddle.float32)  # single precision - float64 for double precision
resolution = 256  # resolution of the lattice, low resolution leads to unstable speeds somewhen after 10 (PU)
flow = flow = lt.Obstacle3D(resolution,resolution,resolution,1600, 0.05, lattice,10.1)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming)
kinE_reporter = EnergyReporter(lattice, flow, interval=1, out=None)
simulation.reporters.append(kinE_reporter)
VTKreport = lt.VTKReporter(lattice, flow, interval=1000, filename_base="./output")
simulation.reporters.append(VTKreport)
# ---------- Simulate until time = 10 (PU) -------------
print("MLUPS: ", simulation.step(10000))
E = np.asarray(kinE_reporter.out)
np.save("TGV3DoutRes" + str(resolution) + "E", E)
