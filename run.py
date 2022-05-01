import sys
sys.path.append('./lettuce')
import lettuce as lt
import paddle
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--reynolds", type=int, default=1600)
args = parser.parse_args()

paddle.set_device('gpu')

def EnergyReporter(lattice, flow, interval=1, starting_iteration=0, out=sys.stdout):
    from lettuce.observables import IncompressibleKineticEnergy
    return lt.ObservableReporter(IncompressibleKineticEnergy(lattice, flow), interval=interval, out=out)

print("start")
device = 'gpu'  # replace with device("cpu"), if no GPU is available
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=paddle.float32)  # single precision - float64 for double precision
flow = lt.TaylorGreenVortex3D(args.resolution, args.reynolds, 0.05, lattice)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming)
kinE_reporter = EnergyReporter(lattice, flow, interval=1, out=None)
simulation.reporters.append(kinE_reporter)
VTKreport = lt.VTKReporter(lattice, flow, interval=1000, filename_base="./output")
simulation.reporters.append(VTKreport)
print("Simulating", int(simulation.flow.units.convert_time_to_lu(10)), "steps! Maybe drink some water in the meantime.")
print("MLUPS: ", simulation.step(int(simulation.flow.units.convert_time_to_lu(10))))
Es = np.asarray(kinE_reporter.out)
np.save("TGV3DoutRes" + str(args.resolution) + "E", Es)