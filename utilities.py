import vtk
from vtk.util import numpy_support as ns
import numpy as np

class VData(object):
    """This class reads the vtk files produces for unstructured grids by fenics,
       and converts the associated mesh and values to numpy arrays"""
    def __init__(self,filename):
        self.f = filename
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(self.f)
        reader.Update()
        data = reader.GetOutput()

        self.npts = reader.GetNumberOfPoints()
        self.ntris = reader.GetNumberOfCells()
        self.npas = reader.GetNumberOfPointArrays()

        self.x = ns.vtk_to_numpy(data.GetPoints().GetData())
        self.tris = ns.vtk_to_numpy(data.GetCells().GetData()).reshape((-1,4))[:,1:]
        self.u = ns.vtk_to_numpy(data.GetPointData().GetArray(0))

    def get_point_area(self):
        area = np.ones_like(self.tris[:,0])
        for i,t in enumerate(self.tris):
            A = np.array([self.x[t[0]],self.x[t[1]],self.x[t[2]]])
            A[:,2] += 1
            area[i] = 0.5*abs(np.linalg.det(A))

        point_area = np.zeros_like(self.x[:,0])
        point_adj = np.zeros_like(self.x[:,0])

        for t,a in zip(self.tris,area):
            point_area[t] += a

        point_area = abs(point_area/3.)
        return point_area


    def plot_U(self,nc=30,colorbar=False):
        fig,axs = plt.subplots()
        p = axs.tricontourf(self.x[:,0],self.x[:,1],self.tris,np.linalg.norm(self.u,axis=1),np.linspace(0,100,31))
        xmin = self.x[:,0].min()
        xmax = self.x[:,0].max()
        ymin = self.x[:,1].min()
        ymax = self.x[:,1].max()
        w = 12
        h = w*(ymax-ymin)/(xmax-xmin)

        fig.set_size_inches(w,h)
        axs.axis('equal')
        return p,fig,axs



