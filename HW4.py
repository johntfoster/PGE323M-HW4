
import yaml

import numpy as np

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, inv

import matplotlib.pyplot as plt

class HW4(object):

    def __init__(self, input_filename):
        
        #Load the input file into dictionary
        with open(input_filename) as f:
            self.input_data = yaml.load(f)

        self.permeability = self.check_input_and_return_data('permeability')
        self.input_data['numerical']['number of grids'] = self.permeability.shape[0]
        self.porosity = self.check_input_and_return_data('porosity')

        self.ngrids = self.permeability.shape[0]

        self.res_height = self.input_data['reservoir']['height']
        self.res_width = self.input_data['reservoir']['width']
        self.res_length = self.input_data['reservoir']['length']
        self.res_area = self.res_height * self.res_width

        self.dx_arr = self.assign_dx_array()
        
        self.grid_numbers = np.arange(self.ngrids)


        self.fluid_viscosity = self.input_data['fluid']['viscosity']
        self.compressibility = self.input_data['fluid']['compressibility']
        self.form_volume_factor = self.input_data['fluid']['formation volume factor']

        self.conv_factor = self.input_data['unit conversion factor']

        self.time_step = self.input_data['numerical']['time step']
        self.final_time = self.input_data['numerical']['final time']

        self.number_of_time_steps = np.int(self.final_time / self.time_step) 

        self.solver_method = self.input_data['numerical']['method']

        self.initial_pressure = self.input_data['reservoir']['initial pressure']

        if 'mixed method theta' not in self.input_data['numerical']:
            self.solver_theta = 0.5
        else:
            self.solver_theta = self.input_data['numerical']['mixed method theta']

        if 'wells' not in self.input_data:
            self.well_grids = None
        else:
            self.well_grids = self.compute_well_index_locations()
            self.well_rates = np.array(self.input_data['wells']['rates'], 
                                       dtype=np.double)
            self.productivity_indices = self.compute_productivity_index()

        self.T, self.B, self.Q = self.assemble_matrices()

        return


    def check_input_and_return_data(self, input_name):

        #Check to see if data is given by a file
        if isinstance(self.input_data['reservoir'][input_name], str):
            #Get filename
            filename = self.input_data['reservoir'][input_name]
            #Load data 
            data = np.loadtxt(filename, dtype=np.double)
            
        #Check to see if data is given by a list
        elif isinstance(self.input_data['reservoir'][input_name], (list, tuple)):
            #Turn the list into numpy array
            data = np.array(self.input_data['reservoir'][input_name], 
                            dtype=np.double)

        #data is a constant array (homogenuous)
        else:
            ngrids = self.input_data['numerical']['number of grids']
            data = (self.input_data['reservoir'][input_name] * 
                              np.ones(ngrids))
        return data


    def assign_dx_array(self):

        #If dx is not defined by user, compute a uniform dx
        if 'delta x' not in self.input_data['numerical']:

            length = self.res_length
            dx = np.float(length) / self.ngrids

            dx_arr = np.ones(self.ngrids) * dx
        else:
            #Convert to numpy array and ensure that the length of 
            #dx matches ngrids
            dx_arr = np.array(self.input_data['numerical']['delta x'], 
                              dtype=np.double)

            length_dx_arr = dx_arr.shape[0]
            
            assert length_dx_arr == self.ngrids, ("User defined 'delta x' array \
                                                   doesn't match 'number of grids'")

        return dx_arr


    def compute_well_index_locations(self):
        
        dx = self.dx_arr

        grid_centers = np.cumsum(dx) - dx[0] / 2.0

        well_locations = np.array(self.input_data['wells']['locations'])

        #Returns True for grids that have a well
        bool_arr = np.all([grid_centers - dx / 2.0 < well_locations[:, None],
                           grid_centers + dx / 2.0 > well_locations[:, None]],
                           axis=0)

        #Find the grid # of the wells marked True above
        return np.array([self.grid_numbers[item] for item in bool_arr], 
                         dtype=np.int).ravel()


    def compute_half_transmissibility(self, i, j):

        #These are just pointers, there is no deep copy here.
        dx = self.dx_arr
        perm = self.permeability
        area = self.res_area
        viscosity = self.fluid_viscosity
        conv_factor = self.conv_factor
        Bw = self.form_volume_factor

        #Compute the half-grid permeability
        k_half = (dx[i] + dx[j]) / (dx[i] / perm[i] + dx[j] / perm[j])

        #Compute the half-grid distance
        dx_half = (dx[i] + dx[j]) / 2.0

        #Return the half-grid transmissibility
        return (k_half * area) / (viscosity * Bw * dx_half) * conv_factor


    def compute_accumulation(self, i):

        #These are just pointers, there is no deep copy here.
        dx = self.dx_arr
        phi = self.porosity
        area = self.res_area
        viscosity = self.fluid_viscosity
        cf = self.compressibility
        Bw = self.form_volume_factor

        return area * dx[i] * phi[i] * cf / Bw


    def assemble_matrices(self):

        bcs = self.input_data['boundary conditions']

        wells = self.input_data['wells']

        N = self.ngrids

        T = lil_matrix((N, N), dtype=np.double)
        B = np.zeros(N, dtype=np.double)
        Q = np.zeros(N, dtype=np.double)

        bc_type_1 = bcs['left']['type'].lower()
        bc_type_2 = bcs['right']['type'].lower()

        bc_value_1 = bcs['left']['value']
        bc_value_2 = bcs['right']['value']

        for i in range(N):

            #Apply left BC
            if i == 0:
                T[i, i+1] = -self.compute_half_transmissibility(i, i + 1)

                if bc_type_1 == 'neumann':
                    T[i, i] = T[i,i] - T[i, i+1]
                elif bc_type_1 == 'dirichlet':
                    T0 = self.compute_half_transmissibility(i, i)
                    T[i, i] = T[i,i] - T[i, i+1] + 2.0 * T0
                    Q[i] = 2.0 * T0 * bc_value_1
                else:
                    pass #Add error checking here if no bc is specified

            #Apply right BC
            elif i == (N - 1):
                T[i, i-1] = -self.compute_half_transmissibility(i, i - 1)

                if bc_type_2 == 'neumann':
                    T[i, i] = T[i,i] - T[i, i-1]
                elif bc_type_2 == 'dirichlet':
                    T0 = self.compute_half_transmissibility(i, i)
                    T[i, i] = T[i, i] - T[i, i-1] + 2.0 * T0
                    Q[i] = 2.0 * T0 * bc_value_2
                else:
                    pass #Add error checking here if no bc is specified

            else:
                T[i, i-1] = -self.compute_half_transmissibility(i, i-1)
                T[i, i+1] = -self.compute_half_transmissibility(i, i+1)
                T[i, i] = (self.compute_half_transmissibility(i, i-1) +
                           self.compute_half_transmissibility(i, i+1))

            B[i] = self.compute_accumulation(i)

        Q[self.well_grids] += self.well_rates

        
        return (T.tocsr(), 
                csr_matrix((B, (np.arange(N), np.arange(N))), shape=(N,N)), 
                Q)

    def compute_productivity_index(self):

        perm = self.permeability
        visc = self.fluid_viscosity
        dx = self.dx_arr
        h = self.res_height
        factor = self.conv_factor
        Bw = self.form_volume_factor
        grids = self.well_grids
        
        r_w = np.array(self.input_data['wells']['radii'], dtype=np.double)
        r_eq = dx[grids] * np.exp(-np.pi / 2.0)

        return ((factor * 2.0 * np.pi * perm[grids] * h) / 
                (visc * Bw * np.log(r_eq / r_w)))


    def compute_time_step(self, P_n):

        method = self.solver_method
        dt = self.time_step
        theta = self.solver_theta
        T = self.T
        B = self.B
        Q = self.Q

        if method == 'mixed':
            A = ((1.0 - theta) * T + B / dt)
            b = (B / dt - theta * T).dot(P_n) + Q
            P_np1 = spsolve(A, b)

        elif method == 'explicit':
            P = P_n + 1 / B * dt * (Q - T.dot(P_n))
        else:
            A = T + B / dt
            b = (B / dt).dot(P_n) + Q
            P_np1 = spsolve(A, b)

        return P_np1


    def run(self, plot_freq=None):
        

        self.P = np.ones(self.ngrids) * self.initial_pressure

        P_plot = []
        self.time = []

        for i in range(self.number_of_time_steps + 1):

            if (plot_freq is not None and i % plot_freq == 0):
                P_plot.append(self.P)
                self.time.append(i * self.time_step)
            elif (i == self.number_of_time_steps):
                P_plot.append(self.P)
                break

            self.P = self.compute_time_step(self.P)

        self.P_plot = np.array(P_plot)

        return 


    def get_solution(self):
        return self.P


    def plot(self, x_unit='ft', y_unit='psi'):

        x_pos = np.cumsum(self.dx_arr) - self.dx_arr[0] / 2.0

        plt.figure()
        for P in self.P_plot:
            plt.plot(x_pos, P)

        plt.xlabel('$x$ position (' + x_unit + ')')
        plt.ylabel('Pressure (' + y_unit + ')')
        plt.xlim([0, self.res_length])
        plt.show()
        
    def plot_BHP(self, x_unit='days', y_unit='psi'):

        time = self.time
        grids = self.well_grids
        rates = self.well_rates
        J = self.productivity_indices


        BHPs = self.P_plot[:,grids].T + (rates / J)[:, None]

        plt.figure()
        for BHP in BHPs:
            plt.plot(time, BHP)

        plt.xlabel('time (' + x_unit + ')')
        plt.ylabel('Bottom Hole Pressure (' + y_unit + ')')
        plt.show()

        
if __name__ == "__main__":
    
    problem = HW4('HW4.in')
    problem.run()
    problem.plot()
