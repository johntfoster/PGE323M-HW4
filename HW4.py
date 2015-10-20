
import yaml

import numpy as np

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, inv

import matplotlib.pyplot as plt

class HW4(object):

    def __init__(self, input_filename):
        """
           Initializes the HW4 class by parsing the input deck and computing
           interblock transmissibilities, accumulation, boundary conditions, 
           and wells.
        """
        
        #Load the input file into dictionary
        with open(input_filename) as f:
            self.input_data = yaml.load(f)

        #Parse permeability and porosity values, can read from file, accept
        #user defined list, or constant values
        self.permeability = self.check_input_and_return_data('permeability')
        self.input_data['numerical']['number of grids'] = self.permeability.shape[0]
        self.porosity = self.check_input_and_return_data('porosity')

        #Sets the 'number of grids' equivalent to the length of the permeability
        #array.  This will override any user defined values in case the
        #permeability was read from file and the length differs from the 
        #'number of grids' in the input deck
        self.ngrids = self.permeability.shape[0]

        #Read in reservoir parameters
        self.res_height = self.input_data['reservoir']['height']
        self.res_width = self.input_data['reservoir']['width']
        self.res_length = self.input_data['reservoir']['length']
        self.res_area = self.res_height * self.res_width

        #Assign/compute the grid block lengths
        self.dx_arr = self.assign_dx_array()
        
        #Assign numbers (indices) to the grids
        self.grid_numbers = np.arange(self.ngrids)

        #Read in fluid properties
        self.fluid_viscosity = self.input_data['fluid']['viscosity']
        self.compressibility = self.input_data['fluid']['compressibility']
        self.form_volume_factor = self.input_data['fluid']['formation volume factor']

        #Read in 'unit conversion factor' if it exists in the input deck, 
        #otherwise set it to 1.0
        if 'unit conversion factor' in self.input_data:
            self.conv_factor = self.input_data['unit conversion factor']
        else:
            self.conv_factor = 1.0

        #Read in and compute numerical parameters
        self.time_step = self.input_data['numerical']['time step']
        self.final_time = self.input_data['numerical']['final time']
        self.number_of_time_steps = np.int(self.final_time / self.time_step) 

        #Check solver type, if mixed method is used, check theta value otherwise
        #set it to 0.5 (Crank-Nicholson)
        self.solver_method = self.input_data['numerical']['method']
        if 'mixed method theta' in self.input_data['numerical']:
            self.solver_theta = self.input_data['numerical']['mixed method theta']
        else:
            self.solver_theta = 0.5

        #Read initial pressure
        self.initial_pressure = self.input_data['reservoir']['initial pressure']

        #If wells are present, find their grid indices, and compute productivity
        #index
        if 'wells' in self.input_data:
            if 'rate' in self.input_data['wells']:
                self.rate_well_grids = self.compute_well_index_locations('rate')
                self.rate_well_values = np.array(self.input_data['wells']['rate']['values'], 
                                                 dtype=np.double)
                self.rate_well_prod_ind = self.compute_productivity_index('rate')
            else:
                self.rate_well_grids = None

            if 'bhp' in self.input_data['wells']:
                self.bhp_well_grids = self.compute_well_index_locations('bhp')
                self.bhp_well_values = np.array(self.input_data['wells']['bhp']['values'], 
                                           dtype=np.double)
                self.bhp_well_prod_ind = self.compute_productivity_index('bhp')
            else:
                self.bhp_well_grids = None
        else:
            self.rate_well_grids = None
            self.bhp_well_grids = None


        #Compute interblock transmissibilities, accumulation, and rate vector
        self.T, self.B, self.Q = self.assemble_matrices()

        return


    def check_input_and_return_data(self, input_name):
        """
           Used to parse the permeability and porosity from the input deck 
           depending on whether they are to be read from file, given by user
           input lists or constants.
        """

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

        #data is a constant array (homogeneous)
        else:
            ngrids = self.input_data['numerical']['number of grids']
            data = (self.input_data['reservoir'][input_name] * 
                              np.ones(ngrids))
        return data


    def assign_dx_array(self):
        """
           Used to assign grid block widths (dx values) after pereability
           and porosity has been assigned.

           Can also accept user defined list of dx values.

           TODO: Add ability to read dx values from file.
        """

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
            
            #For user input 'delta x' array, we need to ensure that its size
            #agress with ngrids as determined from permeability/porosity values
            assert length_dx_arr == self.ngrids, ("User defined 'delta x' array \
                                                   doesn't match 'number of grids'")

        return dx_arr


    def compute_well_index_locations(self, well_type='rate'):
        """
           Used to find well index locations from given coordinate positions.
        """
        
        #Reassignment for convenience, not a deep-copy
        dx = self.dx_arr

        #Compute grid centers
        grid_centers = np.cumsum(dx) - dx[0] / 2.0

        #Coordinate locations of wells
        well_locations = np.array(self.input_data['wells'][well_type]['locations'])

        #Returns True for grids that have a well
        bool_arr = np.all([grid_centers - dx / 2.0 < well_locations[:, None],
                           grid_centers + dx / 2.0 > well_locations[:, None]],
                           axis=0)

        #Find the grid # of the wells marked True above
        return np.array([self.grid_numbers[item] for item in bool_arr], 
                         dtype=np.int).ravel()


    def compute_transmissibility(self, i, j):
        """
           Compute the transmissibility between blocks i and j
        """

        #These are just pointer reassignments, not a deep-copy.
        dx = self.dx_arr
        perm = self.permeability
        area = self.res_area
        viscosity = self.fluid_viscosity
        factor = self.conv_factor
        Bw = self.form_volume_factor

        #Compute the half-grid permeability
        k_half = (dx[i] + dx[j]) / (dx[i] / perm[i] + dx[j] / perm[j])

        #Compute the half-grid distance
        dx_half = (dx[i] + dx[j]) / 2.0

        #Return the half-grid transmissibility
        return (k_half * area) / (viscosity * Bw * dx_half) * factor


    def compute_accumulation(self, i):
        """
           Computes the accumulation value for block i
        """

        #These are just pointer reassignments, not a deep-copy.
        dx = self.dx_arr
        phi = self.porosity
        area = self.res_area
        cf = self.compressibility
        Bw = self.form_volume_factor

        return area * dx[i] * phi[i] * cf / Bw


    def assemble_matrices(self):
        """
           Assemble the transmisibility, accumulation matrices, and the flux
           vector.  Returns sparse data-structures
        """
        
        #Pointer reassignment for convenience
        N = self.ngrids

        #Begin with a linked-list data structure for the transmissibilities,
        #and one-dimenstional arrays for the diagonal of B and the flux vector
        T = lil_matrix((N, N), dtype=np.double)
        B = np.zeros(N, dtype=np.double)
        Q = np.zeros(N, dtype=np.double)

        #Read in boundary condition types and values
        bcs = self.input_data['boundary conditions']
        bc_type_1 = bcs['left']['type'].lower()
        bc_type_2 = bcs['right']['type'].lower()
        bc_value_1 = bcs['left']['value']
        bc_value_2 = bcs['right']['value']
      
        #Loop over all grid cells
        for i in range(N):

            #Apply left BC
            if i == 0:
                T[i, i+1] = -self.compute_transmissibility(i, i + 1)

                if bc_type_1 == 'neumann':
                    T[i, i] = T[i,i] - T[i, i+1]
                elif bc_type_1 == 'dirichlet':
                    #Computes the transmissibility of the ith block
                    T0 = self.compute_transmissibility(i, i)
                    T[i, i] = T[i,i] - T[i, i+1] + 2.0 * T0
                    Q[i] = 2.0 * T0 * bc_value_1
                else:
                    pass #TODO: Add error checking here if no bc is specified

            #Apply right BC
            elif i == (N - 1):
                T[i, i-1] = -self.compute_transmissibility(i, i - 1)

                if bc_type_2 == 'neumann':
                    T[i, i] = T[i,i] - T[i, i-1]
                elif bc_type_2 == 'dirichlet':
                    #Computes the transmissibility of the ith block
                    T0 = self.compute_transmissibility(i, i)
                    T[i, i] = T[i, i] - T[i, i-1] + 2.0 * T0
                    Q[i] = 2.0 * T0 * bc_value_2
                else:
                    pass #TODO:Add error checking here if no bc is specified

            #If there is no boundary condition compute interblock transmissibilties
            else:
                T[i, i-1] = -self.compute_transmissibility(i, i-1)
                T[i, i+1] = -self.compute_transmissibility(i, i+1)
                T[i, i] = (self.compute_transmissibility(i, i-1) +
                           self.compute_transmissibility(i, i+1))

            #Compute accumulations
            B[i] = self.compute_accumulation(i)

        #If constant-rate wells are present, add them to the flux vector
        if self.rate_well_grids is not None:
            Q[self.rate_well_grids] += self.rate_well_values

        
        #Return sparse data-structures
        return (T.tocsr(), 
                csr_matrix((B, (np.arange(N), np.arange(N))), shape=(N,N)), 
                Q)

    def compute_productivity_index(self, well_type='rate'):
        """
           Used to compute productivity indices of wells.  All indices for
           a 'well_type' are computed and returned at once (vectorized)
        """

        #Pointer reassignment for convenience
        perm = self.permeability
        visc = self.fluid_viscosity
        dx = self.dx_arr
        h = self.res_height
        factor = self.conv_factor
        Bw = self.form_volume_factor

        #Get grid indices for 'well_type' wells
        if well_type == 'rate':
            grids = self.rate_well_grids
        elif well_type == 'bhp':
            grids = self.bhp_well_grids
        
        #Read in well radius from input file
        r_w = np.array(self.input_data['wells'][well_type]['radii'], 
                       dtype=np.double)
        #Compute equivalent radius with Peaceman correction
        r_eq = dx[grids] * np.exp(-np.pi / 2.0)

        #Return array of productivity indices for 'well_type' wells
        return ((factor * 2.0 * np.pi * perm[grids] * h) / 
                (visc * Bw * np.log(r_eq / r_w)))


    def compute_time_step(self, P_n):
        """
           Computes a single time-step solution for the choice of solver method
           given in the input deck (implicit, explicit, mixed)
        """

        #Pointer reassignment for convenience
        method = self.solver_method
        dt = self.time_step
        theta = self.solver_theta
        T = self.T
        B = self.B
        Q = self.Q

        #If mixed, or explicit are specified, otherwise default is implicit
        #therefore it's not actually required to be placed in the input deck
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

        #Return solution vector from a single time-step
        return P_np1


    def run(self, plot_freq=None):
        """
           Computes all time steps requested in the simulation.  Also stores
           solution vectors every 'plot_freq' steps (the first and last steps
           are stored by default)
        """
        
        #Initialize the initial pressure
        P = np.ones(self.ngrids) * self.initial_pressure

        #Initialize arrays for storing solution and time every 'plot_freq' steps
        P_plot = []
        self.time = []

        #Loop over time steps, the '+1' is to get the 'plot_freq' storage correct
        for i in range(self.number_of_time_steps + 1):

            #Logic for storing solutions at 'plot_freq' or on the last step
            if (plot_freq is not None and i % plot_freq == 0):
                P_plot.append(P)
                self.time.append(i * self.time_step)
                if (i == self.number_of_time_steps):
                    break
            elif (i == self.number_of_time_steps):
                P_plot.append(P)
                break

            #Compute solution for this step        
            P = self.compute_time_step(P)

        #Ensure stored solution is a numpy array for correct indexing later
        self.P_plot = np.array(P_plot)

        return 


    def get_solution(self):
        """
           Convenience function for finding the solution at the last step.
        """
        return self.P_plot[-1]


    def plot(self, x_unit='ft', y_unit='psi'):
        """
           Plot pressure as a function of reservoir position.  Default units
           are (psi) and (ft), but they can be changed via the arguments.
        """

        #Find the grid centers (where the solution exists)
        x_pos = np.cumsum(self.dx_arr) - self.dx_arr[0] / 2.0

        #Loop over all stored solutions and plot stair-step line (because
        #pressure is constant over grid block).  We skip the first stored values
        #because they are just the initialization values.
        plt.figure()
        for P in self.P_plot:
            plt.plot(x_pos, P)

        #Labels, etc.
        plt.xlabel('Reservoir position (' + x_unit + ')')
        plt.ylabel('Pressure (' + y_unit + ')')
        plt.xlim([0, self.res_length])
        plt.show()
        
    def plot_BHP(self, x_unit='days', y_unit='psi'):
        """
           Plot the bottom hole pressure as a function of time for every
           constant-rate well
        """

        #Raise exception if trying to plot and there are no rate wells defined
        if self.rate_well_grids is None:
            raise ValueError("No constant rate wells are defined.")

        #Pointer reassignments for convenience
        time = self.time
        grids = self.rate_well_grids
        rates = self.rate_well_values
        J = self.rate_well_prod_ind

        #Compute bottom hole pressures
        BHPs = self.P_plot[:,grids].T + (rates / J)[:, None]

        #Plot bottom-hole pressure for each well
        plt.figure()
        for BHP in BHPs:
            plt.plot(time, BHP)

        #Labels, etc.
        plt.xlabel('time (' + x_unit + ')')
        plt.ylabel('Bottom Hole Pressure (' + y_unit + ')')
        plt.show()


#This module is indended to be used as a library, but if it is called as an
#executable script, the solution will be computed and final pressure plotted.
if __name__ == "__main__":
    
    problem = HW4('HW4.in')
    problem.run()
    problem.plot()
