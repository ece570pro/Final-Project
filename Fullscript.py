import pandas as pd
import math
import os
import subprocess
from texgen_utils import create_weave_voxel_mesh
import contextlib
import io


# Function to calculate R
def calculate_R(Vf):
    return math.sqrt(math.sqrt(3) * Vf / (2 * math.pi))

# Function to extract k values
def extract_k_values():
    try:
        with open("Output.sc.k", "r") as f:
            lines = f.readlines()
        
        stiffness_matrix = []
        recording = False
        
        for line in lines:
            if "Effective Stiffness Matrix" in line:
                recording = True
                continue
            elif "Effective Compliance Matrix" in line:
                break
                
            if recording and line.strip():
                values = []
                for part in line.split():
                    try:
                        values.append(float(part))
                    except ValueError:
                        continue
                if values:
                    stiffness_matrix.append(values)
        
        if len(stiffness_matrix) != 3 or any(len(row) != 3 for row in stiffness_matrix):
            raise ValueError("Could not parse complete 3x3 stiffness matrix")
        
        return stiffness_matrix[0][0], stiffness_matrix[2][2]  # k11, k33
        
    except Exception as e:
        print(f"Error extracting k values: {e}")
        return None, None

# Function to read .msh file and create .sc file
def create_sc_file(msh_file, sc_file):
    def read_msh_file(msh_file):
        nodes = []
        elements = []
        with open(msh_file, 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                if lines[i].strip() == "$Nodes":
                    i += 1
                    n_node = int(lines[i].strip())
                    i += 1
                    for _ in range(n_node):
                        node_info = lines[i].strip().split()
                        nodes.append((int(node_info[0]), float(node_info[1]), float(node_info[2])))
                        i += 1
                elif lines[i].strip() == "$Elements":
                    i += 1
                    n_elem = int(lines[i].strip())
                    i += 1
                    for _ in range(n_elem):
                        elem_info = lines[i].strip().split()
                        if elem_info[1] == '2':  # 3-node triangle
                            elements.append((int(elem_info[0]), int(elem_info[3]), int(elem_info[5]), int(elem_info[6]), int(elem_info[7]), 0, 0, 0, 0, 0, 0))
                        elif elem_info[1] == '3':  # 4-node quadrilateral
                            elements.append((int(elem_info[0]), int(elem_info[3]), int(elem_info[5]), int(elem_info[6]), int(elem_info[7]), int(elem_info[8]), 0, 0, 0, 0, 0))
                        i += 1
                else:
                    i += 1
        return nodes, elements, n_node, n_elem

    try:
        nodes, elements, n_node, n_elem = read_msh_file(msh_file)
        
        with open(sc_file, 'w') as file:
            file.write("2 0 0 0 \t # Analysis_type  elem_type trans_flag temp_flag\n\n")
            file.write(f"2 {n_node} {n_elem} 2 0 0 \t # nSG nNode nElem nMat nSlave nLayer\n\n")
            
            for node in nodes:
                file.write(f"{node[0]} {node[1]} {node[2]} \t # node_no  x  y\n")
            file.write("\n")
            
            for elem in elements:
                file.write(f"{elem[0]} {elem[1]} {elem[2]} {elem[3]} {elem[4]} {elem[5]} {elem[6]} {elem[7]} {elem[8]} {elem[9]} {elem[10]}\t # elem_no  mat_type  node1 node2  .... node 9\n")
            file.write("\n")
            
            file.write("1 1 1 \t # mat_type isotropy ntemp (This if for fiber)\n")
            file.write("0 0 # T and Rho\n")
            file.write("10.2 1.256 1.256 # k11 k22 k33\n\n")
            
            file.write("2 0 1 \t # mat_type isotropy ntemp (This if for matrix)\n")
            file.write("0 0 # T and Rho\n")
            file.write("0.180 #k\n\n")

            file.write("1.732 \t #Homogenized SG volume")

    except Exception as e:
        print(f"An error occurred while writing the .sc file: {e}")
        raise

# Main processing function for each row
def process_row(row, index):
    try:
        Vf = row['Vf']
        width_ratio = row['Width to Spacing']
        thickness_ratio = row['Thickness to Spacing']
        
        print(f"\nProcessing Row {index + 1}: Vf={Vf}, Width={width_ratio}, Thickness={thickness_ratio}")

        # Calculate R and generate .geo file
        R = calculate_R(Vf)
        sqrt3 = math.sqrt(3)
        half_sqrt3 = sqrt3 / 2
        half_sqrt3_minus_R = half_sqrt3 - R
        neg_half_sqrt3_plus_R = -half_sqrt3 + R

        # Write the .geo file
        with open('Trial.geo', 'w') as file:
            file.write(f"""// ##1, orthotropic material
// Material name: 1 -- fiber
Physical Point("1 1 1 0 0 10.2 1.256 1.256") = {{}}; 

// ##2, isotropic material
// Material name: 2 -- matrix
Physical Point("2 0 1 0 0 0.180") = {{}}; 

// Define fiber volume
// volume fraction = {Vf};
l = 1;

// Define points
Point(1) = {{ -0.5, 0.866025, 0, 0.1 }};
Point(2) = {{ 0.5, 0.866025, 0, 0.1 }};
Point(3) = {{ 0.5, -0.866025, 0, 0.1 }};
Point(4) = {{ -0.5, -0.866025, 0, 0.1 }};
Point(5) = {{ {-0.5 + R}, 0.866025, 0, 0.1 }};
Point(6) = {{ -0.5, {half_sqrt3_minus_R}, 0, 0.1 }};
Point(7) = {{ {0.5 - R}, 0.866025, 0, 0.1 }};
Point(8) = {{ 0.5, {half_sqrt3_minus_R}, 0, 0.1 }};
Point(9) = {{ 0.5, {neg_half_sqrt3_plus_R}, 0, 0.1 }};
Point(10) = {{ {0.5 - R}, -0.866025, 0, 0.1 }};
Point(11) = {{ {-0.5 + R}, -0.866025, 0, 0.1 }};
Point(12) = {{ -0.5, {neg_half_sqrt3_plus_R}, 0, 0.1 }};
Point(13) = {{ 0, 0, 0, 0.1 }};
Point(14) = {{ 0, {-R}, 0, 0.1 }};
Point(15) = {{ {R}, 0, 0, 0.1 }};
Point(16) = {{ 0, {R}, 0, 0.1 }};
Point(17) = {{ {-R}, 0, 0, 0.1 }};

// Define lines and circles
Line(3) = {{ 2, 7 }}; 
Line(4) = {{ 7, 5 }}; 
Line(5) = {{ 5, 1 }}; 
Line(6) = {{ 1, 6 }}; 
Line(7) = {{ 6, 12 }}; 
Line(8) = {{ 12, 4 }}; 
Line(9) = {{ 4, 11 }}; 
Line(10) = {{ 11, 10 }}; 
Line(11) = {{ 10, 3 }}; 
Line(12) = {{ 3, 9 }}; 
Line(13) = {{ 9, 8 }}; 
Line(14) = {{ 8, 2 }}; 

Circle(15) = {{ 5, 1, 6 }}; 
Circle(16) = {{ 12, 4, 11 }}; 
Circle(17) = {{ 10, 3, 9 }}; 
Circle(18) = {{ 8, 2, 7 }}; 
Circle(19) = {{ 16, 13, 17 }}; 
Circle(20) = {{ 17, 13, 14 }}; 
Circle(21) = {{ 14, 13, 15 }}; 
Circle(22) = {{ 15, 13, 16 }}; 

// Define closed loops properly
Line Loop(23) = {{ 5, 6, -15 }}; 
Plane Surface(24) = {{ 23 }}; 

Line Loop(25) = {{ 3, -18, 14 }}; 
Plane Surface(26) = {{ 25 }}; 

Line Loop(27) = {{ 20, 21, 22, 19 }}; 
Plane Surface(28) = {{ 27 }}; 

Line Loop(29) = {{ 8, 9, -16 }}; 
Plane Surface(30) = {{ 29 }}; 

Line Loop(31) = {{ 11, 12, -17 }}; 
Plane Surface(32) = {{ 31 }}; 

Line Loop(33) = {{ 4, 15, 7, 16, 10, 17, 13, 18 }}; 
Plane Surface(34) = {{ 33, 27 }}; 

// Assign physical surfaces
Physical Surface(1) = {{ 24, 26, 28, 30, 32 }}; 
Physical Surface(2) = {{ 34 }}; 

// Ensure quadrilateral meshing
Recombine Surface{{ 24, 26, 28, 30, 32, 34 }}; 

// Mesh settings
Mesh.Algorithm = 8; // 8 = Quad dominant
Mesh.RecombineAll = 1; 

Mesh.Points = 1;
Mesh.SurfaceFaces = 1;
Mesh.SurfaceEdges = 1;
Mesh.VolumeEdges = 1;
Mesh.ColorCarousel = 2;

Mesh.CharacteristicLengthFactor = 1; // Default is 1, lower values make finer mesh

""")

        # Run GMSH
        geo_file = os.path.abspath("Trial.geo")
        msh_file = os.path.abspath("Trial.msh")
        subprocess.run(["gmsh", "-2", geo_file, "-format", "msh", "-o", msh_file],check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Create .sc file
        create_sc_file("Trial.msh", "Trial.sc")

        # Run SwiftComp
        subprocess.run(["Swiftcomp", "Trial.sc", "3D", "H"], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Run TexGen
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                create_weave_voxel_mesh(width_ratio, thickness_ratio)


        # Run Meso.py
        subprocess.run(["python", "Meso.py"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


        # Run final SwiftComp
        subprocess.run(["Swiftcomp", "Output.sc", "3D", "H"], 
                      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Extract and return k values
        return extract_k_values()
        
    except Exception as e:
        print(f"Error processing row {index + 1}: {e}")
        return None, None

# Main script
def main():
    input_file = 'Vf_data_updated.xlsx'
    
    try:
        # Load the Excel file
        df = pd.read_excel(input_file)
        
        # Add columns for results if they don't exist
        if 'k11' not in df.columns:
            df['k11'] = None
        if 'k33' not in df.columns:
            df['k33'] = None
        
        # Process each row
        for index, row in df.iterrows():
            k11, k33 = process_row(row, index)
            if k11 is not None and k33 is not None:
                df.at[index, 'k11'] = k11
                df.at[index, 'k33'] = k33
                print(f"Row {index + 1} completed: k11={k11:.4E}, k33={k33:.4E}")
            
            # Save progress after each row
            df.to_excel(input_file, index=False)
        
        print("\nAll rows processed successfully!")
        
    except Exception as e:
        print(f"Script failed: {e}")

if __name__ == "__main__":
    main()