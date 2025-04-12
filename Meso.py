import os

def merge_continuation_lines(lines):
    """
    Merge lines that are continued with a trailing comma.
    Returns a new list of lines with continuations merged.
    """
    merged = []
    accumulator = ""
    for line in lines:
        stripped = line.rstrip("\n").strip()
        if not stripped:
            continue
        if accumulator:
            accumulator += " " + stripped
        else:
            accumulator = stripped
        if accumulator.endswith(','):
            accumulator = accumulator.rstrip(',')
            continue
        else:
            merged.append(accumulator)
            accumulator = ""
    if accumulator:
        merged.append(accumulator)
    return merged

def parse_inp(inp_filename):
    """
    Parse the .inp file to extract nodes, element connectivity, and element set definitions.
    Assumes nodes are defined after a "*Node" header and elements after a "*Element" header.
    Handles continuation lines and element set definitions (from "*ElSet" lines).
    Elements in sets with "Yarn" (case-insensitive) get material id 1; those with "Matrix" get 2.
    """
    with open(inp_filename, 'r') as f:
        raw_lines = f.readlines()
    lines = merge_continuation_lines(raw_lines)
    
    nodes = []
    elements = []
    elem_to_mat = {}  # key: element id (as string), value: material id
    current_set = None
    in_set_block = False
    node_section = False
    element_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for element set header.
        if line.lower().startswith("*elset"):
            parts = [p.strip() for p in line.split(',') if p.strip()]
            set_name = None
            for part in parts:
                if "elset=" in part.lower():
                    set_name = part.split('=')[1].strip()
                    break
            if set_name:
                current_set = set_name
                in_set_block = True
            else:
                current_set = None
                in_set_block = False
            continue

        # New header resets section flags.
        if line.startswith("*"):
            in_set_block = False
            if line.lower().startswith("*node"):
                node_section = True
                element_section = False
            elif line.lower().startswith("*element"):
                element_section = True
                node_section = False
            else:
                node_section = False
                element_section = False
            continue

        # Process element-set block lines.
        if in_set_block and current_set:
            nums = line.replace(',', ' ').split()
            for num in nums:
                if "yarn" in current_set.lower():
                    elem_to_mat[num] = 1
                elif "matrix" in current_set.lower():
                    elem_to_mat[num] = 2
            continue

        if node_section:
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) >= 4:
                node_id = parts[0]
                x = parts[1]
                y = parts[2]
                z = parts[3]
                nodes.append((node_id, x, y, z))
        elif element_section:
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) < 9:
                parts = line.split()
            if not parts[0][0].isdigit():
                continue
            if len(parts) >= 9:
                elem_id = parts[0]
                node_ids = parts[1:9]
                elements.append((elem_id, node_ids))
            else:
                raise ValueError("Element line does not have at least 9 entries: " + line)
    return nodes, elements, elem_to_mat

def read_trial_k(trial_k_filename):
    """
    Reads the stiffness file and extracts the effective stiffness matrix values.
    Waits for the line containing "Effective Stiffness Matrix" (ignoring case)
    and then reads the next three nonempty lines, assuming each line is a row 
    of the 3x3 stiffness matrix. Returns the diagonal entries:
      k11 from row1, k22 from row2, k33 from row3.
    Adjust indices if your file layout is different.
    """
    matrix_rows = []
    reading = False
    with open(trial_k_filename, 'r') as file:
        for line in file:
            line = line.strip()
            if "effective stiffness matrix" in line.lower():
                reading = True
                continue
            if reading:
                if "effective compliance matrix" in line.lower():
                    break
                if line:
                    tokens = line.split()
                    row_vals = []
                    for token in tokens:
                        try:
                            row_vals.append(float(token))
                        except ValueError:
                            continue
                    if row_vals:
                        matrix_rows.append(row_vals)
                    if len(matrix_rows) == 3:
                        break
    if len(matrix_rows) != 3:
        raise ValueError("Could not parse complete 3x3 stiffness matrix. Check file format.")
    try:
        k11 = matrix_rows[0][0]
        k22 = matrix_rows[1][1]
        k33 = matrix_rows[2][2]
    except Exception as e:
        raise ValueError("Error extracting diagonal entries: " + str(e))
    return k11, k22, k33

def read_ori(ori_filename):
    """
    Reads the orientation (.ori) file and returns a dictionary mapping element id (as a string)
    to a list of 6 orientation numbers [a1, a2, a3, b1, b2, b3].
    Data lines are expected to be in one of the following forms (ignoring any header):
       1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
    or
       1 1.0 0.0 0.0 0.0 1.0 0.0
    Lines not starting with a digit are skipped.
    """
    ori_data = {}
    with open(ori_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove any leading '#' or '*' characters.
            while line and line[0] in "#*":
                line = line[1:].strip()
            tokens = line.replace(",", " ").split()
            if not tokens:
                continue
            if not tokens[0].isdigit():
                continue
            if len(tokens) < 7:
                continue
            elem_id = tokens[0]
            try:
                # Simply take the first six values as provided.
                values = [tokens[i] for i in range(1, 7)]
            except Exception:
                continue
            ori_data[elem_id] = values
    return ori_data

def write_sc(sc_filename, nodes, elements, elem_to_mat, k11, k22, k33, orientation_data=None):
    """
    Writes the Swiftcomp (.sc) file in the following sections:

    (1) Header:
         2 0 0 0       # Analysis Type, elem type, trans flag, tempflag
         2 nNode nElem 2 0 0   # nSG nNode nElem nMat nSlave nLayer

    (2) Node Data:
         node_no  x  y  z

    (3) Element Connectivity:
         For each element: elem_no, Mat Id, Node1,..., Node8, then 12 zeros (22 entries total)

    (4) Orientation Block (if orientation_data is provided):
         For each element: elem_no, a1, a2, a3, b1, b2, b3, 0, 0, 0
         Here, the first 6 entries after the element id are taken directly from the .ori file.
         (The last 3 entries are zeros.)
    
    (5) Material Property Blocks:
         For fiber (mat id = 1): uses stiffness values read from Trial.sc.k
         For matrix (mat id = 2): uses a hardcoded k of 0.180

    (6) Homogenized SG Volume
    """
    N_node = len(nodes)
    N_elem = len(elements)
    with open(sc_filename, 'w') as f:
        # Write header.
        f.write("2 0 1 0 \t # Analysis_type  elem_type trans_flag temp_flag\n\n")
        f.write("3 {} {} 2 0 0 \t # nSG nNode nElem nMat nSlave nLayer\n".format(N_node, N_elem))
        # Write nodes.
        for node in nodes:
            node_id, x, y, z = node
            f.write("{} {} {} {}\n".format(node_id, x, y, z))
        f.write("\n")
        # Write element connectivity.
        for elem in elements:
            elem_id, node_ids = elem
            mat_id = elem_to_mat.get(elem_id, 2)
            if len(node_ids) != 8:
                raise ValueError("Element {} does not have 8 nodes.".format(elem_id))
            line = "{} {} {} {} {} {} {} {} {} {}".format(
                elem_id, mat_id,
                node_ids[0], node_ids[1], node_ids[2], node_ids[3],
                node_ids[4], node_ids[5], node_ids[6], node_ids[7]
            )
            zeros_str = " ".join(["0"] * 12)
            f.write(line + " " + zeros_str + "\n")
        f.write("\n")
        # Write orientation block if available.
        if orientation_data is not None:
            for elem in elements:
                elem_id, _ = elem
                if elem_id in orientation_data:
                    oris = orientation_data[elem_id]
                    # Write the six orientation values as they appear in the .ori file,
                    # followed by three zeros.
                    ori_line = "{}  {}  {}  {}    {}  {}  {}  0 0 0".format(
                        elem_id, oris[0], oris[1], oris[2],
                        oris[3], oris[4], oris[5]
                    )
                else:
                    ori_line = "{} 0 0 0 0 0 0 0 0 0".format(elem_id)
                f.write(ori_line + "\n")
            f.write("\n")
        # Write material properties block for fiber (mat id = 1)
        f.write("1 1 1 \t # mat_type isotropy ntemp (This is for fiber)\n")
        f.write("0 0 # T and Rho\n")
        f.write("{} {} {} # k11 k22 k33\n\n".format(k11, k22, k33))
        # Write material properties block for matrix (mat id = 2)
        f.write("2 0 1 \t # mat_type isotropy ntemp (This is for matrix)\n")
        f.write("0 0 # T and Rho\n")
        f.write("0.180 # k\n\n")
        # Write homogenized SG Volume.
        f.write("0.44 \t #Homogenized SG Volume")

if __name__ == "__main__":
    inp_filename = "PlainWeave.inp"     # Abaqus .inp file
    trial_k_filename = "Trial.sc.k"      # Stiffness file (.k data)
    sc_filename = "Output.sc"            # Swiftcomp output file
    ori_filename = "PlainWeave.ori"      # Orientation file
    if not os.path.exists(inp_filename):
        raise FileNotFoundError(f"Input file '{inp_filename}' does not exist.")
    if not os.path.exists(trial_k_filename):
        raise FileNotFoundError(f"Stiffness file '{trial_k_filename}' does not exist.")
    try:
        nodes, elements, elem_to_mat = parse_inp(inp_filename)
        if not nodes:
            raise ValueError("No nodes were found in the inp file.")
        if not elements:
            raise ValueError("No elements were found in the inp file.")
        k11, k22, k33 = read_trial_k(trial_k_filename)
        orientation_data = {}
        if os.path.exists(ori_filename):
            orientation_data = read_ori(ori_filename)
        write_sc(sc_filename, nodes, elements, elem_to_mat, k11, k22, k33, orientation_data)
        print("Conversion complete. Swiftcomp file written to", os.path.abspath(sc_filename))
    except Exception as e:
        print("Error processing:", e)
        exit(1)
