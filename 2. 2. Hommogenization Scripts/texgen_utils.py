from TexGen.Core import *
from TexGen.Export import *
# Instead of importing the octree mesh, import the rectangular voxel mesh:
from TexGen.Core import CRectangularVoxelMesh

def create_weave_voxel_mesh(width_to_spacing, thickness_to_spacing, output_prefix="PlainWeave"):
    """
    Create a plain weave textile and voxel mesh with given parameters.
    
    Args:
        width_to_spacing (float): Ratio of yarn width to spacing.
        thickness_to_spacing (float): Ratio of textile thickness to spacing.
        output_prefix (str): Prefix for output files.
    """
    try:
        # Calculate dimensions (assuming spacing = 1 mm)
        spacing = 1.0
        yarnWidth = width_to_spacing * spacing
        thickness = thickness_to_spacing * spacing
        
        print(f"\nCreating weave with:")
        print(f"- Width/Spacing: {width_to_spacing:.3f}")
        print(f"- Thickness/Spacing: {thickness_to_spacing:.3f}")
        
        #----------------------------------------------------------
        # Create textile
        #----------------------------------------------------------
        m = 2  # Number of warp yarns
        n = 2  # Number of weft yarns
        weave = CTextileWeave2D(m, n, spacing, thickness, False)
        weave.SetGapSize(0)
        
        # Set weave pattern (swapping positions for alternate yarns)
        for i in range(m):
            for j in range(n):
                if (i + j) % 2 == 0:
                    weave.SwapPosition(i, j)
        
        # Set yarn properties so that separate yarn regions are defined
        yarnHeight = thickness / 2.0
        weave.SetYarnWidths(yarnWidth)
        weave.SetYarnHeights(yarnHeight)
        
        # Assign a default domain and add textile to TexGen
        weave.AssignDefaultDomain()
        AddTextile(output_prefix, weave)
        
        # Save textile to XML format (for reference)
        SaveToXML(f"{output_prefix}.tg3")
        
        #----------------------------------------------------------
        # Create and export voxel mesh using rectangular voxels
        #----------------------------------------------------------
        nXvoxel = 40
        nYvoxel = 40
        nZvoxel = 20
        
        # Instantiate the rectangular voxel mesh instead of the octree one.
        voxelMesh = CRectangularVoxelMesh()
        
        # Use the 10-parameter SaveVoxelMesh overload:
        #   SaveVoxelMesh(Textile, OutputFilename, nX, nY, nZ,
        #                 bOutputMatrix, bOutputYarns, iBoundaryConditions, iElementType, FileType)
        #
        # Setting both bOutputMatrix and bOutputYarns to True tells TexGen to export separate element sets
        # for the matrix and for the yarns. iBoundaryConditions is set to 0 to avoid periodic BCs that might blend
        # yarn detail. iElementType is set to 0 (often corresponding to C3D8R elements) and INP_EXPORT is a constant.
        voxelMesh.SaveVoxelMesh(weave, f"{output_prefix}.inp", nXvoxel, nYvoxel, nZvoxel, 
                                  True, True, 0, 0, INP_EXPORT)
        
        return True
        
    except Exception as e:
        print(f"\nError creating weave: {e}")
        return False
