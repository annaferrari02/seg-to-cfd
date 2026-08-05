# convert_mesh.py
# Esegui con:  pvpython process_mesh.py    

from paraview.simple import *

#paths 
#INPUT_DIR= sarà output del generate_mesh.py in 3dslicer 
#OUTPUT_DIR= cartella di input per simVascular 


#Parametri
input_file  = "lumen_tree_cfd_cap.vtk"      #mesh di partenza
output_file = "lumen_tree_cfd_clip_cm.vtk"
scale       = 0.1                             #mm -> cm
recenter    = True

#carico mesh
mesh = OpenDataFile(input_file)   # rileva formato e determina il reader 
UpdatePipeline()                  # forza il caricamento 

# Controllo i bounds per capire se è già in cm
b = mesh.GetDataInformation().GetBounds()
print("Bounds iniziali:", b)      # (xmin, xmax, ymin, ymax, zmin, zmax)

# scale-transform
scaled = Transform(Input=mesh)
scaled.Transform.Scale = [scale, scale, scale]
UpdatePipeline()

# recenter (if true)
if recenter:
    b = scaled.GetDataInformation().GetBounds()
    center = [(b[0]+b[1])/2.0, (b[2]+b[3])/2.0, (b[4]+b[5])/2.0]
    centered = Transform(Input=scaled)
    centered.Transform.Translate = [-center[0], -center[1], -center[2]]
    UpdatePipeline()
    current = centered
else:
    current = scaled

# triangulate
tri = Triangulate(Input=current)

# clean
cleaned = Clean(Input=tri)
UpdatePipeline()

b = cleaned.GetDataInformation().GetBounds()
print("Bounds finali:", b)


#mesh check 
info = cleaned.GetDataInformation()
n_points = info.GetNumberOfPoints()
n_cells  = info.GetNumberOfCells()
print(f"Punti: {n_points}  Celle: {n_cells}")

#salvataggio
SaveData(output_file, proxy=cleaned)
print("Salvato:", output_file)