import os
import sys
import slicer
import vtk

def process_patient(patient_dir):
    input_nrrd = os.path.join(patient_dir, "Combined.seg.nrrd")
    output_vtk = os.path.join(patient_dir, "lumen_tree_cfd.vtk")

    if not os.path.exists(input_nrrd):
        raise FileNotFoundError(f"File non trovato: {input_nrrd}")

    # 1. Caricamento del file di segmentazione .seg.nrrd
    segmentation_node = slicer.util.loadSegmentation(input_nrrd)
    segmentation = segmentation_node.GetSegmentation()

    # 2. Identificazione del segmento lumen_cfd
    segment_id = None
    for i in range(segmentation.GetNumberOfSegments()):
        sid = segmentation.GetNthSegmentID(i)
        seg_name = segmentation.GetSegment(sid).GetName()
        if seg_name.lower() == "lumen_cfd":
            segment_id = sid
            break
            
    # Fallback al primo segmento se non trova il nome esatto
    if not segment_id and segmentation.GetNumberOfSegments() > 0:
        segment_id = segmentation.GetNthSegmentID(0)

    if not segment_id:
        raise ValueError("Nessun segmento valido trovato in Combined.seg.nrrd")

    # 3. Generazione e isolamento della superficie 3D (Models)
    segmentation_node.CreateClosedSurfaceRepresentation()
    sh_node = slicer.mrmlScene.GetSubjectHierarchyNode()
    folder_id = sh_node.CreateFolderItem(sh_node.GetSceneItemID(), "ExportFolder")

    # Esporta solo il segmento selezionato come nodo Model
    slicer.modules.segmentations.logic().ExportSegmentsToModels(
        segmentation_node, [segment_id], folder_id
    )

    # 4. Salvataggio del modello in lumen_tree_cfd.vtk
    children = vtk.vtkIdList()
    sh_node.GetItemChildren(folder_id, children)
    if children.GetNumberOfIds() == 0:
        raise RuntimeError("Estrazione della superficie fallita.")

    model_node = sh_node.GetItemDataNode(children.GetId(0))
    slicer.util.saveNode(model_node, output_vtk)
    print(f"[STEP 0 SUCCESS] Esportato: {output_vtk}")

    # Pulizia della scena Slicer
    slicer.mrmlScene.Clear(0)

if __name__ == "__main__":
    # Eseguito via CLI: Slicer --no-main-window --python-script step0_export_lumen.py <patient_dir>
    if len(sys.argv) > 1:
        patient_folder = sys.argv[-1]
        process_patient(patient_folder)