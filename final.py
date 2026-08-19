import vtk
import numpy as np
import logging
import functools
import os
import sys
import scipy.ndimage
import slicer
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from CurvedPlanarReformat import CurvedPlanarReformatLogic
import SegmentStatistics 
from typing import List, Optional
import torch
from monai.transforms import (
    Compose, EnsureTyped, ResizeWithPadOrCropD, MedianSmoothd,
    ScaleIntensityd, Activationsd, AsDiscreted, DivisiblePadd
)
from monai.networks.nets import SegResNet
from slicer.util import getNode, arrayFromVolume, arrayFromSegmentBinaryLabelmap, updateSegmentBinaryLabelmapFromArray
import SimpleITK as sitk
import pandas as pd
from scipy import stats
from pprint import pprint

import csv

import datetime


import SimpleITK as sitk
from radiomics import featureextractor


def log_method_call(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        logging.debug(f"→ Calling `{method.__name__}` with args={args}, kwargs={kwargs}")
        result = method(self, *args, **kwargs)
        logging.debug(f"← Finished `{method.__name__}`")
        return result
    return wrapper

class CoordinateTransformer:
    
    # x,y,z -> R A S (mm) 
    # R A S -> x y z
   
    def __init__(self, volume_node):
        self.volume_node = volume_node
        self.logger = logging.getLogger(self.__class__.__name__)

    @log_method_call
    def ijk_to_ras(self, point_kji):
        point_ijk = [point_kji[2], point_kji[1], point_kji[0]]
        #point_ijk = point_kji
        matrix = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASMatrix(matrix)
        point_ras = [0, 0, 0, 1]
        matrix.MultiplyPoint(np.append(point_ijk, 1.0), point_ras)

        self.logger.debug(f"Called IJKtoRAS for point {point_ijk}")
        self.logger.debug(f"IJK to RAS matrix:\n{matrix}")
        self.logger.debug(f"Result: {point_ijk} → {point_ras[:-1]}")
        return point_ras[:-1]
    
    @log_method_call
    def ras_to_ijk(self, point_ras):
        matrix = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASMatrix(matrix)
        matrix.Invert()
        point_ijk = [0, 0, 0, 1]
        matrix.MultiplyPoint(np.append(point_ras, 1.0), point_ijk)

        self.logger.debug(f"Called RAStoIJK for point {point_ras}")
        self.logger.debug(f"RAS to IJK matrix (inverted):\n{matrix}")
        self.logger.debug(f"Result: {point_ras} → {point_ijk[:-1]}")
        return [round(x) for x in point_ijk[:-1]]
    

class CPRProcessor:
    # curved planar reformat
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logic = CurvedPlanarReformatLogic()

    def do_cpr(
        self,
        centerline_node,
        volume_node,
        volume_name,
        field_of_view=[55.0, 55.0],
        output_spacing=[1.0, 1.0, 0.5],
    ):
        self.logger.debug(
            f"Calculating CPR for centerline {centerline_node.GetName()} and volume {volume_node.GetName()}"
        )

        straightening_transform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTransformNode", f"Straightening_transform_{volume_name}"
        )

        self.logic.computeStraighteningTransform(
            straightening_transform, centerline_node, field_of_view, output_spacing[2]
        )

        straightened_volume = slicer.modules.volumes.logic().CloneVolume(
            volume_node, f"{volume_node.GetName()}_{volume_name}_straightened"
        )
        self.logic.straightenVolume(
            straightened_volume, volume_node, output_spacing, straightening_transform
        )

        panoramic_volume = slicer.modules.volumes.logic().CloneVolume(
            straightened_volume, f"{straightened_volume.GetName()}_{volume_name}_panoramic"
        )
        self.logic.projectVolume(panoramic_volume, straightened_volume)

        slicer.util.setSliceViewerLayers(
            background=straightened_volume, fit=True, rotateToVolumePlane=True
        )

        self.logger.debug(f"Straightened volume: {straightened_volume.GetName()}")
        self.logger.debug(f"Transform: {straightening_transform.GetName()}")

        return straightened_volume, straightening_transform, panoramic_volume

    def transfer_segmentation_world_to_cpr(self, segment_names, from_seg_node, to_seg_node, volume_node):
        self.logger.debug(f"Transferring segments {segment_names} to CPR space")
        for name in segment_names:
            seg_array = arrayFromSegmentBinaryLabelmap(from_seg_node, name, volume_node)
            updateSegmentBinaryLabelmapFromArray(seg_array, to_seg_node, name, volume_node)

    def transfer_segmentation_cpr_to_world(self, segment_names, from_seg_node, to_seg_node):
        self.logger.debug(f"Transferring segments {segment_names} back to world space")
        for name in segment_names:
            seg_id = from_seg_node.GetSegmentation().GetSegmentIdBySegmentName(name)
            to_seg_node.GetSegmentation().CopySegmentFromSegmentation(
                from_seg_node.GetSegmentation(), seg_id
            )

    def invert_transform(self, transform_node):
        self.logger.debug(f"Inverting transform {transform_node.GetName()}")
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        itemID = shNode.GetItemByDataNode(transform_node)
        clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemID)
        cloned_node = shNode.GetItemDataNode(clonedItemID)
        cloned_node.Inverse()
        self.logger.debug(f"Inverted transform created: {cloned_node.GetName()}")
        return cloned_node

    def from_cpr_to_world(
        self,
        cpr_seg_node,
        main_seg_node,
        volume_node,
        inverse_transform,
        save_path,
        segments_to_export=None,
    ):
        if segments_to_export:
            self.make_only_segments_visible(segments_to_export, cpr_seg_node)

        visible_segments = []
        display_node = cpr_seg_node.GetDisplayNode()
        for seg_id in cpr_seg_node.GetSegmentation().GetSegmentIDs():
            if display_node.GetSegmentVisibility(seg_id):
                visible_segments.append(seg_id)

        cpr_seg_node.SetAndObserveTransformNodeID(inverse_transform.GetID())

        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", cpr_seg_node.GetName()
        )

        
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            cpr_seg_node, labelmap_node
        )
        slicer.util.exportNode(
            labelmap_node,
            os.path.join(save_path, f"{labelmap_node.GetName()}.nii.gz"),
            {"useCompression": 0},
            world=True,
        )

     

        loaded_seg = slicer.util.loadSegmentation(
            os.path.join(save_path, f"{labelmap_node.GetName()}.nii.gz")
        )

        for i, seg_id in enumerate(loaded_seg.GetSegmentation().GetSegmentIDs()):
            loaded_seg.GetSegmentation().GetSegment(seg_id).SetName(visible_segments[i])

        self.transfer_segmentation_cpr_to_world(
            visible_segments,
            loaded_seg,
            main_seg_node,
        )

        cpr_seg_node.SetAndObserveTransformNodeID(None)
        slicer.mrmlScene.RemoveNode(loaded_seg)
        slicer.mrmlScene.RemoveNode(labelmap_node)

    def make_only_segments_visible(self, segment_names, segmentation_node):
        display_node = segmentation_node.GetDisplayNode()
        for seg_id in segmentation_node.GetSegmentation().GetSegmentIDs():
            is_visible = segmentation_node.GetSegmentation().GetSegment(seg_id).GetName() in segment_names
            display_node.SetSegmentVisibility(seg_id, is_visible)





class IntensityThresholdEstimator:
    def __init__(self, num_components=2, contamination=0.05):
        self.num_components = num_components
        self.contamination = contamination
        self.logger = logging.getLogger(self.__class__.__name__)

    def _merge_segmentations(self, segmentations):
        merged = np.zeros_like(segmentations[0], dtype=bool)
        for s in segmentations:
            merged = np.logical_or(merged, s)
        return merged

    def fit_gmm(self, vol_arr, segmentations):
        merged_mask = self._merge_segmentations(segmentations)
        hus = vol_arr[merged_mask]

        self.logger.debug("Running IsolationForest to remove outliers")
        iso_forest = IsolationForest(contamination=self.contamination, random_state=0)
        iso_forest.fit(hus.reshape(-1, 1))
        inliers = iso_forest.predict(hus.reshape(-1, 1)) == 1
        hus = hus[inliers]

        self.logger.debug("Fitting Gaussian Mixture Model")
        gm = GaussianMixture(n_components=self.num_components, random_state=0)
        gm.fit(hus.reshape(-1, 1))

        means = gm.means_.flatten()
        vars_ = gm.covariances_.reshape(self.num_components, -1)

        mdc_index = np.argmax(means)
        non_mdc_index = np.argmin(means)

        mdc_mean = means[mdc_index]
        mdc_var = vars_[mdc_index][0]

        non_mdc_mean = means[non_mdc_index]
        non_mdc_var = vars_[non_mdc_index][0]

        # Estimate the threshold as the intensity value where the GMM label changes
        threshold = None
        label = gm.predict([[non_mdc_mean - 1]])[0]
        for i in range(int(non_mdc_mean), int(mdc_mean)):
            new_label = gm.predict([[i]])[0]
            if new_label != label:
                threshold = i
                break

        self.logger.debug(f"GMM means: {means}")
        self.logger.debug(f"GMM variances: {vars_}")
        self.logger.debug(f"Estimated threshold: {threshold}")

        return {
            "gm": gm,
            "means": means,
            "variances": vars_,
            "threshold": threshold,
            "mdc": {"mean": mdc_mean, "var": mdc_var},
            "non_mdc": {"mean": non_mdc_mean, "var": non_mdc_var},
        }

    def find_threshold(self, vol_array, segmentations, is_aorta=False):
        self.logger.debug("Finding threshold between mdc and thrombus")

        result = self.fit_gmm(vol_array, segmentations)
        means = result["means"]
        variances = result["variances"]

        mu1_index = np.argmin(means)
        mu2_index = np.argmax(means)

        seg_mask = self._merge_segmentations(segmentations)
        voxels = vol_array[seg_mask]

        n_bins = 100 if is_aorta else 50
        hist_counts, bin_edges = np.histogram(voxels, bins=n_bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        between = (bin_centers > means[mu1_index]) & (bin_centers < means[mu2_index])
        between_indices = np.where(between)[0]

        if len(between_indices) == 0:
            raise ValueError("No bins found between GMM component means.")

        min_bin_idx = between_indices[np.argmin(hist_counts[between])]
        threshold = int(bin_centers[min_bin_idx])

        self.logger.debug(f"Selected threshold: {threshold}")
        self.logger.debug(f"mu1: {means[mu1_index]}, mu2: {means[mu2_index]}")
        self.logger.debug(f"Variance mu1: {variances[mu1_index]}, mu2: {variances[mu2_index]}")

        return {
            "threshold": threshold,
            "mu1_mean": means[mu1_index],
            "mu1_var": variances[mu1_index][0],
            "mu2_mean": means[mu2_index],
            "mu2_var": variances[mu2_index][0],
            "gmm": result["gm"],
        }




class SegmentEditorManager:
    def __init__(self, segmentation_node, volume_node=None, mask_mode=None, overwrite=None, mask_segment_id=None):
        self.segmentation_node = segmentation_node
        self.volume_node = volume_node
        self.ct = CoordinateTransformer(volume_node) if volume_node else None
        self.segment_editor_widget = slicer.qMRMLSegmentEditorWidget()
        self.segment_editor_node = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(self.segment_editor_node)
        
        self.segment_editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setMRMLScene(slicer.mrmlScene)

        if volume_node:
            self.segment_editor_widget.setSourceVolumeNode(volume_node)

        self.segment_editor_node.SetOverwriteMode(
            overwrite if overwrite is not None else slicer.vtkMRMLSegmentEditorNode.OverwriteNone
        )

        if mask_mode is not None:
            self.segment_editor_node.SetMaskMode(mask_mode)

        if mask_segment_id is not None:
            self.segment_editor_node.SetMaskMode(5)
            self.segment_editor_node.SetMaskSegmentID(mask_segment_id)

        self.segmentation_node.RemoveClosedSurfaceRepresentation()

    @log_method_call
    def set_segment(self, segment_id):
        self.segment_editor_node.SetSelectedSegmentID(segment_id)
        
    @log_method_call
    def apply_effect(self, effect_name, params: dict):
        self.segment_editor_widget.setActiveEffectByName(effect_name)
        effect = self.segment_editor_widget.activeEffect()
        for key, val in params.items():
            effect.setParameter(key, val)
        effect.self().onApply()
        
    @log_method_call
    def island_effect(self, segment_id, operation="KEEP_LARGEST_ISLAND", min_size=10000):
        logging.debug(f"Island effect on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Islands", {
            "Operation": operation,
            "MinimumSize": str(min_size)
        })
    @log_method_call
    def close_holes(self, segment_id, kernel_size=10):
        logging.debug(f"Close holes on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Smoothing", {
            "SmoothingMethod": "MORPHOLOGICAL_CLOSING",
            "KernelSizeMm": kernel_size
        })
    @log_method_call
    def smoothing(self, segment_id, method="GAUSSIAN", kernel_size=1):
        logging.debug(f"Smoothing on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Smoothing", {
            "SmoothingMethod": method,
            "GaussianStandardDeviationMm": kernel_size
        })
    @log_method_call
    def hollow(self, segment_id, thickness_mm=1, mode="OUTSIDE_SURFACE"):
        logging.debug(f"Hollowing on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Hollow", {
            "ShellThicknessMm": thickness_mm,
            "ShellMode": mode,
            "ApplyToAllVisibleSegments": 0
        })
    @log_method_call
    def margin(self, segment_id, kernel_size=1):
        logging.debug(f"Margining on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Margin", {
            "MarginSizeMm": kernel_size,
            "ApplyToAllVisibleSegments": 0
        })
    @log_method_call
    def copy_segment(self, new_segment_name, from_segment_id):
        logging.debug(f"Copying {from_segment_id} -> {new_segment_name}")
        new_id = self.segmentation_node.GetSegmentation().AddEmptySegment(new_segment_name)
        self.segment_editor_node.SetSelectedSegmentID(new_id)
        
        from_id = from_segment_id
        if not from_segment_id.startswith("Segment"):
            from_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(from_segment_id)
            
        self.apply_effect("Logical operators", {
            "Operation": SegmentEditorEffects.LOGICAL_COPY,
            "ModifierSegmentID": from_id,
            "BypassMasking": 1
        })
        return new_id
    @log_method_call
    def subtract_segment(self, segment_id, subtract_id):
        logging.debug(f"Subtracting {subtract_id} from {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Logical operators", {
            "Operation": SegmentEditorEffects.LOGICAL_SUBTRACT,
            "ModifierSegmentID": self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(subtract_id),
            "BypassMasking": 1
        })
    @log_method_call
    def add_segment(self, segment_id, add_id):
        logging.debug(f"Adding {add_id} into {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Logical operators", {
            "Operation": SegmentEditorEffects.LOGICAL_UNION,
            "ModifierSegmentID": self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(add_id),
            "BypassMasking": 1
        })
    @log_method_call
    def threshold(self, new_segment_name, min_t, max_t, mask_segment_id=None):
        logging.debug(f"Thresholding {new_segment_name} with range {min_t}-{max_t}")
        new_id = self.segmentation_node.GetSegmentation().AddEmptySegment(new_segment_name)
        self.segment_editor_node.SetSelectedSegmentID(new_id)
        if mask_segment_id:
            self.segment_editor_node.SetMaskMode(5)
            self.segment_editor_node.SetMaskSegmentID(mask_segment_id)
        self.apply_effect("Threshold", {
            "MinimumThreshold": str(min_t),
            "MaximumThreshold": str(max_t)
        })
        return new_id
    
    @log_method_call
    def cleanup(self):
        slicer.mrmlScene.RemoveNode(self.segment_editor_node)
        
    @log_method_call
    def get_centroids_of_segment(self):
        logging.debug(f"get_centroids_of_segment | computing centroids for {self.segmentation_node.GetName()} on volume {self.volume_node.GetName()}")
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter("Segmentation", self.segmentation_node.GetID())
        segStatLogic.getParameterNode().SetParameter("ScalarVolume", self.volume_node.GetID())
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.centroid_ras.enabled", "True")

        for key in [
            "LabelmapSegmentStatisticsPlugin.volume_mm3",
            "LabelmapSegmentStatisticsPlugin.volume_cm3",
            "ScalarVolumeSegmentStatisticsPlugin.voxel_count",
            "ScalarVolumeSegmentStatisticsPlugin.volume_mm3",
            "ScalarVolumeSegmentStatisticsPlugin.volume_cm3",
            "ScalarVolumeSegmentStatisticsPlugin.min",
            "ScalarVolumeSegmentStatisticsPlugin.max",
            "ScalarVolumeSegmentStatisticsPlugin.mean",
            "ScalarVolumeSegmentStatisticsPlugin.stdev",
            "ScalarVolumeSegmentStatisticsPlugin.median",
        ]:
            segStatLogic.getParameterNode().SetParameter(key, "False")

        segStatLogic.computeStatistics()
        data = segStatLogic.getStatistics()
        logging.debug(f"get_centroids_of_segment | centroids computed: {data}")
        return data

    @log_method_call
    def get_centroids_of_segment_numpy(self, segment_name, slice_height=None,volume_node = None):
        logging.debug(f"get_centroids_of_segment_numpy | {segment_name}")
        vol = self.volume_node if volume_node is None else volume_node

        seg_array = arrayFromSegmentBinaryLabelmap(self.segmentation_node, self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name), vol)
        if slice_height is not None:
            h = min(slice_height, seg_array.shape[0] - 1)
            centroids = np.mean(np.argwhere(seg_array[int(h), :, :] == 1), axis=0)
            return int(centroids[0]), int(centroids[1])
        
        centroids = np.mean(np.argwhere(seg_array == 1), axis=0)
        return (
            int(centroids[2]) if not np.isnan(centroids[2]) else np.nan,
            int(centroids[1]) if not np.isnan(centroids[1]) else np.nan,
            int(centroids[0]) if not np.isnan(centroids[0]) else np.nan,
        )

    @log_method_call
    def get_segment_centroid_ras(self, segment_name: str, slice_height=None,coord_t=None):
        """Returns the RAS-space centroid of a segment at given slice height."""

        ct = self.ct if coord_t is None else coord_t
        if slice_height is not None:
            x, y = self.get_centroids_of_segment_numpy(segment_name, slice_height= slice_height,volume_node=ct.volume_node)
            ras = ct.ijk_to_ras([slice_height, x,y])
        else:
            ijk = self.get_centroids_of_segment_numpy(segment_name,slice_height=None,volume_node=ct.volume_node)
            ras = ct.ijk_to_ras(ijk)
       
        return np.array(ras)
    
    def get_segment_id(self, segment_name: str):
        return self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)


    def get_volume_spacing(self):
        return self.volume_node.GetSpacing()
        
    
    @log_method_call
    def _set_only_segments_visible(self, segments: list[str]):
        logging.debug(f"_set_only_segments_visible | Target segments: {segments} on node: {self.segmentation_node.GetName()}")
        display_node = self.segmentation_node.GetDisplayNode()

        if display_node is None:
            logging.warning("_set_only_segments_visible | No display node found for segmentation")
            return

        display_node.SetAllSegmentsVisibility(False)
        for segment_id in segments:
            logging.debug(f"_set_only_segments_visible | Setting segment '{segment_id}' to visible")
            display_node.SetSegmentVisibility(segment_id, True)
            

    @log_method_call
    def extract_centerline(
        self,
        segment_id,
        case_id,
        setting,
        centerline_endpoints,
        name="full_centerline",
        sampling_distance=1,
    ):
        logging.debug(
            f"extract_centerline | creating centerline '{name}' for segment '{segment_id}' and case '{case_id}'"
        )
        logging.debug(f"endpoints: {centerline_endpoints}")
        logging.debug(f"segmentation node: {self.segmentation_node.GetName()}")
        if self.volume_node:
            logging.debug(f"volume node: {self.volume_node.GetName()}")
        logging.debug(f"sampling distance: {sampling_distance}")

        ecWidgetRepresentation = slicer.modules.extractcenterline.widgetRepresentation()
        ecUI = ecWidgetRepresentation.self().ui

        ecUI.inputSurfaceSelector.setCurrentNode(self.segmentation_node)
        ecUI.inputSegmentSelectorWidget.setCurrentSegmentID(segment_id)

        centerline_endpoints_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"{case_id}_{setting}_{name}_endpoints"
        )
        for point in centerline_endpoints:
            centerline_endpoints_node.AddControlPoint(point)

        model = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", f"{case_id}_{setting}_{name}_model"
        )
        curve = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", f"{case_id}_{setting}_{name}_centerline"
        )

        ecUI.preprocessInputSurfaceModelCheckBox.checked = True
        ecUI.endPointsMarkupsSelector.setCurrentNode(centerline_endpoints_node)
        ecUI.outputCenterlineModelSelector.setCurrentNode(model)
        ecUI.outputCenterlineCurveSelector.setCurrentNode(curve)
        # Optional nodes can be set here if needed
        ecUI.decimationAggressivenessWidget.setValue(4)
        ecUI.targetKPointCountWidget.value = 100.0
        ecUI.curveSamplingDistanceSpinBox.setValue(sampling_distance)

        print(f"ready to extract centerline {name}")
        import pdb;pdb.set_trace()
        ecUI.applyButton.click()

        logging.debug(f"curve created: {curve.GetName()}")
        logging.debug(f"model created: {model.GetName()}")

        # Return the created centerline curve node
        return slicer.util.getNode(f"{case_id}_{setting}_{name}_centerline (0)")
    
    
    @log_method_call
    def compute_segment_volumes(self, features: dict) -> dict:
        logging.debug("compute_segment_volumes | Calculating volumes for predefined segments")

        segments_of_interest = [
            "located_renal_artery_right",
            "located_renal_artery_left",
            "located_sma",
            "located_celiac",
            "located_common_iliac_right",
            "located_common_iliac_left",
            "located_neck",
            "located_internal_iliac_left",
            "located_internal_iliac_right",
            "located_aneurysm_sac_lumen",
            "located_aneurysm_sac_thrombus",
            "located_aneurysm_sac_calc",
            "located_external_iliac_right",
            "located_external_iliac_left",
            "located_distal_sealing_right",
            "located_distal_sealing_left",
        ]

        spacing = self.volume_node.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # in mm³

        for segment_name in segments_of_interest:
            try:
                segment_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
                if segment_id is None:
                    logging.warning(f"Segment '{segment_name}' not found in segmentation")
                    continue

                seg_array = arrayFromSegmentBinaryLabelmap(self.segmentation_node, segment_id, self.volume_node)
                n_voxels = np.sum(seg_array)
                volume_mm3 = n_voxels * voxel_volume
                volume_cm3 = volume_mm3 / 1000.0

                features[f"{segment_name}_volume"] = volume_cm3
                logging.debug(f"{segment_name}: {volume_cm3:.2f} cm³")

            except Exception as e:
                logging.error(f"Error processing segment '{segment_name}': {str(e)}")

        # Compute total infrarenal aortic volume
        try:
            total_volume = (
                features["located_neck_volume"]
                + features["located_aneurysm_sac_thrombus_volume"]
                + features["located_aneurysm_sac_calc_volume"]
                + features["located_aneurysm_sac_lumen_volume"]
            )
            features["infrarenal_aorta_global_volume"] = total_volume
            logging.debug(f"infrarenal_aorta_global_volume: {total_volume:.2f} cm³")
        except KeyError as e:
            logging.warning(f"Missing component for infrarenal aorta volume: {str(e)}")

        return features


    @log_method_call
    def subdivide_aneurysm_sac(self, threshold_aorta: float, min_threshold_calc: float):
        """
        Subdivide the 'located_aneurysm_sac' segment into thrombus, lumen, and calcification
        based on voxel intensity thresholds.
        """
        logging.debug(f"subdivide_aneurysm_sac | Calcification if HU > {min_threshold_calc}")
        logging.debug(f"subdivide_aneurysm_sac | Thrombus if HU < {threshold_aorta}")

        volume_np = slicer.util.arrayFromVolume(self.volume_node)
        sac_seg_array = arrayFromSegmentBinaryLabelmap(
            self.segmentation_node,self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName("located_aneurysm_sac") , self.volume_node
        )

        thrombus_array = np.zeros_like(sac_seg_array, dtype=np.uint8)
        lumen_array = np.zeros_like(sac_seg_array, dtype=np.uint8)
        calc_array = np.zeros_like(sac_seg_array, dtype=np.uint8)

        masked_volume = np.copy(volume_np)
        masked_volume[sac_seg_array == 0] = -1000  # Background

        # Thresholding
        thrombus_indices = np.where((masked_volume < threshold_aorta) & (masked_volume != -1000))
        lumen_indices = np.where((masked_volume >= threshold_aorta) & (masked_volume < min_threshold_calc))
        calc_indices = np.where(masked_volume >= min_threshold_calc)

        thrombus_array[thrombus_indices] = 1
        lumen_array[lumen_indices] = 1
        calc_array[calc_indices] = 1

        # Create new empty segments
        self.segmentation_node.GetSegmentation().AddEmptySegment("located_aneurysm_sac_thrombus")
        self.segmentation_node.GetSegmentation().AddEmptySegment("located_aneurysm_sac_lumen")
        self.segmentation_node.GetSegmentation().AddEmptySegment("located_aneurysm_sac_calc")

        # Update segments from arrays
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            thrombus_array, self.segmentation_node, "located_aneurysm_sac_thrombus", self.volume_node
        )
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            lumen_array, self.segmentation_node, "located_aneurysm_sac_lumen", self.volume_node
        )

        self.close_holes("located_aneurysm_sac_lumen", kernel_size=3)

        lumen_pre_island = arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, "located_aneurysm_sac_lumen", self.volume_node
        )

        self.island_effect("located_aneurysm_sac_lumen", min_size=5000)

        lumen_post_island = arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, "located_aneurysm_sac_lumen", self.volume_node
        )

        # Identify dropped lumen regions and add them to calc
        dropped_voxels = (lumen_pre_island == 1) & (lumen_post_island == 0)
        retained_voxels = (lumen_post_island == 1)

        calc_array[dropped_voxels] = 1
        calc_array[retained_voxels] = 0  # Just in case

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            calc_array, self.segmentation_node, "located_aneurysm_sac_calc", self.volume_node
        )

        logging.debug("subdivide_aneurysm_sac | Subdivision complete")


    @log_method_call
    def get_bounds_of_segment(self, segment_id: str, box_size: int = None):
        """
        Get Z, Y, X bounds of a binary segment in IJK space.

        If box_size is given:
            - First crop the volume around the VOLUME CENTER (not segment center)
            - Only X and Y are cropped; Z remains full
            - Then compute segment bounds inside the cropped array
        """

        # ------------------------------------------------------------------
        # Load full segmentation labelmap array
        # ------------------------------------------------------------------
        seg_array = arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, segment_id, self.volume_node
        )

        Z, Y, X = seg_array.shape

        # ------------------------------------------------------------------
        # Step 1: CROP AROUND VOLUME CENTER (if requested)
        # ------------------------------------------------------------------
        if box_size is not None:
            y_c = Y // 2
            x_c = X // 2

            y_min = max(0, y_c - box_size)
            y_max = min(Y, y_c + box_size)

            x_min = max(0, x_c - box_size)
            x_max = min(X, x_c + box_size)

            seg_array = seg_array[:, y_min:y_max, x_min:x_max]

            # Shape now changed after crop
            Z, Y, X = seg_array.shape

        # ------------------------------------------------------------------
        # Step 2: FIND SEGMENT BOUNDS INSIDE seg_array (cropped or not)
        # ------------------------------------------------------------------
        def find_bounds(arr):
            z_min = z_max = y_min = y_max = x_min = x_max = None

            # Z bounds
            for i in range(arr.shape[0]):
                if np.any(arr[i, :, :]):
                    z_min = i
                    break
            for i in range(arr.shape[0] - 1, -1, -1):
                if np.any(arr[i, :, :]):
                    z_max = i
                    break

            # Segment is empty
            if z_min is None:
                return None, None, None, None, None, None

            # Y bounds
            for j in range(arr.shape[1]):
                if np.any(arr[z_min:z_max+1, j, :]):
                    y_min = j
                    break
            for j in range(arr.shape[1] - 1, -1, -1):
                if np.any(arr[z_min:z_max+1, j, :]):
                    y_max = j
                    break

            # X bounds
            for k in range(arr.shape[2]):
                if np.any(arr[z_min:z_max+1, :, k]):
                    x_min = k
                    break
            for k in range(arr.shape[2] - 1, -1, -1):
                if np.any(arr[z_min:z_max+1, :, k]):
                    x_max = k
                    break

            return z_min, z_max, y_min, y_max, x_min, x_max

        # Compute segment bounds inside the (possibly cropped) array
        z_min, z_max, y_min, y_max, x_min, x_max = find_bounds(seg_array)

        # ------------------------------------------------------------------
        # Return
        # ------------------------------------------------------------------
        logging.debug(
            f"Final segment bounds in cropped volume: "
            f"Z=[{z_min},{z_max}]  Y=[{y_min},{y_max}]  X=[{x_min},{x_max}]"
        )

        return {
            "z_min": z_min, "z_max": z_max,
            "y_min": y_min, "y_max": y_max,
            "x_min": x_min, "x_max": x_max,
            "seg_array": seg_array
        }


    @log_method_call
    def extract_walls_of_segment(
        self,
        segment_id_to_extract_walls_from: str,
        helper_segment_id: str,
        final_segment_name: str,
        thickness: float = 2.0,
    ):
        """
        Extract the wall of a segment by:
        1. Keeping the largest island
        2. Smoothing it
        3. Copying it to a new segment
        4. Hollowing it
        5. Subtracting a helper mask
        """
        logging.debug(f"extract_walls_of_segment | Extracting walls from: {segment_id_to_extract_walls_from}")
        logging.debug(f"Segmentation node: {self.segmentation_node.GetName()}")
        logging.debug(f"Volume node: {self.volume_node.GetName()}")
        logging.debug(f"Helper segment to subtract: {helper_segment_id}")
        logging.debug(f"Final wall segment name: {final_segment_name}")

        self.island_effect(
            segment_id=segment_id_to_extract_walls_from,
            operation="KEEP_LARGEST_ISLAND",
            min_size=100,
        )

        self.smoothing(
            segment_id=segment_id_to_extract_walls_from,
            method="GAUSSIAN",
            kernel_size=1,
        )

        self.copy_segment(final_segment_name, segment_id_to_extract_walls_from)

        self.hollow(
            segment_id=final_segment_name,
            thickness_mm=thickness,
            mode="OUTSIDE_SURFACE",
        )

        self.subtract_segment(final_segment_name, helper_segment_id)

        logging.debug(f"extract_walls_of_segment | Final wall segment '{final_segment_name}' created successfully.")


    @log_method_call
    def generate_walls_segmentations(self, thresholds_aorta, thresholds_iliac_left, thresholds_iliac_right,features):
        
  
        vol_arr = arrayFromVolume(self.volume_node)

        threshold_aorta = thresholds_aorta['threshold']
        calc_threshold_aorta = thresholds_aorta['mu2_mean'] + 2*np.sqrt(thresholds_aorta['mu2_var'])
        
        threshold_iliac_left = thresholds_iliac_left['threshold']
        calc_threshold_iliac_left = thresholds_iliac_left['mu2_mean'] + 2*np.sqrt(thresholds_iliac_left['mu2_var'])
        
        threshold_iliac_right = thresholds_iliac_right['threshold']
        calc_threshold_iliac_right = thresholds_iliac_right['mu2_mean'] + 2*np.sqrt(thresholds_iliac_right['mu2_var'])
        
        seg_aorta = arrayFromSegmentBinaryLabelmap(self.segmentation_node, "proximal_sealing_zone_walls", self.volume_node)
        seg_iliac_right = arrayFromSegmentBinaryLabelmap(self.segmentation_node, "right_distal_sealing_zone_walls", self.volume_node)
        seg_iliac_left = arrayFromSegmentBinaryLabelmap(self.segmentation_node, "left_distal_sealing_zone_walls", self.volume_node)

        thrombus_aorta_ind = np.where((vol_arr > -20) & (vol_arr <= threshold_aorta) & (seg_aorta == 1))
        thrombus_iliac_left_ind = np.where((vol_arr > -20) & (vol_arr <= threshold_iliac_left) & (seg_iliac_left == 1))
        thrombus_iliac_right_ind = np.where((vol_arr > -20) & (vol_arr <= threshold_iliac_right) & (seg_iliac_right == 1))

        calc_aorta_ind = np.where((vol_arr > calc_threshold_aorta) & (seg_aorta == 1))
        calc_iliac_left_ind = np.where((vol_arr > calc_threshold_iliac_left) & (seg_iliac_left == 1))
        calc_iliac_right_ind = np.where((vol_arr > calc_threshold_iliac_right) & (seg_iliac_right == 1))

        seg_aorta_calc = np.zeros_like(seg_aorta)
        seg_iliac_left_calc = np.zeros_like(seg_aorta)
        seg_iliac_right_calc = np.zeros_like(seg_aorta)

        seg_aorta_thrombus = np.zeros_like(seg_aorta)
        seg_iliac_left_thrombus = np.zeros_like(seg_aorta)
        seg_iliac_right_thrombus = np.zeros_like(seg_aorta)

        seg_aorta_calc[calc_aorta_ind] = 1
        seg_iliac_left_calc[calc_iliac_left_ind] = 1
        seg_iliac_right_calc[calc_iliac_right_ind] = 1

        seg_aorta_thrombus[thrombus_aorta_ind] = 1
        seg_iliac_left_thrombus[thrombus_iliac_left_ind] = 1
        seg_iliac_right_thrombus[thrombus_iliac_right_ind] = 1

        eroded_seg_aorta_thrombus = self.erode_itk(seg_aorta_thrombus)
        eroded_seg_iliac_left_thrombus = self.erode_itk(seg_iliac_left_thrombus)
        eroded_seg_iliac_right_thrombus = self.erode_itk(seg_iliac_right_thrombus)

        updateSegmentBinaryLabelmapFromArray(seg_aorta_calc, self.segmentation_node, "aorta_walls_calc", self.volume_node)
        updateSegmentBinaryLabelmapFromArray(seg_iliac_left_calc, self.segmentation_node, "iliac_left_walls_calc", self.volume_node)
        updateSegmentBinaryLabelmapFromArray(seg_iliac_right_calc, self.segmentation_node, "iliac_right_walls_calc", self.volume_node)

        updateSegmentBinaryLabelmapFromArray(eroded_seg_aorta_thrombus, self.segmentation_node, "aorta_walls_thrombus", self.volume_node)
        updateSegmentBinaryLabelmapFromArray(eroded_seg_iliac_left_thrombus, self.segmentation_node, "iliac_left_walls_thrombus", self.volume_node)
        updateSegmentBinaryLabelmapFromArray(eroded_seg_iliac_right_thrombus, self.segmentation_node, "iliac_right_walls_thrombus", self.volume_node)

        updateSegmentBinaryLabelmapFromArray(seg_aorta_thrombus, self.segmentation_node, "aorta_walls_thrombus_ne", self.volume_node)
        updateSegmentBinaryLabelmapFromArray(seg_iliac_left_thrombus, self.segmentation_node, "iliac_left_walls_thrombus_ne", self.volume_node)
        updateSegmentBinaryLabelmapFromArray(seg_iliac_right_thrombus, self.segmentation_node, "iliac_right_walls_thrombus_ne", self.volume_node)

        # Logging statistics
        def log_stats(name, eroded, original, segment):
            eroded_pct = np.sum(eroded) / (np.sum(segment) / 2) if np.sum(segment) > 0 else 0
            original_pct = np.sum(original) / (np.sum(segment)) if np.sum(segment) > 0 else 0
            
            features[f"perc_thrombo_{name}_(erosion)"] = eroded_pct
            features[f"perc_thrombo_{name}"] = original_pct
            logging.info(f"perc_thrombo_{name}_(erosion): {eroded_pct}")
            logging.info(f"perc_thrombo_{name}: {original_pct}")
        
        log_stats("neck", eroded_seg_aorta_thrombus, seg_aorta_thrombus, seg_aorta)
        log_stats("left iliac", eroded_seg_iliac_left_thrombus, seg_iliac_left_thrombus, seg_iliac_left)
        log_stats("right iliac", eroded_seg_iliac_right_thrombus, seg_iliac_right_thrombus, seg_iliac_right)

        def log_calc(name, calc_seg, seg):
            pct = np.sum(calc_seg) / np.sum(seg) if np.sum(seg) > 0 else 0
            features[f"perc_calc_{name}"] = pct 
            logging.info(f"perc_calc_{name}: {pct}")

        log_calc("neck", seg_aorta_calc, seg_aorta)
        log_calc("left iliac", seg_iliac_left_calc, seg_iliac_left)
        log_calc("right iliac", seg_iliac_right_calc, seg_iliac_right)

        logging.info(f"thresholds aorta: {thresholds_aorta}")
        logging.info(f"thresholds left iliac: {thresholds_iliac_left}")
        logging.info(f"thresholds right iliac: {thresholds_iliac_right}")
        
    @log_method_call
    def extract_diameters_fedez(self, seg_name, ref_fetta, points_of_interest, offset=0, ascending=True,straight_vol = None,straight_transform = None,save_csv_path = None,view="axial",retain_biggest=False,crop_spatial=None):
        
        from feret.main import Calculater
        
        import numpy as np
        from skimage.measure import label, regionprops

        def keep_largest_component(mask):
            labeled = label(mask)               # label connected components
            if labeled.max() == 0:
                return mask                     # no components
            
            # get the region with the largest area
            largest_region = max(regionprops(labeled), key=lambda r: r.area)
            largest_label = largest_region.label
            
            # keep only that region
            return (labeled == largest_label).astype(mask.dtype)

        
        
        def feret_diameters_and_coords(img, edge=False):
            calc = Calculater(img, edge)
            calc.calculate_minferet()
            calc.calculate_maxferet()
            
            min_dist, min_endpoints = calc.calculate_distances(
                calc.minf_angle - np.pi / 2
            )

            return {
                "min_diameter": calc.minf,
                "min_endpoints": min_endpoints, 
                "min_coords": calc.minf_coords,
                "min_angle": calc.minf_angle,  
                "max_diameter": calc.maxf,
                "max_coords": calc.maxf_coords,
                "max_angle": calc.maxf_angle       
            }
        
        vol = self.volume_node if straight_vol is None else straight_vol
        if straight_transform is not None:
            self.segmentation_node.SetAndObserveTransformNodeID(
                    straight_transform.GetID()
                )
            
        seg_array = arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, seg_name, vol
        )
        
        if crop_spatial is not None:

            x_start = (64 - crop_spatial) // 2
            y_start = (64 - crop_spatial) // 2
            mask = np.zeros((64, 64), dtype=bool)
            mask[x_start:x_start + crop_spatial, y_start:y_start + crop_spatial] = True
            seg_array = seg_array*mask
            
            
        spacings = vol.GetSpacing()  # (x,y,z)
        depth = seg_array.shape[0]

        toRtnMaxFerets = []
        toRtnMinFerets = []
        rows_to_save = []
        
        for p in points_of_interest:
            slice_height = int(p / spacings[2])
            if not ascending:
                slice_height = ref_fetta - slice_height
            else:
                slice_height = ref_fetta + slice_height

            offsetUp = offset
            offsetDown = offset
            if p == 0:
                if ascending:
                    offsetUp = 0
                else:
                    offsetDown = 0

            start_idx = max(0, slice_height - offsetUp)
            end_idx = min(depth - 1, slice_height + offsetDown)
            indices_of_interest = range(start_idx, end_idx + 1)

            min_ferets = []
            max_ferets = []

            for idx in indices_of_interest:
                seg_slice = seg_array[idx, :, :]
                if retain_biggest:
                    seg_slice = keep_largest_component(seg_slice)
                if np.all(seg_slice == 0):
                    continue
                feret_info = feret_diameters_and_coords(seg_array[idx, :, :])

                # Convert diameter from pixels to mm
                min_dia = feret_info['min_diameter'] * spacings[0]
                max_dia = feret_info['max_diameter'] * spacings[0]

                min_ferets.append(min_dia)
                max_ferets.append(max_dia)

                # Save detailed row for CSV
                min_coords = feret_info['min_coords']
                max_coords = feret_info['max_coords']
                
                min_pts = feret_info["min_endpoints"]

                row_min = [
                    p, idx, "min", min_dia,
                    min_pts[0][0], min_pts[0][1],
                    min_pts[1][0], min_pts[1][1],
                    "", "",                      # no p3
                    feret_info["min_angle"], view
                ]
                row_max = [
                    p, idx, "max", max_dia,
                    *max_coords[0], *max_coords[1], "", "", feret_info['max_angle'],view
                ]
                rows_to_save.append(row_min)
                rows_to_save.append(row_max)

            toRtnMinFerets.append(np.mean(min_ferets))
            toRtnMaxFerets.append(np.mean(max_ferets))

        if straight_transform is not None:
            self.segmentation_node.SetAndObserveTransformNodeID(None)

        # Save CSV if requested
        if save_csv_path:
            dirpath = os.path.dirname(save_csv_path)
            if dirpath:  # Only create directories if one is specified
                os.makedirs(dirpath, exist_ok=True)
            with open(save_csv_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "point_distance", "slice_index", "feret_type", "diameter_mm",
                    "p1_y", "p1_x", "p2_y", "p2_x", "p3_y", "p3_x", "angle_rad","view"
                ])
                writer.writerows(rows_to_save)
            logging.info(f"Saved Feret diameter data to: {save_csv_path}")

        return toRtnMinFerets, toRtnMaxFerets
    
    
    @staticmethod
    def transfer_segmentation_from_seg_nodes_world_to_cpr(segmentations,sm_from,sm_to,transform):
        sm_from.segmentation_node.SetAndObserveTransformNodeID(transform.GetID())
        SegmentEditorManager.transfer_segmentations(segmentations,sm_from,sm_to)
        sm_from.segmentation_node.SetAndObserveTransformNodeID(None)
    
    @staticmethod
    def transfer_segmentations(segmentations,sm_from,sm_to):

        logging.debug(f"transfering segmentations {segmentations}")
        logging.debug(f"from {sm_from.segmentation_node.GetName()}")
        logging.debug(f"to {sm_to.segmentation_node.GetName()}")

        for segmentation in segmentations:
            seg_array = arrayFromSegmentBinaryLabelmap(
                sm_from.segmentation_node, segmentation, sm_from.volume_node
            )

            updateSegmentBinaryLabelmapFromArray(
                seg_array,
                sm_to.segmentation_node,
                segmentation,
                sm_from.volume_node,
            )

    
    
    @staticmethod
    def load_segmentation_from_node(name):
        try:
            return slicer.util.getNode(f"{name}.nii.gz")
        except Exception as e:
            logging.error(f"Failed to load segmentation {name}: {e}")
            return None

    @log_method_call
    def erode_itk(self,binary_array,kernel_radius=(1,1,0)):
        """
        Apply a 3D binary erosion to the input binary mask using SimpleITK.

        Erosion is applied with a (1, 1, 0) kernel radius and a ball-shaped kernel.
        This performs 2D erosion across slices rather than full 3D.

        Args:
            binary_array (np.ndarray): 3D NumPy array representing the binary mask (values: 0 and 1).

        Returns:
            np.ndarray: The eroded binary mask as a NumPy array.
        """
        
        binary_array = np.asarray(binary_array, dtype=np.uint8)
        
        input_image = sitk.GetImageFromArray(binary_array)

        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetBackgroundValue(0.0)
        erode_filter.SetForegroundValue(1.0)
        erode_filter.SetBoundaryToForeground(True)
        erode_filter.SetKernelRadius(kernel_radius)  # 2D erosion in axial slices
        erode_filter.SetKernelType(sitk.sitkBall)
        erode_filter.SetNumberOfThreads(12)
        erode_filter.SetNumberOfWorkUnits(0)

        output_image = erode_filter.Execute(input_image)
        return sitk.GetArrayFromImage(output_image)
    
    @log_method_call
    def get_index_of_most_overlapped_labelmap(
        self,
        target_segment_name: str,
        candidate_segment_names: List[str],) -> Optional[int]:
        """
        Find the index of the labelmap in `candidate_masks` that has the highest overlap
        (based on voxel-wise logical AND) with `target_mask`.

        Args:
            target_mask (np.ndarray): The binary mask to match against.
            candidate_masks (List[np.ndarray]): A list of binary masks to compare.

        Returns:
            int: Index of the mask in `candidate_masks` with the highest overlap.

        Raises:
            SystemExit: If no overlapping mask is found.
        """
        max_overlap = -1
        max_index = None

        target_segment_mask = arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, target_segment_name, self.volume_node
        )
        
        for i, seg_name in enumerate(candidate_segment_names):
            seg_mask = arrayFromSegmentBinaryLabelmap(
                self.segmentation_node, seg_name, self.volume_node
            )
            overlap = np.logical_and(target_segment_mask, seg_mask).sum()
            if overlap > max_overlap:
                max_overlap = overlap
                max_index = i

        if max_overlap == 0:
            logging.error("No overlapping labelmap found for the target mask.")
            sys.exit()

        return max_index
    
    @log_method_call
    def mask_volume(self,
                    segmentations,
                    blacklist = {"aorta","iliac_artery_left","iliac_artery_right","iliac_vena_left","iliac_vena_right"},
                    mask_value=-1000):
        logging.debug(
            f"Masking volume {self.volume_node.GetName()} based on segmentations in {self.segmentation_node.GetName()}"
        )

        # blacklist = {
        #     "aorta",
        #     "kidney_left",
        #     "kidney_right",
        #     "iliac_artery_left",
        #     "iliac_artery_right",
        #     "iliac_vena_left",
        #     "iliac_vena_right",
        # }

        vol_array = arrayFromVolume(
            self.volume_node
        )

        for seg_name in segmentations:
            if seg_name in blacklist:
                continue

            segment = self.segmentation_node.GetSegmentation().GetSegment(seg_name)
            if segment is None:
                logging.warning(f"Segment '{seg_name}' not found in combined segmentation.")
                continue

            mask = arrayFromSegmentBinaryLabelmap(
                self.segmentation_node, seg_name, self.volume_node
            )

            vol_array[mask != 0] = mask_value

        updateVolumeFromArray(self.volume_node, vol_array)  
        logging.debug("Volume masking completed.")

        return self.volume_node, vol_array
    
    @log_method_call
    def merge_segmentations(self,segmentation_names):

        vertebrae_mask = np.zeros_like(arrayFromVolume(self.volume_node), dtype=bool)
    
        for name in segmentation_names:
            seg_node = self.load_segmentation_from_node(name)
            if seg_node is None:
                logging.warning(f"Could not load segmentation: {name}")
                continue

            segment = seg_node.GetSegmentation().GetSegment("Segment_1")
            if segment is None:
                #missing_kidneys = check_kidney(name, missing_kidneys)
                slicer.mrmlScene.RemoveNode(seg_node)
                continue

            segment.SetName(name)

            self.segmentation_node.GetSegmentation().AddSegment(segment)

            if "vertebrae" in name:
                mask = arrayFromSegmentBinaryLabelmap(
                    self.segmentation_node, name, self.volume_node
                )
                vertebrae_mask = np.logical_or(vertebrae_mask, mask)

            slicer.mrmlScene.RemoveNode(seg_node)

        # Add combined vertebrae segment
        self.segmentation_node.GetSegmentation().AddEmptySegment("vertebrae")
        updateSegmentBinaryLabelmapFromArray(
            vertebrae_mask.astype(np.uint8), self.segmentation_node, "vertebrae", self.volume_node
        )
        logging.debug("Finished merge_segmentations()")
        
        displayNode = self.segmentation_node.GetDisplayNode()
        displayNode.SetAllSegmentsVisibility2DFill(False)
        displayNode.SetSliceIntersectionThickness(2)


    @log_method_call
    def add_empty_segment(self, segment_name: str):
        """
        Add an empty segment to the segmentation node.
        """
        if self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name):
            logging.warning(f"Segment '{segment_name}' already exists. Skipping addition.")
            return

        self.segmentation_node.GetSegmentation().AddEmptySegment(segment_name)
        logging.debug(f"Added empty segment '{segment_name}' to segmentation node.")
        
    def retain_segment_inside_box(self, segment_id: str, box_bounds: dict):
        """
        Retain only the part of the segment that lies within the specified box bounds.
        Box bounds should be provided as a dictionary with keys:
        'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max' in IJK coordinates.
        """
        seg_array = arrayFromSegmentBinaryLabelmap(self.segmentation_node, segment_id, self.volume_node)
        z_len, y_len, x_len = seg_array.shape
        z_min = box_bounds["z_min"] if "z_min" in box_bounds else 0
        z_max = box_bounds["z_max"] if "z_max" in box_bounds else z_len - 1
        y_min = box_bounds["y_min"] if "y_min" in box_bounds else 0
        y_max = box_bounds["y_max"] if "y_max" in box_bounds else y_len - 1
        x_min = box_bounds["x_min"] if "x_min" in box_bounds else 0
        x_max = box_bounds["x_max"] if "x_max" in box_bounds else x_len - 1

        # Create a mask for the box
        box_mask = np.zeros_like(seg_array, dtype=bool)
        box_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = True

        # Retain only the part of the segment inside the box
        new_seg_array = np.where(box_mask, seg_array, 0)

        # Update the segment with the new array
        slicer.util.updateSegmentBinaryLabelmapFromArray(new_seg_array, self.segmentation_node, segment_id, self.volume_node)
        logging.debug(f"Retained part of segment '{segment_id}' inside specified box bounds.")
        
        
    def add_segment_from_numpy(self,segment_id,segmentation):
        self.add_empty_segment(segment_id)
        
        slicer.util.updateSegmentBinaryLabelmapFromArray(segmentation, self.segmentation_node, segment_id, self.volume_node)
        
    def set_segment_name(self,segment_id,new_name):
        self.segmentation_node.GetSegmentation().GetSegment(segment_id).SetName(new_name)
        logging.debug(f"changed name of segment {segment_id} to {new_name}")
        
        
    def intersect_segment(self,seg_id,modifier_seg_id):
        
        seg_array = arrayFromSegmentBinaryLabelmap(self.segmentation_node, seg_id, self.volume_node)
        seg_modifier_array = arrayFromSegmentBinaryLabelmap(
        self.segmentation_node, modifier_seg_id, self.volume_node
    )

        seg_array = seg_array & seg_modifier_array
        updateSegmentBinaryLabelmapFromArray(
        seg_array, self.segmentation_node, seg_id, self.volume_node
    )
        
    def getSegmentationAtRASNeighborhood(self, ras, segmentation_names, neighborhood_radius=2):
        """
        Parameters:
            ras (list or np.ndarray): RAS coordinate [x, y, z]
            segmentation_names (list or None): list of segmentation names to check.
                                          
            neighborhood_radius (int): how many voxels to include around the center (default: 1)
        Returns:
            list of (segmentation_name, most_frequent_label) tuples
        """
        ijk = self.ct.ras_to_ijk(ras)
        i, j, k = map(int, map(round, ijk))

        max_count = 0
        best_segmentation = None
        voxel_counts = 0
        for name in segmentation_names:
            
            seg = arrayFromSegmentBinaryLabelmap(self.segmentation_node, name, self.volume_node)
            
            i_min = max(0, i - neighborhood_radius)
            i_max = min(seg.shape[2], i + neighborhood_radius + 1)
            j_min = max(0, j - neighborhood_radius)
            j_max = min(seg.shape[1], j + neighborhood_radius + 1)
            k_min = max(0, k - neighborhood_radius)
            k_max = min(seg.shape[0], k + neighborhood_radius + 1)

            neighborhood = seg[k_min:k_max, j_min:j_max, i_min:i_max]
            count = np.count_nonzero(neighborhood == 1)

            if count > max_count:
                max_count = count
                best_segmentation = name
                voxel_counts = np.count_nonzero(seg)

        return best_segmentation, max_count,voxel_counts

    def export_segment_as_vol(self,segment_name,path):

        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLScalarVolumeNode", self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
                )

        m = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASDirectionMatrix(m)
        labelmap_node.SetIJKToRASDirectionMatrix(m)
        labelmap_node.SetSpacing(self.volume_node.GetSpacing())
        labelmap_node.SetOrigin(self.volume_node.GetOrigin())


        segmentation_arr_to_export  = arrayFromSegmentBinaryLabelmap(self.segmentation_node, self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name), self.volume_node)
        segmentation_arr_to_export = segmentation_arr_to_export[:,:,:]
        updateVolumeFromArray(labelmap_node,segmentation_arr_to_export)


        slicer.util.exportNode(
                    labelmap_node,
                    os.path.join(path, f"{labelmap_node.GetName()}.nii.gz"),
                    {"useCompression": 0},
                    world=True,
                )

        slicer.mrmlScene.RemoveNode(labelmap_node)

        return True 


from skimage.transform import resize


class ThrombusSegmentationPredictor:
    def __init__(self, non_mdc_mean, non_mdc_var, crop_x=64, crop_y=64):
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.non_mdc_mean = non_mdc_mean
        self.non_mdc_var = non_mdc_var
        self.device = torch.device("cpu")
        self.logger = logging.getLogger(self.__class__.__name__)

    # ----------------------------------------------------------------------------
    # MODEL LOADING
    # ----------------------------------------------------------------------------
    def _load_model(self, model_path, spatial_dims):
        model = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=2,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2 if spatial_dims == 3 else 0.0,
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval().to(self.device)
        return model

    # ----------------------------------------------------------------------------
    # PREPROCESSING PIPELINES
    # ----------------------------------------------------------------------------
    def _get_preprocessing(self, spatial_dims):
        pad_size = (self.crop_x, self.crop_y, -1) if spatial_dims == 3 else (self.crop_x, self.crop_y)
        return Compose([
            EnsureTyped(keys=["image"]),
            ResizeWithPadOrCropD(keys=["image"], spatial_size=pad_size, allow_missing_keys=True),
            DivisiblePadd(keys=["image"], k=(-1, -1, 8) if spatial_dims == 3 else (-1, -1)),
            MedianSmoothd(keys=["image"], radius=1),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ])

    def _get_postprocessing(self):
        return Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys=["pred"], argmax=True),
        ])

    # ----------------------------------------------------------------------------
    # CT WINDOWING & NORMALIZATION
    # ----------------------------------------------------------------------------
    def _vectorize(self, L, W):
        norm = lambda t: min(255, max(0, ((t - (L - (W / 2))) * (255 / W))))
        return np.vectorize(norm)

    def _process_ct_scan(self, volume_data):
        L = self.non_mdc_mean
        lambda_l = self.non_mdc_mean - 2 * np.sqrt(self.non_mdc_var)
        W = -2 * (lambda_l - L)
        normalizer = self._vectorize(L, W)
        return normalizer(volume_data)

    # ----------------------------------------------------------------------------
    # VOLUME LOADING AND BOUNDING BOX HANDLING
    # ----------------------------------------------------------------------------
    def _prepare_volume(self, volume_node, bounding_box):
        vol = arrayFromVolume(volume_node)
        vol = np.swapaxes(vol, 0, 2)  # Slicer ZYX → XYZ

        full_shape = vol.shape  # shape BEFORE cropping

        if bounding_box:
            z0, z1 = bounding_box
            # Safety check
            assert 0 <= z0 < z1 <= full_shape[2], \
                f"Invalid bounding box {bounding_box}, full Z={full_shape[2]}"

            print(f"[prepare_volume] Cropping: full Z={full_shape[2]} --> {z1 - z0}")
            vol = vol[:, :, z0:z1]

        processed = self._process_ct_scan(vol)

        return processed, full_shape


    # ----------------------------------------------------------------------------
    # FINAL VOLUME RECONSTRUCTION
    # ----------------------------------------------------------------------------
    def _finalize_output(self, raw_output, full_shape, bounding_box):
        full_vol = np.zeros(full_shape, dtype=raw_output.dtype)

        if bounding_box:
            z0, z1 = bounding_box
            expected_z = z1 - z0
            actual_z = raw_output.shape[2]

            if actual_z != expected_z:
                raise ValueError(
                    f"[finalize_output] Z mismatch: expected {expected_z} slices (from bbox "
                    f"{bounding_box}) but raw_output has {actual_z} slices"
                )

            full_vol[:, :, z0:z1] = raw_output
            return full_vol

        else:
            return raw_output

    # ----------------------------------------------------------------------------
    # 3D MODEL PREDICTION
    # ----------------------------------------------------------------------------
    def predict_volume_3d(self, volume_node, segmentation_node, segment_name, bounding_box=None):

        volume, full_shape = self._prepare_volume(volume_node, bounding_box)

        model = self._load_model(r"C:\tmp\AAA\thrombus_seg_3d\models\3d_best_metric_model_2.pth", spatial_dims=3)

        preprocessing = self._get_preprocessing(spatial_dims=3)
        postprocessing = self._get_postprocessing()

        with torch.no_grad():
            data = preprocessing({"image": np.expand_dims(volume, 0)})
            img = data["image"]
            pred = model(img.unsqueeze(0))

            data = postprocessing({"pred": pred[0]})

            data["image"] = preprocessing.transforms[2].inverse({"image": data["pred"]})["image"]
            data["image"] = preprocessing.transforms[1].inverse({"image": data["image"]})["image"]

            prediction = data["image"][0].numpy()

        final_output = self._finalize_output(prediction, full_shape, bounding_box)

        segmentation_node.GetSegmentation().AddEmptySegment(segment_name)
        updateSegmentBinaryLabelmapFromArray(
            np.swapaxes(final_output, 0, 2).astype(np.uint8),
            segmentation_node, segment_name, volume_node
        )

    # ----------------------------------------------------------------------------
    # 2D MODEL PREDICTION
    # ----------------------------------------------------------------------------
    def predict_volume_2d(self, volume_node, segmentation_node, segment_name, bounding_box=None):

        self.logger.debug(f"Running 2D prediction on {volume_node.GetName()}")

        # Prepare volume + original full shape
        volume, full_shape = self._prepare_volume(volume_node, bounding_box)

        X_full, Y_full, Z_full = full_shape           # full volume dims
        Z_cropped = volume.shape[2]                   # cropped Z

        print(f"[2D] Full Z={Z_full}, Cropped Z={Z_cropped}")

        # ----------------------------
        # Load model
        # ----------------------------
        model = self._load_model(r"C:\tmp\AAA\thrombus_seg_2d\models\2d_best_metric_model_2.pth", spatial_dims=2)


        # ----------------------------
        # Preprocessing & postprocessing
        # ----------------------------
        pad = ResizeWithPadOrCropD(keys=["image"], spatial_size=(self.crop_x, self.crop_y))

        preprocessing = Compose([
            EnsureTyped(keys=["image"]),
            pad,
            MedianSmoothd(keys=["image"], radius=1),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ])

        postprocessing = Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys=["pred"], argmax=True),
        ])

        # Volume for model: (1, X, Y, Z_cropped)
        volume = np.expand_dims(volume, 0)

        # Output volume for cropped part
        roi_volume = np.zeros((X_full, Y_full, Z_cropped), dtype=np.uint8)

        # ----------------------------
        # Process each slice
        # ----------------------------
        with torch.no_grad():
            for i in range(Z_cropped):
                print(f"[2D] Processing slice {i+1}/{Z_cropped}")

                # Preprocess slice
                data = preprocessing({"image": volume[:, :, :, i]})
                img = data["image"]

                # Model inference
                pred = model(img.unsqueeze(0))
                data = postprocessing({"pred": pred[0]})

                # Undo padding (2D only affects XY)
                slice_pred = pad.inverse({"image": data["pred"]})["image"][0].numpy()

                # Resize back to full original XY resolution
                slice_full = resize(
                    slice_pred,
                    (X_full, Y_full),
                    order=0,
                    preserve_range=True
                ).astype(np.uint8)

                roi_volume[:, :, i] = slice_full

        # ----------------------------
        # Reconstruct full output volume
        # ----------------------------
        final_output = self._finalize_output(roi_volume, full_shape, bounding_box)

        # ----------------------------
        # Write to segmentation
        # ----------------------------
        segmentation_node.GetSegmentation().AddEmptySegment(segment_name)
        updateSegmentBinaryLabelmapFromArray(
            np.swapaxes(final_output, 0, 2).astype(np.uint8),
            segmentation_node,
            segment_name,
            volume_node
        )


    # ----------------------------------------------------------------------------
    # RUN BOTH
    # ----------------------------------------------------------------------------
    def predict(self, volume_node, segmentation_node, segment_base_name, bounding_box=None):
        self.predict_volume_2d(volume_node, segmentation_node, segment_base_name + "_2d", bounding_box)
        self.predict_volume_3d(volume_node, segmentation_node, segment_base_name + "_3d", bounding_box)






def apply_color(segment, name):
    colors = {
        "inferior_vena_cava": (0, 151, 206),
        "aorta_extended": (224, 97, 76),
        "iliac_artery_left": (152, 55, 13),
        "iliac_artery_right": (151, 85, 57),
        "renal_arteries": (0, 0, 255),
    }
    if name in colors:
        segment.SetColor(np.array(colors[name]) / 255.0)

# def check_kidney(segmentation_name, missing_kidneys):
#     logging.debug("Verifying kidney presence...")

#     if segmentation_name in {"kidney_left", "kidney_right"}:
#         missing_kidneys[segmentation_name] = True
#         logging.debug(f"Marked '{segmentation_name}' as missing.")
#     else:
#         logging.debug(f"'{segmentation_name}' is not a kidney segment.")

#     return missing_kidneys





def hide_scene():
    for node in list(slicer.mrmlScene.GetNodesByClass('vtkMRMLDisplayableNode')):
        if node.GetName() in ["Combined","Red Transform", "Red Volume Slice","Green Transform","Green Volume Slice","Yellow Transform","Yellow Volume Slice"]:
            continue
        print(node.GetName())
        
        d_s = node.GetDisplayNode()
        if d_s:
            d_s.SetVisibility(False)
            




def load_data(folder_path, case_id, setting, segmentations):
    """
    Load a CT scan volume and its associated segmentations from a given folder.

    Args:
        folder_path (str): Path to the dataset folder.
        case_id (str): Case identifier (e.g., '001').
        setting (str): Identifier for the prediction setting (e.g., 'baseline').
        segmentations (list of str): List of segmentation names to load.

    Returns:
        None
    """
    ct_filename = f"CT_{case_id}_{setting}_001_0002.nii.gz"
    ct_path = os.path.join(folder_path,case_id, ct_filename)

    seg_folder = os.path.join(folder_path,case_id, f"Predictions_{setting}")

    logging.debug(f"Loading CT scan: {ct_filename}")
    if not os.path.exists(ct_path):
        logging.error(f"CT file not found: {ct_path}")
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    slicer.util.loadVolume(ct_path)

    for seg_name in segmentations:
        seg_filename = f"{seg_name}.nii.gz"
        seg_path = os.path.join(seg_folder, seg_filename)

        logging.debug(f"Loading segmentation '{seg_name}' from {seg_path}")
        if not os.path.exists(seg_path):
            logging.error(f"Segmentation file not found: {seg_path}")
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

        slicer.util.loadSegmentation(seg_path)


def save_features(features, folder_path, case_id, setting):
    """
    Save extracted features to a CSV file.

    Args:
        features (list): List of feature values.
        folder_path (str): Path to the root folder where features will be saved.
        case_id (str): Identifier of the case.
        setting (str): Processing or prediction setting name.

    Returns:
        None
    """
    save_dir = os.path.join(folder_path, case_id)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "features_estratte.csv")
    logging.debug(f"Saving features to {save_path}")

    df = pd.DataFrame({
        "Misura": list(features.keys()),
        "Valore": list(features.values())
    })

    try:
        df.to_csv(save_path, index=False)
    except Exception as e:
        logging.error(f"Failed to save features CSV: {e}")
        raise


def get_max_hu_from_volume(volume_node):
    """
    Get the maximum Hounsfield Unit (HU) value from a volume node.

    Args:
        volume_node (vtkMRMLScalarVolumeNode): The volume node to analyze.

    Returns:
        float: The maximum HU value in the volume.
    """
    if not volume_node:
        logging.error("Volume node is None.")
        raise ValueError("Volume node is None.")

    vol_array = arrayFromVolume(volume_node)
    max_hu = np.max(vol_array)

    logging.debug(f"Maximum HU value in the volume: {max_hu}")

    return max_hu

# --- Helper function to rename most overlapped island segment ---
def rename_island_segment(label_index, islands, new_name,sm):
    idx = sm.get_index_of_most_overlapped_labelmap(label_index, islands)
    segment_id = sm.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(islands[idx])
    segment = sm.segmentation_node.GetSegmentation().GetSegment(segment_id)
    segment.SetName(new_name)
    
    
    
    
    



def calculate_diameter(diameters, i):
    """Helper function to calculate the mean diameter."""
    return np.mean(diameters[i - 1:i + 2])


def extract_neck_statistics(
    mis_diameters,
    ce_diameters,
    init_index,
    end_index,
    centerline_node,
    features,
    debug=False,
):
    logging.debug("Extracting measures from identified neck")

    points_of_interests = [0, 0.15, 5, 10, 15, 20]
    current_distance_to_look = 0
    length_travelled = 0

    centerline_curve = centerline_node.GetCurve()

    # Extract diameters at specific points of interest along the neck
    for i in range(init_index, end_index):
        delta_x = np.linalg.norm(
            np.array(centerline_curve.GetPoint(i))
            - np.array(centerline_curve.GetPoint(i + 1))
        )

        # Check if it's time to calculate the diameter for the next point of interest
        if length_travelled >= points_of_interests[current_distance_to_look]:
            point_of_interest = points_of_interests[current_distance_to_look]

            # Update features for the current point of interest
            features[f"neck_diameter_minus_{point_of_interest}_ce"] = (
                calculate_diameter(ce_diameters, i)
            )
            features[f"neck_diameter_minus_{point_of_interest}_mis"] = (
                calculate_diameter(mis_diameters, i)
            )

            logging.debug(
                f"Estimated diameter (CE) at {point_of_interest}mm: "
                f"{features[f'neck_diameter_minus_{point_of_interest}_ce']}"
            )
            logging.debug(
                f"Estimated diameter (MIS) at {point_of_interest}mm: "
                f"{features[f'neck_diameter_minus_{point_of_interest}_mis']}"
            )

            if debug:
                F = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsFiducialNode", "F"
                )
                F.AddControlPoint(centerline_curve.GetPoint(i))

            current_distance_to_look += 1

        length_travelled += delta_x

        if current_distance_to_look == len(points_of_interests):
            break

    # End diameter measurements
    features["neck_diameter_end_ce"] = calculate_diameter(
        ce_diameters, end_index
    )
    features["neck_diameter_end_mis"] = calculate_diameter(
        mis_diameters, end_index
    )

    logging.debug(
        f"Diameter (CE) at the end of neck: "
        f"{features['neck_diameter_end_ce']}"
    )
    logging.debug(
        f"Diameter (MIS) at the end of neck: "
        f"{features['neck_diameter_end_mis']}"
    )

    # Mean diameters for the entire neck region
    features["neck_diameter_mean_mis"] = np.mean(
        mis_diameters[init_index:end_index]
    )
    features["neck_diameter_mean_ce"] = np.mean(
        ce_diameters[init_index:end_index]
    )

    logging.debug(
        f"Mean diameter (CE) of neck: "
        f"{features['neck_diameter_mean_ce']}"
    )
    logging.debug(
        f"Mean diameter (MIS) of neck: "
        f"{features['neck_diameter_mean_mis']}"
    )

    # Median diameters
    features["neck_diameter_median_mis"] = np.median(
        mis_diameters[init_index:end_index]
    )
    features["neck_diameter_median_ce"] = np.median(
        ce_diameters[init_index:end_index]
    )

    logging.debug(
        f"Median diameter (CE) of neck: "
        f"{features['neck_diameter_median_ce']}"
    )
    logging.debug(
        f"Median diameter (MIS) of neck: "
        f"{features['neck_diameter_median_mis']}"
    )

    # Max diameters
    features["neck_diameter_max_mis"] = np.max(
        mis_diameters[init_index:end_index]
    )
    features["neck_diameter_max_ce"] = np.max(
        ce_diameters[init_index:end_index]
    )

    logging.debug(
        f"Max diameter (CE) of neck: "
        f"{features['neck_diameter_max_ce']}"
    )
    logging.debug(
        f"Max diameter (MIS) of neck: "
        f"{features['neck_diameter_max_mis']}"
    )

    # Min diameters
    features["neck_diameter_min_mis"] = np.min(
        mis_diameters[init_index:end_index]
    )
    features["neck_diameter_min_ce"] = np.min(
        ce_diameters[init_index:end_index]
    )

    logging.debug(
        f"Min diameter (CE) of neck: "
        f"{features['neck_diameter_min_ce']}"
    )
    logging.debug(
        f"Min diameter (MIS) of neck: "
        f"{features['neck_diameter_min_mis']}"
    )

    # Calculate the total length of the neck
    length_travelled = sum(
        np.linalg.norm(
            np.array(centerline_curve.GetPoint(i))
            - np.array(centerline_curve.GetPoint(i + 1))
        )
        for i in range(init_index, end_index)
    )
    features["neck_length"] = length_travelled

    logging.debug(
        f"Estimated neck length: {features['neck_length']}"
    )

    # Suprarenal aorta diameter
    for i in range(init_index, 0, -1):
        delta_x = np.linalg.norm(
            np.array(centerline_curve.GetPoint(i))
            - np.array(centerline_curve.GetPoint(i - 1))
        )

        if length_travelled >= 15:
            features["suprarenal_aorta_diameter_ce"] = ce_diameters[i]
            features["suprarenal_aorta_diameter_mis"] = mis_diameters[i]
            break

        length_travelled += delta_x

    logging.debug(
        f"Diameter (CE) 15mm above the neck: "
        f"{features['suprarenal_aorta_diameter_ce']}"
    )
    logging.debug(
        f"Diameter (MIS) 15mm above the neck: "
        f"{features['suprarenal_aorta_diameter_mis']}"
    )

    # Spearman correlation and conical diameter classification
    spearman_corr = stats.spearmanr(
        range(init_index, end_index + 1),
        mis_diameters[init_index:end_index + 1],
    )
    features["neck_conical_diameter_mis"] = "Straight"

    if (
        spearman_corr.correlation > 0.5
        and (
            features["neck_diameter_max_mis"]
            / features["neck_diameter_minus_0_mis"]
        )
        > 1.24
    ):
        features["neck_conical_diameter_mis"] = "Tapered"

    elif (
        spearman_corr.correlation < -0.5
        and (
            features["neck_diameter_max_mis"]
            / features["neck_diameter_minus_0_mis"]
        )
        < 0.74
    ):
        features["neck_conical_diameter_mis"] = "Reversed Tapered"

    logging.debug(f"Spearman correlation: {spearman_corr.correlation}")
    logging.debug(
        f"Max to initial diameter ratio: "
        f"{features['neck_diameter_max_mis'] / features['neck_diameter_minus_0_mis']}"
    )
    logging.debug("End neck measurements")

    return features















def extract_diameters_of_centerline(centerline_node):
    logging.debug(f"Extracting diameters from {centerline_node.GetName()}")
    points = slicer.util.arrayFromMarkupsCurvePoints(centerline_node, world=True)
    indices = centerline_node.GetCurveWorld().GetPointData().GetArray("PedigreeIDs")
    n_points = len(points)

    mis_radii, ce_radii = np.zeros(n_points), np.zeros(n_points)
    radius_vals = centerline_node.GetMeasurement("Radius").GetControlPointValues()

    for i in range(n_points - 1):
        poly = compute_cross_section_polydata(i, centerline_node, slicer.util.getNode("Combined"))
        if not poly:
            continue
        mass = vtk.vtkMassProperties()
        mass.SetInputData(poly)
        mass.Update()
        ce_radii[i] = np.sqrt(mass.GetSurfaceArea() / np.pi)

        float_idx = indices.GetValue(i)
        idx_a, idx_b = int(float_idx), int(float_idx) + 1
        r_a, r_b = radius_vals.GetValue(idx_a), radius_vals.GetValue(idx_b)
        interpolated = r_a * (float_idx - idx_a) + r_b * (idx_b - float_idx)
        mis_radii[i] = interpolated

    mis_radii[-1] = radius_vals.GetValue(radius_vals.GetNumberOfValues() - 1)
    return mis_radii * 2, ce_radii * 2

def get_curve_point_to_world_transform(point_index, node):
    mtx = vtk.vtkMatrix4x4()
    node.GetCurvePointToWorldTransformAtPointIndex(point_index, mtx)
    return mtx

def compute_cross_section_polydata(point_index, centerline_node, segmentation_node):
    mtx = get_curve_point_to_world_transform(point_index, centerline_node)
    center, normal = [mtx.GetElement(i, 3) for i in range(3)], [mtx.GetElement(i, 2) for i in range(3)]

    plane = vtk.vtkPlane()
    plane.SetOrigin(center)
    plane.SetNormal(normal)

    segmentation_node.CreateClosedSurfaceRepresentation()
    surface = vtk.vtkPolyData()
    segmentation_node.GetClosedSurfaceRepresentation("aorta_iliacs", surface)

    if segmentation_node.GetParentTransformNode():
        transform = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            segmentation_node.GetParentTransformNode(), None, transform
        )
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetTransform(transform)
        tf_filter.SetInputData(surface)
        tf_filter.Update()
        surface = tf_filter.GetOutput()

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface)
    cutter.SetCutFunction(plane)
    cutter.Update()

    if not cutter.GetOutput().GetPoints() or cutter.GetOutput().GetNumberOfPoints() < 3:
        slicer.util.showStatusMessage("Could not cut segment. Is it visible in 3D view?", 3000)
        return None

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(cutter.GetOutput())
    conn.SetClosestPoint(center)
    conn.SetExtractionModeToClosestPointRegion()
    conn.Update()

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(conn.GetOutput())
    triangulator.Update()

    return triangulator.GetOutput()

def get_angle_features(
    init_index,
    end_index,
    centerline_node,
    features,
):
    """
    Compute alpha and beta angles along a centerline.
    Alpha: above renal to neck to end of collar.
    Beta: start of collar to end of collar to bifurcation.
    """

    logging.debug(f"Centerline node: {centerline_node.GetName()}")

    # Create angle nodes
    alpha_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsAngleNode", "alpha_angle")
    beta_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsAngleNode", "beta_angle")

    curve = centerline_node.GetCurve()

    def find_point_at_distance(start_idx, direction=1, target_distance=25):
        """Traverse centerline to find a point approximately target_distance mm away."""
        length = 0
        i = start_idx
        n_points = curve.GetNumberOfPoints()
        while 0 <= i < n_points - 1:
            next_idx = i + direction
            delta = np.linalg.norm(np.array(curve.GetPoint(i)) - np.array(curve.GetPoint(next_idx)))
            length += delta
            if length >= target_distance:
                return curve.GetPoint(next_idx), next_idx
            i = next_idx
        return curve.GetPoint(i), i

    # --- Alpha angle points ---
    # First point: ~25mm above init_index
    alpha_first, alpha_first_idx = find_point_at_distance(init_index, direction=-1, target_distance=25)
    # Second point: slightly before init_index
    alpha_second = curve.GetPoint(max(init_index - 10, 0))
    # Third point: end_index
    alpha_third = curve.GetPoint(end_index)

    for pt in [alpha_first, alpha_second, alpha_third]:
        alpha_node.AddControlPoint(pt)

    logging.debug(f"Alpha angle points: first {alpha_first} (idx {alpha_first_idx}), second {alpha_second}, third {alpha_third}")

    # --- Beta angle points ---
    # First point: ~25mm before end_index
    beta_first, beta_first_idx = find_point_at_distance(end_index, direction=-1, target_distance=25)
    # Second point: end_index
    beta_second = curve.GetPoint(end_index)
    # Third point: ~25mm after end_index
    beta_third, beta_third_idx = find_point_at_distance(end_index, direction=1, target_distance=25)

    for pt in [beta_first, beta_second, beta_third]:
        beta_node.AddControlPoint(pt)

    logging.debug(f"Beta angle points: first {beta_first} (idx {beta_first_idx}), second {beta_second}, third {beta_third} (idx {beta_third_idx})")

    # --- Compute angles ---
    features["angle_alpha"] = 180 - alpha_node.GetAngleDegrees()
    features["angle_beta"] = 180 - beta_node.GetAngleDegrees()

    logging.debug(f'ALPHA ANGLE: {features["angle_alpha"]}')
    logging.debug(f'BETA ANGLE: {features["angle_beta"]}')

    return features


def get_neck_points_centerline(diameters, diameters_ce, init_neck_index, centerline_node):
    """
    Identify the optimal 'neck' region along a centerline based on diameter measurements.

    Args:
        diameters (list or np.ndarray): Diameter values along the centerline.
        diameters_ce (list or np.ndarray): CE diameters (currently unused).
        init_neck_index (int): Initial guess for the neck region index.
        centerline_node: A VTK-like node with GetCurve().GetPoint(i) method.

    Returns:
        int: Index in the centerline indicating the end of the neck.
    """
    logging.debug("Starting neck region identification.")

    # Find local maximum after the initial neck guess
    init_search = np.argmax(diameters[init_neck_index:]) - 1
    init_search += init_neck_index
    logging.debug(f"Initial search peak found at index: {init_search}")

    # First threshold: 25% increase from reference diameter
    threshold_25 = diameters[init_neck_index] * 1.25
    init_index = init_neck_index  # default fallback

    for i in range(init_search, -1, -1):
        if diameters[i] < threshold_25:
            logging.debug(
                f"Index {i}: diameter {diameters[i]} is below 25% threshold ({threshold_25})"
            )
            init_index = i
            break

    # Estimate physical length from init_neck_index to init_index
    neck_length = 0
    increase_threshold = True

    # for i in range(init_neck_index + 1, init_index):
    #     pt1 = np.array(centerline_node.GetCurve().GetPoint(i - 1))
    #     pt2 = np.array(centerline_node.GetCurve().GetPoint(i))
    #     neck_length += np.linalg.norm(pt1 - pt2)

    #     if neck_length > 20:
    #         logging.debug("Neck is at least 20 mm long. No need to increase threshold.")
    #         #increase_threshold = False
            
    #         break

    # Second threshold: 30% increase, if neck was too short
    if increase_threshold:
        logging.debug("Neck too short; increasing threshold to 30%.")
        threshold_30 = diameters[init_neck_index] * 1.40                    #che era 30
        for i in range(init_search, -1, -1):
            if diameters[i] < threshold_30:
                logging.debug(
                    f"Index {i}: diameter {diameters[i]} is below 30% threshold ({threshold_30})"
                )
                init_index = i
                break

    # Gradient-based analysis
    diameters_cropped = diameters[:init_search]
    if len(diameters_cropped) < 2:
        logging.warning("Insufficient data for gradient computation.")
        return init_index

    sp = np.diff(diameters_cropped)[0]
    grad = np.gradient(diameters_cropped, sp)
    grad = scipy.ndimage.median_filter(grad, size=10)
    logging.debug(f"Smoothed gradient of diameters: {grad}")

    end_neck = init_index
    for i in range(init_index, 0, -1):
        if i < len(grad) and abs(grad[i]) < 0.5:
            end_neck = i
            logging.debug(f"Gradient threshold met at index {end_neck}.")
            break

    # Ensure estimated neck length is realistic
    estimated_length = (end_neck - init_neck_index) * 0.08  # assuming ~0.08 mm spacing
    if estimated_length < 10:
        end_neck = init_index
        logging.debug("Estimated neck length too short; reverting to init_index.")

    logging.debug(
        f"Final neck identified between indices {init_neck_index}-{end_neck} "
        f"on centerline '{centerline_node.GetName()}'."
    )

    return end_neck
    
    
def process_margin(seg_mgr: SegmentEditorManager, base_segment="aorta_iliacs", margin_segment="aorta_iliacs_margin_m_3", wall_segments=None):
    """
    Process the aorta_iliacs segment by copying, smoothing, margining, and extracting wall segments.

    Parameters
    ----------
    seg_mgr : SegmentEditorManager
        Manager for segmentation operations (holds CombinedSegmentationNode)
    volume_node : vtkMRMLScalarVolumeNode
        Reference volume node
    base_segment : str
        Name of the base segment to copy
    margin_segment : str
        Name of the new segment with margin applied
    wall_segments : dict
        Dictionary mapping reference segments to output wall segment names and optional thickness
        Example:
        {
            "located_neck": {"name": "proximal_sealing_zone_walls", "thickness": 0},
            "located_distal_sealing_right": {"name": "right_distal_sealing_zone_walls", "thickness": 1},
            "located_distal_sealing_left": {"name": "left_distal_sealing_zone_walls", "thickness": 1}
        }
    """
    # Step 1: Copy the base segment
    seg_mgr.copy_segment(margin_segment, base_segment)

    # Step 2: Apply smoothing
    seg_mgr.smoothing(margin_segment, method="MEDIAN", kernel_size=1)

    # Step 3: Apply margin
    seg_mgr.margin(margin_segment, kernel_size=-3)
    
    volume_node = seg_mgr.volume_node

    # Step 4: Extract walls if specified
    if wall_segments is not None:
        for ref_segment_name, params in wall_segments.items():
            seg_mgr.extract_walls_of_segment(
                seg_mgr.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(ref_segment_name),margin_segment,
                params['name'],
                thickness=params['thickness'])



from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing


import numpy as np
from scipy.ndimage import binary_closing, median_filter
from skimage.measure import label, regionprops
from feret.main import Calculater

def detect_bifurcation_straightened(
        lumen_vol,
        spacing=(1.0,1.0,1.0),
        noise_area_mm2=3.0,
        stability_window=3,
        z_range=None,
        debug=False):
    """
    lumen_vol: 3D numpy array (binary mask) already straightened.
    spacing: (sx, sy, sz) in mm.
    """

    def feret_diameters_and_coords(img, edge=False):
        calc = Calculater(img, edge)
        calc.calculate_minferet()
        calc.calculate_maxferet()
        return {
            "min_diameter": calc.minf,
            "min_coords": calc.minf_coords,
            "min_angle": calc.minf_angle,
            "max_diameter": calc.maxf,
            "max_coords": calc.maxf_coords,
            "max_angle": calc.maxf_angle
        }

    # Restrict to z-range if provided
    if z_range is not None:
        z_start, z_end = z_range
        vol_roi = lumen_vol[z_start:z_end, :, :]
    else:
        z_start = 0
        vol_roi = lumen_vol

    sx, sy, sz = spacing
    voxel_area = sx * sy

    # Light closing to remove pinholes
    lumen_clean = binary_closing(vol_roi, structure=np.ones((3,3,3)))

    nz = vol_roi.shape[0]
    num_components = np.zeros(nz, dtype=int)
    slice_diameters = np.zeros(nz, dtype=float)

    for z in range(nz):
        slice_mask = lumen_clean[z,:,:]

        if slice_mask.size == 0 or np.all(slice_mask == 0):
            num_components[z] = 0
            slice_diameters[z] = 0.0
            continue

        # Connected components
        lbl = label(slice_mask)
        props = regionprops(lbl)

        # Filter components by area threshold
        valid = [p for p in props if p.area * voxel_area > noise_area_mm2]
        num_components[z] = len(valid)

        if len(valid) == 0:
            slice_diameters[z] = 0
            continue

        # Use largest component for diameter estimate
        largest = max(valid, key=lambda p: p.area)
        coords = largest.coords
        pts_mm = np.column_stack([coords[:,1]*sx, coords[:,0]*sy])

        diam = feret_diameters_and_coords(slice_mask, False)
        slice_diameters[z] = diam['min_diameter']

    # ---------- Smooth component curve ----------
    smoothed_components = median_filter(num_components, size=stability_window)

    if debug:
        print("num_components:", num_components)
        print("smoothed_components:", smoothed_components)

    # ---------- Detect bifurcation ----------
    bif_slice = None
    for z in range(stability_window, nz - stability_window):
        window = smoothed_components[z-stability_window:z+stability_window+1]
        if np.any(window >= 2):  # relaxed: at least one multi-component slice
            bif_slice = z
            break

    if bif_slice is None:
        # fallback: pick slice with max components
        bif_slice = np.argmax(smoothed_components)
        if smoothed_components[bif_slice] < 2:
            raise RuntimeError("No robust bifurcation identified even after fallback.")

    # Backtrack to last single-lumen slice
    true_bif_local = None
    for z in range(bif_slice, -1, -1):
        if smoothed_components[z] <= 1:
            true_bif_local = z
            break
    if true_bif_local is None:
        true_bif_local = bif_slice

    true_bif_global = z_start + true_bif_local

    return {
        "bifurcation_index": true_bif_global,
        "bifurcation_diameter_mm": slice_diameters[true_bif_local],
        "diameter_curve_mm": slice_diameters,
        "component_curve": num_components,
    }


# ============================================================
#                   CPR GENERATION (LEFT / RIGHT)
# ============================================================

def run_cpr_for_side(side: str, sm, centerline, thresholds):
    """
    Runs CPR for one side (left or right) and transfers required segments.
    Returns a dictionary with volume, seg manager, transforms, thresholds.
    """

    cpr_proc = CPRProcessor()

    # ---- CPR ----
    volume, transform, _ = cpr_proc.do_cpr(
        centerline,
        sm.volume_node,
        f"aorta_iliac_{side}_str",
        [64, 64],
        [1, 1, 1],
    )
    
    volume_arr = np.copy(arrayFromVolume(volume))

    transform_inv = cpr_proc.invert_transform(transform)

    # ---- Create segmentation in CPR space ----
    seg_mgr = SegmentEditorManager(
        slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            f"aorta_iliac_{side}_seg"
        ),
        volume
    )

    # ---- Transfer segments into CPR ----
    needed_segments = [
        "aorta_iliacs",
        "located_renal_artery_left_ostio",
        "located_renal_artery_right_ostio",
        "located_internal_iliac_right_ostio",
        "located_internal_iliac_left_ostio",
        "iliac_vena_left",
        "iliac_vena_right",
        "iliac_artery_left",
        "iliac_artery_right",
        "aorta",
        "inferior_vena_cava",
        "located_sma_ostio",
    ]

    SegmentEditorManager.transfer_segmentation_from_seg_nodes_world_to_cpr(
        needed_segments,
        sm,
        seg_mgr,
        transform
    )

    seg_mgr.island_effect(f"iliac_artery_{side}",min_size=1000)
    
    return {
        "side": side,
        "volume": volume,
        "seg": seg_mgr,
        "transform": transform,
        "transform_inv": transform_inv,
        "thresholds": thresholds,
        "centerline": centerline,
        "original_volume_array":volume_arr
    }


# ============================================================
#                   GLOBAL FEATURES (ON RIGHT CPR)
# ============================================================

def compute_global_features(cpr,sm,thresholds,save_folder):
    """
    Compute global features (renal index, SMA length,
    suprarenal diameters, bifurcation) on the RIGHT CPR only.
    """
    seg = cpr["seg"]
    volume = cpr["volume"]

    features = {}

    # --------------------------------------------------
    # 1) Renal centroids → lowest renal artery
    # --------------------------------------------------
    c_left = seg.get_centroids_of_segment_numpy("located_renal_artery_left_ostio")
    c_right = seg.get_centroids_of_segment_numpy("located_renal_artery_right_ostio")

    if c_left[2] > c_right[2]:
        features["lowest_renal_artery"] = "Left"
    else:
        features["lowest_renal_artery"] = "Right"

    # --------------------------------------------------
    # 2) Lowest renal slice index
    # --------------------------------------------------
    arr_left = arrayFromSegmentBinaryLabelmap(seg.segmentation_node,
                                              "located_renal_artery_left_ostio",
                                              seg.volume_node)
    arr_right = arrayFromSegmentBinaryLabelmap(seg.segmentation_node,
                                               "located_renal_artery_right_ostio",
                                               seg.volume_node)
    renals = arr_left | arr_right

    lowest_idx = np.max(np.where(renals == 1)[0])
    features["lowest_renal_idx"] = int(lowest_idx)

    # --------------------------------------------------
    # 3) Suprarenal diameters
    # --------------------------------------------------
    features["suprarenal_diameters"] = seg.extract_diameters_fedez(
        "aorta_iliacs",
        max(0, lowest_idx - 14),
        [0],
        2,
        ascending=False,
        save_csv_path=os.path.join(save_folder,"suprarenal_diameters.csv")
    )

    # --------------------------------------------------
    # 4) SMA length
    # --------------------------------------------------
    sma_bounds = seg.get_bounds_of_segment("located_sma_ostio")
    sma_ras = seg.ct.ijk_to_ras([sma_bounds['z_max'],32,32])
    renals_ras = seg.ct.ijk_to_ras([lowest_idx,32,32])
    features["sma_length"] = np.linalg.norm(np.array(sma_ras[2])-np.array(renals_ras[2]))

    # --------------------------------------------------
    # 5) Lumen cleanup + bifurcation detection
    # --------------------------------------------------
    vol_arr = arrayFromVolume(seg.volume_node)
    original = vol_arr.copy()

    thresholds = cpr["thresholds"]
    aorta_thr_mu = thresholds["mu2_mean"]
    aorta_thr_var = thresholds["mu2_var"]

    seg.copy_segment("lumen", "aorta_iliacs")

    lumen = arrayFromSegmentBinaryLabelmap(seg.segmentation_node,
                                           "lumen",
                                           seg.volume_node)

    # Remove dark pixels → probable thrombus
    mask = (lumen == 1) & (original < aorta_thr_mu - 2 * np.sqrt(aorta_thr_var))
    lumen[mask] = 0

    updateSegmentBinaryLabelmapFromArray(lumen, seg.segmentation_node, "lumen", seg.volume_node)
    seg.close_holes("lumen", 3)

    lumen = arrayFromSegmentBinaryLabelmap(seg.segmentation_node, "lumen", seg.volume_node)

    # Iliac bounds
    iliac_L = seg.get_bounds_of_segment("iliac_artery_left")
    iliac_R = seg.get_bounds_of_segment("iliac_artery_right")

    zmin = min(iliac_L["z_min"], iliac_R["z_min"])
    zmax = max(iliac_L["z_max"], iliac_R["z_max"])

    bif = detect_bifurcation_straightened(
        lumen,
        spacing=(1.0, 1.0, 1.0),
        noise_area_mm2=1.0,
        stability_window=2,
        z_range=[zmin - 10, zmin + 40],
    )

    features["bifurcation_idx"] = int(bif["bifurcation_index"])
    
    cpr['seg'].island_effect("lumen", operation="KEEP_LARGEST_ISLAND", min_size=200)
    
    seg.extract_diameters_fedez("lumen",features["bifurcation_idx"]-1,[1,2,3,4,5],offset=2,ascending=False,save_csv_path=os.path.join(save_folder,"bifurcation_diameters.csv"))
    
    thrombus_predictor = ThrombusSegmentationPredictor(thresholds['mu1_mean'],thresholds['mu1_var'],64,64)

    thrombus_predictor.predict(cpr['seg'].volume_node,cpr['seg'].segmentation_node,"aorta_refined",bounding_box=[lowest_idx-1,zmin])
    
    
    cpr['seg'].copy_segment("aorta_refined", "aorta_refined_2d")
    cpr['seg'].add_segment("aorta_refined", "aorta_refined_3d")
    
    cpr['seg'].island_effect("aorta_refined", operation="KEEP_LARGEST_ISLAND", min_size=200)
    cpr['seg'].smoothing("aorta_refined", "MORPHOLOGICAL_OPENING", 3)
    cpr_proc = CPRProcessor()
    
    cpr_proc.from_cpr_to_world(cpr['seg'].segmentation_node,sm.segmentation_node,sm.volume_node,cpr['transform_inv'],os.path.join(folder_path,case_id),segments_to_export=["aorta_refined"])
    sm.add_segment("aorta_iliacs","aorta_refined")
    # neck 
    
    init_neck_found_markups_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"init_neck_centerline"
        )

    init_neck_found_markups_node.SetAndObserveTransformNodeID(
            cpr['transform'].GetID()
        )

    init_neck_ras_straight = None

    

    centroid_aorta_x, centroid_aorta_y = cpr['seg'].get_centroids_of_segment_numpy("aorta_iliacs",lowest_idx)

    init_neck_ras_straight = cpr['seg'].ct.ijk_to_ras([lowest_idx,centroid_aorta_y,centroid_aorta_x])


    init_neck_found_markups_node.AddControlPointWorld(init_neck_ras_straight)

    init_neck_found_markups_node.SetAndObserveTransformNodeID(
            None
        )


    # Cache the neck position outside the loop
    init_neck_position = np.array(init_neck_found_markups_node.GetNthControlPointPositionWorld(0))

    min_dist_to_init_neck = float("inf")
    min_dist_to_init_neck_index = None

    # Get the curve points once to avoid repeated function calls inside the loop
    curve = cpr['centerline'].GetCurve()
    num_points = curve.GetNumberOfPoints()

    for i in range(num_points):
        # Calculate the distance between current point and the neck position
        point = np.array(curve.GetPoint(i))
        dist = np.linalg.norm(point - init_neck_position)
        
        if dist < min_dist_to_init_neck:
            min_dist_to_init_neck = dist
            min_dist_to_init_neck_index = i
            
    
    mis_diameters,ce_diameters = extract_diameters_of_centerline(cpr['centerline'])

    end_index = get_neck_points_centerline(mis_diameters,ce_diameters,min_dist_to_init_neck_index,cpr['centerline'])
    cpr['seg'].extract_diameters_fedez("aorta_refined",lowest_idx,[0,5,10,15,20],offset=1,ascending=True,save_csv_path=os.path.join(save_folder,"proximal_diameters.csv"))


    features = extract_neck_statistics(mis_diameters,ce_diameters,min_dist_to_init_neck_index,end_index,cpr['centerline'],features,debug=False)

    features["max_diameter_mis"] = np.max(mis_diameters)
    features["max_diameter_ce"] = np.max(ce_diameters)
    
    max_diameter_position_ras = cpr['centerline'].GetCurve().GetPoint(np.argmax(ce_diameters))

    max_aneurism_fiducial = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"max_aneurysm_fiducial")
    max_aneurism_fiducial.AddControlPointWorld(max_diameter_position_ras)
    
    max_aneurism_fiducial.SetAndObserveTransformNodeID(cpr['transform'].GetID())
    
    max_aneurism_ras = max_aneurism_fiducial.GetNthControlPointPositionWorld(0)
    max_aneurism_ijk = cpr['seg'].ct.ras_to_ijk(max_aneurism_ras)
    
    
    
    cpr['seg'].extract_diameters_fedez("lumen",max_aneurism_ijk[2],[0],ascending=True,save_csv_path=os.path.join(save_folder,"minimum_lumen_diam.csv"))
    
    
    end_neck_markups_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"end_neck_centerline"
        )

    end_neck_markups_node.AddControlPointWorld(
            cpr['centerline'].GetCurve().GetPoint(end_index)
        )

    end_neck_markups_node.SetAndObserveTransformNodeID(
            cpr['transform'].GetID()
        )
    end_neck_ras_straight = end_neck_markups_node.GetNthControlPointPositionWorld(0)

    end_neck_ijk_straight = cpr['seg'].ct.ras_to_ijk(end_neck_ras_straight)
    lower_bound_neck = end_neck_ijk_straight[2]

    cpr['seg'].copy_segment("located_neck","aorta_refined")
    cpr['seg'].retain_segment_inside_box("located_neck",{"z_max":lower_bound_neck,"z_min":lowest_idx-1})
    cpr_proc.from_cpr_to_world(cpr['seg'].segmentation_node,sm.segmentation_node,sm.volume_node,cpr['transform_inv'],os.path.join(folder_path,case_id),segments_to_export=["located_neck"])
    
    cpr['seg'].add_segment("aorta_refined","aorta_iliacs")
    cpr['seg'].copy_segment("located_aneurysm_sac","aorta_refined")
    cpr['seg'].subtract_segment("located_aneurysm_sac","iliac_artery_left")
    cpr['seg'].retain_segment_inside_box("located_aneurysm_sac",{"z_max":zmin,"z_min":lower_bound_neck})
    cpr_proc.from_cpr_to_world(cpr['seg'].segmentation_node,sm.segmentation_node,sm.volume_node,cpr['transform_inv'],os.path.join(folder_path,case_id),segments_to_export=["located_aneurysm_sac"])
    
    init_neck_found_markups_node.SetAndObserveTransformNodeID(
            cpr['transform'].GetID()
        )

    end_neck_markups_node.SetAndObserveTransformNodeID(
            cpr['transform'].GetID()
        )
    
    neck_length = np.linalg.norm(
        np.array(init_neck_found_markups_node.GetNthControlPointPositionWorld(0))-np.array(end_neck_markups_node.GetNthControlPointPositionWorld(0))
        )
    
    features['proximal_neck_length'] = neck_length
    
    features = get_angle_features(
    min_dist_to_init_neck_index,
    end_index,
    cpr['centerline'],
    features)
    
    return features


def compute_iliac_features(cpr,sm, global_features,features,save_folder, side: str):
    """
    Compute iliac-specific features for either left or right CPR,
    depending on `side`. 
    
    side ∈ {"left", "right"}
    """

    assert side in ["left", "right"], "side must be 'left' or 'right'"

    seg   = cpr["seg"]
    thr   = cpr["thresholds"]
    lowest_idx = global_features["lowest_renal_idx"]


    # ----------------------------------------------------------------------
    # 1. Select segments based on SIDE
    # ----------------------------------------------------------------------
    if side == "left":
        iliac_ostio_name = "located_internal_iliac_left_ostio"
        refined_name     = "located_common_iliac_left"
        opposite_iliac_name = "iliac_artery_right"
    else:
        iliac_ostio_name = "located_internal_iliac_right_ostio"
        refined_name     = "located_common_iliac_right"
        opposite_iliac_name = "iliac_artery_left"

    # ----------------------------------------------------------------------
    # 2. Bounds
    # ----------------------------------------------------------------------
    iliac_bounds = seg.get_bounds_of_segment(iliac_ostio_name,15)
    aorta_bounds = seg.get_bounds_of_segment("aorta")

    common_bounds = [
        aorta_bounds["z_max"],   # top of common iliac
        iliac_bounds["z_min"],   # bottom (threshold start)
    ]

    # Lengths
    # common 179,233
    lower_common_ras = seg.ct.ijk_to_ras([common_bounds[1],32,32])
    upper_common_ras = seg.ct.ijk_to_ras([common_bounds[0],32,32])
    renals_ras = seg.ct.ijk_to_ras([lowest_idx,32,32])
    features[f"{side}_iliac_length"]  = np.linalg.norm(np.array(renals_ras)-np.array(lower_common_ras))
    features[f"{side}_common_length"] = np.linalg.norm(np.array(upper_common_ras)-np.array(lower_common_ras))

    # ----------------------------------------------------------------------
    # 3. Thrombus estimation (side-specific)
    # ----------------------------------------------------------------------
    predictor = ThrombusSegmentationPredictor(
        thr["mu1_mean"], thr["mu1_var"], 72, 72
    )

    vena_arr = arrayFromSegmentBinaryLabelmap(seg.segmentation_node,f"iliac_vena_right",seg.volume_node)
    if "iliac_id" in cpr:
        seg_other_iliac_artery = seg.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(cpr['iliac_id'])
        if seg_other_iliac_artery != "":

            vena_arr = np.logical_or(vena_arr,arrayFromSegmentBinaryLabelmap(seg.segmentation_node,seg_other_iliac_artery,seg.volume_node))
    vol_arr = arrayFromVolume(seg.volume_node)
    
    prev_vol_arr = vol_arr.copy()
    
    
    vol_arr[np.where(vena_arr == 1)] = -1024

    vena_arr = arrayFromSegmentBinaryLabelmap(seg.segmentation_node,f"iliac_vena_left",seg.volume_node)    
    
    vol_arr[np.where(vena_arr == 1)] = -1024
    
    updateVolumeFromArray(seg.volume_node, vol_arr)  
    try:
        predictor.predict(
            seg.volume_node,
            seg.segmentation_node,
            refined_name,
            common_bounds
        )

        # cleanup
        seg.copy_segment(refined_name, refined_name + "_2d")
        seg.add_segment(refined_name, refined_name + "_3d")
        seg.subtract_segment(refined_name,opposite_iliac_name)
        seg.smoothing(refined_name, "MORPHOLOGICAL_OPENING", 3)
        seg.island_effect(refined_name, operation="KEEP_LARGEST_ISLAND", min_size=200)
        seg.close_holes(refined_name, 3)

        updateVolumeFromArray(seg.volume_node, prev_vol_arr)  
    except AssertionError:
        refined_name = f"iliac_artery_{side}"
    
    
    
    refined_seg_arr = arrayFromSegmentBinaryLabelmap(seg.segmentation_node,refined_name,seg.volume_node)
    distal_arr = np.zeros_like(refined_seg_arr)
    distal_arr[common_bounds[1]-30:common_bounds[1]+1] = refined_seg_arr[common_bounds[1]-30:common_bounds[1]+1]
    
    seg.copy_segment(f"located_distal_sealing_{side}", refined_name)
    
    updateSegmentBinaryLabelmapFromArray(distal_arr, seg.segmentation_node, f"located_distal_sealing_{side}", seg.volume_node)    
    
    cpr_proc = CPRProcessor()
    
    cpr_proc.from_cpr_to_world(cpr['seg'].segmentation_node,sm.segmentation_node,sm.volume_node,cpr['transform_inv'],os.path.join(folder_path,case_id),segments_to_export=[f"located_distal_sealing_{side}"])
    
    cpr_proc.from_cpr_to_world(cpr['seg'].segmentation_node,sm.segmentation_node,sm.volume_node,cpr['transform_inv'],os.path.join(folder_path,case_id),segments_to_export=[refined_name])
    
    seg.extract_diameters_fedez(refined_name,common_bounds[1]-1,[0,10,20,30],ascending=False,save_csv_path=os.path.join(save_folder,f"common_iliac_{side}_sealing_zone_diameters.csv"))
    
    n_slices = common_bounds[1]-common_bounds[0]+10
    
    seg.extract_diameters_fedez(refined_name,common_bounds[0]+10,[i for i in range(n_slices)],ascending=True,save_csv_path=os.path.join(save_folder,f"common_iliac_{side}_all_diameters.csv"))
    
    pprint(features)
    
    
    
    point_ras = cpr['seg'].ct.ijk_to_ras([common_bounds[0],32,32])
    
    
    init_tortuosity_fiducial = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"init_tortuosity_{side}"
        )
    
    init_tortuosity_fiducial.AddControlPointWorld(point_ras)
    
    
    init_tortuosity_fiducial.SetAndObserveTransformNodeID(
            cpr['transform_inv'].GetID()
        )

    
    init_point_to_look = init_tortuosity_fiducial.GetNthControlPointPositionWorld(0)
    init_point_to_look = np.array(init_point_to_look)
    d = float("inf")
    init_p = None
    idx_p = 0
    for i in range(cpr['centerline'].GetCurve().GetNumberOfPoints()):
        point = cpr['centerline'].GetCurve().GetPoint(i)
        dist = np.linalg.norm(np.array(point)-init_point_to_look)
        if dist < d:
            d = dist
            init_p = point
            idx_p = i
    
    init_p = np.array(init_p)
    end = np.array(cpr['centerline'].GetCurve().GetPoint(cpr['centerline'].GetCurve().GetNumberOfPoints()-1))
    # total_length / np.linalg.norm(np.array(curve.GetPoint(start_idx)) - np.array(end_point))
    
    total_l = 0
    
    for i in range(idx_p+1,cpr['centerline'].GetCurve().GetNumberOfPoints()):
        p_prev = np.array(cpr['centerline'].GetCurve().GetPoint(i-1))
        p = np.array(cpr['centerline'].GetCurve().GetPoint(i))
        
        total_l += np.linalg.norm(p-p_prev)
        
    features[f"tortuosity_{side}"] = total_l / np.linalg.norm(end-init_p)
    
    
    cpr['seg'].smoothing(f"iliac_artery_{side}","GAUSSIAN",1)
    
    iliac_artery_bounds = cpr['seg'].get_bounds_of_segment(f"iliac_artery_{side}")
    
    n_slices = iliac_artery_bounds["z_max"] - 25 - common_bounds[1] + 10
    seg.extract_diameters_fedez(f"iliac_artery_{side}",common_bounds[1]+10,[i for i in range(n_slices)],ascending=True,save_csv_path=os.path.join(save_folder,f"external_iliac_{side}_diameters.csv"),crop_spatial=20)
    
    
    
    return features

def save_radiomics_csv(case_id, vol_path, segmentations, root):
    # Create radiomics folder if it does not exist
    radiomics_folder = os.path.join(root, case_id, "radiomics")
    os.makedirs(radiomics_folder, exist_ok=True)

    # Initialize extractor

    #extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()     #aggiunto
    extractor.settings['label'] = 1   #aggiunto
    extractor.settings['resampledPixelSpacing']= None
    extractor.settings['enableAllImageTypes']= True
    #print(dir(extractor))
    
    img = sitk.ReadImage(vol_path)
    img = sitk.Cast(img, sitk.sitkFloat32)
    ct_nrrd=sitk.Image(img)
    

    # Iterate over segmentations
    for s in segmentations:
        seg_path = os.path.join(root, case_id, "Predictions_PRE", f"{s}.nii.gz")
        if not os.path.isfile(seg_path):
            print(f"Warning: segmentation file not found: {seg_path}")
            continue
        # Execute feature extraction (voxelBased=True returns SimpleITK.Image for some features)
        mask = sitk.ReadImage(seg_path)

        # Force matching geometry
        mask.SetOrigin(img.GetOrigin())
        mask.SetSpacing(img.GetSpacing())
        mask.SetDirection(img.GetDirection())

        # Resample to match size
        mask = sitk.Resample(mask, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        
        # Assicurati sia un intero binario
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        mask = sitk.BinaryThreshold(mask, 1, 9999, 1, 0)
        mask_nrrd=sitk.Image(mask)




        result = extractor.execute(ct_nrrd, mask_nrrd, voxelBased=False)
        # Convert results dictionary to dataframe
        # Only keep numeric features
        data = {k: v for k, v in result.items()}
        df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
        df.index.name = 'Feature'

        # Save CSV
        csv_path = os.path.join(radiomics_folder, f"{s}_radiomics.csv")
        df.to_csv(csv_path)
        print(f"Saved radiomics features for {s} to {csv_path}")


def save_lumen_tree(sm,vol_arr,th=500):
    branches = ['located_celiac','located_sma','located_renal_artery_left','located_renal_artery_right','located_internal_iliac_left','located_internal_iliac_right']
    sm.copy_segment("lumen_tree","lumen")
    for b in branches:
        sm.add_segment("lumen_tree",b)

    sm.copy_segment("lumen_cfd","lumen_tree")
    sm.smoothing("lumen_cfd", "MORPHOLOGICAL_OPENING", 3)
    sm.close_holes("lumen_cfd",kernel_size=3)
    sm.smoothing("lumen_cfd", method="GAUSSIAN",kernel_size=1)

    lumen_arr = arrayFromSegmentBinaryLabelmap(sm.segmentation_node,"lumen_cfd")
    lumen_arr[np.where((lumen_arr == 1) & (vol_arr > 500))] = 0
    updateSegmentBinaryLabelmapFromArray(lumen_arr, sm.segmentation_node, "lumen_cfd", sm.volume_node) 

    # sm.threshold("lumen_cfd",
    #             th,
    #             get_max_hu_from_volume(sm.volume_node))


    
def main(case_id,folder_path,_setting):

    logging.info(f"Beginning features extraction of patient {case_id}")
    logging.info(f"Patient saved in {os.path.join(folder_path,case_id)}")

    start_time = datetime.datetime.now()

    segmentations = [   
            "kidney_right",
            "kidney_left",
            "aorta",
            "inferior_vena_cava",
            "portal_vein_and_splenic_vein",
            "vertebrae_L5",
            "vertebrae_L4",
            "vertebrae_L3",
            "vertebrae_L2",
            "vertebrae_L1",
            "vertebrae_T12",
            "vertebrae_T11",
            "vertebrae_T10",
            "vertebrae_T9",
            "vertebrae_T8",
            "vertebrae_T7",
            "vertebrae_T6",
            "vertebrae_T5",
            "vertebrae_T4",
            "vertebrae_T3",
            "vertebrae_T2",
            "vertebrae_T1",
            "vertebrae_C7",
            "vertebrae_C6",
            "vertebrae_C5",
            "vertebrae_C4",
            "vertebrae_C3",
            "vertebrae_C2",
            "vertebrae_C1",
            "vertebrae_S1",
            "iliac_artery_left",
            "iliac_artery_right",
            "femur_left",
            "femur_right",
            "hip_left",
            "hip_right",
            "sacrum",
            "gluteus_maximus_left",
            "gluteus_medius_left",
            "gluteus_minimus_left",
            "autochthon_left",
            "iliopsoas_left",
            "gluteus_maximus_right",
            "gluteus_medius_right",
            "gluteus_minimus_right",
            "autochthon_right",
            "iliopsoas_right",
            "iliac_vena_left",
            "iliac_vena_right"
        ]


    features = {}
    features["ct_name"] = f"CT_{case_id}_{_setting}"

    load_data(folder_path, case_id, _setting, segmentations)

    sm = SegmentEditorManager(
            slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", "Combined"
            ), 
            getNode(f"CT_{case_id}_{_setting}_001_0002")
        )
    
    original_vol_arr = np.copy(arrayFromVolume(sm.volume_node))

    sm.merge_segmentations(segmentations)

    sm.close_holes("vertebrae",kernel_size=10)

    estimator = IntensityThresholdEstimator(num_components=2, contamination=0.05)

    aorta_thresholds = estimator.find_threshold(arrayFromVolume(sm.volume_node),
                                                [
            arrayFromSegmentBinaryLabelmap(sm.segmentation_node,"aorta"),
            arrayFromSegmentBinaryLabelmap(sm.segmentation_node,"inferior_vena_cava"),
                                                    ],
                                                is_aorta=True)

    iliac_left_thresholds = estimator.find_threshold(arrayFromVolume(sm.volume_node),
                                                [
                                                    arrayFromSegmentBinaryLabelmap(sm.segmentation_node,"iliac_artery_left"),
                                                    arrayFromSegmentBinaryLabelmap(sm.segmentation_node,"iliac_vena_left"),
                                                    ],
                                                is_aorta=False)

    iliac_right_thresholds = estimator.find_threshold(arrayFromVolume(sm.volume_node),
                                                [
                                                    arrayFromSegmentBinaryLabelmap(sm.segmentation_node,"iliac_artery_right"),
                                                    arrayFromSegmentBinaryLabelmap(sm.segmentation_node,"iliac_vena_right"),
                                                    ],
                                                is_aorta=False)

    aorta_bounds = sm.get_bounds_of_segment("aorta")

    sm.mask_volume(segmentations+['vertebrae'],mask_value=-1000)

    sm.copy_segment("aorta_iliacs","aorta")
    sm.add_segment("aorta_iliacs","iliac_artery_left")
    sm.add_segment("aorta_iliacs","iliac_artery_right")
    import pdb
    print("verifica che iliac_artery_left/right siano fatte bene...........! unifica i popoli!")
    print("controlla aorta iliacs e aorta (non deve esserci segmentazione dopo biforcazione!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)")
    pdb.set_trace()

    kidney_bounds = {"left": sm.get_bounds_of_segment("kidney_left"),
                    "right": sm.get_bounds_of_segment("kidney_right")
                    }

    iliacs_bounds = {"left": sm.get_bounds_of_segment("iliac_artery_left"),
                    "right": sm.get_bounds_of_segment("iliac_artery_right")
                    }

    sm.copy_segment("aorta_iliacs_margin_30","aorta_iliacs")
    sm.copy_segment("aorta_iliacs_margin_10","aorta_iliacs")
    sm.copy_segment("aorta_iliacs_margin_3","aorta_iliacs")

    sm.margin("aorta_iliacs_margin_30",kernel_size=30)
    sm.margin("aorta_iliacs_margin_10",kernel_size=15)
    sm.margin("aorta_iliacs_margin_3",kernel_size=3)


    sm.threshold("thresholded_aorta_iliacs",aorta_thresholds["threshold"],get_max_hu_from_volume(sm.volume_node))
    #"aorta_iliacs_margin_30"
    thresholded_aorta_iliacs_arr = arrayFromSegmentBinaryLabelmap(sm.segmentation_node, "thresholded_aorta_iliacs", sm.volume_node)
    aorta_iliacs_margin_30_arr = arrayFromSegmentBinaryLabelmap(sm.segmentation_node, "aorta_iliacs_margin_30", sm.volume_node)
    new_s = np.zeros_like(thresholded_aorta_iliacs_arr)
    new_s[np.where((thresholded_aorta_iliacs_arr==1)&(aorta_iliacs_margin_30_arr==1))] = 1
    updateSegmentBinaryLabelmapFromArray(new_s, sm.segmentation_node, "thresholded_aorta_iliacs", sm.volume_node)


    sm.island_effect("thresholded_aorta_iliacs",min_size=5000)
    sm.copy_segment("thresholded_aorta_iliacs_localization","thresholded_aorta_iliacs")
    sm.subtract_segment("thresholded_aorta_iliacs","aorta_iliacs")
    sm.subtract_segment("thresholded_aorta_iliacs_localization","aorta_iliacs_margin_3")

    sm.copy_segment("iliac_artery_left_top_half","iliac_artery_left")
    sm.copy_segment("iliac_artery_right_top_half","iliac_artery_right")

    sm.retain_segment_inside_box("iliac_artery_left_top_half",{
                                "z_min": iliacs_bounds["left"]["z_min"],
                                "z_max": int((iliacs_bounds["left"]["z_min"]+iliacs_bounds["left"]["z_max"])/2),
                                "y_min": iliacs_bounds["left"]["y_min"],
                                "y_max": iliacs_bounds["left"]["y_max"],
                                "x_min": iliacs_bounds["left"]["x_min"],
                                "x_max": iliacs_bounds["left"]["x_max"]
                                })  

    sm.retain_segment_inside_box("iliac_artery_right_top_half",{
                                "z_min": iliacs_bounds["right"]["z_min"],
                                "z_max": int((iliacs_bounds["right"]["z_min"]+iliacs_bounds["right"]["z_max"])/2),
                                "y_min": iliacs_bounds["right"]["y_min"],
                                "y_max": iliacs_bounds["right"]["y_max"],
                                "x_min": iliacs_bounds["right"]["x_min"],
                                "x_max": iliacs_bounds["right"]["x_max"]
                                })  

    sm._set_only_segments_visible([
                "thresholded_aorta_iliacs",
                "kidney_left",
                "kidney_right",
                "iliac_artery_left_top_half",
                "iliac_artery_right_top_half",
                # "iliac_artery_left",
                # "iliac_artery_right",
            ])
    
    print("controlla, e eventualmente aggiungi se manca segmento o segmento troppo piccolo...................")
    import pdb;pdb.set_trace()
    sm.island_effect("thresholded_aorta_iliacs",min_size=100,operation="SPLIT_ISLANDS_TO_SEGMENTS")
    
    islands_names_loc = ["thresholded_aorta_iliacs"]

    i = 2 
    while sm.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(f"thresholded_aorta_iliacs_{i}") != "":
        islands_names_loc.append(f"thresholded_aorta_iliacs_{i}")
        i += 1
    
    
    infos_secondary_vessels = [["celiac"],["sma"],["renal_artery_right"],["renal_artery_left"],["internal_iliac_right"],["internal_iliac_left"]]
    view=slicer.app.layoutManager().threeDWidget(0).threeDView()

    selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
    selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
    interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
    placeModePersistence = 1
    interactionNode.SetPlaceModePersistence(placeModePersistence)
    # mode 1 is Place, can also be accessed via slicer.vtkMRMLInteractionNode().Place
    interactionNode.SetCurrentInteractionMode(1)
    pprint(infos_secondary_vessels)
    sm.segmentation_node.CreateClosedSurfaceRepresentation()
    
    print("Piazza marker")
    import pdb;pdb.set_trace()
    
    interactionNode.SwitchToViewTransformMode()
    interactionNode.SetPlaceModePersistence(0)

    node = slicer.util.getNode("F")
    
    node.SetName("anatomical_markers")
    
    nPoints = node.GetNumberOfControlPoints()
    
    for i in range(nPoints):
        ras = node.GetNthControlPointPosition(i)
        infos_secondary_vessels[i].append(ras)
        infos_secondary_vessels[i].append(sm.getSegmentationAtRASNeighborhood(ras, islands_names_loc, neighborhood_radius=10))


    for label_key,ras,res in infos_secondary_vessels:
        sm.set_segment_name(res[0],f"located_{label_key}")

    
    ostio_segments = {
        "sma": ("located_sma_ostio","located_sma"),
        "renal_artery_right": ("located_renal_artery_right_ostio","located_renal_artery_right"),
        "renal_artery_left": ("located_renal_artery_left_ostio","located_renal_artery_left"),
        "celiac": ("located_celiac_ostio","located_celiac"),
        "internal_iliac_right": ("located_internal_iliac_right_ostio","located_internal_iliac_right"),
        "internal_iliac_left": ("located_internal_iliac_left_ostio","located_internal_iliac_left")
    }

    pprint(infos_secondary_vessels)
    for label_key, ras, res in infos_secondary_vessels:
        if label_key not in ostio_segments:
            continue
        sm.copy_segment(ostio_segments[label_key][0],ostio_segments[label_key][1])
        sm.subtract_segment(ostio_segments[label_key][0], "thresholded_aorta_iliacs_localization")
    import pdb;pdb.set_trace()

    aorta_slice_height = int(
        max(kidney_bounds["left"]["z_max"], kidney_bounds["right"]["z_max"])
        + 15 // sm.get_volume_spacing()[0]
    )

    # Compute RAS centroid for aorta
    upper_centerline_point_RAS = sm.get_segment_centroid_ras("aorta", slice_height=aorta_slice_height)

    # Iliac offset
    iliac_offset = 10 // sm.get_volume_spacing()[2]

    iliac_left_z = iliacs_bounds["left"]["z_min"] + iliac_offset
    iliac_right_z = iliacs_bounds["right"]["z_min"] + iliac_offset

    iliac_left_centerline_point_RAS = sm.get_segment_centroid_ras("iliac_artery_left", slice_height=iliac_left_z)
    iliac_right_centerline_point_RAS = sm.get_segment_centroid_ras("iliac_artery_right", slice_height=iliac_right_z)

    # Internal iliac - LEFT
    iil_seg_id = sm.get_segment_id("located_internal_iliac_left")
    IIL_seg = arrayFromSegmentBinaryLabelmap(sm.segmentation_node, iil_seg_id, sm.volume_node)
    IIL_centroid_numpy = np.mean(np.argwhere(IIL_seg == 1), axis=0)
    fetta_media_h_ill = IIL_centroid_numpy[0]
    IIL_centroid_RAS = sm.get_segment_centroid_ras("located_internal_iliac_left", slice_height=fetta_media_h_ill)

    # Internal iliac - RIGHT
    iir_seg_id = sm.get_segment_id("located_internal_iliac_right")
    IIR_seg = arrayFromSegmentBinaryLabelmap(sm.segmentation_node, iir_seg_id, sm.volume_node)
    IIR_centroid_numpy = np.mean(np.argwhere(IIR_seg == 1), axis=0)
    fetta_media_h_iir = IIR_centroid_numpy[0]
    IIR_centroid_RAS = sm.get_segment_centroid_ras("located_internal_iliac_right", slice_height=fetta_media_h_iir)

    # Assemble centerline endpoints
    centerline_endpoints = [
        upper_centerline_point_RAS,
        iliac_left_centerline_point_RAS,
        IIL_centroid_RAS,
        iliac_right_centerline_point_RAS,
        IIR_centroid_RAS,
    ]

    # Add internal iliacs to aorta_iliacs segment
    for segment_name in ["located_internal_iliac_right", "located_internal_iliac_left"]:
        sm.add_segment("aorta_iliacs", segment_name)

    # Fill holes
    sm.close_holes("aorta_iliacs", kernel_size=3)

    # Extract centerline

    left_iliac_cl = sm.extract_centerline("aorta_iliacs",case_id,_setting,[upper_centerline_point_RAS,iliac_left_centerline_point_RAS],"aorta_iliac_left",0.8)
    
    right_iliac_cl = sm.extract_centerline("aorta_iliacs",case_id,_setting,[upper_centerline_point_RAS,iliac_right_centerline_point_RAS],"aorta_iliac_right",0.8)

    #tmp = np.copy(arrayFromVolume(sm.volume_node))
    
    #updateVolumeFromArray(sm.volume_node, original_vol_arr)  
    
    cpr_left = run_cpr_for_side("left",sm,left_iliac_cl,iliac_left_thresholds)

    cpr_right = run_cpr_for_side("right",sm,right_iliac_cl,iliac_right_thresholds)
    
    #updateVolumeFromArray(sm.volume_node, tmp)  
    
    global_features = compute_global_features(cpr_right,sm,aorta_thresholds,os.path.join(folder_path,case_id,"data"))
    
    sm.subdivide_aneurysm_sac(aorta_thresholds['threshold'],aorta_thresholds['mu2_mean'] + (2 * np.sqrt(aorta_thresholds["mu2_var"])))
    features = global_features.copy()
    
    compute_iliac_features(cpr_right,sm,global_features,features,os.path.join(folder_path,case_id,"data"),"right")
    cpr_left['iliac_id'] = sm.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName("located_common_iliac_right")
    SegmentEditorManager.transfer_segmentation_from_seg_nodes_world_to_cpr(
        [sm.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName("located_common_iliac_right")],
        sm,
        cpr_left['seg'],
        cpr_left['transform']
    )

    compute_iliac_features(cpr_left,sm,global_features,features,os.path.join(folder_path,case_id,"data"),"left")    

    wall_segments_dict = {
        "located_neck": {
            "name": "proximal_sealing_zone_walls",
            "thickness":2},
        "located_distal_sealing_right": {
            "name": "right_distal_sealing_zone_walls", 
            "thickness": 2},
        "located_distal_sealing_left": {
            "name": 
            "left_distal_sealing_zone_walls", 
            "thickness": 2},
    }
    
    process_margin(sm, base_segment="aorta_iliacs", margin_segment="aorta_iliacs_margin_m_3", wall_segments=wall_segments_dict)
    
    sm.generate_walls_segmentations(aorta_thresholds,iliac_left_thresholds,iliac_right_thresholds,features)
        
    features['time'] = datetime.datetime.now() - start_time
    
    save_features(features,folder_path,case_id,None)
    
    updateVolumeFromArray(sm.volume_node,original_vol_arr)
    
    updateVolumeFromArray(cpr_left['volume'],cpr_left["original_volume_array"])
    
    updateVolumeFromArray(cpr_right['volume'],cpr_right["original_volume_array"])



    sm.copy_segment("lumen", "aorta_iliacs")

    lumen = arrayFromSegmentBinaryLabelmap(sm.segmentation_node,
                                           "lumen",
                                           sm.volume_node)

    # Remove dark pixels → probable thrombus
    mask = (lumen == 1) & (original_vol_arr < aorta_thresholds['mu2_mean'] - 2 * np.sqrt(aorta_thresholds['mu2_var']))
    lumen[mask] = 0

    updateSegmentBinaryLabelmapFromArray(lumen, sm.segmentation_node, "lumen", sm.volume_node)
    sm.island_effect("lumen", operation="KEEP_LARGEST_ISLAND", min_size=200)
    
    sm.close_holes("lumen", 3)

   
    
    save_lumen_tree(sm,original_vol_arr)

    print("Finished")
    
    
    to_save_dir = os.path.join(folder_path,case_id,"Scene")
    os.makedirs(to_save_dir,exist_ok=True)
    if slicer.app.applicationLogic().SaveSceneToSlicerDataBundleDirectory(to_save_dir, None):
        logging.debug("Scene saved correctly")
        print("Scene saved correctly")
    else:
        logging.error("error in saving the scene")
        print("ERROR: SCENE NOT SAVED")
    
    segmentNames = [
        "located_renal_artery_left",
        "located_renal_artery_right",
        "located_aneurysm_sac_thrombus",
        "located_aneurysm_sac_lumen",
        "lumen",
        "aorta_iliacs",
        "located_common_iliac_right",
        "located_common_iliac_left",
        "located_celiac",
        "located_sma",
        "located_internal_iliac_right",
        "located_internal_iliac_left",
        "located_neck",
        "located_distal_sealing_left",
        "located_distal_sealing_right"
    ]
    
    for segmentName in segmentNames:
        
        sm.export_segment_as_vol(segmentName,os.path.join(folder_path,case_id,"Predictions_PRE"))



    vol_path = os.path.join(folder_path, case_id, f"CT_{case_id}_PRE_001_0002.nii.gz")

    radiomics_segmentations = [
        "located_aneurysm_sac_thrombus",
        "located_aneurysm_sac_lumen",
        "located_neck",
        "located_distal_sealing_left",
        "located_distal_sealing_right"
    ]

    save_radiomics_csv(case_id, vol_path, radiomics_segmentations, folder_path)

    
    logging.debug("Ended")
    
    
    
    import pdb;pdb.set_trace()
    
    
        
folder_path = r"C:\pzapps\PatientsToPlan" # CAMBIA QUA
_setting = "PRE"

case_id = sys.argv[1]  # alphanumeric digit
logging.basicConfig(
            force=True,
            filename=os.path.join(folder_path, case_id, "app.log"),
            encoding="utf-8",
            filemode="w",
            format="{asctime} - {levelname} - {funcName} - {lineno} - {message}",
            style="{",
            datefmt="%d-%m-%Y %H:%M",
        )



main(case_id,folder_path,_setting)