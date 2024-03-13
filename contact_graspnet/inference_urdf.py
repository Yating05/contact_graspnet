import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import numpy as np
import time
import glob
import cv2
import open3d as o3d
import re

import tensorflow.compat.v1 as tf
from urdf_models import models_data
import h5py

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

def inference(global_config, checkpoint_dir, input_paths, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    os.makedirs('results', exist_ok=True)

    # Process example test scenes
    grasp = {}
    for p in glob.glob(input_paths):
        print('Loading ', p)

        pc_segments = {}
        # segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)


        models = models_data.model_lib()
        pathlist = models.model_path_list
        namelist = models.model_name_list
        for i in range(len(namelist)):

            random_index = i
            print('asset_object_name:',namelist[random_index])
            obj_name  = namelist[random_index]
            # if obj_name !='medium_clamp':
            #     continue
            path = pathlist[random_index]
            asset_path = path[:-10] + 'collision.obj'
        
            mesh = o3d.io.read_triangle_mesh(asset_path)

        # Sample points uniformly from the surface
            pc_full =np.asarray( mesh.sample_points_uniformly(number_of_points=10000).points)
    
        
        # if segmap is None and (local_regions or filter_grasps):
        #     raise ValueError('Need segmentation map to extract local regions or filter grasps')

        # if pc_full is None:
        #     print('Converting depth to point cloud(s)...')
        #     pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
        #                                                                             skip_border_objects=skip_border_objects, z_range=z_range)

            print('Generating Grasps...')
            pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

        # Save results
            # np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                #   pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)
            

        #select grasp
            if scores[-1].size == 0:
                print('No grasps found')
                grasp[obj_name] = 0
                continue
            max_score_idx = np.argmax(scores[-1])
            max_score_grasp ={-1: pred_grasps_cam[-1][max_score_idx,:,:].reshape(1,4,4)}
            grasp[obj_name] = max_score_grasp[-1]
            # pred_grasps_cam[-1]= np.identity(4).reshape(1,4,4)
        # Visualize results          
        # show_image(rgb, segmap)
        # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=None)
            visualize_grasps(pc_full, max_score_grasp, scores, plot_opencv_cam=True, pc_colors=None)
            print('Max score grasp:', max_score_grasp)
            print("Done")
            # break
        ['blue_plate','knife','plate_holder','gelatin_box','phillips_screwdriver','soap_dish',\
        'plastic_orange','blue_marker','doraemon_plate','medium_clamp','remote_controller_1','cleanser'\
        'two_color_hammer','clear_box_1','doraemon_bowl','suger_3','stapler_1','book_holder_3',]


        # store_h5_dict(f'./results/grasp_YCB.h5',grasp)
    if not glob.glob(input_paths):
        print('No files found: ', input_paths)

def store_h5_dict(path, data_dict):
    hf = h5py.File(path, 'w')
    for k, v in data_dict.items():
        hf.create_dataset(k, data=v)
    hf.close()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/0.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id, 
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

