import numpy as np
import argparse,os,copy 
from tqdm import tqdm 
from pysdf import SDF   
# import open3d as o3d
  
# def get_mesh_from_plys(data_path: str,unicolor=False) -> o3d.geometry.TriangleMesh:
#     mesh = o3d.io.read_triangle_mesh(data_path)
#     mesh.compute_vertex_normals()
#     if unicolor:
#         mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
#     return mesh

 

def collision_detection_sdf(obj_vs, obj_faces, 
                            query_vs,
                            col_tolerance=0.005,
                            touch_thresh=0.005 ):
    """
    a function to compute collision flag and distances 
    args:
        obj_vs (N,3)
        obj_faces (M,3)
        query_vs (K,3)
    return:
        collision_flag (K,): binary collision flag. 1 for collision 0 otherwise
        non_collision_flag (K,): binary non collision flag. 1 for non collision 0 otherwise
        collision_dist_sum (float): sum of collision distances
        collision_dist_avg (float): mean of collision distances
    """
    f = SDF(obj_vs, obj_faces) 
    sdf_multi_point = f(query_vs)
    
    ######### need to eliminate the leaking sdf values #########
    sdf_multi_point[sdf_multi_point>=0.05]=0
    #sdf_multi_point[sdf_multi_point>=0]=1
    #sdf_multi_point[sdf_multi_point<0]=0
    
    
    
    ########## collision flag computation #########
    collision_flag_tolerance = copy.copy(sdf_multi_point) 
    collision_flag_tolerance[collision_flag_tolerance >= col_tolerance] = 1
    collision_flag_tolerance[collision_flag_tolerance < col_tolerance] = 0
    #print(collision_flag_tolerance)
    collision_flag = copy.copy(sdf_multi_point) 
    collision_flag[collision_flag >= 0] = 1
    collision_flag[collision_flag < 0] = 0
     
    n_colliding_vs = np.sum(collision_flag)
    
    ######### collision dist. computation ##########
    collision_dist_sum = np.sum(np.abs(sdf_multi_point*collision_flag))
    collision_dist_avg = np.average(np.abs(sdf_multi_point*collision_flag))
     
    ########## non touching flag computation ######
    non_touching = _non_touching_judgement_from_sdf(sdf_values=sdf_multi_point,
                                                    touch_thresh=touch_thresh)
    
    return collision_flag,collision_flag_tolerance, collision_dist_sum, \
           collision_dist_avg, non_touching,n_colliding_vs


def check_touch_presence(obj_vs, obj_faces,query_vs,touch_thresh=0.005):
  """
    a function to compute collision flag and distances 
    args:
        obj_vs (N,3)
        obj_faces (M,3)
        query_vs (K,3)
    return:
        touching judgement (int)
    """
  f = SDF(obj_vs, obj_faces) 
  sdf_values = f(query_vs)
  
  ######### need to eliminate the leaking sdf values #########
  sdf_values[sdf_values>=0.05]=0
  sdf_values=np.abs(sdf_values)
  
  
  non_touch_flag= copy.copy(sdf_values)
  non_touch_flag[non_touch_flag>touch_thresh]=1
  non_touch_flag[non_touch_flag<=touch_thresh]=0
  touch_flag = copy.copy(non_touch_flag)
  touch_flag-=1
  touch_flag*=-1  
  if np.sum(touch_flag)>0:
    return 1
  else:
    return 0


def _non_touching_judgement_from_sdf(sdf_values,touch_thresh):
  """
  args:
    sdf_values (N,): sdf values
    touch_thresh (float): thresholding value to judge the presence of touchings
  return
    judgement flag (int): binary flag. 1 for non touching 0 for otherwise
  """
  sdf_values=np.abs(sdf_values)
  non_touch_flag= copy.copy(sdf_values)
  non_touch_flag[non_touch_flag>touch_thresh]=1
  non_touch_flag[non_touch_flag<=touch_thresh]=0
  touch_flag = copy.copy(non_touch_flag)
  touch_flag-=1
  touch_flag*=-1  
  if np.sum(touch_flag)>0:
    return 0
  else:
    return 1


def get_plausibility_metrics(gt_head_vs, gt_rh_vs, pred_head_vs, pred_rh_vs, flame_faces):
  col_tolerance=0.005
  touch_thresh=0.005

  non_col_count = 0 
  count=0
  total_non_touching=0.0
  gt_touch_count = 0
  total_collision=0.0

  HAND_DIM = 778
  
  for i in range(gt_head_vs.shape[0]): 
    gt_touch=check_touch_presence(obj_vs=gt_head_vs[i],
                                            obj_faces= flame_faces,
                                            query_vs=gt_rh_vs[i],
                                    touch_thresh=touch_thresh)
      
    collision_flag , collision_flag_tolerance, collision_dist_sum, \
    collision_dist_avg,non_touching,n_colliding_vs = \
      collision_detection_sdf(obj_faces=flame_faces,
                                        obj_vs=pred_head_vs[i] ,
                                        query_vs=pred_rh_vs[i],
                                        col_tolerance=col_tolerance,
                                        touch_thresh=touch_thresh ) 
      
    if np.sum(collision_flag_tolerance)==0:
      non_col_count+=1 
    
    if gt_touch: 
      gt_touch_count += 1
      total_non_touching += non_touching
      
    if n_colliding_vs!=0:
      total_collision += (collision_dist_sum/HAND_DIM)
    count+=1

  return non_col_count, total_non_touching, gt_touch_count, total_collision, count

# def main():
  
#   ######## calculation w/o alignment #########  
 
#   col_tolerance=0.005
#   touch_thresh=0.005
 
   
#   for sub in subs:
      
     
#     n_files =100# len([x for x in os.listdir(pred_path) ])
#     #n_frames = len(gt_hand_cons)
    
 
#     non_col_count = 0 
#     count=0
#     total_non_touching=0.0
#     total_collision=0.0
    
#     for i in tqdm(range(n_files)): 
      
       
#       ####### load data #######
     
#       """TODO
#       provide your own code to load the vertex positions of the head and the right hand in a camera frame 
#       as well as the corresponding ground truth vertex positions.
#       """
#       head_vs = None# np.zeros((head_dim,3))
#       rh_vs = None ##np.zeros((hand_dim,3))
#       gt_head_vs =None #np.zeros((head_dim,3))
#       gt_rh_vs = None #np.zeros((hand_dim,3))
      
#       gt_touch=check_touch_presence(obj_vs=gt_head_vs,
#                                               obj_faces= flame_faces,
#                                               query_vs=gt_rh_vs,
#                                       touch_thresh=touch_thresh)
        
#       collision_flag , collision_flag_tolerance, collision_dist_sum, \
#       collision_dist_avg,non_touching,n_colliding_vs = \
#         collision_detection_sdf(obj_faces=flame_faces,
#                                           obj_vs=head_vs ,
#                                           query_vs=rh_vs,
#                                           col_tolerance=col_tolerance,
#                                           touch_thresh=touch_thresh ) 
        
#       if np.sum(collision_flag_tolerance)==0:
#         non_col_count+=1 
      
#       if gt_touch: 
#         total_non_touching += non_touching
        
#       if n_colliding_vs!=0:
#         total_collision += (collision_dist_sum/EI.hand_dim)
#       count+=1
 
#     ########## calc measurement ####################
#     print("==========",sub,"==========")
#     print(count)
#     non_col_ratio = 100*non_col_count/count
#     col_dist = 100*total_collision/count#,total_collision
#     non_touching_ratio = 100*total_non_touching/count 
#     print("non_col_ratio",non_col_ratio)
#     print("col_dist",col_dist)
#     print("non_touching_ratio",non_touching_ratio) 
#   return 
 
# if __name__ == '__main__':
#   parser = argparse.ArgumentParser(description='configuratoins')
#   parser.add_argument('--dataset_path', type=str, default="/mnt/d/DecafDataset/")  
   
#   args = parser.parse_args()
#   head_dim=5023
#   hand_dim=778
#   subs  = ['S1','S3','S6' ]
#   flame_faces = np.array( get_mesh_from_plys(
#       args.dataset_path+"assets/default_mesh.ply").triangles)
#   pred_path="/PAHT/TO/YOUR/PREDICTIONS/" #in this script, we assume that the predictions are saved in .ply format
 
  main()
