from typing import Tuple, TypeVar
import numpy as np
import copy
import torch
from pytorch3d import transforms
from src.decaf import system_util

def rot6d2rot_angle(rot6d:torch.Tensor):
  """ 
  Args:
      rot_angle (*,6):  
  return
      rotation matrix (*,3)
  """
  return transforms.quaternion_to_axis_angle( 
        transforms.matrix_to_quaternion(
        transforms.rotation_6d_to_matrix(rot6d )
        ))
  
def rot_angle2rotmat(rot_angle:torch.Tensor):
  """ 
  Args:
      rot_angle (*,3):  
  return
      rotation 6d (*,3,3)
  """
  return  transforms.axis_angle_to_matrix(rot_angle )


def rot_angle2rot6d(rot_angle:torch.Tensor):
  """ 
  Args:
      rot_angle (*,3):  
  return
      rotation 6d (*,6)
  """
  return transforms.matrix_to_rotation_6d(
        transforms.axis_angle_to_matrix(rot_angle ) )

def rot6d2rot_angle_np(rot6d:np.array):
  """ 
  Args:
      rot_angle (6):  
  return
      rotation matrix (3)
  """
  rot6d = system_util.np2tor(rot6d).view(1,-1)
  return rot6d2rot_angle(rot6d)[0].numpy()

def rot_angle2rot6d_np(rot_angle:np.array):
  """ 
  Args:
      rot_angle (3):  
  return
      rotation 6d (6)
  """
  rot_angle = system_util.np2tor(rot_angle).view(1,-1)
  return rot_angle2rot6d(rot_angle=rot_angle)[0].numpy()

def rot6d2rotmat_np(rot6d:np.array):
  """ 
  Args:
      rotation 6d (6):  
  return
      rotation matrix (3,3)
  """
  rot6d = system_util.np2tor(rot6d).view(1,-1)
  return  transforms.rotation_6d_to_matrix(rot6d)[0].numpy()
 
def rot_angle2rotmat_np(rot_angle:np.array):
  """ 
  Args:
      rotation angle (3):  
  return
      rotation matrix (3,3)
  """
  rot_angle = system_util.np2tor(rot_angle).view(1,-1)
  return  transforms.axis_angle_to_matrix(rot_angle)[0].numpy()
def rotmat2rot6d_np(rotmat:np.array):
  """ 
  Args:
      rotmat (3x3):  
  return
      rotation 6d (6):
  """ 
  rotmat = system_util.np2tor(rotmat).view(1,3,3)
  return  transforms.matrix_to_rotation_6d(rotmat)[0].numpy()

def rotmat2rotangle_np(rotmat:np.array):
  """ 
  Args:
      rotmat (3x3):  
  return
      rotation angle (3):
  """ 
  rotmat = system_util.np2tor(rotmat).view(1,3,3)
  return  transforms.matrix_to_axis_angle(rotmat)[0].numpy()

def rot_angle2rot6d(rot_angle:torch.Tensor):
  """ 
  Args:
      rot_angle (*,3):  
  return
      rotation 6d (*,6)
  """
  return transforms.matrix_to_rotation_6d(
        transforms.axis_angle_to_matrix(rot_angle ) )
  
def convert_R_T_to_RT4x4_np(rot_mat:np.ndarray,transl:np.ndarray)->np.ndarray:
  """
  args:
    rot_mat (3,3)
    transl (3,)
  returns:
    RT matrix (4,4)
  """
  rot_mat = rot_mat.reshape(3,3)
  transl= transl.reshape(3,1)
  RT = np.concatenate((rot_mat,transl),axis=1)
  RT=np.concatenate((RT,np.array([[0,0,0,1]])),axis=0) 
  return RT

def convert_R_T_to_RT4x4(rot_mat:torch.Tensor,transl:torch.Tensor)->torch.Tensor:
  """
  args:
    rot_mat (*,3,3)
    transl (*,3,)
  returns:
    RT matrix (*,4,4)
  """
  b=len(rot_mat)
  rot_mat = rot_mat.view(b,3,3)
  transl= transl.view(b,3,1)
  RT = torch.cat((rot_mat,transl),dim=2)
  bottom = torch.FloatTensor(
    [0,0,0,1]).view(1,1,4).expand(b,-1,-1).to(rot_mat.device)
  RT=torch.cat((RT,bottom),dim=1) 
  return RT


def apply_transform_np(data:np.ndarray,RT:np.ndarray)->np.ndarray:
    """
    a function to apply 3D transformation on a 3D data
    Args:
      data: n_vs,3
      RT: 4,4
    Return ( n_vs,3):
      data after transformation
    """ 
    n_vs,_=data.shape
    data = np.concatenate((data,np.ones((n_vs,1))),axis=-1) 
    return (np.dot(RT,data.T).T)[:,:-1]
  
def apply_transform_batch(data:torch.tensor,RT:torch.tensor)->torch.tensor:
    """
    a function to apply 3D transformation on a 3D data
    Args:
      data (*,n_vs,3):
      RT: (*,4,4)
    Return (*,n_vs,3):
      data after transformation
    """ 
    b,n_vs,_=data.shape
    
    homo=torch.ones(b,n_vs,1).to(data.device)
    data = torch.cat((data,homo),dim=-1)
     
    return torch.bmm(RT, data.permute(0,2,1)).permute(0,2,1)[:,:,:-1] 
  
def rigid_transform(R:torch.tensor, T:torch.tensor, data:torch.tensor):
    """
    R:Bx3x3; T:Bx3, data:BxNx3, s:Bx1
    """
    B, N, _ = data.shape
    centroid = data.clone().detach().mean(1)
    center_data = data-centroid.view(B, 1, 3)
    data_transformed_T = torch.bmm(R, torch.transpose(
        center_data, 2, 1))+centroid.detach().view(B, 3, 1) + T.view(B, 3, 1)
    return data_transformed_T

#def rigid_transform_np(R:np.ndarray, T:np.ndarray, data:np.ndarray):
#    """
#    R:3x3; T:3, data:Nx3 
#    """ 
#    return np.dot(R,data.T).T+T.reshape(1,3)

def _rigid_transform_3D(A: np.ndarray, 
                        B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert len(A) == len(B)

    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(
            num_rows, num_cols))

    [num_rows, num_cols] = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(
            num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1).reshape(3, 1)
    centroid_B = np.mean(B, axis=1).reshape(3, 1)
    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am.dot(np.transpose(Bm))

    # sanity check
    if np.linalg.matrix_rank(H) < 3:
        raise ValueError("rank of H = {}, expecting 3".format(
            np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A + centroid_B

    return R, t


def procrastes(pred_points, gt_points):
    """
    args:
      pred_points: NxD
      gt_points  : NxD
    return:
      rotaiton
      translation
    """
    assert pred_points.shape == gt_points.shape
    A = np.mat(copy.copy(pred_points.T))
    B = np.mat(copy.copy(gt_points.T))

    ret_R, ret_t = _rigid_transform_3D(A, B)
    n = len(pred_points)
    B2 = (ret_R * A) + np.tile(ret_t, (1, n))
    B2 = np.array(B2).T
    return np.array(ret_R), np.array(ret_t)


def multiview_projection(Ps,points,device):
    """
    Ps:n_view x 3 x 4
    points: 1 x n_points x 3
    returns: n_view x n_points x 2
    """
    n_view,_,_=Ps.shape
    _,n_p,d = points.shape
    points=points.expand(n_view,-1,-1)
    points_homo=torch.cat((points,torch.ones(n_view,n_p,1).to(device)),2) 
    proj_homo = torch.transpose(torch.bmm(Ps,torch.transpose(points_homo,2,1)),2,1)
    proj_homo = proj_homo/(proj_homo[:,:,-1].view(n_view,n_p,1)) 
    return proj_homo[:,:,:-1]
  
import cv2
def image_flipper_with_image_center(k,img):
    """
    A function to flip image vertically considering the image center.
    The image center needs to be taken into account when you flip the 3D geometry
    horizontally and want to obtain the corresponding flipped image.
    Note: this function works only when w/2 > k[0][2] !!!!!!!
    args:
      k (3,3): intrinsic matrix
      img (h,w,c): image
    """
     
    h,w,c=img.shape
    n =int( k[0][2] ) 
    sub_img = img[:,:2*n]
    sub_img = cv2.flip(sub_img, 1)
    end_img = np.zeros((h, w-2*n, 3), dtype = np.uint8) #img[:,:w-2*n] 
    return  np.concatenate((sub_img,end_img),axis=1)
   
def convert_canonical_deforms_into_cam_space(can_deforms,
                                             head_pose, 
                                              cam_transform
                                              ):
    """
    cannonical frame -> world frame -> camera frame
    
    can_deform (N,3): deformations in a FLAME canoincal space
    head_pose (4,4): head pose matrix 
    cam_transform (4,4): camera transformation matrix
    """ 
     
    head_pose[:3, -1] = 0 
    posed_deforms = apply_transform_np(data=can_deforms,
                                          RT=head_pose)
    ##### posed -> cam space ###
    cam_transform=copy.copy(cam_transform)
    cam_transform[:3, -1] = 0 
    cam_space_deforms = apply_transform_np(data=posed_deforms,
                                              RT=cam_transform)

    return cam_space_deforms #cam_space_deforms
  
def convert_canonical_deforms_into_world_space( can_deforms,
                                             head_pose, 
                                              ):
    """
    cannonical -> 3D space with the given poses -> camera frame
    """ 
    head_pose[:3, -1] = 0 
    posed_deforms = apply_transform_np(data=can_deforms,
                                          RT=head_pose)
    ##### posed -> cam space ###
     
    return posed_deforms 
  
def convert_world_deforms_into_canonical_space( world_deforms,
                                             head_pose, 
                                              ):
    """
    cannonical -> 3D space with the given poses -> camera frame
    """ 
    head_pose[:3, -1] = 0 
    posed_deforms = apply_transform_np(data=world_deforms,
                                          RT=np.linalg.inv(head_pose))
    ##### posed -> cam space ###
     
    return posed_deforms 

from scipy.spatial.transform import Rotation as R
def get_RT_from_z_angle(angle:float) -> np.ndarray:
  """a funciton to obtain RT matrix from a rotaion agnle on z axis

  Args:
      angle (float): amount of rotation in DEGREE, not radian

  Returns:
      transformation matrix (4x4)
  """
  r = R.from_euler('xyz', [0, 0, angle], degrees=True) 
  z_rot_transform = np.eye(4)
  z_rot_transform[:3, :3] = r.as_matrix()
  return z_rot_transform