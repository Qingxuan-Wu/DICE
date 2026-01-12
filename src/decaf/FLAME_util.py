import pytorch3d.transforms as transforms
import os 
import torch
import torch.nn as nn
import numpy as np
import pickle
from src.decaf.lbs import (
  lbs, batch_rodrigues, vertices2landmarks, rot_mat_to_euler
)
 
from src.decaf.tracking_util import (
  normalize_keys_batch,
  multiview_projection_batch,
  rigid_transform_with_scaling
) 
 
def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, model_path,landmark_path):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.np_face=to_np(flame_model.f, dtype=np.int64)
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:100], shapedirs[:,:,300:300+50]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        # 
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(landmark_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long())
        self.register_buffer('lmk_bary_coords', torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long())
        self.register_buffer('full_lmk_bary_coords', torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype))

        neck_kin_chain = []; NECK_IDX=1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
    
     
    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:dd2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(
            device=vertices.device) * num_verts

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                       self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                       self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([pose_params[:, :3], 
                               self.neck_pose.expand(batch_size, -1), 
                               pose_params[:, 3:], 
                               eye_pose_params], dim=1)
        
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        bz = vertices.shape[0] 
 
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                       self.full_lmk_faces_idx.repeat(bz, 1),
                                       self.full_lmk_bary_coords.repeat(bz, 1, 1))
        return vertices, landmarks2d, landmarks3d
    def convert_vs2landmarks(self, vertices):
        bz=vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                       self.full_lmk_faces_idx.repeat(bz, 1),
                                       self.full_lmk_bary_coords.repeat(bz, 1, 1))
        return landmarks3d

def get_FLAME_faces(file_path):
    with open(file_path, 'rb') as f:
        ss = pickle.load(f, encoding='latin1')
        flame_model = Struct(**ss)
    return torch.LongTensor(np.array(flame_model.f).astype(int))
 

def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # print(vertices.shape, faces.shape, lmk_faces_idx.shape, lmk_bary_coords.shape)
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(batch_size, dtype=torch.long,
                              device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.reshape(-1,
                                    3)[lmk_faces].view(batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def flame_forwarding(flame_model,
                     head_shape_params,
                     head_pose_params,
                     head_expression_params,
                     head_transl,
                     head_rotation,
                     head_scale_params,
                     device=None,
                     img_size=None,
                     Ps=None,
                     return2d=False, 
                     return_2d_verts=False
                     ):
    """_summary_

    Args:
        flame_model (_type_): FLAME parametric model
        head_shape_params (_type_): _description_
        head_pose_params (_type_): _description_
        head_expression_params (_type_): _description_
        head_transl (_type_): _description_
        head_rotation (_type_): _description_
        head_scale_params (_type_): _description_
        device (_type_, optional): _description_. Defaults to None.
        img_size (_type_, optional): _description_. Defaults to None.
        Ps (_type_, optional): _description_. Defaults to None.
        return2d (bool, optional): _description_. Defaults to False.

    Returns:
        [3D face verticec (*,P,3),
            3D face land marks (*,N,3),
        optionally: 2D NORMALIZED keypoints (*,M,2)]
    """
    b=len(head_shape_params)
    #print(head_shape_params,head_pose_params,head_expression_params,flame_model)
    head_verts, landmarks2d, _ = flame_model(
        shape_params=head_shape_params,
        pose_params=head_pose_params,
        expression_params=head_expression_params)  # ,
    ######## important!!! ########
    # head_pose_params[:, :3] = head_pose_params[:, :3] * 0 

    head_rotation_mat = transforms.axis_angle_to_matrix(head_rotation)
    head_verts_transformed = torch.transpose(
        rigid_transform_with_scaling(
            head_rotation_mat, head_transl, head_scale_params, head_verts),2,1)
    
    landmarks3d = \
        vertices2landmarks(
            head_verts_transformed,
            flame_model.faces_tensor,
            flame_model.full_lmk_faces_idx.repeat( b, 1),
            flame_model.full_lmk_bary_coords.repeat(b, 1, 1))
    # print(head_verts.shape, "head_verts")
    # print(landmarks3d.shape, "landmarks3d")
    if return2d:
        # print(landmarks3d.shape,Ps.shape)
        # torch.Size([1, 68, 3]) torch.Size([1, 3, 4])
        # b=len(landmarks3d)
        # Ps=Ps.view(b,-1,3,4)
        # landmarks3d = landmarks3d.view(b,-1,3)
        head_keys_proj = multiview_projection_batch(
            Ps.detach(), landmarks3d, device=device) 
        
        head_keys_proj = normalize_keys_batch(
            head_keys_proj, img_size[0], img_size[1]) 
        if return_2d_verts:
            head_vs_proj = multiview_projection_batch(
            Ps.detach(), head_verts_transformed, device=device) 
        
            head_vs_proj = normalize_keys_batch(
                head_vs_proj, img_size[0], img_size[1]) 
            return [head_verts_transformed, landmarks3d, head_keys_proj, head_vs_proj]
        else:
            return [head_verts_transformed, landmarks3d, head_keys_proj]
    else:
        return [head_verts_transformed,landmarks3d]