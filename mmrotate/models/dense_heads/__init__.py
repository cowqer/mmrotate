# Copyright (c) OpenMMLab. All rights reserved.
from .csl_rotated_fcos_head import CSLRFCOSHead
from .csl_rotated_retina_head import CSLRRetinaHead
from .kfiou_odm_refine_head import KFIoUODMRefineHead
from .kfiou_rotate_retina_head import KFIoURRetinaHead
from .kfiou_rotate_retina_refine_head import KFIoURRetinaRefineHead
from .odm_refine_head import ODMRefineHead
from .oriented_reppoints_head import OrientedRepPointsHead
from .oriented_rpn_head import OrientedRPNHead
from .Pconv_oriented_rpn_head import PconvOrientedRPNHead
from .AR_oriented_rpn_head import AR_OrientedRPNHead
from .AR2_oriented_rpn_head import AR2_OrientedRPNHead
from .ADR_oriented_rpn_head_1_testing import ADRconvOrientedRPNHead1
from .ADR_oriented_rpn_head_xy import ADRconvOrientedRPNHeadxy
# from .AP_oriented_rpn_head import APconvOrientedRPNHead
from .ADR_oriented_rpn_head import ADRconvOrientedRPNHead
from .ADR_P_oriented_rpn_head import ADRPconvOrientedRPNHead
from .ADR_loss_oriented_rpn_head import ADRP_test_OrientedRPNHead
from .ADR_loss_oriented_rpn_head2 import ADRconvLossOrientedRPNHead2
from .rotated_anchor_free_head import RotatedAnchorFreeHead
from .rotated_anchor_head import RotatedAnchorHead
from .rotated_atss_head import RotatedATSSHead
from .rotated_fcos_head import RotatedFCOSHead
from .rotated_reppoints_head import RotatedRepPointsHead
from .rotated_retina_head import RotatedRetinaHead
from .rotated_retina_refine_head import RotatedRetinaRefineHead
from .rotated_rpn_head import RotatedRPNHead
from .sam_reppoints_head import SAMRepPointsHead
from .ADR_stn_oriented_rpn_head import ADRSTNOrientedRPNHead
from .ADR_oriented_rpn_head_pro import ADRPOrientedRPNHead
from .SADR_oriented_rpn_head import SADRPOrientedRPNHead
from .ADR_ms_oriented_rpn_head_pro import MSADRPOrientedRPNHead
from .GADR_oriented_rpn_head_pro import ADRPGOrientedRPNHead
from .A_stn_DR_oriented_rpn_head_pro import STN_ADRPOrientedRPNHead
from .A_group_DR_oriented_rpn_head_pro import Group_ADRPOrientedRPNHead
from .A_chunk_DR_oriented_rpn_head_pro import Chunk_ADRPOrientedRPNHead
from .A_max_DR_oriented_rpn_head_pro import Max_ADRPOrientedRPNHead
from .GADR_oriented_rpn_head_pro_mlp import MLP_ADRPGOrientedRPNHead
from .A_tran_DR_oriented_rpn_head_pro import TRANS_ADRPOrientedRPNHead
from .A_SCSA_DR_oriented_rpn_head_pro import SCSA_ADRPOrientedRPNHead
from .A_promax_DR_oriented_rpn_head_pro import Promax_ADRPOrientedRPNHead
from .ADR1_oriented_rpn_head_pro import ADR1POrientedRPNHead
from .ADR2_oriented_rpn_head_pro import ADR2POrientedRPNHead
from .ADR3_oriented_rpn_head_pro import ADR3POrientedRPNHead
from .ADR4_oriented_rpn_head_pro import ADR4POrientedRPNHead
from .ADR5_oriented_rpn_head_pro import ADR5POrientedRPNHead
from .ADR6_oriented_rpn_head_pro import ADR6POrientedRPNHead
from .ADR7_oriented_rpn_head_pro import ADR7POrientedRPNHead
from .ADR8_oriented_rpn_head_pro import ADR8POrientedRPNHead
from .ADR9_oriented_rpn_head_pro import ADR9POrientedRPNHead
from .MADR_oriented_rpn_head_pro import MADRPOrientedRPNHead
from .HW_ADR_oriented_rpn_head_pro import HWADRPOrientedRPNHead
from .new_ADR_oriented_rpn_head_pro import NewADRPOrientedRPNHead
from .A_attn_DR_oriented_rpn_head_pro import MSCA1_ADRPOrientedRPNHead
from .A_MSCA_DR_oriented_rpn_head_pro import MSCA_ADRPOrientedRPNHead

__all__ = [
    'RotatedAnchorHead', 'RotatedRetinaHead', 'RotatedRPNHead',
    'OrientedRPNHead', 'RotatedRetinaRefineHead', 'ODMRefineHead',
    'KFIoURRetinaHead', 'KFIoURRetinaRefineHead', 'KFIoUODMRefineHead',
    'RotatedRepPointsHead', 'SAMRepPointsHead', 'CSLRRetinaHead',
    'RotatedATSSHead', 'RotatedAnchorFreeHead', 'RotatedFCOSHead',
    'CSLRFCOSHead', 'OrientedRepPointsHead','AR_OrientedRPNHead',
    'AR2_OrientedRPNHead','PconvOrientedRPNHead','ADRconvOrientedRPNHead',
    'ADRPconvOrientedRPNHead','ADRP_test_OrientedRPNHead',
    'ADRconvLossOrientedRPNHead2','ADRconvOrientedRPNHead1','ADRconvOrientedRPNHeadxy',
    'ADRSTNOrientedRPNHead','ADRPOrientedRPNHead','SADRPOrientedRPNHead',
    'MSADRPOrientedRPNHead','ADRPGOrientedRPNHead','STN_ADRPOrientedRPNHead',
    'Group_ADRPOrientedRPNHead','Chunk_ADRPOrientedRPNHead','Max_ADRPOrientedRPNHead',
    'MLP_ADRPGOrientedRPNHead','TRANS_ADRPOrientedRPNHead','SCSA_ADRPOrientedRPNHead',
    'Promax_ADRPOrientedRPNHead','ADR1POrientedRPNHead','ADR2POrientedRPNHead','ADR3POrientedRPNHead',
    'MADRPOrientedRPNHead','HWADRPOrientedRPNHead','NewADRPOrientedRPNHead','MSCA_ADRPOrientedRPNHead',
    'ADR4POrientedRPNHead','ADR5POrientedRPNHead','ADR6POrientedRPNHead','ADR7POrientedRPNHead',
    'ADR8POrientedRPNHead','ADR9POrientedRPNHead','MSCA1_ADRPOrientedRPNHead'
]
