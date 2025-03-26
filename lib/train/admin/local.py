class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/wangjun/code/RAAT_VC'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'  # Directory for tensorboard files.
        self.lasot_dir = '/media/wangjun/新加卷/train/LaSOT'
        self.got10k_dir = '/media/wangjun/新加卷/train/GOT-10K/train'
        self.got10k_val_dir = '/media/wangjun/新加卷/train/GOT-10K/val_data/val'
        self.trackingnet_dir = '/media/wangjun/新加卷/train/TrackingNet'
        self.coco_dir = '/media/wangjun/新加卷/train/COCO/images'  # Directory for tensorboard files.
        self.pretrained_networks = '/home/wangjun/code/LiteTrack-main/pretrained_models'
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir =  ''
        self.imagenet_lmdb_dir =  ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''




