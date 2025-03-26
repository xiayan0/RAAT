from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    #settings.prj_dir = 'D:/tracking\RAAT_VC'
    #settings.save_dir = 'D:/tracking\RAAT_VC'
    # Set your local paths here.

    settings.davis_dir = '/home/wangjun/code/RAAT_VC'
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/mnt/sda1/xy-test/Test/GOT-10K'
    settings.got_packed_results_path = '/home/wangjun/code/RAAT_VC/lib/train/output'
    settings.got_reports_path = ''
    settings.trackingnet_path = '/media/wangjun/E/TrackingNet/TEST'
    settings.uav_path = '/mnt/sda1/xy-test/Test/UAV123'
    settings.itb_path = ''
    settings.lasot_extension_subset_path_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/mnt/sda1/xy-test/Test/LaSOT/TEST'    #
    settings.nfs_path = '/mnt/sda1/xy-test/Test/Nfs'          #
    settings.otb_path = '/mnt/sda1/xy-test/Test/OTB2015/OTB2015/OTB100'
    settings.network_path = '/home/wangjun/code/RAAT_VC'  # Where tracking networks are stored.
    settings.prj_dir = '/home/wangjun/code/RAAT_VC'
    settings.result_plot_path = '/home/wangjun/code/RAAT_VC/lib/train/output/test/result_plots'
    settings.results_path = '/home/wangjun/code/RAAT_VC/output/test/tracking_results'  # Where to store  # Where to store tracking results
    settings.save_dir = '/home/wangjun/code/RAAT_VC/lib/train/output'
    settings.tc128_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.vot18_path = '/mnt/sda1/xy-test/Test/VOT2018'
    settings.vot22_path = '/mnt/sda1/xy-test/Test/VOT2020'
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings






