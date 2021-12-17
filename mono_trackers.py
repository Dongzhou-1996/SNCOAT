from trackers.SiamFC.siamfc import TrackerSiamFC
from trackers.SiamRPN.siamrpn import TrackerSiamRPN
from trackers.cftracker.mosse import MOSSE
from trackers.cftracker.csk import CSK
from trackers.cftracker.kcf import KCF
from trackers.cftracker.cn import CN
from trackers.cftracker.dsst import DSST
from trackers.cftracker.staple import Staple
from trackers.cftracker.dat import DAT
from trackers.cftracker.eco import ECO
from trackers.cftracker.bacf import BACF
from trackers.cftracker.csrdcf import CSRDCF
from trackers.cftracker.samf import SAMF
from trackers.cftracker.ldes import LDES
from trackers.cftracker.mkcfup import MKCFup
from trackers.cftracker.strcf import STRCF
from trackers.cftracker.mccth_staple import MCCTHStaple
from trackers.lib.eco.config import otb_deep_config, otb_hc_config
from trackers.cftracker.config import staple_config, ldes_config, dsst_config, csrdcf_config, mkcf_up_config, mccth_staple_config


class MonoTracker(object):

    def __init__(self, mono_tracker_name='SiamRPN'):
        support_trackers = ['SiamRPN', 'SiamFC', 'KCF_CN', 'KCF_GRAY', 'KCF_HOG',
                            'MOSSE', 'CSK', 'CN', 'DCF_GRAY', 'DCF_HOG', 'DAT',
                            'DSST', 'DSST-LP', 'MKCFup', 'MKCFup-LP', 'STRCF', 'LDES',
                            'ECO-HC', 'ECO', 'CSRDCF', 'CSRDCF-LP', 'BACF', 'SAMF',
                            'Staple', 'Staple-CA', 'MCCTH-Staple']

        assert mono_tracker_name in support_trackers
        self.name = mono_tracker_name

        if mono_tracker_name == 'SiamRPN':
            self.tracker = TrackerSiamRPN('trackers/SiamRPN/model.pth')

        elif mono_tracker_name == 'SiamFC':
            self.tracker = TrackerSiamFC('trackers/SiamFC/model.pth')

        elif mono_tracker_name == 'MOSSE':
            self.tracker = MOSSE()

        elif mono_tracker_name == 'CSK':
            self.tracker = CSK()

        elif mono_tracker_name == 'CN':
            self.tracker = CN()

        elif mono_tracker_name == 'DSST':
            self.tracker = DSST(dsst_config.DSSTConfig())

        elif mono_tracker_name == 'Staple':
            self.tracker = Staple(config=staple_config.StapleConfig())

        elif mono_tracker_name == 'Staple-CA':
            self.tracker = Staple(config=staple_config.StapleCAConfig())

        elif mono_tracker_name == 'KCF_CN':
            self.tracker = KCF(features='cn', kernel='gaussian')

        elif mono_tracker_name == 'KCF_GRAY':
            self.tracker = KCF(features='gray', kernel='gaussian')

        elif mono_tracker_name == 'KCF_HOG':
            self.tracker = KCF(features='hog', kernel='gaussian')

        elif mono_tracker_name == 'DCF_GRAY':
            self.tracker = KCF(features='gray', kernel='linear')

        elif mono_tracker_name == 'DCF_HOG':
            self.tracker = KCF(features='hog', kernel='linear')

        elif mono_tracker_name == 'DAT':
            self.tracker = DAT()

        elif mono_tracker_name == 'ECO-HC':
            self.tracker = ECO(config=otb_hc_config.OTBHCConfig())

        elif mono_tracker_name == 'ECO':
            self.tracker = ECO(config=otb_deep_config.OTBDeepConfig())

        elif mono_tracker_name == 'BACF':
            self.tracker = BACF()

        elif mono_tracker_name == 'CSRDCF':
            self.tracker = CSRDCF(config=csrdcf_config.CSRDCFConfig())

        elif mono_tracker_name == 'CSRDCF-LP':
            self.tracker = CSRDCF(config=csrdcf_config.CSRDCFLPConfig())

        elif mono_tracker_name == 'SAMF':
            self.tracker = SAMF()

        elif mono_tracker_name == 'LDES':
            self.tracker = LDES(ldes_config.LDESDemoLinearConfig())

        elif mono_tracker_name == 'DSST-LP':
            self.tracker = DSST(dsst_config.DSSTLPConfig())

        elif mono_tracker_name == 'MKCFup':
            self.tracker = MKCFup(config=mkcf_up_config.MKCFupConfig())

        elif mono_tracker_name == 'MKCFup-LP':
            self.tracker = MKCFup(config=mkcf_up_config.MKCFupLPConfig())

        elif mono_tracker_name == 'STRCF':
            self.tracker = STRCF()

        elif mono_tracker_name == 'MCCTH-Staple':
            self.tracker = MCCTHStaple(config=mccth_staple_config.MCCTHOTBConfig())

        else:
            print('=> mono-tracker {} is not supported!'.format(mono_tracker_name))
            print('=> supported trackers list: \n{}'.format(support_trackers))
        pass

    def init(self, img, roi):
        print('=> start to initialize monocular tracker ...')
        self.tracker.init(img, roi)
        print('=> monocular tracker have been initialized successfully!')

    def update(self, img):
        bbox2d = self.tracker.update(img)
        return bbox2d


if __name__ == '__main__':
    mono_trackers = [
        MonoTracker('SiamFC'),
        MonoTracker('SiamRPN'),
        MonoTracker('ECO'),
        MonoTracker('DAT'),
        MonoTracker('MOSSE'),
        MonoTracker('Staple'),
        MonoTracker('CSK'),
        MonoTracker('Staple-CA'),
        MonoTracker('KCF_GRAY'),
        MonoTracker('KCF_HOG'),
        MonoTracker('DCF_GRAY'),
        MonoTracker('DCF_HOG'),
        MonoTracker('ECO-HC'),
        MonoTracker('BACF'),
        MonoTracker('CSRDCF'),
        MonoTracker('CSRDCF-LP'),
        MonoTracker('SAMF'),
        MonoTracker('MKCFup'),
        MonoTracker('MKCFup-LP'),
        MonoTracker('STRCF'),
        MonoTracker('MCCTH-Staple'),
        MonoTracker('CN'),
        MonoTracker('KCF_CN'),
        MonoTracker('LDES'),
        MonoTracker('DSST'),
        MonoTracker('DSST-LP')
    ]