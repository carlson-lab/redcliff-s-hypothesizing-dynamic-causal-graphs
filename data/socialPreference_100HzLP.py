import numpy as np
import scipy.io as scio
import os
import random
import pickle as pkl
import argparse
import json

from general_utils.time_series import *

# fix random seed(s) to 1337 -- see https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
np.random.seed(1337)
random.seed(1337)


proper_s_label_files = [
    'Mouse8881_100617_SocialPreference_Class.mat', 'Mouse699L_061616_SocialPreference_Class.mat', 'Mouse6992_061416_SocialPreference_Class.mat', 'Mouse6662_032618_SocialPreference_Class.mat', 'Mouse8894_100517_SocialPreference_Class.mat', 'Mouse8881_092917_SocialPreference_Class.mat', 'Mouse8893_090917_SocialPreference_Class.mat', 'Mouse8884_100617_SocialPreference_Class.mat', 'Mouse6992_060916_SocialPreference_Class.mat', 'Mouse5321_052616_SocialPreference_Class.mat', 'Mouse6662_040218_SocialPreference_Class.mat', 'Mouse0632_092517_SocialPreference_Class.mat', 'Mouse8882_092217_SocialPreference_Class.mat', 'Mouse6674_041118_SocialPreference_Class.mat', 'Mouse0634_092017_SocialPreference_Class.mat', 'Mouse8882_100217_SocialPreference_Class.mat', 'Mouse8894_091417_SocialPreference_Class.mat', 'Mouse0633_100417_SocialPreference_Class.mat', 'Mouse0631_100617_SocialPreference_Class.mat', 'Mouse0643_091917_SocialPreference_Class.mat', 'Mouse0632_091817_SocialPreference_Class.mat', 'Mouse0634_100917_SocialPreference_Class.mat', 'Mouse533L_070716_SocialPreference_Class.mat', 'Mouse6662_040418_SocialPreference_Class.mat', 'Mouse5333_060916_SocialPreference_Class.mat', 'Mouse6992_052416_SocialPreference_Class.mat', 'Mouse5331_070716_SocialPreference_Class.mat', 'Mouse6662_040918_SocialPreference_Class.mat', 'Mouse5333_061416_SocialPreference_Class.mat', 'Mouse6664_040218_SocialPreference_Class.mat', 'Mouse0634_092917_SocialPreference_Class.mat', 'Mouse6991_060216_SocialPreference_Class.mat', 'Mouse8893_091417_SocialPreference_Class.mat', 'Mouse0640_100517_SocialPreference_Class.mat', 'Mouse5321_060716_SocialPreference_Class.mat', 'Mouse0644_101017_SocialPreference_Class.mat', 'Mouse0631_092017_SocialPreference_Class.mat', 'Mouse8893_091617_SocialPreference_Class.mat', 'Mouse6664_040618_SocialPreference_Class.mat', 'Mouse0632_092917_SocialPreference_Class.mat', 'Mouse533L_052616_SocialPreference_Class.mat', 'Mouse8884_100417_SocialPreference_Class.mat', 'Mouse0631_092217_SocialPreference_Class.mat', 'Mouse5321_060216_SocialPreference_Class.mat', 'Mouse0643_091417_SocialPreference_Class.mat', 'Mouse8882_091117_SocialPreference_Class.mat', 'Mouse8893_092617_SocialPreference_Class.mat', 'Mouse8884_092917_SocialPreference_Class.mat', 'Mouse6662_033018_SocialPreference_Class.mat', 'Mouse0632_100617_SocialPreference_Class.mat', 'Mouse8891_100317_SocialPreference_Class.mat', 'Mouse5321_060916_SocialPreference_Class.mat', 'Mouse6674_041618_SocialPreference_Class.mat', 'Mouse6662_041118_SocialPreference_Class.mat', 'Mouse0641_091917_SocialPreference_Class.mat', 'Mouse533L_060216_SocialPreference_Class.mat', 'Mouse6664_032618_SocialPreference_Class.mat', 'Mouse0642_091617_SocialPreference_Class.mat', 'Mouse8893_093017_SocialPreference_Class.mat', 'Mouse6992_060716_SocialPreference_Class.mat', 'Mouse6991_070716_SocialPreference_Class.mat', 'Mouse8884_091117_SocialPreference_Class.mat', 'Mouse0644_093017_SocialPreference_Class.mat', 'Mouse0640_100717_SocialPreference_Class.mat', 'Mouse8891_090917_SocialPreference_Class.mat', 'Mouse5332_060216_SocialPreference_Class.mat', 'Mouse0642_093017_SocialPreference_Class.mat', 'Mouse0634_091817_SocialPreference_Class.mat', 'Mouse6990_060916_SocialPreference_Class.mat', 'Mouse0634_092517_SocialPreference_Class.mat', 'Mouse0644_100517_SocialPreference_Class.mat', 'Mouse6992_053116_SocialPreference_Class.mat', 'Mouse8891_091617_SocialPreference_Class.mat', 'Mouse8884_092217_SocialPreference_Class.mat', 'Mouse5321_061416_SocialPreference_Class.mat', 'Mouse8881_092217_SocialPreference_Class.mat', 'Mouse0633_092017_SocialPreference_Class.mat', 'Mouse0633_092917_SocialPreference_Class.mat', 'Mouse0640_092617_SocialPreference_Class.mat', 'Mouse0632_100417_SocialPreference_Class.mat', 'Mouse8891_092617_SocialPreference_Class.mat', 'Mouse533L_061416_SocialPreference_Class.mat', 'Mouse0633_092217_SocialPreference_Class.mat', 'Mouse6674_040418_SocialPreference_Class.mat', 'Mouse0631_100217_SocialPreference_Class.mat', 'Mouse0632_091517_SocialPreference_Class.mat', 'Mouse0631_100417_SocialPreference_Class.mat', 'Mouse6674_040618_SocialPreference_Class.mat', 'Mouse0633_091517_SocialPreference_Class.mat', 'Mouse6990_060716_SocialPreference_Class.mat', 'Mouse8881_091117_SocialPreference_Class.mat', 'Mouse0631_091817_SocialPreference_Class.mat', 'Mouse0642_092117_SocialPreference_Class.mat', 'Mouse6674_040218_SocialPreference_Class.mat', 'Mouse5321_061616_SocialPreference_Class.mat', 'Mouse8893_091217_SocialPreference_Class.mat', 'Mouse0634_092217_SocialPreference_Class.mat', 'Mouse8882_092517_SocialPreference_Class.mat', 'Mouse5331_070516_SocialPreference_Class.mat', 'Mouse6990_061416_SocialPreference_Class.mat', 'Mouse8882_092917_SocialPreference_Class.mat', 'Mouse0640_091417_SocialPreference_Class.mat', 'Mouse0642_100717_SocialPreference_Class.mat', 'Mouse5331_060716_SocialPreference_Class.mat', 'Mouse6990_052616_SocialPreference_Class.mat', 'Mouse6991_052416_SocialPreference_Class.mat', 'Mouse0644_092317_SocialPreference_Class.mat', 'Mouse6674_032618_SocialPreference_Class.mat', 'Mouse6662_041618_SocialPreference_Class.mat', 'Mouse8884_091517_SocialPreference_Class.mat', 'Mouse8891_093017_SocialPreference_Class.mat', 'Mouse0633_100917_SocialPreference_Class.mat', 'Mouse0643_092617_SocialPreference_Class.mat', 'Mouse8894_091617_SocialPreference_Class.mat', 'Mouse0630_091517_SocialPreference_Class.mat', 'Mouse8893_092317_SocialPreference_Class.mat', 'Mouse8882_100417_SocialPreference_Class.mat', 'Mouse0630_100217_SocialPreference_Class.mat', 'Mouse0640_092317_SocialPreference_Class.mat', 'Mouse6990_061616_SocialPreference_Class.mat', 'Mouse8882_091317_SocialPreference_Class.mat', 'Mouse0632_100217_SocialPreference_Class.mat', 'Mouse0630_092217_SocialPreference_Class.mat', 'Mouse0642_091917_SocialPreference_Class.mat', 'Mouse8884_100217_SocialPreference_Class.mat', 'Mouse6992_061616_SocialPreference_Class.mat', 'Mouse0644_100717_SocialPreference_Class.mat', 'Mouse0642_091417_SocialPreference_Class.mat', 'Mouse0641_100517_SocialPreference_Class.mat', 'Mouse0632_092017_SocialPreference_Class.mat', 'Mouse0631_092517_SocialPreference_Class.mat', 'Mouse8894_093017_SocialPreference_Class.mat', 'Mouse0631_092917_SocialPreference_Class.mat', 'Mouse699L_060216_SocialPreference_Class.mat', 'Mouse6991_060716_SocialPreference_Class.mat', 'Mouse0633_100217_SocialPreference_Class.mat', 'Mouse8884_092517_SocialPreference_Class.mat', 'Mouse0633_100617_SocialPreference_Class.mat', 'Mouse6674_041318_SocialPreference_Class.mat', 'Mouse5332_052616_SocialPreference_Class.mat', 'Mouse8891_091417_SocialPreference_Class.mat', 'Mouse6664_040918_SocialPreference_Class.mat', 'Mouse699L_061416_SocialPreference_Class.mat', 'Mouse8891_100517_SocialPreference_Class.mat', 'Mouse0641_100317_SocialPreference_Class.mat', 'Mouse0634_091517_SocialPreference_Class.mat', 'Mouse6662_040618_SocialPreference_Class.mat', 'Mouse0630_100617_SocialPreference_Class.mat', 'Mouse8894_092617_SocialPreference_Class.mat', 'Mouse8894_092317_SocialPreference_Class.mat', 'Mouse6992_052616_SocialPreference_Class.mat', 'Mouse8882_090817_SocialPreference_Class.mat', 'Mouse8893_100517_SocialPreference_Class.mat', 'Mouse5333_060716_SocialPreference_Class.mat', 'Mouse533L_070516_SocialPreference_Class.mat', 'Mouse8884_090817_SocialPreference_Class.mat', 'Mouse8882_100617_SocialPreference_Class.mat', 'Mouse0634_100617_SocialPreference_Class.mat', 'Mouse0630_092517_SocialPreference_Class.mat', 'Mouse0632_092217_SocialPreference_Class.mat', 'Mouse0642_100317_SocialPreference_Class.mat', 'Mouse5333_070516_SocialPreference_Class.mat', 'Mouse699L_053116_SocialPreference_Class.mat', 'Mouse8894_100717_SocialPreference_Class.mat', 'Mouse8881_100217_SocialPreference_Class.mat', 'Mouse0641_091417_SocialPreference_Class.mat', 'Mouse0630_091817_SocialPreference_Class.mat', 'Mouse6991_052616_SocialPreference_Class.mat', 'Mouse0633_091817_SocialPreference_Class.mat', 'Mouse0642_101017_SocialPreference_Class.mat', 'Mouse0641_092117_SocialPreference_Class.mat', 'Mouse0641_092617_SocialPreference_Class.mat', 'Mouse5332_061416_SocialPreference_Class.mat', 'Mouse699L_070516_SocialPreference_Class.mat', 'Mouse0641_100717_SocialPreference_Class.mat', 'Mouse0632_100917_SocialPreference_Class.mat', 'Mouse0640_091617_SocialPreference_Class.mat', 'Mouse5333_060216_SocialPreference_Class.mat', 'Mouse699L_052616_SocialPreference_Class.mat', 'Mouse5332_052416_SocialPreference_Class.mat', 'Mouse6664_040418_SocialPreference_Class.mat', 'Mouse0641_091617_SocialPreference_Class.mat', 'Mouse8881_091317_SocialPreference_Class.mat', 'Mouse0643_092317_SocialPreference_Class.mat', 'Mouse5332_070716_SocialPreference_Class.mat', 'Mouse6992_060216_SocialPreference_Class.mat', 'Mouse0633_092517_SocialPreference_Class.mat', 'Mouse6674_040918_SocialPreference_Class.mat', 'Mouse6662_041318_SocialPreference_Class.mat', 'Mouse6991_070516_SocialPreference_Class.mat', 'Mouse0634_100417_SocialPreference_Class.mat', 'Mouse533L_060716_SocialPreference_Class.mat', 'Mouse8891_092317_SocialPreference_Class.mat', 'Mouse5321_053116_SocialPreference_Class.mat', 'Mouse0643_092117_SocialPreference_Class.mat', 'Mouse0633_091417_SocialPreference_Class.mat', 'Mouse8881_090817_SocialPreference_Class.mat', 'Mouse8893_100317_SocialPreference_Class.mat', 'Mouse0644_091617_SocialPreference_Class.mat', 'Mouse5331_061616_SocialPreference_Class.mat', 'Mouse0632_091417_SocialPreference_Class.mat', 'Mouse0642_092317_SocialPreference_Class.mat', 'Mouse0631_091417_SocialPreference_Class.mat', 'Mouse0642_100517_SocialPreference_Class.mat', 'Mouse0634_100217_SocialPreference_Class.mat', 'Mouse0630_100917_SocialPreference_Class.mat', 'Mouse0643_100317_SocialPreference_Class.mat', 'Mouse0630_091417_SocialPreference_Class.mat', 'Mouse0643_100717_SocialPreference_Class.mat', 'Mouse5333_061616_SocialPreference_Class.mat', 'Mouse0643_093017_SocialPreference_Class.mat', 'Mouse0644_092117_SocialPreference_Class.mat', 'Mouse6992_070716_SocialPreference_Class.mat', 'Mouse5333_070716_SocialPreference_Class.mat', 'Mouse0630_100417_SocialPreference_Class.mat', 'Mouse6664_041118_SocialPreference_Class.mat', 'Mouse8884_091317_SocialPreference_Class.mat', 'Mouse0634_091417_SocialPreference_Class.mat', 'Mouse0641_101017_SocialPreference_Class.mat', 'Mouse0644_100317_SocialPreference_Class.mat', 'Mouse8894_100317_SocialPreference_Class.mat', 'Mouse533L_053116_SocialPreference_Class.mat', 'Mouse0644_091917_SocialPreference_Class.mat', 'Mouse5331_061416_SocialPreference_Class.mat', 'Mouse5332_061616_SocialPreference_Class.mat', 'Mouse8894_090917_SocialPreference_Class.mat', 'Mouse5332_053116_SocialPreference_Class.mat', 'Mouse8893_100717_SocialPreference_Class.mat', 'Mouse6991_061616_SocialPreference_Class.mat', 'Mouse0630_092017_SocialPreference_Class.mat', 'Mouse8891_091217_SocialPreference_Class.mat', 'Mouse8881_091517_SocialPreference_Class.mat', 'Mouse5331_052416_SocialPreference_Class.mat', 'Mouse0640_092117_SocialPreference_Class.mat', 'Mouse699L_052416_SocialPreference_Class.mat', 'Mouse5331_053116_SocialPreference_Class.mat', 'Mouse8882_091517_SocialPreference_Class.mat', 'Mouse0640_093017_SocialPreference_Class.mat', 'Mouse8891_100717_SocialPreference_Class.mat', 'Mouse0631_100917_SocialPreference_Class.mat', 'Mouse0640_101017_SocialPreference_Class.mat', 'Mouse0640_100317_SocialPreference_Class.mat', 'Mouse0641_092317_SocialPreference_Class.mat', 'Mouse0641_093017_SocialPreference_Class.mat', 'Mouse6664_041318_SocialPreference_Class.mat', 'Mouse6664_041618_SocialPreference_Class.mat', 'Mouse8894_091217_SocialPreference_Class.mat', 'Mouse8881_092517_SocialPreference_Class.mat', 'Mouse5333_052616_SocialPreference_Class.mat', 'Mouse0640_091917_SocialPreference_Class.mat', 'Mouse5333_052416_SocialPreference_Class.mat', 'Mouse6991_061416_SocialPreference_Class.mat', 'Mouse6991_060916_SocialPreference_Class.mat', 'Mouse5332_070516_SocialPreference_Class.mat', 'Mouse0644_092617_SocialPreference_Class.mat', 'Mouse8881_100417_SocialPreference_Class.mat', 'Mouse0631_091517_SocialPreference_Class.mat', 'Mouse0644_091417_SocialPreference_Class.mat', 'Mouse5332_060716_SocialPreference_Class.mat', 'Mouse0630_092917_SocialPreference_Class.mat', 'Mouse6991_053116_SocialPreference_Class.mat', 'Mouse0643_100517_SocialPreference_Class.mat', 'Mouse0643_091617_SocialPreference_Class.mat', 'Mouse0642_092617_SocialPreference_Class.mat', 'Mouse6992_070516_SocialPreference_Class.mat', 'Mouse0643_101017_SocialPreference_Class.mat', 'Mouse699L_070716_SocialPreference_Class.mat'
]
proper_o_label_files = [
    'Mouse8881_100617_SocialPreference_Class.mat', 'Mouse699L_061616_SocialPreference_Class.mat', 'Mouse6992_061416_SocialPreference_Class.mat', 'Mouse6662_032618_SocialPreference_Class.mat', 'Mouse8894_100517_SocialPreference_Class.mat', 'Mouse8881_092917_SocialPreference_Class.mat', 'Mouse8893_090917_SocialPreference_Class.mat', 'Mouse8884_100617_SocialPreference_Class.mat', 'Mouse6992_060916_SocialPreference_Class.mat', 'Mouse5321_052616_SocialPreference_Class.mat', 'Mouse6662_040218_SocialPreference_Class.mat', 'Mouse0632_092517_SocialPreference_Class.mat', 'Mouse8882_092217_SocialPreference_Class.mat', 'Mouse6674_041118_SocialPreference_Class.mat', 'Mouse0634_092017_SocialPreference_Class.mat', 'Mouse8882_100217_SocialPreference_Class.mat', 'Mouse8894_091417_SocialPreference_Class.mat', 'Mouse0633_100417_SocialPreference_Class.mat', 'Mouse0631_100617_SocialPreference_Class.mat', 'Mouse0643_091917_SocialPreference_Class.mat', 'Mouse0632_091817_SocialPreference_Class.mat', 'Mouse0634_100917_SocialPreference_Class.mat', 'Mouse533L_070716_SocialPreference_Class.mat', 'Mouse6662_040418_SocialPreference_Class.mat', 'Mouse5333_060916_SocialPreference_Class.mat', 'Mouse6992_052416_SocialPreference_Class.mat', 'Mouse5331_070716_SocialPreference_Class.mat', 'Mouse6662_040918_SocialPreference_Class.mat', 'Mouse5333_061416_SocialPreference_Class.mat', 'Mouse6664_040218_SocialPreference_Class.mat', 'Mouse0634_092917_SocialPreference_Class.mat', 'Mouse6991_060216_SocialPreference_Class.mat', 'Mouse8893_091417_SocialPreference_Class.mat', 'Mouse0640_100517_SocialPreference_Class.mat', 'Mouse5321_060716_SocialPreference_Class.mat', 'Mouse0644_101017_SocialPreference_Class.mat', 'Mouse0631_092017_SocialPreference_Class.mat', 'Mouse8893_091617_SocialPreference_Class.mat', 'Mouse6664_040618_SocialPreference_Class.mat', 'Mouse0632_092917_SocialPreference_Class.mat', 'Mouse533L_052616_SocialPreference_Class.mat', 'Mouse8884_100417_SocialPreference_Class.mat', 'Mouse0631_092217_SocialPreference_Class.mat', 'Mouse5321_060216_SocialPreference_Class.mat', 'Mouse0643_091417_SocialPreference_Class.mat', 'Mouse8882_091117_SocialPreference_Class.mat', 'Mouse8893_092617_SocialPreference_Class.mat', 'Mouse8884_092917_SocialPreference_Class.mat', 'Mouse6662_033018_SocialPreference_Class.mat', 'Mouse0632_100617_SocialPreference_Class.mat', 'Mouse8891_100317_SocialPreference_Class.mat', 'Mouse5321_060916_SocialPreference_Class.mat', 'Mouse6674_041618_SocialPreference_Class.mat', 'Mouse6662_041118_SocialPreference_Class.mat', 'Mouse0641_091917_SocialPreference_Class.mat', 'Mouse533L_060216_SocialPreference_Class.mat', 'Mouse6664_032618_SocialPreference_Class.mat', 'Mouse0642_091617_SocialPreference_Class.mat', 'Mouse8893_093017_SocialPreference_Class.mat', 'Mouse6992_060716_SocialPreference_Class.mat', 'Mouse6991_070716_SocialPreference_Class.mat', 'Mouse8884_091117_SocialPreference_Class.mat', 'Mouse0644_093017_SocialPreference_Class.mat', 'Mouse0640_100717_SocialPreference_Class.mat', 'Mouse8891_090917_SocialPreference_Class.mat', 'Mouse5332_060216_SocialPreference_Class.mat', 'Mouse0642_093017_SocialPreference_Class.mat', 'Mouse0634_091817_SocialPreference_Class.mat', 'Mouse6990_060916_SocialPreference_Class.mat', 'Mouse0634_092517_SocialPreference_Class.mat', 'Mouse0644_100517_SocialPreference_Class.mat', 'Mouse6992_053116_SocialPreference_Class.mat', 'Mouse8891_091617_SocialPreference_Class.mat', 'Mouse8884_092217_SocialPreference_Class.mat', 'Mouse5321_061416_SocialPreference_Class.mat', 'Mouse8881_092217_SocialPreference_Class.mat', 'Mouse0633_092017_SocialPreference_Class.mat', 'Mouse0633_092917_SocialPreference_Class.mat', 'Mouse0640_092617_SocialPreference_Class.mat', 'Mouse0632_100417_SocialPreference_Class.mat', 'Mouse8891_092617_SocialPreference_Class.mat', 'Mouse533L_061416_SocialPreference_Class.mat', 'Mouse0633_092217_SocialPreference_Class.mat', 'Mouse6674_040418_SocialPreference_Class.mat', 'Mouse0631_100217_SocialPreference_Class.mat', 'Mouse0632_091517_SocialPreference_Class.mat', 'Mouse0631_100417_SocialPreference_Class.mat', 'Mouse6674_040618_SocialPreference_Class.mat', 'Mouse0633_091517_SocialPreference_Class.mat', 'Mouse6990_060716_SocialPreference_Class.mat', 'Mouse8881_091117_SocialPreference_Class.mat', 'Mouse0631_091817_SocialPreference_Class.mat', 'Mouse0642_092117_SocialPreference_Class.mat', 'Mouse6674_040218_SocialPreference_Class.mat', 'Mouse5321_061616_SocialPreference_Class.mat', 'Mouse8893_091217_SocialPreference_Class.mat', 'Mouse0634_092217_SocialPreference_Class.mat', 'Mouse8882_092517_SocialPreference_Class.mat', 'Mouse5331_070516_SocialPreference_Class.mat', 'Mouse6990_061416_SocialPreference_Class.mat', 'Mouse8882_092917_SocialPreference_Class.mat', 'Mouse0640_091417_SocialPreference_Class.mat', 'Mouse0642_100717_SocialPreference_Class.mat', 'Mouse5331_060716_SocialPreference_Class.mat', 'Mouse6990_052616_SocialPreference_Class.mat', 'Mouse6991_052416_SocialPreference_Class.mat', 'Mouse0644_092317_SocialPreference_Class.mat', 'Mouse6674_032618_SocialPreference_Class.mat', 'Mouse6662_041618_SocialPreference_Class.mat', 'Mouse8884_091517_SocialPreference_Class.mat', 'Mouse8891_093017_SocialPreference_Class.mat', 'Mouse0633_100917_SocialPreference_Class.mat', 'Mouse0643_092617_SocialPreference_Class.mat', 'Mouse8894_091617_SocialPreference_Class.mat', 'Mouse0630_091517_SocialPreference_Class.mat', 'Mouse8893_092317_SocialPreference_Class.mat', 'Mouse8882_100417_SocialPreference_Class.mat', 'Mouse0630_100217_SocialPreference_Class.mat', 'Mouse0640_092317_SocialPreference_Class.mat', 'Mouse6990_061616_SocialPreference_Class.mat', 'Mouse8882_091317_SocialPreference_Class.mat', 'Mouse0632_100217_SocialPreference_Class.mat', 'Mouse0630_092217_SocialPreference_Class.mat', 'Mouse0642_091917_SocialPreference_Class.mat', 'Mouse8884_100217_SocialPreference_Class.mat', 'Mouse6992_061616_SocialPreference_Class.mat', 'Mouse0644_100717_SocialPreference_Class.mat', 'Mouse0642_091417_SocialPreference_Class.mat', 'Mouse0641_100517_SocialPreference_Class.mat', 'Mouse0632_092017_SocialPreference_Class.mat', 'Mouse0631_092517_SocialPreference_Class.mat', 'Mouse8894_093017_SocialPreference_Class.mat', 'Mouse0631_092917_SocialPreference_Class.mat', 'Mouse699L_060216_SocialPreference_Class.mat', 'Mouse6991_060716_SocialPreference_Class.mat', 'Mouse0633_100217_SocialPreference_Class.mat', 'Mouse8884_092517_SocialPreference_Class.mat', 'Mouse0633_100617_SocialPreference_Class.mat', 'Mouse6674_041318_SocialPreference_Class.mat', 'Mouse5332_052616_SocialPreference_Class.mat', 'Mouse8891_091417_SocialPreference_Class.mat', 'Mouse6664_040918_SocialPreference_Class.mat', 'Mouse699L_061416_SocialPreference_Class.mat', 'Mouse8891_100517_SocialPreference_Class.mat', 'Mouse0641_100317_SocialPreference_Class.mat', 'Mouse0634_091517_SocialPreference_Class.mat', 'Mouse6662_040618_SocialPreference_Class.mat', 'Mouse0630_100617_SocialPreference_Class.mat', 'Mouse8894_092617_SocialPreference_Class.mat', 'Mouse8894_092317_SocialPreference_Class.mat', 'Mouse6992_052616_SocialPreference_Class.mat', 'Mouse8882_090817_SocialPreference_Class.mat', 'Mouse8893_100517_SocialPreference_Class.mat', 'Mouse5333_060716_SocialPreference_Class.mat', 'Mouse533L_070516_SocialPreference_Class.mat', 'Mouse8884_090817_SocialPreference_Class.mat', 'Mouse8882_100617_SocialPreference_Class.mat', 'Mouse0634_100617_SocialPreference_Class.mat', 'Mouse0630_092517_SocialPreference_Class.mat', 'Mouse0632_092217_SocialPreference_Class.mat', 'Mouse0642_100317_SocialPreference_Class.mat', 'Mouse5333_070516_SocialPreference_Class.mat', 'Mouse699L_053116_SocialPreference_Class.mat', 'Mouse8894_100717_SocialPreference_Class.mat', 'Mouse8881_100217_SocialPreference_Class.mat', 'Mouse0641_091417_SocialPreference_Class.mat', 'Mouse0630_091817_SocialPreference_Class.mat', 'Mouse6991_052616_SocialPreference_Class.mat', 'Mouse0633_091817_SocialPreference_Class.mat', 'Mouse0642_101017_SocialPreference_Class.mat', 'Mouse0641_092117_SocialPreference_Class.mat', 'Mouse0641_092617_SocialPreference_Class.mat', 'Mouse5332_061416_SocialPreference_Class.mat', 'Mouse699L_070516_SocialPreference_Class.mat', 'Mouse0641_100717_SocialPreference_Class.mat', 'Mouse0632_100917_SocialPreference_Class.mat', 'Mouse0640_091617_SocialPreference_Class.mat', 'Mouse5333_060216_SocialPreference_Class.mat', 'Mouse699L_052616_SocialPreference_Class.mat', 'Mouse5332_052416_SocialPreference_Class.mat', 'Mouse6664_040418_SocialPreference_Class.mat', 'Mouse0641_091617_SocialPreference_Class.mat', 'Mouse8881_091317_SocialPreference_Class.mat', 'Mouse0643_092317_SocialPreference_Class.mat', 'Mouse5332_070716_SocialPreference_Class.mat', 'Mouse6992_060216_SocialPreference_Class.mat', 'Mouse0633_092517_SocialPreference_Class.mat', 'Mouse6674_040918_SocialPreference_Class.mat', 'Mouse6662_041318_SocialPreference_Class.mat', 'Mouse6991_070516_SocialPreference_Class.mat', 'Mouse0634_100417_SocialPreference_Class.mat', 'Mouse533L_060716_SocialPreference_Class.mat', 'Mouse8891_092317_SocialPreference_Class.mat', 'Mouse5321_053116_SocialPreference_Class.mat', 'Mouse0643_092117_SocialPreference_Class.mat', 'Mouse0633_091417_SocialPreference_Class.mat', 'Mouse8881_090817_SocialPreference_Class.mat', 'Mouse8893_100317_SocialPreference_Class.mat', 'Mouse0644_091617_SocialPreference_Class.mat', 'Mouse5331_061616_SocialPreference_Class.mat', 'Mouse0632_091417_SocialPreference_Class.mat', 'Mouse0642_092317_SocialPreference_Class.mat', 'Mouse0631_091417_SocialPreference_Class.mat', 'Mouse0642_100517_SocialPreference_Class.mat', 'Mouse0634_100217_SocialPreference_Class.mat', 'Mouse0630_100917_SocialPreference_Class.mat', 'Mouse0643_100317_SocialPreference_Class.mat', 'Mouse0630_091417_SocialPreference_Class.mat', 'Mouse0643_100717_SocialPreference_Class.mat', 'Mouse5333_061616_SocialPreference_Class.mat', 'Mouse0643_093017_SocialPreference_Class.mat', 'Mouse0644_092117_SocialPreference_Class.mat', 'Mouse6992_070716_SocialPreference_Class.mat', 'Mouse5333_070716_SocialPreference_Class.mat', 'Mouse0630_100417_SocialPreference_Class.mat', 'Mouse6664_041118_SocialPreference_Class.mat', 'Mouse8884_091317_SocialPreference_Class.mat', 'Mouse0634_091417_SocialPreference_Class.mat', 'Mouse0641_101017_SocialPreference_Class.mat', 'Mouse0644_100317_SocialPreference_Class.mat', 'Mouse8894_100317_SocialPreference_Class.mat', 'Mouse533L_053116_SocialPreference_Class.mat', 'Mouse0644_091917_SocialPreference_Class.mat', 'Mouse5331_061416_SocialPreference_Class.mat', 'Mouse5332_061616_SocialPreference_Class.mat', 'Mouse8894_090917_SocialPreference_Class.mat', 'Mouse5332_053116_SocialPreference_Class.mat', 'Mouse8893_100717_SocialPreference_Class.mat', 'Mouse6991_061616_SocialPreference_Class.mat', 'Mouse0630_092017_SocialPreference_Class.mat', 'Mouse8891_091217_SocialPreference_Class.mat', 'Mouse8881_091517_SocialPreference_Class.mat', 'Mouse5331_052416_SocialPreference_Class.mat', 'Mouse0640_092117_SocialPreference_Class.mat', 'Mouse699L_052416_SocialPreference_Class.mat', 'Mouse5331_053116_SocialPreference_Class.mat', 'Mouse8882_091517_SocialPreference_Class.mat', 'Mouse0640_093017_SocialPreference_Class.mat', 'Mouse8891_100717_SocialPreference_Class.mat', 'Mouse0631_100917_SocialPreference_Class.mat', 'Mouse0640_101017_SocialPreference_Class.mat', 'Mouse0640_100317_SocialPreference_Class.mat', 'Mouse0641_092317_SocialPreference_Class.mat', 'Mouse0641_093017_SocialPreference_Class.mat', 'Mouse6664_041318_SocialPreference_Class.mat', 'Mouse6664_041618_SocialPreference_Class.mat', 'Mouse8894_091217_SocialPreference_Class.mat', 'Mouse8881_092517_SocialPreference_Class.mat', 'Mouse5333_052616_SocialPreference_Class.mat', 'Mouse0640_091917_SocialPreference_Class.mat', 'Mouse5333_052416_SocialPreference_Class.mat', 'Mouse6991_061416_SocialPreference_Class.mat', 'Mouse6991_060916_SocialPreference_Class.mat', 'Mouse5332_070516_SocialPreference_Class.mat', 'Mouse0644_092617_SocialPreference_Class.mat', 'Mouse8881_100417_SocialPreference_Class.mat', 'Mouse0631_091517_SocialPreference_Class.mat', 'Mouse0644_091417_SocialPreference_Class.mat', 'Mouse5332_060716_SocialPreference_Class.mat', 'Mouse0630_092917_SocialPreference_Class.mat', 'Mouse0643_100517_SocialPreference_Class.mat', 'Mouse0643_091617_SocialPreference_Class.mat', 'Mouse0642_092617_SocialPreference_Class.mat', 'Mouse6992_070516_SocialPreference_Class.mat', 'Mouse0643_101017_SocialPreference_Class.mat', 'Mouse699L_070716_SocialPreference_Class.mat'
]

PROPER_LABEL_FILES = list(set([x for x in proper_s_label_files+proper_o_label_files if x in proper_s_label_files and x in proper_o_label_files]))




def load_lfp_data_matrix(raw_data_path, raw_file_name, keys_of_interest, num_channels_in_samples, sample_freq=1000, 
                         cutoff=LOW_PASS_CUTOFF, lowcut=LOWCUT, highcut=HIGHCUT, mad_threshold=DEFAULT_MAD_TRESHOLD, 
                         q=Q, order=ORDER, apply_notch_filters=True, filter_type="lowpass"):
    # get raw-data, replacing outlying values with np.nan
    raw_data = scio.loadmat(os.path.join(raw_data_path, raw_file_name))
    raw_data = {key:raw_data[key].reshape(-1).astype(float) for key in keys_of_interest}
    raw_data = mark_outliers(
        raw_data, 
        sample_freq, 
        cutoff=cutoff, 
        lowcut=lowcut, 
        highcut=highcut, 
        mad_threshold=mad_threshold, 
        filter_type=filter_type
    )

    # combine raw data into a single matrix
    raw_data_combined = filter_signal(
        raw_data[keys_of_interest[0]], 
        sample_freq, 
        cutoff=cutoff, 
        lowcut=lowcut, 
        highcut=highcut, 
        q=q, 
        order=order, 
        apply_notch_filters=apply_notch_filters, 
        filter_type=filter_type
    ).reshape(1,-1)
    for key in keys_of_interest[1:]:
        raw_data_combined = np.vstack([
            raw_data_combined, 
            filter_signal(
                raw_data[key], 
                sample_freq, 
                cutoff=cutoff, 
                lowcut=lowcut, 
                highcut=highcut, 
                q=q, 
                order=order, 
                apply_notch_filters=apply_notch_filters, 
                filter_type=filter_type
            ).reshape(1,-1)
        ])

    assert raw_data_combined.shape[0] == num_channels_in_samples # sanity check
    return raw_data_combined


def determine_keys_of_interest(files_to_process, raw_data_path):
    keys_of_interest = set()
    keys_to_remove = set()
    for i, raw_file_name in enumerate(files_to_process):
        raw_data = scio.loadmat(os.path.join(raw_data_path, raw_file_name))
        raw_data_useful_keys = [x for x in raw_data.keys() if "__" not in x]
        if i == 0:
            keys_of_interest = set(raw_data_useful_keys)
        else:
            for key in keys_of_interest:
                if key not in raw_data_useful_keys:
                    keys_to_remove.add(key)
    for key in keys_to_remove:
        keys_of_interest.remove(key)
    return sorted(list(keys_of_interest))


def preprocess_socPref_raw_lfps_for_windowed_training(lfp_data_path, label_data_path, preprocessed_data_save_path, post_processing_sample_freq, 
                                                  num_processed_samples=10000, sample_temp_window_size=1000, 
                                                  max_num_samps_per_preprocessed_file=100, sample_freq=1000, max_num_samp_attempts=10, 
                                                  cutoff=LOW_PASS_CUTOFF, lowcut=LOWCUT, highcut=HIGHCUT, mad_threshold=DEFAULT_MAD_TRESHOLD, 
                                                  q=Q, order=ORDER, apply_notch_filters=True, filter_type="lowpass"):
    RECORDING_DURRATION_STEPS = 10*60*sample_freq # 10-minute recordings sampled at sample_freq (in samps per sec)
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  START")
    assert sample_freq > post_processing_sample_freq
    downsampling_step_size = sample_freq // post_processing_sample_freq
    
    raw_lfp_files_to_load = []
    for lfp_file in os.listdir(lfp_data_path):
        if "_LFP" in lfp_file and ".mat" in lfp_file:
            for plf in PROPER_LABEL_FILES:
                if lfp_file[:23] == plf[:23]:
                    raw_lfp_files_to_load.append(lfp_file)
                    break
    raw_lfp_files_to_load = sorted(raw_lfp_files_to_load)
    
    behavioral_label_files_to_load = []
    for b, behavior_file in enumerate(os.listdir(label_data_path)):
        if "_Class" in behavior_file and ".mat" in behavior_file:
            for lfp_file in raw_lfp_files_to_load:
                if behavior_file[:23] == lfp_file[:23]:
                    behavioral_label_files_to_load.append(behavior_file)
                    break
    behavioral_label_files_to_load = sorted(behavioral_label_files_to_load)
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  len(raw_lfp_files_to_load) == ", len(raw_lfp_files_to_load))
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  raw_lfp_files_to_load == ", raw_lfp_files_to_load)
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  len(behavioral_label_files_to_load) == ", len(behavioral_label_files_to_load))
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  behavioral_label_files_to_load == ", behavioral_label_files_to_load, flush=True)

    unique_mice_names = list(set([x.split('_')[0] for x in raw_lfp_files_to_load]))
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  len(unique_mice_names) == ", len(unique_mice_names))
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  unique_mice_names == ", unique_mice_names, flush=True)
    random.shuffle(unique_mice_names)
    num_samps_per_mouse = num_processed_samples // len(unique_mice_names)
    num_samples_per_label_type = num_samps_per_mouse//2
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  num_samps_per_mouse == ", num_samps_per_mouse)
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  num_samples_per_label_type == ", num_samples_per_label_type, flush=True)

    keys_of_interest = determine_keys_of_interest(raw_lfp_files_to_load, lfp_data_path)
    num_channels_in_samples = len(keys_of_interest)
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: num_channels_in_samples == ", num_channels_in_samples)
    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: keys_of_interest == ", keys_of_interest, flush=True)

    print("\nsocPref.preprocess_socPref_raw_lfps_for_windowed_training: PROCESSING RAW FILES AND SAVING CLEANED SAMPLES\n", flush=True)
    for i, mouse_name in enumerate(unique_mice_names):
        print("\nprocessing mouse samples: now processing files for mouse i==", i, " of n==", len(unique_mice_names), " with mouse_name == ", mouse_name, flush=True)
        mouse_lfp_files = [x for x in raw_lfp_files_to_load if mouse_name in x]
        mouse_label_files = [x for x in behavioral_label_files_to_load if mouse_name in x]
        total_soc_samples_for_curr_mouse = 0
        total_obj_samples_for_curr_mouse = 0
        if len(mouse_lfp_files) == len(mouse_label_files):
            soc_samples = []
            obj_samples = []
            soc_subset_counter = 0
            obj_subset_counter = 0
            for recording_ind, (curr_lfp_file_name, curr_class_file_name) in enumerate(zip(mouse_lfp_files, mouse_label_files)):
                assert curr_lfp_file_name[:23] == curr_class_file_name[:23] # ensure files are properly aligned/matched
                print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: pairing curr_lfp_file_name == ", curr_lfp_file_name, " with curr_class_file_name == ", curr_class_file_name, flush=True)

                # read in data
                curr_mat_file_data = scio.loadmat(os.path.join(label_data_path, curr_class_file_name))
                start_time_step = sample_freq * int(curr_mat_file_data['StartTime'])
                raw_data_combined = load_lfp_data_matrix(
                    lfp_data_path, 
                    curr_lfp_file_name, 
                    keys_of_interest, 
                    num_channels_in_samples, 
                    sample_freq=sample_freq, 
                    cutoff=cutoff, 
                    lowcut=lowcut, 
                    highcut=highcut, 
                    mad_threshold=mad_threshold, 
                    q=q, 
                    order=order, 
                    apply_notch_filters=apply_notch_filters, 
                    filter_type=filter_type
                )
                raw_data_combined = raw_data_combined[:,start_time_step:start_time_step+RECORDING_DURRATION_STEPS]

                # split data by labels ----------------------------
                soc_class_labels_by_time_step = curr_mat_file_data['S_Class'][0,start_time_step:start_time_step+RECORDING_DURRATION_STEPS]
                obj_class_labels_by_time_step = curr_mat_file_data['O_Class'][0,start_time_step:start_time_step+RECORDING_DURRATION_STEPS]
                
                soc_zero_locs = [l for l in range(len(soc_class_labels_by_time_step)) if np.isnan(soc_class_labels_by_time_step[l])]
                obj_zero_locs = [l for l in range(len(obj_class_labels_by_time_step)) if np.isnan(obj_class_labels_by_time_step[l])]
                lfp_nan_locs = [l for l in range(raw_data_combined.shape[1]) if np.isnan(np.sum(raw_data_combined[:, l]))]
                soc_exclude_locs = sorted(list(set(soc_zero_locs + lfp_nan_locs)))
                obj_exclude_locs = sorted(list(set(obj_zero_locs + lfp_nan_locs)))
                
                soc_sample_start_inds = draw_timesteps_to_sample_from_using_label_reference(
                    soc_class_labels_by_time_step, 
                    sample_temp_window_size, 
                    num_samples_per_label_type, 
                    soc_exclude_locs, 
                    max_num_draws=10
                )
                print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: len(soc_sample_start_inds) == ", len(soc_sample_start_inds), flush=True)
                obj_sample_start_inds = draw_timesteps_to_sample_from_using_label_reference(
                    obj_class_labels_by_time_step, 
                    sample_temp_window_size, 
                    num_samples_per_label_type, 
                    obj_exclude_locs, 
                    max_num_draws=10
                )
                print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: len(obj_sample_start_inds) == ", len(obj_sample_start_inds), flush=True)
                
                print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: DRAWING RANDOM SAMPLES", flush=True)
                for j in range(num_samples_per_label_type):
                    if j%10 == 0:
                        print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t sample j == ", j, " of ", num_samples_per_label_type)
                    # draw random sample from raw_data_combined
                    if j < len(soc_sample_start_inds) and total_soc_samples_for_curr_mouse < num_samples_per_label_type:
                        curr_soc_samp = raw_data_combined[:,soc_sample_start_inds[j]:soc_sample_start_inds[j]+sample_temp_window_size]
                        curr_soc_samp = np.transpose(curr_soc_samp, axes=(1,0)) # ensure all samples are of shape (num_time_steps, num_channels) - see https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
                        if np.isnan(np.sum(curr_soc_samp)):
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t NAN VALUE DETECTED IN SAMPLED SIGNAL")
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t soc_sample_start_inds[j] == ", soc_sample_start_inds[j])
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t soc_sample_start_inds[j]+sample_temp_window_size == ", soc_sample_start_inds[j]+sample_temp_window_size)
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t [x for x in soc_exclude_locs if x>=soc_sample_start_inds[j] and x<=soc_sample_start_inds[j]+sample_temp_window_size] == ", [x for x in soc_exclude_locs if x>=soc_sample_start_inds[j] and x<=soc_sample_start_inds[j]+sample_temp_window_size])
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t ENDING SAMPLE COLLECTION ASSOCIATED WITH CURRENT FILE DUE TO NAN DETECTION - curr_lfp_file_name == ", curr_lfp_file_name, flush=True)
                            break # prevent more nan-samples from being drawn
                        curr_soc_label = np.array([1.,0.])
                        if downsampling_step_size > 1:
                            curr_soc_samp = curr_soc_samp[::downsampling_step_size, :]
                        soc_samples.append([curr_soc_samp, curr_soc_label])
                        total_soc_samples_for_curr_mouse += 1

                    if j < len(obj_sample_start_inds) and total_obj_samples_for_curr_mouse < num_samples_per_label_type:
                        curr_obj_samp = raw_data_combined[:,obj_sample_start_inds[j]:obj_sample_start_inds[j]+sample_temp_window_size]
                        curr_obj_samp = np.transpose(curr_obj_samp, axes=(1,0)) # ensure all samples are of shape (num_time_steps, num_channels) - see https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
                        if np.isnan(np.sum(curr_obj_samp)):
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t NAN VALUE DETECTED IN SAMPLED SIGNAL")
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t obj_sample_start_inds[j] == ", obj_sample_start_inds[j])
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t obj_sample_start_inds[j]+sample_temp_window_size == ", obj_sample_start_inds[j]+sample_temp_window_size)
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t [x for x in obj_exclude_locs if x>=obj_sample_start_inds[j] and x<=obj_sample_start_inds[j]+sample_temp_window_size] == ", [x for x in obj_exclude_locs if x>=obj_sample_start_inds[j] and x<=obj_sample_start_inds[j]+sample_temp_window_size])
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t ENDING SAMPLE COLLECTION ASSOCIATED WITH CURRENT FILE DUE TO NAN DETECTION - curr_lfp_file_name == ", curr_lfp_file_name, flush=True)
                            break # prevent more nan-samples from being drawn
                        curr_obj_label = np.array([0.,1.])
                        if downsampling_step_size > 1:
                            curr_obj_samp = curr_obj_samp[::downsampling_step_size, :]
                        obj_samples.append([curr_obj_samp, curr_obj_label])
                        total_obj_samples_for_curr_mouse += 1
                    
                    # periodically save sample sets to external files so that full dataset need not be loaded all at once
                    if j+1 == max_num_samps_per_preprocessed_file or j == num_samples_per_label_type-1:
                        
                        if len(soc_samples) > 0:
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t now saving soc_subset_counter==", soc_subset_counter)
                            curr_soc_set_name = mouse_name+"_social_processed_data_subset_"+str(soc_subset_counter)+".pkl"
                            with open(os.path.join(preprocessed_data_save_path, curr_soc_set_name), 'wb') as outfile:
                                pkl.dump(soc_samples, outfile)
                            del soc_samples
                            soc_samples = []
                            soc_subset_counter += 1
                        
                        if len(obj_samples) > 0:
                            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t now saving obj_subset_counter==", obj_subset_counter)
                            curr_obj_set_name = mouse_name+"_object_processed_data_subset_"+str(obj_subset_counter)+".pkl"
                            with open(os.path.join(preprocessed_data_save_path, curr_obj_set_name), 'wb') as outfile:
                                pkl.dump(obj_samples, outfile)
                            del obj_samples
                            obj_samples = []
                            obj_subset_counter += 1
                            
                        pass # end periodic save if statement
                    pass # end loop for saving random samples
                
                del raw_data_combined
                del soc_class_labels_by_time_step
                del obj_class_labels_by_time_step
                del soc_zero_locs
                del obj_zero_locs
                del lfp_nan_locs
                del soc_exclude_locs
                del obj_exclude_locs
                del soc_sample_start_inds
                del obj_sample_start_inds
                
                if total_soc_samples_for_curr_mouse == num_samples_per_label_type and len(soc_samples)==0 and total_obj_samples_for_curr_mouse == num_samples_per_label_type and len(obj_samples)==0:
                    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: \t breaking loop now that all samples have been collected", flush=True)
                    break
                pass # end loop over files corresponding to mice id
        else:
            print("socPref.preprocess_socPref_raw_lfps_for_windowed_training: IGNORING ANY FILES ASSOCIATED WITH mouse_name == ", mouse_name, " DUE TO MISMATCH IN CONTENTS OF mouse_lfp_files == ", mouse_lfp_files, " AND mouse_label_files == ", mouse_label_files, flush=True)
            
        print("total_soc_samples_for_curr_mouse == ", total_soc_samples_for_curr_mouse)
        print("total_obj_samples_for_curr_mouse == ", total_obj_samples_for_curr_mouse, flush=True)
        pass # end loop over mice ids

    print("socPref.preprocess_socPref_raw_lfps_for_windowed_training:  STOP", flush=True)
    pass



if __name__ == '__main__':
    parse=argparse.ArgumentParser(description='preprocess tst data')
    parse.add_argument(
        "-cached_args_file",
        default="socialPreference_100HzLP_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()

    print("__MAIN__: LOADING VALUES OF ARGUMENTS FOUND IN ", args.cached_args_file, flush=True)
    with open(args.cached_args_file, 'r') as infile:
        new_args_dict = json.load(infile)
        args.original_lfp_data_path = new_args_dict["original_lfp_data_path"]
        args.original_behavioral_label_data_path = new_args_dict["original_behavioral_label_data_path"]
        args.save_path = new_args_dict["save_path"]
        args.num_processed_samples = int(new_args_dict["num_processed_samples"])
        args.sample_temp_window_size = int(new_args_dict["sample_temp_window_size"]) # the number of time steps to capture in each time series recording window in final preprocessed dataset
        args.max_num_samps_per_preprocessed_file = int(new_args_dict["max_num_samps_per_preprocessed_file"])
        args.sample_freq = int(new_args_dict["sample_freq"]) # this represents the frequency in Hz that recordings were made at

        args.post_processing_sample_freq = int(new_args_dict["post_processing_sample_freq"]) # this represents the frequency (in Hz) that the final, preprocessed samples will be in
        args.max_num_samp_attempts = int(new_args_dict["max_num_samp_attempts"])
        args.cutoff = None if new_args_dict["cutoff"]=="None" else float(new_args_dict["cutoff"])
        args.lowcut = None if new_args_dict["lowcut"]=="None" else float(new_args_dict["lowcut"]) 
        args.highcut = None if new_args_dict["highcut"]=="None" else float(new_args_dict["highcut"]) 
        args.mad_threshold = float(new_args_dict["mad_threshold"]) 
        args.q = float(new_args_dict["q"]) 
        args.order = int(new_args_dict["order"]) 
        args.apply_notch_filters = True if new_args_dict["apply_notch_filters"]=="True" else False
        args.filter_type = new_args_dict["filter_type"]
    
    preprocess_socPref_raw_lfps_for_windowed_training(
        args.original_lfp_data_path, 
        args.original_behavioral_label_data_path, 
        args.save_path, 
        args.post_processing_sample_freq, 
        num_processed_samples=args.num_processed_samples, 
        sample_temp_window_size=args.sample_temp_window_size, 
        max_num_samps_per_preprocessed_file=args.max_num_samps_per_preprocessed_file, 
        sample_freq=args.sample_freq, 
        max_num_samp_attempts=args.max_num_samp_attempts, 
        cutoff=args.cutoff, 
        lowcut=args.lowcut, 
        highcut=args.highcut, 
        mad_threshold=args.mad_threshold, 
        q=args.q, 
        order=args.order, 
        apply_notch_filters=args.apply_notch_filters, 
        filter_type=args.filter_type
    )
    print("__MAIN__: FINISHED !!!!!!", flush=True)
    pass