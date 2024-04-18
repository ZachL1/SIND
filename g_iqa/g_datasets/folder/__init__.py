from .PIQ23_base import PIQ23Folder # 0

# IQA datasets
from .SPAQ import SPAQFolder # 100
from .LIVEChallenge import LIVEChallengeFolder # 200
from .Koniq_10k import Koniq_10kFolder # 300
from .BID import BIDFolder # 400

# IAA datasets
from .PARA import PARAFolder # 1000
from .EVA import EVAFolder # 1100

__all__ = [
    'PIQ23Folder',
    'SPAQFolder',
    'LIVEChallengeFolder',
    'Koniq_10kFolder',
    'BIDFolder',
    'PARAFolder',
    'EVAFolder'
]