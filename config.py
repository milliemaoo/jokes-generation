import os
# 参数配置
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAXLEN = 20
REVERSE = False
teacher_forcing_ratio = 0  # 强制学习阈值 1为一定强制学习