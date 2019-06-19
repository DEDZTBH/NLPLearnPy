import time
from bert_base.client import BertClient

with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
    start_t = time.perf_counter()
    str = '职位描述：独立负责客户的项目需求沟通，整理项目简报，撰写策划方案；独立管理客户项目，把控项目执行进度，确保项目执行质量。'
    rst = bc.encode([str, str])
    print('rst:', rst)
    print(time.perf_counter() - start_t)