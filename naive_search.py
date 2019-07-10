#! usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas

from extract_util import valid_label

data = pandas.read_csv("data/tiny_label.csv", delimiter="	", index_col="jd.jd_id")
# remove duplicate index
data = data.loc[~data.index.duplicated(keep='first')]
data = data[['desc_clean', 'labels']]

data['labels'] = [list(filter(valid_label, eval(l))) for l in data['labels']]

labels_set = set()
for label_list in data['labels']:
    for label in label_list:
        labels_set.add(label)

# str = '职位描述： 1.软件测试工程师主要职责是测试公司自主研发的各种软件产品，提供详细的测试结果； 2.能够学习相关的主流技术，结合产品进行测试； 3.能够根据客户需求，完成测试用例的编写和执行，同时能够按照测试计划，完成测试任务。 职位要求： 1.本科及以上学历，计算机或软件专业，对黑盒测试感兴趣，愿意从事测试工作； 2.逻辑清晰，表达清楚；有较强的责任心，抗压能力和团队合作精神，快速学习和适应环境能力； 3.具备一定的计算机基础知识，对操作系统和数据库有基础的了解和认识； 4英语四级以上，能够阅读英文文档，英语或日语熟练表达使用者优先； 5.对有编程经验，了解VM，SharePoint，Online Service，SQL Server，或者有过相关工作经验者优先。'

# found_result = set(filter(lambda l: l in str, labels_set))

# new_found = {'黑盒', '日语', '编程', '软件测试', 'SQL', 'VM', '责任心', '操作系统'} - found_result

# print(new_found)

magic = {'写作能力', '项目管理', '英语口语', '信息处理', '责任心', '地球物理'}

print(magic - labels_set)
