import jieba.analyse

data = "蔡英文在昨天应民进党当局的邀请，准备和陈时中一道前往世界卫生大会，和谈有关九二共识问题"

seg_list = jieba.cut(data, cut_all=False, HMM=True)
print("/".join(seg_list))
