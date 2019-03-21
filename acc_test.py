# -*- coding: utf-8 -*-
"""

@author: wgshun

"""
from __future__ import division
import os
import sys
if 2 == sys.version_info[0]:
    reload(sys)
    sys.setdefaultencoding('utf-8')

cou_res = open('count_result.txt', 'w')
tr_count = 0
fa_count = 0
tr_count1 = []
fa_count1 = []
root_dir = 'count/'
label_tr = {}
label_fa = {}
for i in os.listdir(root_dir):
    te = open(os.path.join(root_dir, i), 'r')
    te_list = te.readlines()
    for tl in te_list:
        if tl.strip().split('#')[-1] == 'true':
	    tr_count1.append(tl.strip().split('#')[1])
	    tr_count += int(tl.strip().split('#')[2])
	    if label_tr.get(tl.strip().split('#')[1]) == None:
	        label_tr[tl.strip().split('#')[1]] = int(tl.strip().split('#')[2])
	    else:
		label_tr[tl.strip().split('#')[1]] = int(tl.strip().split('#')[2]) + label_tr[tl.strip().split('#')[1]]
        else:
	    fa_count1.append(tl.strip().split('#')[1])
	    fa_count += int(tl.strip().split('#')[2])
	    if label_fa.get(tl.strip().split('#')[1]) == None:
	        label_fa[tl.strip().split('#')[1]] = int(tl.strip().split('#')[2])
	    else:
		label_fa[tl.strip().split('#')[1]] = int(tl.strip().split('#')[2]) + label_fa[tl.strip().split('#')[1]]

print('----------基于图片准确率----------')
cou_res.write('----------基于图片准确率----------\n')
print 'picture true:', tr_count
cou_res.write(('picture true:'+str(tr_count)+'\n'))

print 'picture false:', fa_count
cou_res.write(('picture false:'+str(fa_count)+'\n'))

acc = float(tr_count/(tr_count+fa_count))
print('picture acc:%f' % acc)
cou_res.write(('picture acc:'+str(acc)+'\n'))

print('\n----------基于标签准确率----------')
temp1 = len(list(set(tr_count1)))
temp2 = len(list(set(fa_count1)))
cou_res.write('\n----------基于标签准确率----------\n')
print 'label true:', temp1
cou_res.write(('label true:'+str(temp1)+'\n'))

print 'label false:', temp2
cou_res.write(('label false:'+str(temp2)+'\n'))

acc1 = float(temp1/(temp1+temp2))
print('label acc:%f' % acc1)
cou_res.write(('label acc:'+str(acc1)+'\n'))

# print('\n---------每个类别的准确率---------')
cou_res.write('\n---------每个类别的准确率---------\n')
for i in label_tr.keys():
    if label_fa.get(i) != None:
        label_ac = i+':'+str(label_tr[i])+'/'+str(label_fa[i]+label_tr[i])+'='+str(label_tr[i]/(label_fa[i]+label_tr[i]))
        # print label_ac
	cou_res.write(label_ac+'\n')
    else:
	label_ac1 = i+':'+str(label_tr[i])+'/'+str(label_tr[i])+'='+str(label_tr[i]/label_tr[i])
	# print label_ac1
	cou_res.write(label_ac1+'\n')

for i in label_fa.keys():
    if label_tr.get(i) == None:
	label_ac2 = i+':'+str(0)+'/'+str(label_fa[i])+'='+str(0.0)
	# print label_ac2
	cou_res.write(label_ac2+'\n')



