import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
 
x_values= [1,2,3,4,5,6,7,8,9,10]
cdt=[81.12,82.19,82.82,82.01,81.31,81.03,81.66,81.74,81.66,82.01]
asp_gcn = [73.70,77.73,78.09,80.67,80.94,80.05,80.49,80.76,79.86,77.90]
asp_bilstm = [80.6,80.6,80.6,80.6,80.6,80.6,80.6,80.6,80.6,80.6]
plt.plot(x_values,cdt,c='red',label="CDT")
plt.plot(x_values,asp_gcn,c='blue',label="ASP_GCN")
plt.plot(x_values,asp_bilstm,c='purple',label="ASP-BILSTM")
#plt.tick_params(axis='both',which='major',labelsize=14)
plt.xlabel('number of GCN layers',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
x_major_locator=MultipleLocator(1)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
#plt.xlim(-0.5,11)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(72,84)
plt.legend()
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
plt.show()
