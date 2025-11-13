import os
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def draw_all():
    # 设置字体为 Times New Roman，字号和字体加粗
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 1  # 加粗边框

    dataset_num = ["A", "B", "C"]
    subjects_num = [8, 28, 30]

    # 创建画布，设置为三列排列
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1行3列

    # 设置标记形状和颜色
    markers = ['o', 's', '^', 'D']  # 圆形、正方形、三角形、菱形
    colors = ['red', 'blue', 'green', 'purple']  # 红色、蓝色、绿色、紫色
    # 遍历数据集
    for i, dataset in enumerate(dataset_num):
        # 文件夹路径和模型列表
        base_path = f"save/{dataset}/LOSO"
        models = ["fNIRS-T", "fNIRS-PreT", 'CT-Net-old',"fNIRS_TTT_M", "fNIRS_TTT_L"]  # 添加其他模型名称
        subjects = range(1, subjects_num[i] + 1)  # 根据实际受试者数量调整

        # 存储所有模型的准确率，以便动态设置刻度
        all_accuracies = []

        # 遍历模型
        for j, model in enumerate(models):
            accuracies = []
            for subject in subjects:
                file_path = os.path.join(base_path, model, str(subject), "test_acc.txt")
                try:
                    with open(file_path, 'r') as f:
                        accuracy = float(f.read().strip())
                        accuracies.append(accuracy)
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    accuracies.append(None)  # 如果文件不存在，记录为空值

            # 绘制折线
            if model == "fNIRS_TTT_M": model = "fNIRS-TTT-M"
            if model == "fNIRS_TTT_L": model = "fNIRS-TTT-L"
            if model == "CT-Net-old" : model = "CT-Net"
            axes[i].plot(subjects, accuracies, 
                          marker=markers[j % len(markers)], 
                          linestyle='-', 
                          color=colors[j % len(colors)], 
                          label=model)
            
            # 保存所有模型的准确率
            all_accuracies.extend(accuracies)

        # 根据所有准确率动态设置纵轴刻度
        if all_accuracies:  # 确保有有效的准确率数据
            min_accuracy = np.nanmin(all_accuracies)
            max_accuracy = np.nanmax(all_accuracies)
            
            # 确定适当的范围和步长
            ticks = np.arange(np.floor(min_accuracy / 10) * 10, np.ceil(max_accuracy / 10) * 10 + 1, 10)
            axes[i].set_yticks(ticks)  # 设置y轴刻度

            # 设置主刻度和次刻度样式
            axes[i].tick_params(axis='y', which='major', length=8, width=1.5)
            axes[i].tick_params(axis='y', which='minor', length=4, width=1)
            axes[i].set_yticks(np.arange(min(ticks) + 5, max(ticks), 5), minor=True)  # 次要刻度

        # 设置图例、标签和标题
        axes[i].set_xlabel("Subject", fontsize=16, fontweight='bold')
        axes[i].set_ylabel("Classification Accuracy (%)", fontsize=16, fontweight='bold')
        axes[i].set_title(f"Dataset {dataset}", fontsize=18, fontweight='bold')
        axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, 0.19), ncol=2, fontsize=12, edgecolor='black')

        # 保留完整的黑色边框（包括顶部和右侧）
        axes[i].spines['top'].set_color('black')
        axes[i].spines['right'].set_color('black')
        axes[i].spines['bottom'].set_color('black')
        axes[i].spines['left'].set_color('black')

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig("loso_accuracy_all_datasets.tiff")
    plt.savefig("loso_accuracy_all_datasets.png",dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_all()
