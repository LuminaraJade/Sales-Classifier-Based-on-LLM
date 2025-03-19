###初始化设定和随机种子
import os
os.environ["CUDAVISIBLE_DEVICES"] = "0, 1"

import torch
import numpy as np
import pandas as pd
import random
import json
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_dataset

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


### 1. 构造数据集
# 读取数据集和json
data_path = 'gsj/QWEN2.5Classification/datav1_feature15.xlsx'
description_path = 'gsj/QWEN2.5Classification/description.json'
# 加载属性描述的 JSON 文件
with open(description_path, 'r', encoding='utf-8') as f:
    attribute_description = json.load(f)


# 初始化x_data和y_data
x_data = []
y_data = []

# 读取Excel数据
df = pd.read_excel(data_path)



# 定义属性列表（确保顺序与需求一致）
attributes = [
    "tb_push",
    "scenario_1",
    "tb_pay_money",
    "tb_cvr",
    "reserve_price",
    "weekly_browse_ratio",
    "daily_browse_ratio",
    "daily_browse_cnt",
    "monthly_browse_ratio",
    "final_promotion_price",
    "shop_title",
    "feature1",
    "tb_ctr",
    "level_one_category_name",
    "category_name",
    "zk_final_price",
    "convert_rate"
]

# 遍历每一行数据
for index, row in df.iterrows():
    # 构造消息
    message = (
        "你是一名淘宝商品推荐工作人员。你的职责是判断商品是否容易被用户接受,即容易被卖出去。商品有17个属性,分别是:"
        + ", ".join(attributes)
        + "\n"
        + "<tb_push>第1个属性名称是“淘宝推送”。指通过淘宝平台向用户推送的某个品类所有商品的次数。推送策略可能基于用户行为、偏好和历史数据进行优化。数据越高越好.\n"
        + "<scenario_1>第2个属性名称是“场景1”。表示商品推荐所处的特定应用场景或情境,为分类变量。不同的场景可能对应不同的用户需求和推荐策略,例如烘焙表示该商品主要用于烹饪场景\n"
        + "<tb_pay_money>第3个属性名称是“淘宝支付金额”。指用户在淘宝平台上为某商品支付的金额。这一指标可以反映商品的实际销售额,帮助评估商品的经济价值和受欢迎程度。\n"
        + "<tb_cvr>第4个属性名称是转化率（Conversion Rate, CVR）是指某一个二级品类的所有商品在用户点击推荐广告后,完成购买动作的比率。它是通过将完成购买动作的用户数除以点击广告的用户数来计算的。\n"
        + "<reserve_price>第5个属性名称是“保留价格”。通常指商品的最低售出价格,或卖家设置的底价。在拍卖或促销活动中,保留价格是确保商品不低于某一价格出售的机制。\n"
        + "<weekly_browse_ratio>第6个属性名称是“周浏览比例”。表示商品在一周内被浏览的次数占总浏览次数的比例。这有助于分析商品的周度受欢迎程度和用户关注度的变化趋势。\n"
        + "<daily_browse_ratio>第7个属性名称是“日浏览比例”。表示商品在一天内被浏览的次数占总浏览次数的比例。用于监控商品的日常受欢迎程度,识别高峰期和低谷期。\n"
        + "<daily_browse_cnt>第8个属性名称是“每日浏览次数”。指商品每天被浏览的总次数。直接反映商品的日活跃度和曝光率。\n"
        + "<monthly_browse_ratio>第9个属性名称是“月浏览比例”。表示商品在一个月内被浏览的次数占总浏览次数的比例。用于分析商品的月度趋势和长期受欢迎程度。\n"
        + "<final_promotion_price>第10个属性名称是“最终促销价格”。指商品在促销活动结束后的最终销售价格。这一指标可以用于评估促销活动的效果以及商品的价格敏感度。\n"
        + "<shop_title>第11个属性名称是“店铺标题”。是卖家店铺的名称或标题。店铺标题通常包含品牌名称、主营类目或特色,影响用户对店铺的第一印象和点击意愿。\n"
        + "<feature1>第12个属性名称是“特征1”。这是一个通用的特征名称,具体含义需根据数据集的定义确定。通常用于表示商品的某一特定属性或用户的某一行为特征。\n"
        + "<tb_ctr>第13个属性名称是“淘宝点击率”。点击率（Click-Through Rate）表示商品展示后被点击的比例。在淘宝平台上,CTR 用于衡量推荐商品的吸引力和广告效果。\n"
        + "<level_one_category_name>第14个属性名称是“一级分类名称”。指商品所属的一级分类名称,如“服装”、“电子产品”、“家居”等。分类信息有助于推荐系统进行类别过滤和相关性匹配。\n"
        + "<category_name>第15个属性名称是“二级分类名称”。指商品所属的二级品类名称,如“牙膏”、“手机”、“家居”等。分类信息有助于推荐系统进行类别过滤和相关性匹配。\n"
        + "<zk_final_price>第16个属性名称是“折扣最终价格”。指商品在折扣或优惠活动后的最终销售价格。反映了促销力度和用户购买的价格优惠程度。\n"
        + "<convert_rate>第17个属性名称是“转化率”。表示转化率,即浏览商品后实际完成购买的比例。高转化率说明商品具有较强的吸引力和购买动机,是评估推荐效果的重要指标。\n"
        + "上述属性中,tb_cvr,tb_ctr,tb_push,tb_pay_money表示与商品二级品类相关的统计信息。\n"
        + "上述属性中,weekly_browse_ratio,daily_browse_ratio,daily_browse_cnt,monthly_browse_ratio表示商品随时间维度的统计信息。\n"
        + "上述属性中,final_promotion_price,shop_title,feature1,zk_final_price,convert_rate, reserve_price, scenario_1表示与商品二级品类相关的统计信息。\n"
        + "上述属性中,level_one_category_name,category_name表示与商品二级品类相关的统计信息。\n"
        + "用户输入为 \n"  #请告诉我你是如何一步一步推理得到结果的。 # type: ignore
        + f"<tb_push>:淘宝推送的值为:{row['tb_push']}\n"
        + f"<scenario_1> 场景1是:{row['scenario_1']}\n"
        + f"<tb_pay_money> 淘宝支付金额的值为:{row['tb_pay_money']}\n"
        + f"<reserve_price> 保留价格的值为:{row['reserve_price']}\n"
        + f"<tb_cvr> 转化率:{row['tb_cvr']}\n"
        + f"<weekly_browse_ratio> 周浏览比例的值为:{row['weekly_browse_ratio']}\n"
        + f"<daily_browse_ratio> 日浏览比例的值为:{row['daily_browse_ratio']}\n"
        + f"<daily_browse_cnt> 每日浏览次数的值为:{row['daily_browse_cnt']}\n"
        + f"<monthly_browse_ratio> 月浏览比例的值为:{row['monthly_browse_ratio']}\n"
        + f"<final_promotion_price> 最终促销价格为:{row['final_promotion_price']}\n"
        + f"<shop_title> 店铺标题为:{row['shop_title']}\n"
        + f"<feature1> 特征1为:{row['feature1']}\n"
        + f"<tb_ctr> 淘宝点击率为:{row['tb_ctr']}\n"
        + f"<level_one_category_name> 一级分类名称为:{row['level_one_category_name']}\n"
        + f"<category_name> 二级分类名称为:{row['category_name']}\n"
        + f"<zk_final_price> 折扣最终价格的值为:{row['zk_final_price']}\n"
        + f"<convert_rate> 转化率为:{row['convert_rate']}\n"
        # + "<END>"
    )
    
    # 根据'label_x'的值分配到x_data或y_data
    label = row['lable_x']
    if label == 0:
        x_data.append({
            "content": message,
            "label": 0,
            "标注类别": "负例"
        })
    elif label == 1:
        y_data.append({
            "content": message,
            "label": 1,
            "标注类别": "正例"
        })
    else:
        print(f"第{index}行的'label_x'值为无效值: {label}")

# 输出结果（可选）
print(f"x_data 共有 {len(x_data)} 条记录")
print(f"y_data 共有 {len(y_data)} 条记录")


def save_json(path, data):
    # 将Python字典转换为JSON字符串
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


save_json('gsj/QWEN2.5Classification/data/goods_train.json', x_data[:800]+y_data[:800])
save_json('gsj/QWEN2.5Classification/data/goods_valid.json', x_data[800:900]+y_data[800:900])
save_json('gsj/QWEN2.5Classification/data/goods_test.json', x_data[9:]+y_data[9:])