import collections
import random
import re

import numpy as np
from tqdm import tqdm

BASE_DIR = "./track1/"  # 数据根目录
TRAIN_LEN = 100000  # 训练和测试所用数据量
PROPORTION = 0.8  # 训练集占总数据量的多少
INT_PATTERN = "^-?[0-9]+$"
BASE_CATAGORY = 21  # 对catagory计量时的基
MIN_USERID = 100044 # 最小user_id
MAX_USERID = 2421043 # 最大user_id
USER_NUM = 1392873 # user总数
ITEM_NUM = 4710 # item总数

class DataProcessor:

    user_tag_dict = [[]] * USER_NUM  # 存储 rec_log_train.txt 里的信息, 顺序为itemid，res，time
    item_dict = {}  # 存储 item.txt 里的信息，顺序为catagory，tags
    user_dict = []  # 存储 user_profile.txt 里的信息, 顺序为birth，gender，tweetnum，tags
    user_action_dict = {}  # 存储 user_action.txt 里的信息，顺序为at, re, co
    user_sns_dict = {}  # 存储 user_sns.txt 里的信息
    user_key_dict = {}  # 存储 user_key_word.txt 里的信息
    user_index = [0] * (MAX_USERID - MIN_USERID) # user_id到index的映射
    tag_user_dict = {} # 关注该key的user

    def strList2intList(self, strList, correctNum):
        # 将string的list转成int的list存储以缩小内存，若不合int形式则用correctNum代替。
        ret = []
        for ch in strList:
            if re.match(INT_PATTERN, ch):
                ret.append(int(ch))
            else:
                ret.append(correctNum)
        return ret

    def str2int(self, str, correctNum):
        # 将string转成int存储以缩小内存，若不合int形式则用correctNum代替。
        if re.match(INT_PATTERN, str):
            return int(str)
        else:
            return correctNum

    def __init__(self):

        # 读 user_profile.txt
        # 任务：1. 由于user_id有序，故建立user_id到index的映射（数组占用率差不多50%，感觉还行），并直接用数组存储数据
        print('loading user_profile.txt')
        user_profile_txt = open(BASE_DIR + "user_profile.txt")
        for i, user_profile_line in enumerate(tqdm(user_profile_txt)):
            # user_profile_line = user_profile_txt.readline()
            if not user_profile_line:
                break
            else:
                # 根据\t分割
                user_profile_msg = user_profile_line.split('\t')
                # 存储用户信息及映射
                user_index[int(user_profile_msg[0])] = i
                self.user_dict.append([self.str2int(user_profile_msg[1], 2000), self.str2int(
                    user_profile_msg[2], 0), self.str2int(user_profile_msg[3], 0), set(self.strList2intList(user_profile_msg[4].split(';'), 0))])

        # 读 rec_log_train.txt
        print('loading rec_log_train.txt')
        rec_log_train_txt = open(BASE_DIR + "rec_log_train.txt")
        for i, train_line in enumerate(tqdm(rec_log_train_txt)):
            if train_line:
                # 根据\t分割
                train_msg = train_line.split('\t')
                # 存储所有正向数据
                if train_msg[2] == '1':
                    self.user_tag_dict[self.user_index[int(train_msg[0])]].append(
                        [int(train_msg[1]), int(train_msg[2]), int(train_msg[3])})]
                    self.tag_user_dict.setdefault(int(train_msg[1]), [])
                    self.tag_user_dict(int(train_msg[1])).append(int(train_msg[0]))
                # 由于原数据集中有大约92%的负向数据，故为了保持正负向数据的均衡，以10%的概率随机挑选负向数据
                else:
                    ran = random.randint(0, 9)
                    if ran == 0:
                        self.user_tag_dict[self.user_index[int(train_msg[0])]].append(
                            [int(train_msg[1]), int(train_msg[2]), int(train_msg[3])})]
            else:
                break

        # 读 item.txt
        print('loading item.txt')
        item_txt = open(BASE_DIR + "item.txt")
        for i, item_line in enumerate(tqdm(item_txt)):
            # item_line = item_txt.readline()
            if not item_line:
                break
            else:
                # 根据\t分割
                item_msg = item_line.split('\t')
                # 存储分类目录和相关关键词
                self.item_dict.setdefault(int(item_msg[0]), [])
                self.item_dict[int(item_msg[0])] = [self.strList2intList(item_msg[1].split('.'), 0), set(self.strList2intList(item_msg[2].split(';'), 0))]

        # 读 user_action.txt
        print('loading user_action.txt')
        user_action_txt = open(BASE_DIR + 'user_action.txt')
        for i, user_action_line in enumerate(tqdm(user_action_txt)):
            # user_action_line = user_action_txt.readline()
            if not user_action_line:
                break
            else:
                user_action_msg = user_action_line.split('\t')
                hash = int(user_action_msg[0]) * USER_NUM + int(user_action_msg[1])
                self.user_action_dict.setdefault(hash, [])
                self.user_action_dict[hash] = [int(user_action_msg[2], int(user_action_msg[3], int(user_action_msg[4]]

        # 读 user_sns.txt
        print('loading user_sns.txt')
        user_sns_txt = open(BASE_DIR + 'user_sns.txt')
        for i, user_sns_line in enumerate(tqdm(user_sns_txt)):
            user_sns_line = user_sns_txt.readline()
            if not user_sns_line:
                break
            else:
                user_sns_msg = user_sns_line.split('\t')
                self.user_sns_dict.setdefault(int(user_sns_msg[0]), [])
                self.user_sns_dict[int(user_sns_msg[0])].append(
                    int(user_sns_msg[1]))

        # 读 user_key_word.txt
        print('loading user_keyword.txt')
        user_key_word_txt = open(BASE_DIR + 'user_key_word.txt')
        for i, user_key_word_line in enumerate(tqdm(user_key_word_txt)):
            user_key_word_line = user_key_word_txt.readline()
            if not user_key_word_line:
                break
            else:
                user_key_word_msg = user_key_word_line.split('\t')
                key_words = user_key_word_msg[1].split(';')
                for kw in key_words:
                    kw_split = kw.split(':')
                    self.user_key_dict.setdefault(
                        int(user_key_word_msg[0]), {})
                    (self.user_key_dict[int(user_key_word_msg[0])])[
                        int(kw_split[0])] = float(kw_split[1])

    # 计算标签的关键字和用户的关键字之间的重合度
    # 计算方法是：二者交集中的所有元素的权重之和 / 二者并集大小

    def get_key_overlap(self, key_weight_dict, item_key_set):
        ans = 0.0
        for key in key_weight_dict:
            if key in item_key_set:
                ans += float(key_weight_dict[key])

        element_num = len(set(key_weight_dict_key) | set(item_key_set))
        return ans / element_num

    # 从训练数据找到所有关注该标签的用户
    def get_user_by_tag(self, item):
        users = []
        for key in self.user_tag_dict:
            # 避免多次访问self.user_tag_dict
            tempDict1 = self.user_tag_dict[key]
            temp11 = self.user_tag_dict.get(key, [])
            for index in range(len(temp11)):
                if (tempDict1[index]['itemid'] == item):
                    users.append(key)
        return users

    # 计算兴趣标签的重合度
    # 计算方法是：二者的交集大小 / 二者的并集大小
    def get_tag_overlap(self, temp7, user):
        # 避免重复访问
        temp2 = temp7['tags']
        temp3 = self.user_dict[user]['tags']
        return len(temp2 & temp3) * 1.0 / len(temp2 | temp3)

    # 归一化sigmoid函数
    def sigmoid(self, n):
        return 1.0 / (1 + np.exp(-n))

    # 计算分类的重合度
    # 居然有不够四层的，辣鸡腾讯
    def get_tag_value(self, temp10):
        item_catagory = temp10[0]
        ret = 0
        for i in range(len(item_catagory)):
            ret = ret * BASE_CATAGORY
            ret = ret + item_catagory[i]
        return ret

    # 获得所有特征值
    def get_feature(self, key, key_weight_dict, item):
        # key: 训练集中的用户编号
        # key_weight_dict: 用户关键词及权重的字典
        # index: 训练集中该用户关注标签情况列表的下标
        # item: 训练集中的标签
        key = int(key)
        item = int(item)
        temp10 = self.item_dict[item]
        item_key_set = temp10[1]

        # 特征1：标签的关键字和用户的关键字之间的重合度
        key_overlap = self.get_key_overlap(key_weight_dict, item_key_set)

        # 特征2：用户与所有关注该关键字的用户的关联度
        # 该特征由一些子特征计算得出，参数的设置需要依赖网络，这里给出所有需要的子特征
        # 先挑出所有关注该标签的用户
        user_tag_list = self.get_user_by_tag(item)
        user_tag_list_len = len(user_tag_list)

        # 子特征2.1：该用户与这些用户之间兴趣标签的重合度的平均值
        tag_overlap = 0.0
        # 子特征2.2：该用户关注的人中这些用户所占的比例
        followee_portion = 0.0
        # 子特征2.3：关注该用户的人中这些用户所占的比例
        follower_portion = 0.0
        # 子特征2.4：该用户@，转发，评论这些用户的数量的平均值
        at_user = 0.0
        re_user = 0.0
        co_user = 0.0
        # 子特征2.5：这些用户@，转发，评论该用户的数量的平均值
        user_at = 0.0
        user_re = 0.0
        user_co = 0.0
        tag_value = 0

        temp7 = self.user_dict[key]
        if user_tag_list_len > 0:
            for user in user_tag_list:
                user = int(user)
                # 子特征2.1
                tag_overlap = tag_overlap + self.get_tag_overlap(temp7, user)
                # 子特征2.2
                # 只要用了两次以上的都暂存起来
                temp6 = self.user_sns_dict.get(key, [])
                if user in temp6:
                    followee_portion = followee_portion + 1
                # 子特征2.3
                if key in self.user_sns_dict.get(user, []):
                    follower_portion = follower_portion + 1
                # 子特征2.4
                if self.user_action_dict.get(key, {}).get(user, []):
                    # 避免多次访问
                    temp4 = (self.user_action_dict[key])[user]
                    at_user = at_user + \
                        int(temp4['at'])
                    re_user = re_user + \
                        int(temp4['re'])
                    co_user = co_user + \
                        int(temp4['co'])

                # 子特征2.5
                if self.user_action_dict.get(user, {}).get(key, []):
                    temp5 = (self.user_action_dict[user])[key]
                    user_at = user_at + \
                        int(temp5['at'])
                    user_re = user_re + \
                        int(temp5['re'])
                    user_co = user_co + \
                        int(temp5['co'])

            tag_overlap = tag_overlap / user_tag_list_len if user_tag_list_len != 0 else 0.0
            followee_portion = followee_portion / \
                len(self.user_sns_dict[key]) if len(temp6) != 0 else 0.0
            follower_sum = 0.00
            for u in self.user_sns_dict:
                if key in self.user_sns_dict[u]:
                    follower_sum = follower_sum + 1
            follower_portion = follower_portion / follower_sum if follower_sum != 0 else 0.0
            at_user = self.sigmoid(at_user / user_tag_list_len)
            re_user = self.sigmoid(re_user / user_tag_list_len)
            co_user = self.sigmoid(co_user / user_tag_list_len)
            user_at = self.sigmoid(user_at / user_tag_list_len)
            user_re = self.sigmoid(user_re / user_tag_list_len)
            user_co = self.sigmoid(user_co / user_tag_list_len)

            # 特征3：标签本身的特性
            # 将标签的分类树映射到一个值，保证两个标签分类上越接近，值就越接近。
            tag_value = self.sigmoid(self.get_tag_value(temp10))

        # 特征四：用户本身的特性
        # 即用户的出生年份，性别和发微博数量
        birth = self.sigmoid(temp7[0])
        gender = temp7[1]
        tweetnum = self.sigmoid(temp7[2])

        return [key_overlap, tag_overlap, followee_portion, follower_portion, at_user, re_user, co_user, user_at, user_re, user_co, tag_value, birth, gender, tweetnum]

    def write_dataset(self):
        with open('train.csv', 'w') as out:
            for i in tqdm(range(TRAIN_LEN)):
                key = random.choice(list(self.user_tag_dict.keys()))
                print(key)
                # key_weight_dict = self.user_key_dict[key]
                outs = []
                if self.user_key_dict.__contains__(key):
                    key_weight_dict = self.user_key_dict[key]
                else:
                    continue
                for temp12 in tqdm(self.user_tag_dict[key]):
                    # temp12 = (self.user_tag_dict[key])[index]
                    item = int(temp12['itemid'])
                    res = int(temp12['res'])
                    features = self.get_feature(
                        key, key_weight_dict, item)

                    vals = []

                    for i in range(0, len(features)):
                        vals.append("{0:.6f}".format(features[i]).rstrip('0')
                                    .rstrip('.'))

                    vals = ','.join(vals)
                    outs.append(','.join([vals, str(res)]) + '\n')
                out.write(''.join(outs))
