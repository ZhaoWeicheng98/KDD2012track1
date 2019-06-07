import random
import collections
import numpy as np
from tqdm import tqdm
BASE_DIR = "./track1/"  # 数据根目录
TRAIN_LEN = 50000000  # 训练和测试所用数据量
PROPORTION = 0.8  # 训练集占总数据量的多少


class DataProcessor:

    user_tag_dict = {}  # 存储 rec_log_train.txt 里的信息
    item_dict = {}  # 存储 item.txt 里的信息
    user_dict = {}  # 存储 user_profile.txt 里的信息
    user_action_dict = {}  # 存储 user_action.txt 里的信息
    user_sns_dict = {}  # 存储 user_sns.txt 里的信息
    user_key_dict = {}  # 存储 user_key_word.txt 里的信息

    def __init__(self):
        # 读 rec_log_train.txt
        print('loading rec_log_train.txt')
        rec_log_train_txt = open(BASE_DIR + "rec_log_train.txt")
        for i in tqdm(range(TRAIN_LEN)):
            train_line = rec_log_train_txt.readline()
            if train_line:
                # 根据\t分割
                train_msg = train_line.split('\t')
                # 存储所有正向数据
                if train_msg[2] == '1':
                    if not self.user_tag_dict.__contains__(train_msg[0]):
                        self.user_tag_dict[train_msg[0]] = []
                    self.user_tag_dict[train_msg[0]].append(
                        {'itemid': int(train_msg[1]), 'res': int(train_msg[2]), 'time': int(train_msg[3])})
                # 由于原数据集中有大约92%的负向数据，故为了保持正负向数据的均衡，以10%的概率随机挑选负向数据
                else:
                    ran = random.randint(0, 9)
                    if ran == 0:
                        if not self.user_tag_dict.__contains__(train_msg[0]):
                            self.user_tag_dict[train_msg[0]] = []
                        self.user_tag_dict[train_msg[0]].append(
                            {'itemid': int(train_msg[1]), 'res': int(train_msg[2]), 'time': int(train_msg[3])})
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
                if not self.item_dict.__contains__(item_msg[0]):
                    self.item_dict[item_msg[0]] = []
                self.item_dict[item_msg[0]].append(
                    {'catagory': item_msg[1].split('.'), 'tags': set(item_msg[2].split(';'))})

        # 读 user_profile.txt
        print('loading user_profile.txt')
        user_profile_txt = open(BASE_DIR + "user_profile.txt")
        for i, user_profile_line in enumerate(tqdm(user_profile_txt)):
            # user_profile_line = user_profile_txt.readline()
            if not user_profile_line:
                break
            else:
                # 根据\t分割
                user_profile_msg = user_profile_line.split('\t')
                # 存储用户信息
                if not self.user_dict.__contains__(user_profile_msg[0]):
                    self.user_dict[user_profile_msg[0]] = {}
                self.user_dict[user_profile_msg[0]] = {'birth': user_profile_msg[1], 'gender': user_profile_msg[2],
                                                       'tweetnum': user_profile_msg[3], 'tags': set(user_profile_msg[4].split(';'))}

        # 读 user_action.txt
        print('loading user_action.txt')
        user_action_txt = open(BASE_DIR + 'user_action.txt')
        for i, user_action_line in enumerate(tqdm(user_action_txt)):
            # user_action_line = user_action_txt.readline()
            if not user_action_line:
                break
            else:
                user_action_msg = user_action_line.split('\t')
                if not self.user_action_dict.__contains__(user_action_msg[0]):
                    self.user_action_dict[user_action_msg[0]] = {}
                (self.user_action_dict[user_action_msg[0]])[user_action_msg[1]] = {
                    'at': user_action_msg[2], 're': user_action_msg[3], 'co': user_action_msg[4]}

        # 读 user_sns.txt
        print('loading user_sns.txt')
        user_sns_txt = open(BASE_DIR + 'user_sns.txt')
        for i, user_sns_line in enumerate(tqdm(user_sns_txt)):
            user_sns_line = user_sns_txt.readline()
            if not user_sns_line:
                break
            else:
                user_sns_msg = user_sns_line.split('\t')
                if not self.user_sns_dict.__contains__(user_sns_msg[0]):
                    self.user_sns_dict[user_sns_msg[0]] = []
                self.user_sns_dict[user_sns_msg[0]].append(user_sns_msg[1])

        # 读 user_key_word.txt
        print('loading user_keyword.txt')
        user_key_word_txt = open(BASE_DIR + 'user_key_word.txt')
        for i, user_key_word_line in enumerate(tqdm(user_key_word_txt )):
            user_key_word_line = user_key_word_txt.readline()
            if not user_key_word_line:
                break
            else:
                user_key_word_msg = user_key_word_line.split('\t')
                key_words = user_key_word_msg[1].split(';')
                if not self.user_key_dict.__contains__(user_key_word_msg[0]):
                    self.user_key_dict[user_key_word_msg[0]] = {}
                for kw in key_words:
                    kw_split = kw.split(':')
                    (self.user_key_dict[user_key_word_msg[0]])[
                        kw_split[0]] = kw_split[1]

    # 计算标签的关键字和用户的关键字之间的重合度
    # 计算方法是：二者交集中的所有元素的权重之和 / 二者并集大小

    def get_key_overlap(self, key_weight_dict, item_key_set):
        ans = 0.0
        for key in key_weight_dict:
            if key in item_key_set:
                ans += float(key_weight_dict[key])
        element_num = len(set(key_weight_dict.keys()) | item_key_set)
        return ans / element_num

    # 从训练数据找到所有关注该标签的用户
    def get_user_by_tag(self, item):
        users = []
        for key in range(len(self.user_tag_dict)):
            if(self.user_tag_dict[key]['itemid'] == item):
                users.append(key)
        return users

    # 计算兴趣标签的重合度
    # 计算方法是：二者的交集大小 / 二者的并集大小
    def get_tag_overlap(self, key, user):
        return len(self.user_dict[key] & self.user_dict[user]) * 1.0 / len(self.user_dict[key] | self.user_dict[user])

    # 归一化sigmoid函数
    def sigmoid(self, n):
        return 1.0 / (1 + np.exp(-n))

    # 获得所有特征值
    def get_feature(self, key, key_weight_dict, index, item):
        item_key_set = self.item_dict[item]['tags']

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

        if user_tag_list_len > 0:
            for user in user_tag_list:
                # 子特征2.1
                tag_overlap = tag_overlap + self.get_tag_overlap(key, user)
                # 子特征2.2
                if user in self.user_sns_dict[key]:
                    followee_portion = followee_portion + 1
                # 子特征2.3
                if key in self.user_sns_dict[user]:
                    follower_portion = follower_portion + 1
                # 子特征2.4
                if (self.user_action_dict[key])[user]:
                    at_user = at_user + \
                        int((self.user_action_dict[key])[user]['at'])
                    re_user = re_user + \
                        int((self.user_action_dict[key])[user]['re'])
                    co_user = co_user + \
                        int((self.user_action_dict[key])[user]['co'])
                # 子特征2.5
                if (self.user_action_dict[user])[key]:
                    user_at = user_at + \
                        int((self.user_action_dict[user])[key]['at'])
                    user_re = user_re + \
                        int((self.user_action_dict[user])[key]['re'])
                    user_co = user_co + \
                        int((self.user_action_dict[user])[key]['co'])

            tag_overlap = tag_overlap / user_tag_list_len
            followee_portion = followee_portion / len(self.user_sns_dict[key])
            follower_sum = 0.0
            for u in self.user_sns_dict:
                if key in self.user_sns_dict[u]:
                    follower_sum = follower_sum + 1
            follower_portion = follower_portion / follower_sum
            at_user = self.sigmoid(at_user / user_tag_list_len)
            re_user = self.sigmoid(re_user / user_tag_list_len)
            co_user = self.sigmoid(co_user / user_tag_list_len)
            user_at = self.sigmoid(user_at / user_tag_list_len)
            user_re = self.sigmoid(user_re / user_tag_list_len)
            user_co = self.sigmoid(user_co / user_tag_list_len)
            return [key_overlap, tag_overlap, followee_portion, follower_portion, at_user, re_user, co_user, user_at, user_re, user_co]

    def write_dataset(self):
        with open('train.csv', 'w') as out:
            for key in self.user_tag_dict:
                if self.user_key_dict.__contains__(key):
                    key_weight_dict = self.user_key_dict[key]
                else:
                    continue
                for index in range(len(self.user_tag_dict[key])):
                    item = (self.user_tag_dict[key])[index]['itemid']
                    res = int((self.user_tag_dict[key])[index]['res'])
                    features = self.get_feature(
                        key, key_weight_dict, index, item)

                    vals = []
                    for i in range(0, len(features)):
                        vals.append("{0:.6f}".format(features[i]).rstrip('0')
                                    .rstrip('.'))

                    vals = ','.join(vals)
                    out.write(','.join([vals, res]) + '\n')
