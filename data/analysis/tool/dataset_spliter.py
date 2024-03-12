# coding:utf8
import re
import numpy as np

class DataSetSpliter(object):
    def split_trainset(self, data, label_name):
        '''
        数据集划分
        :param data:
        :param label_name:
        :return:
        '''
        # 生成类别与数据list的字典
        label_to_data_list = {}
        for index, row in data.iterrows():
            if (row['data'] is np.nan) or (len(row['data']) <= 5):
                continue
            if row[label_name] in label_to_data_list:
                label_to_data_list[row[label_name]].append(row['data'])
            else:
                label_to_data_list[row[label_name]] = [row['data']]

        trainset = []
        devset = []
        testset = []

        # 数据均衡
        # pos_num = len(label_to_data_list[1])
        # label_to_data_list[0] = label_to_data_list[0][0:pos_num * 2]

        try:
            for k, v in label_to_data_list.items():
                data_size = len(v)
                if data_size <= 10:
                    for content in v:
                        trainset.append(content + '\t' + str(k) + '\n')
                    continue

                train_size = int(0.8 * data_size)
                dev_size = int(0.8 * data_size)
                for content in v[0:train_size]:
                    trainset.append(content + '\t' + str(k) + '\n')
                # for content in v[train_size:dev_size]:
                #     devset.append(content + '\t' + str(k) + '\n')
                for content in v[dev_size:data_size]:
                    testset.append(content + '\t' + str(k) + '\n')
        except:
            print(content)
            raise Exception

        return trainset, devset, testset

    def write_file(self, path, datas):
        with open(path, 'w') as out:
            for data in datas:
                out.write(data)

    def read_file(self, path):
        datas = []
        with open(path, 'r') as file:
            for data in file:
                text = data.split("\t")[0]
                text = re.sub("[0-9a-zA-Z]+", "", text)
                label = data.split("\t")[1]
                datas.append(text + "\t" + label)
        return datas
