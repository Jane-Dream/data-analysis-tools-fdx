import fasttext
import jieba
import re
import os


def process_words(text):
    words = jieba.cut(text)
    filter_words = []
    for w in words:
        if len(w) < 2:
            continue
        if re.match("^[0-9]+$", w):
            continue
        if re.match("\d+.\d?", w):
            continue
        filter_words.append(w)
    return filter_words


def is_noise_data(predict, score):
    '''
    判断噪音召回模型的结果，是否属于无关数据
    :param predict:
    :param score:
    :return:
    '''
    if "__label__其他" in predict.split(':')[0] or "__label__" == predict.split(':')[0]:
        return True
    elif float(predict.split(':')[1]) < score:
        return True

    return False


def recall_fasttext_predict(model_path, seg_data_list):
    print(model_path)
    model = fasttext.load_model(model_path)
    print(model.labels)
    print()

    if isinstance(seg_data_list, list):
        results = model.predict(seg_data_list)
        print(results)
        out_list = []
        for labels, scores in zip(results[0], results[1]):
            out_list.append(labels[0] + ":" + str(scores[0]))
        return out_list

    elif isinstance(seg_data_list, str):
        print(seg_data_list)
        print()

        result = model.predict(seg_data_list)
        print("模型预测结果：", str(result))
        return result[0] + ":" + str(result[1])
    else:
        print("seg data type error")


if __name__ == '__main__':
    model_name = 'ft_20231107_ts_correction_auto_seg_0.8.bin'
    model_path = os.path.join('/data/sjb/jiaomengshu/analysis/model/online_model/' + model_name)

    data = "".join(process_words(
        "投诉他，平台会处理，退款不退货。我现在终于明白，在网上买东西时，为什么商家要把，一斤重的商品分成两个半斤，两个包装了。 其实原因很简单，就是每个包装袋里，都有一个100多克的干燥剂。这样算来，一斤的商品就能少装200克。 最近我在拼夕夕，买了一斤 夏威夷果花了32块钱，选的最大果。到货后发现是两个袋子，每个袋是250克，当我打开时真的被惊到了。 果子很小不说，里面还有一包干燥剂，拿在手上还挺有份量，打开另一袋也有同样的干燥剂。两袋干燥剂就有200多克。 这样算下来，买一斤夏威夷果，除去干燥剂就只有不到八两，而且这还不算坏果和空壳的。 真是买的不如卖的精，不得不说，现在的商家真是用心良苦，不择手段。你说气不气人，这些无良商家真是太可恨了"))
    seg_data = ["习 就 这 德性 还 模仿 毛贼 东 呢 他 连 毛 的 包皮 垢 都 不如 ", data]

    results = recall_fasttext_predict(model_path, seg_data)
    for result in results:
        print(result)
        is_noise = is_noise_data(result, 0.9)
        print("是否为无关数据：", str(is_noise))
